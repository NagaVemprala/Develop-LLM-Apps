import json
from typing import Literal, TypedDict
# from dotenv import load_dotenv  # Commented out: using Streamlit st.secrets instead of .env
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from schemas import JobSearchRequest, JobListing, JobListingBatch

# load_dotenv()  # Commented out: keys are loaded via st.secrets in ui.py

DEFAULT_ATS_BOARDS = [
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "icims.com",
    "smartrecruiters.com",
    "builtin.com",
]

NOISE_DOMAINS = [
    "youtube.com",
    "simplilearn.com",
    "coursera.org",
    "udemy.com",
    "medium.com",
]

class AgentState(TypedDict):
    criteria: JobSearchRequest
    search_query: str
    raw_results: str
    structured_jobs: list[JobListing]
    approved_jobs: list[JobListing]
    action_summary: str
    attempt: int


llm = ChatOpenAI(model="gpt-6-luna")
structured_llm = llm.with_structured_output(JobListingBatch)

def search_tavily(state: AgentState) -> dict:
    c = state["criteria"]
    kw = " ".join(c.keywords)
    exp = c.experience_level or ""
    # Adding "apply" steers Tavily toward live postings rather than career advice articles
    query = f"{exp} {c.role} {c.location} {kw} job apply".strip()

    # Use explicit target_boards if provided; otherwise fall back to DEFAULT_ATS_BOARDS
    domains_to_include = c.target_boards if c.target_boards else DEFAULT_ATS_BOARDS

    tavily_tool = TavilySearch(
        max_results=10,
        topic="general",
        search_depth="advanced",
        include_domains=domains_to_include,
        exclude_domains=NOISE_DOMAINS,
    )
    response = tavily_tool.invoke({"query": query})
    return {
        "search_query": query,
        "raw_results": json.dumps(response, indent=2),
        "attempt": state.get("attempt", 0) + 1,
    }

def extract_jobs(state: AgentState) -> dict:
    prompt = (
        "Extract specific, active job postings from the Tavily search results below.\n"
        "Rules:\n"
        "1. Extract every real job opening mentioned—even if multiple jobs are listed inside a "
        "job-board search page snippet (such as Built In, SimplyHired, or ZipRecruiter).\n"
        "2. For each job, extract the job title, company name (use 'Unknown / Listed on Board' "
        "if not explicitly named), location, the result's URL, and a concise summary.\n"
        "3. Ignore pure career-advice articles, YouTube videos, and social media posts.\n"
        "4. Be flexible with keywords and seniority: include closely matching roles (e.g., Entry Level, "
        "Junior, Associate, or Analyst II) even if not every keyword appears in the short snippet.\n\n"
        f"Target Criteria: {state['criteria'].model_dump_json()}\n\n"
        f"Tavily Results:\n{state['raw_results']}"
    )
    batch: JobListingBatch = structured_llm.invoke(prompt)
    return {"structured_jobs": batch.jobs}

def human_review(
    state: AgentState,
) -> Command[Literal["search_tavily", "finalize_applications"]]:
    """Pauses for human review; routes either to finalization or back to search."""
    jobs = state["structured_jobs"]

    human_decision = interrupt({
        "attempt": state.get("attempt", 1),
        "search_query": state["search_query"],
        "jobs": [job.model_dump() for job in jobs],
    })

    if human_decision.get("decision") == "reject":
        updated_criteria = state["criteria"].model_copy(
            update={
                "keywords": human_decision.get("keywords", state["criteria"].keywords),
                "location": human_decision.get("location", state["criteria"].location),
                "target_boards": (
                    state["criteria"].target_boards
                    if human_decision.get("keep_board_filter", True)
                    else []
                ),
            }
        )
        return Command(
            goto="search_tavily",
            update={
                "criteria": updated_criteria,
                "structured_jobs": [],
            },
        )

    approved_indices: list[int] = human_decision.get("approved_indices", [])
    selected = [jobs[i] for i in approved_indices if 0 <= i < len(jobs)]
    return Command(
        goto="finalize_applications",
        update={"approved_jobs": selected},
    )

def finalize_applications(state: AgentState) -> dict:
    approved = state["approved_jobs"]
    if not approved:
        return {"action_summary": "No jobs were approved. Workflow ended."}

    companies = ", ".join(job.company for job in approved)
    return {
        "action_summary": f"Approved {len(approved)} job(s) on attempt #{state.get('attempt', 1)}: {companies}."
    }

workflow = StateGraph(AgentState)
workflow.add_node("search_tavily", search_tavily)
workflow.add_node("extract_jobs", extract_jobs)
workflow.add_node("human_review", human_review)
workflow.add_node("finalize_applications", finalize_applications)

workflow.add_edge(START, "search_tavily")
workflow.add_edge("search_tavily", "extract_jobs")
workflow.add_edge("extract_jobs", "human_review")
workflow.add_edge("finalize_applications", END)

checkpointer = MemorySaver()
job_search_graph = workflow.compile(checkpointer=checkpointer)