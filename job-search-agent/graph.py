import json
from typing import Literal, TypedDict
# from dotenv import load_dotenv  # Commented out: using Streamlit st.secrets instead of .env
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from schemas import (
    JobSearchRequest,
    JobListing,
    EnrichedJobDetails,
    EnrichedJobBatch,
    is_specific_job_url,
)

# load_dotenv()  # Commented out: keys are loaded via st.secrets in ui.py

DEFAULT_ATS_BOARDS = [
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "myworkdayjobs.com",
    "icims.com",
    "jobs.smartrecruiters.com",
    "linkedin.com/jobs/view",
]

NOISE_DOMAINS = [
    "youtube.com",
    "simplilearn.com",
    "coursera.org",
    "udemy.com",
    "medium.com",
    "simplyhired.com",
    "ziprecruiter.com",
]


# Intermediate unvalidated batch so one generic URL doesn't discard valid jobs in the same batch
class _RawJob(BaseModel):
    title: str
    company: str
    location: str
    url: str
    summary: str


class _RawJobBatch(BaseModel):
    jobs: list[_RawJob] = Field(default_factory=list)


class AgentState(TypedDict):
    criteria: JobSearchRequest
    search_query: str
    raw_results: str
    structured_jobs: list[JobListing]
    approved_jobs: list[JobListing]
    enriched_jobs: list[EnrichedJobDetails]
    action_summary: str
    attempt: int

llm = ChatOpenAI(model="gpt-6-luna")
structured_llm = llm.with_structured_output(_RawJobBatch)
enrichment_llm = llm.with_structured_output(EnrichedJobBatch)

def search_tavily(state: AgentState) -> dict:
    c = state["criteria"]
    kw = " ".join(c.keywords)
    exp = c.experience_level or ""
    query = f"{exp} {c.role} {c.location} {kw} apply".strip()

    domains_to_include = c.target_boards if c.target_boards else DEFAULT_ATS_BOARDS

    tavily_tool = TavilySearch(
        max_results=12,
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
        "Extract ONLY specific, single-role job postings from the Tavily search results below.\n"
        "Strict Rules:\n"
        "1. ONLY include results whose URL points directly to a single job application/description page "
        "(e.g., boards.greenhouse.io/company/jobs/123, jobs.lever.co/company/uuid, "
        "jobs.ashbyhq.com/company/uuid, linkedin.com/jobs/view/123, or a specific company career req URL).\n"
        "2. DO NOT extract jobs from generic search results or category listing URLs "
        "(e.g., linkedin.com/jobs/remote-..., builtin.com/jobs/remote/..., simplyhired.com/search?...). "
        "Never attach a search-index URL to a job listing.\n"
        "3. Ignore blog posts, career advice articles, and videos.\n\n"
        f"Target Criteria: {state['criteria'].model_dump_json()}\n\n"
        f"Tavily Results:\n{state['raw_results']}"
    )
    raw_batch: _RawJobBatch = structured_llm.invoke(prompt)

    # Programmatically enforce direct-job URL filtering and deduplicate by URL
    seen_urls: set[str] = set()
    valid_jobs: list[JobListing] = []
    for item in raw_batch.jobs:
        if item.url not in seen_urls and is_specific_job_url(item.url):
            seen_urls.add(item.url)
            valid_jobs.append(JobListing(**item.model_dump()))

    return {"structured_jobs": valid_jobs}

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