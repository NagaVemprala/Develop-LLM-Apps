import json
from typing import Literal, TypedDict
# from dotenv import load_dotenv  # Commented out: using Streamlit st.secrets instead of .env
from pydantic import BaseModel, Field
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
    match_score: int = Field(default=5, ge=1, le=10)
    matched_keywords: list[str] = Field(default_factory=list)


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
    raw_payload = json.loads(state["raw_results"])
    candidate_results = [
        {
            "url": r.get("url"),
            "title": r.get("title"),
            "content": (r.get("content") or "")[:2500],
        }
        for r in raw_payload.get("results", [])
        if is_specific_job_url(r.get("url", ""))
    ]

    if not candidate_results:
        return {"structured_jobs": []}

    prompt = (
        "You are parsing pre-filtered job posting pages returned by a web search.\n"
        "Extract ONE job listing object for EACH result below that represents a real job opening.\n\n"
        "Extraction & Scoring Guidelines:\n"
        "1. Do NOT filter out partial matches—extract all valid job postings so the human reviewer can decide.\n"
        "2. Ignore any 'Similar jobs' sidebar links inside a snippet; extract ONLY the primary job "
        "described by that result's `title` and `url`.\n"
        "3. Copy the exact `url` from the result item without modifying it.\n"
        "4. Assign a `match_score` from 1 to 10 based on how well the role title, seniority, work mode/location, "
        "and required skills match the User Search Context (9-10 = strong match across role, level, location, and "
        "keywords; 6-8 = solid match with minor gaps; 1-5 = adjacent role or seniority mismatch).\n"
        "5. Populate `matched_keywords` with the exact keywords from the user's criteria (or direct equivalents) "
        "that appear in the job's title or snippet.\n\n"
        f"User Search Context: {state['criteria'].model_dump_json()}\n\n"
        f"Candidate Job Pages ({len(candidate_results)} items):\n"
        f"{json.dumps(candidate_results, indent=2)}"
    )

    raw_batch: _RawJobBatch = structured_llm.invoke(prompt)

    seen_urls: set[str] = set()
    valid_jobs: list[JobListing] = []
    for item in raw_batch.jobs:
        if item.url not in seen_urls and is_specific_job_url(item.url):
            seen_urls.add(item.url)
            valid_jobs.append(JobListing(**item.model_dump()))

    # Sort highest match_score first
    valid_jobs.sort(key=lambda j: j.match_score, reverse=True)
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