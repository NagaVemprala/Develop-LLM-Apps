import json
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
# from dotenv import load_dotenv  # Commented out: keys loaded via st.secrets in ui.py
from langchain_tavily import TavilySearch, TavilyExtract
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

# load_dotenv()

ATS_DOMAINS = [
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "myworkdayjobs.com",
    "icims.com",
    "jobs.smartrecruiters.com",
]

DEFAULT_ALL_BOARDS = ATS_DOMAINS + ["linkedin.com/jobs/view"]

NOISE_DOMAINS = [
    "youtube.com",
    "simplilearn.com",
    "coursera.org",
    "udemy.com",
    "medium.com",
    "simplyhired.com",
    "ziprecruiter.com",
]


def classify_source(url: str) -> str:
    lower_url = url.lower()
    if any(domain in lower_url for domain in ATS_DOMAINS):
        return "ATS"
    if "linkedin.com" in lower_url:
        return "LinkedIn"
    return "ATS"


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
    fallback_triggered: bool
    fallback_notice: str

llm = ChatOpenAI(model="gpt-6-luna")
structured_llm = llm.with_structured_output(_RawJobBatch)
enrichment_llm = llm.with_structured_output(EnrichedJobBatch)

def search_tavily(state: AgentState) -> dict:
    c = state["criteria"]
    kw = " ".join(c.keywords)
    exp = c.experience_level or ""
    loc = c.location.strip()
    query = f"{exp} {c.role} {loc} {kw} apply".strip()

    domains_to_include = c.target_boards if c.target_boards else DEFAULT_ALL_BOARDS

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
        "1. Extract all valid single-role job postings so the human reviewer can decide.\n"
        "2. Ignore any 'Similar jobs' sidebar links inside a snippet; extract ONLY the primary job "
        "described by that result's `title` and `url`.\n"
        "3. Copy the exact `url` from the result item without modifying it.\n"
        "4. Assign a `match_score` from 1 to 10 based on alignment with the User Search Context.\n"
        "5. Populate `matched_keywords` with the exact keywords from the user's criteria (or direct "
        "equivalents) found in the job's title or snippet.\n\n"
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
            src_type = classify_source(item.url)
            valid_jobs.append(
                JobListing(
                    **item.model_dump(),
                    source_type=src_type,
                )
            )

    # Prioritize ATS boards first (0), then LinkedIn (1), and sort by match_score descending
    valid_jobs.sort(
        key=lambda j: (0 if j.source_type == "ATS" else 1, -j.match_score)
    )
    return {"structured_jobs": valid_jobs}


def route_after_extract(
    state: AgentState,
) -> Literal["auto_correct_otm", "human_review"]:
    """Auto-triggers OTM Business student fallback if 0 jobs were extracted."""
    if not state.get("structured_jobs") and not state.get("fallback_triggered", False):
        return "auto_correct_otm"
    return "human_review"


def auto_correct_otm(state: AgentState) -> dict:
    """Broadens empty searches to suitable Operations & Technology Management (OTM) roles."""
    c = state["criteria"]
    exp = c.experience_level or "Entry Level"

    # Build a broader OTM-aligned query tailored to Business School OTM students
    otm_role = "Operations Analyst OR Business Systems Analyst OR Supply Chain Analyst"
    otm_keywords = ["SQL", "Excel", "Data Analytics", "Process Improvement"]

    updated_criteria = c.model_copy(
        update={
            "role": otm_role,
            "keywords": otm_keywords,
            "target_boards": DEFAULT_ALL_BOARDS,
        }
    )

    notice = (
        f"No exact direct matches were found for '{c.role}'. "
        f"Auto-corrected to Operations & Technology Management (OTM) roles "
        f"(Operations, Supply Chain, and Business Systems Analyst) for '{exp}'."
    )
    return {
        "criteria": updated_criteria,
        "fallback_triggered": True,
        "fallback_notice": notice,
    }


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
                    DEFAULT_ALL_BOARDS
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
                "fallback_triggered": False,  # Allow auto-correct again if manual retry yields 0
                "fallback_notice": "",
            },
        )

    approved_indices: list[int] = human_decision.get("approved_indices", [])
    selected = [jobs[i] for i in approved_indices if 0 <= i < len(jobs)]
    return Command(
        goto="finalize_applications",
        update={"approved_jobs": selected},
    )


def finalize_applications(state: AgentState) -> dict:
    """Scrapes full job pages via TavilyExtract with automatic snippet fallback for LinkedIn."""
    approved = state["approved_jobs"]
    if not approved:
        return {
            "enriched_jobs": [],
            "action_summary": "No jobs were approved. Workflow ended.",
        }

    # Build snippet lookup from initial Tavily search results for LinkedIn fallback
    raw_search_payload = json.loads(state.get("raw_results") or "{}")
    snippet_by_url = {
        r.get("url"): r.get("content", "")
        for r in raw_search_payload.get("results", [])
        if r.get("url")
    }

    unique_urls = list(dict.fromkeys(job.url for job in approved if job.url))
    extract_tool = TavilyExtract(extract_depth="advanced")
    extract_response = extract_tool.invoke({"urls": unique_urls})

    extracted_by_url = {
        page.get("url"): (page.get("raw_content") or "")
        for page in extract_response.get("results", [])
        if page.get("url")
    }

    combined_sources = []
    fallback_count = 0
    for job in approved:
        scraped_text = extracted_by_url.get(job.url, "").strip()
        # LinkedIn frequently blocks full scraping or returns login walls; fall back to search snippet
        is_blocked_or_empty = (
            len(scraped_text) < 250
            or "sign in to tailor your resume" in scraped_text.lower()
            or "you’re signed out" in scraped_text.lower()
        )
        if is_blocked_or_empty:
            fallback_count += 1
            fallback_snippet = snippet_by_url.get(job.url, job.summary)
            combined_sources.append({
                "url": job.url,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "extraction_mode": "search_snippet_fallback",
                "content": f"Title: {job.title}\nCompany: {job.company}\nLocation: {job.location}\nSummary: {job.summary}\nSnippet:\n{fallback_snippet[:4000]}",
            })
        else:
            combined_sources.append({
                "url": job.url,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "extraction_mode": "full_page_extract",
                "content": scraped_text[:12000],
            })

    prompt = (
        "Parse the job posting content below into structured requirements for each approved role.\n"
        "- Even if `extraction_mode` is `search_snippet_fallback` (e.g., LinkedIn), extract all available "
        "responsibilities and qualifications from the snippet and always populate `full_description_markdown` "
        "with the Company Name, Location, Job Title, and Description.\n"
        "- Extract bulleted lists of `key_responsibilities`, `required_qualifications`, and "
        "`preferred_qualifications`.\n"
        "- Include `compensation_and_benefits` if mentioned.\n\n"
        f"Job Sources:\n{json.dumps(combined_sources, indent=2)}"
    )

    enriched_batch: EnrichedJobBatch = enrichment_llm.invoke(prompt)

    companies = ", ".join(job.company for job in approved)
    summary_msg = (
        f"Processed {len(approved)} approved job(s) on attempt #{state.get('attempt', 1)}: {companies}."
    )
    if fallback_count:
        summary_msg += (
            f" ({fallback_count} LinkedIn/restricted posting(s) used search snippet fallback for details.)"
        )

    return {
        "enriched_jobs": enriched_batch.enriched_jobs,
        "action_summary": summary_msg,
    }


workflow = StateGraph(AgentState)
workflow.add_node("search_tavily", search_tavily)
workflow.add_node("extract_jobs", extract_jobs)
workflow.add_node("auto_correct_otm", auto_correct_otm)
workflow.add_node("human_review", human_review)
workflow.add_node("finalize_applications", finalize_applications)

workflow.add_edge(START, "search_tavily")
workflow.add_edge("search_tavily", "extract_jobs")
workflow.add_conditional_edges(
    "extract_jobs",
    route_after_extract,
    {
        "auto_correct_otm": "auto_correct_otm",
        "human_review": "human_review",
    },
)
workflow.add_edge("auto_correct_otm", "search_tavily")
workflow.add_edge("finalize_applications", END)

checkpointer = MemorySaver()
job_search_graph = workflow.compile(checkpointer=checkpointer)