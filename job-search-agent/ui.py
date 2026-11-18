import os
import uuid
import streamlit as st
from langgraph.types import Command
from langchain_core.tracers.context import collect_runs
from langsmith import Client as LangSmithClient
from datetime import date
import io 
import csv 

# --- 1. Load API keys & LangSmith config BEFORE importing graph.py ---
SECRET_KEYS = [
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING_V2",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
]
for key in SECRET_KEYS:
    if key in st.secrets:
        os.environ[key] = (
            str(st.secrets[key]).lower()
            if isinstance(st.secrets[key], bool)
            else str(st.secrets[key])
        )

from schemas import JobSearchRequest  # noqa: E402
from graph import job_search_graph  # noqa: E402


def resolve_location(mode: str, custom_text: str) -> str:
    """Maps the location dropdown choice into the search location string."""
    if mode == "Any":
        return ""
    if mode == "Other":
        return custom_text.strip()
    return mode  # "Remote" or "Onsite"


def render_job_badges(job: dict) -> None:
    """Renders source priority badge, match score badge, and matched keyword badges."""
    src = job.get("source_type", "ATS")
    src_badge = (
        ":violet-badge[:material/corporate_fare: ATS Direct]"
        if src == "ATS"
        else ":blue-badge[:material/work: LinkedIn]"
    )

    score = job.get("match_score", 5)
    if score >= 8:
        score_badge = f":green-badge[:material/verified: Match: {score}/10]"
    elif score >= 5:
        score_badge = f":orange-badge[:material/trending_up: Match: {score}/10]"
    else:
        score_badge = f":red-badge[:material/info: Match: {score}/10]"

    keywords = job.get("matched_keywords") or []
    kw_badges = " ".join(f":gray-badge[{kw}]" for kw in keywords)

    st.markdown(f"{src_badge} {score_badge} {kw_badges}")


def invoke_with_trace(graph_input, config: dict) -> tuple[dict, str | None]:
    """Executes the graph and captures the exact LangSmith trace URL."""
    with collect_runs() as cb:
        result = job_search_graph.invoke(graph_input, config=config)

    trace_url = None
    if cb.traced_runs:
        run_id = cb.traced_runs[0].id
        try:
            ls_client = LangSmithClient()
            trace_url = ls_client.read_run(run_id).url
        except Exception:
            project = os.environ.get("LANGSMITH_PROJECT", "default")
            trace_url = f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_id}"

    return result, trace_url

def build_jobs_csv(approved_jobs: list[dict], enriched_by_url: dict[str, dict]) -> str:
    """Serializes approved jobs and their enriched requirements into a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Job Title",
        "Company",
        "Location",
        "Source",
        "Match Score (1-10)",
        "Matched Keywords",
        "Direct URL",
        "Compensation & Benefits",
        "Required Qualifications",
        "Preferred Qualifications",
        "Key Responsibilities",
        "Summary",
    ])

    for job in approved_jobs:
        details = enriched_by_url.get(job["url"], {})
        writer.writerow([
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            job.get("source_type", ""),
            job.get("match_score", ""),
            ", ".join(job.get("matched_keywords") or []),
            job.get("url", ""),
            details.get("compensation_and_benefits") or "",
            " | ".join(details.get("required_qualifications") or []),
            " | ".join(details.get("preferred_qualifications") or []),
            " | ".join(details.get("key_responsibilities") or []),
            job.get("summary", ""),
        ])

    return output.getvalue()

st.set_page_config(page_title="HITL Job Search Agent", layout="centered")
st.title("Job Search Agent (Human-in-the-Loop)")

# Initialize session state safely
defaults = {
    "thread_id": None,
    "extracted_jobs": [],
    "search_query": "",
    "attempt": 1,
    "keywords_str": "aws, sql, excel",
    "location_mode": "Remote",
    "custom_location": "Portland, OR",
    "fallback_notice": "",
    "final_result": None,
    "trace_urls": [],
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# --- Sidebar: LangSmith Trace History ---
with st.sidebar:
    st.header("Observability")
    trace_urls = st.session_state.get("trace_urls", [])
    if trace_urls:
        st.caption("LangSmith Traces (Current Thread)")
        for label, url in trace_urls:
            st.link_button(f"🔍 {label}", url, use_container_width=True)
    else:
        st.info("Run a search to generate a LangSmith trace link.")

# --- Step 1: Initial Search Criteria ---
st.subheader("1. Search Criteria")
col1, col2 = st.columns(2)
role = col1.text_input("Role", value="Business Analyst")
experience = col1.text_input("Experience Level", value="Entry Level")

location_mode = col2.selectbox(
    "Work Mode / Location",
    options=["Remote", "Onsite", "Any", "Other"],
    index=["Remote", "Onsite", "Any", "Other"].index(st.session_state.location_mode),
)
custom_location = ""
if location_mode == "Other":
    custom_location = col2.text_input(
        "Specify City, State, or Region",
        value=st.session_state.custom_location,
        placeholder="e.g., Portland, OR",
    )

keywords_raw = st.text_input("Keywords (comma-separated)", value=st.session_state.keywords_str)

if st.button("1. Start New Search", type="primary"):
    if location_mode == "Other" and not custom_location.strip():
        st.error("Please enter a location when 'Other' is selected.")
        st.stop()

    resolved_loc = resolve_location(location_mode, custom_location)
    st.session_state.final_result = None
    st.session_state.trace_urls = []
    st.session_state.keywords_str = keywords_raw
    st.session_state.location_mode = location_mode
    if location_mode == "Other":
        st.session_state.custom_location = custom_location.strip()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    req = JobSearchRequest(
        role=role,
        location=resolved_loc,
        experience_level=experience or None,
        keywords=[k.strip() for k in keywords_raw.split(",") if k.strip()],
    )
    initial_state = {
        "criteria": req,
        "search_query": "",
        "raw_results": "",
        "structured_jobs": [],
        "approved_jobs": [],
        "enriched_jobs": [],
        "action_summary": "",
        "attempt": 0,
        "fallback_triggered": False,
        "fallback_notice": "",
    }

    with st.spinner("Searching ATS boards & LinkedIn via Tavily..."):
        result, trace_url = invoke_with_trace(initial_state, config=config)

    st.session_state.thread_id = thread_id
    st.session_state.extracted_jobs = [j.model_dump() for j in result.get("structured_jobs", [])]
    st.session_state.search_query = result.get("search_query", "")
    st.session_state.attempt = result.get("attempt", 1)
    st.session_state.fallback_notice = result.get("fallback_notice", "")
    if trace_url:
        st.session_state.trace_urls.append(("Attempt #1 — Search & Extract", trace_url))
    st.rerun()

# --- Step 2: Human Review (Approve OR Reject & Re-Run) ---
if st.session_state.thread_id and not st.session_state.final_result:
    st.divider()

    header_col, btn_col = st.columns([3, 1])
    header_col.subheader(f"2. Human Review (Attempt #{st.session_state.attempt})")
    if st.session_state.trace_urls:
        _, latest_url = st.session_state.trace_urls[-1]
        btn_col.link_button("View Trace in LangSmith", latest_url)

    st.caption(
        f"Thread: `{st.session_state.thread_id}` | Query used: `{st.session_state.search_query}`"
    )

    if st.session_state.fallback_notice:
        st.info(f"🔄 **Auto-Correct Fallback Triggered:** {st.session_state.fallback_notice}")

    jobs = st.session_state.extracted_jobs
    approved_indices = []

    if not jobs:
        st.warning("No matching jobs were extracted even after OTM fallback. Try broadening keywords below.")
    else:
        for idx, job in enumerate(jobs):
            with st.container(border=True):
                checked = st.checkbox(
                    f"**{job['title']}** — {job['company']} ({job['location']})",
                    key=f"job_{st.session_state.thread_id}_{st.session_state.attempt}_{idx}",
                )
                render_job_badges(job)
                st.write(job["summary"])
                st.markdown(f"[View Posting]({job['url']})")
                if checked:
                    approved_indices.append(idx)

        if st.button("Approve Selected & Extract Full Requirements", type="primary"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Scraping full job postings via TavilyExtract..."):
                final_state, trace_url = invoke_with_trace(
                    Command(
                        resume={
                            "decision": "approve",
                            "approved_indices": approved_indices,
                        }
                    ),
                    config=config,
                )
            if trace_url:
                st.session_state.trace_urls.append(("Resume — TavilyExtract Details", trace_url))
            st.session_state.final_result = {
                "action_summary": final_state.get("action_summary", ""),
                "approved_jobs": [j.model_dump() for j in final_state.get("approved_jobs", [])],
                "enriched_jobs": [e.model_dump() for e in final_state.get("enriched_jobs", [])],
            }
            st.rerun()

    # --- Reject & Tweak Loop ---
    with st.expander("Not satisfied? Reject results & re-run search", expanded=(len(jobs) == 0)):
        new_keywords = st.text_input(
            "Tweak Keywords (comma-separated)",
            value=st.session_state.keywords_str,
            key=f"kw_edit_{st.session_state.attempt}",
        )
        retry_mode = st.selectbox(
            "Adjust Work Mode / Location",
            options=["Remote", "Onsite", "Any", "Other"],
            index=["Remote", "Onsite", "Any", "Other"].index(st.session_state.location_mode),
            key=f"loc_mode_edit_{st.session_state.attempt}",
        )
        retry_custom_loc = ""
        if retry_mode == "Other":
            retry_custom_loc = st.text_input(
                "Specify City, State, or Region",
                value=st.session_state.custom_location,
                key=f"loc_custom_edit_{st.session_state.attempt}",
            )

        keep_boards = st.checkbox(
            "Restrict to prioritized ATS boards & LinkedIn",
            value=True,
            key=f"boards_{st.session_state.attempt}",
        )

        if st.button("Reject & Re-Run Tavily Search"):
            if retry_mode == "Other" and not retry_custom_loc.strip():
                st.error("Please enter a location when 'Other' is selected.")
                st.stop()

            resolved_retry_loc = resolve_location(retry_mode, retry_custom_loc)
            st.session_state.keywords_str = new_keywords
            st.session_state.location_mode = retry_mode
            if retry_mode == "Other":
                st.session_state.custom_location = retry_custom_loc.strip()

            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            with st.spinner("Looping back to Tavily with updated criteria..."):
                updated_state, trace_url = invoke_with_trace(
                    Command(
                        resume={
                            "decision": "reject",
                            "keywords": [k.strip() for k in new_keywords.split(",") if k.strip()],
                            "location": resolved_retry_loc,
                            "keep_board_filter": keep_boards,
                        }
                    ),
                    config=config,
                )

            next_attempt = updated_state.get("attempt", st.session_state.attempt + 1)
            st.session_state.extracted_jobs = [
                j.model_dump() for j in updated_state.get("structured_jobs", [])
            ]
            st.session_state.search_query = updated_state.get("search_query", "")
            st.session_state.attempt = next_attempt
            st.session_state.fallback_notice = updated_state.get("fallback_notice", "")
            if trace_url:
                st.session_state.trace_urls.append((f"Attempt #{next_attempt} — Retry Search", trace_url))
            st.rerun()

# --- Step 3: Full Extracted Job Requirements ---
if st.session_state.final_result:
    st.divider()
    done_col, done_btn_col = st.columns([3, 1])
    done_col.subheader("3. Full Job Requirements")
    if st.session_state.trace_urls:
        _, final_url = st.session_state.trace_urls[-1]
        done_btn_col.link_button("View Trace in LangSmith", final_url)

    st.success(st.session_state.final_result["action_summary"])

    approved_jobs = st.session_state.final_result["approved_jobs"]
    enriched_by_url = {
        item["url"]: item for item in st.session_state.final_result.get("enriched_jobs", [])
    }

    if approved_jobs:
        csv_data = build_jobs_csv(approved_jobs, enriched_by_url)
        st.download_button(
            label="📥 Download Approved Jobs (CSV)",
            data=csv_data.encode("utf-8-sig"),  # utf-8-sig ensures clean display in Excel
            file_name=f"approved_jobs_{date.today().isoformat()}.csv",
            mime="text/csv",
            type="primary",
        )
        
    for idx, job in enumerate(approved_jobs):
        details = enriched_by_url.get(job["url"], {})
        with st.expander(
            f"{job['title']} — {job['company']} ({job['location']})",
            expanded=(idx == 0),
        ):
            render_job_badges(job)
            st.markdown(f"**Company:** {job['company']}  |  **Location:** {job['location']}  |  **Role:** {job['title']}")
            st.markdown(f"**Direct Posting:** [{job['url']}]({job['url']})")

            if details.get("compensation_and_benefits"):
                st.info(f"**Compensation & Benefits:** {details['compensation_and_benefits']}")

            req_col, pref_col = st.columns(2)
            with req_col:
                st.markdown("#### Required Qualifications")
                reqs = details.get("required_qualifications", [])
                if reqs:
                    for item in reqs:
                        st.markdown(f"- {item}")
                else:
                    st.caption("Not explicitly listed in extracted snippet.")

            with pref_col:
                st.markdown("#### Preferred Qualifications")
                prefs = details.get("preferred_qualifications", [])
                if prefs:
                    for item in prefs:
                        st.markdown(f"- {item}")
                else:
                    st.caption("None specified.")

            st.markdown("#### Key Responsibilities & Description")
            resps = details.get("key_responsibilities", [])
            if resps:
                for item in resps:
                    st.markdown(f"- {item}")
            else:
                st.write(job["summary"])

            if details.get("full_description_markdown"):
                with st.popover("View Full Extracted Description"):
                    st.markdown(details["full_description_markdown"])