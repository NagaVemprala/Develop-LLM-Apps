import os
import uuid
import streamlit as st
from langgraph.types import Command
from langchain_core.tracers.context import collect_runs
from langsmith import Client as LangSmithClient

from schemas import JobSearchRequest  # noqa: E402
from graph import job_search_graph  # noqa: E402

# Reads from .streamlit/secrets.toml locally or App Secrets on Streamlit Cloud
# --- Load API keys & LangSmith config from Streamlit Secrets BEFORE importing graph.py ---
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

st.set_page_config(page_title="HITL Job Search Agent", layout="centered")
st.title("Job Search Agent (Human-in-the-Loop)")

# Initialize session state safely (survives live code reloads)
defaults = {
    "thread_id": None,
    "extracted_jobs": [],
    "search_query": "",
    "attempt": 1,
    "keywords_str": "Python, LangGraph, FastAPI",
    "location_str": "Remote",
    "final_result": None,
    "trace_urls": [],
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# --- Sidebar: LangSmith Trace History for Current Thread ---
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
with st.form("search_form"):
    col1, col2 = st.columns(2)
    role = col1.text_input("Role", value="Business Analyst")
    location = col2.text_input("Location", value=st.session_state.location_str)
    experience = col1.text_input("Experience Level", value="Entry Level")
    keywords_raw = col2.text_input("Keywords (comma-separated)", value=st.session_state.keywords_str)
    submitted = st.form_submit_button("1. Start New Search")

if submitted:
    st.session_state.final_result = None
    st.session_state.trace_urls = []
    st.session_state.keywords_str = keywords_raw
    st.session_state.location_str = location

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    req = JobSearchRequest(
        role=role,
        location=location,
        experience_level=experience or None,
        keywords=[k.strip() for k in keywords_raw.split(",") if k.strip()],
    )
    initial_state = {
        "criteria": req,
        "search_query": "",
        "raw_results": "",
        "structured_jobs": [],
        "approved_jobs": [],
        "action_summary": "",
        "attempt": 0,
    }

    with st.spinner("Searching Online using Tavily and extracting roles..."):
        result = job_search_graph.invoke(initial_state, config=config)

    st.session_state.thread_id = thread_id
    st.session_state.extracted_jobs = [j.model_dump() for j in result.get("structured_jobs", [])]
    st.session_state.search_query = result.get("search_query", "")
    st.session_state.attempt = result.get("attempt", 1)

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

    jobs = st.session_state.extracted_jobs
    approved_indices = []

    if not jobs:
        st.warning("No matching jobs were extracted on this attempt.")
    else:
        for idx, job in enumerate(jobs):
            with st.container(border=True):
                checked = st.checkbox(
                    f"**{job['title']}** — {job['company']} ({job['location']})",
                    key=f"job_{st.session_state.thread_id}_{st.session_state.attempt}_{idx}",
                )
                st.write(job["summary"])
                st.markdown(f"[View Posting]({job['url']})")
                if checked:
                    approved_indices.append(idx)

        if st.button("Approve Selected & Complete", type="primary"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Finalizing approved roles..."):
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
                st.session_state.trace_urls.append(("Resume — Finalize Approved", trace_url))
            st.session_state.final_result = {
                "action_summary": final_state.get("action_summary", ""),
                "approved_jobs": [j.model_dump() for j in final_state.get("approved_jobs", [])],
            }
            st.rerun()

    # --- Reject & Tweak Loop ---
    with st.expander("Not satisfied? Reject results & re-run search", expanded=(len(jobs) == 0)):
        new_keywords = st.text_input(
            "Tweak Keywords (comma-separated)",
            value=st.session_state.keywords_str,
            key=f"kw_edit_{st.session_state.attempt}",
        )
        new_location = st.text_input(
            "Adjust Location",
            value=st.session_state.location_str,
            key=f"loc_edit_{st.session_state.attempt}",
        )
        keep_boards = st.checkbox(
            "Restrict to ATS boards (Greenhouse, Lever, Ashby, LinkedIn, Indeed)",
            value=True,
            key=f"boards_{st.session_state.attempt}",
        )

        if st.button("Reject & Re-Run Tavily Search"):
            st.session_state.keywords_str = new_keywords
            st.session_state.location_str = new_location
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            with st.spinner("Looping back to Tavily with updated criteria..."):
                updated_state, trace_url = invoke_with_trace(
                    Command(
                        resume={
                            "decision": "reject",
                            "keywords": [k.strip() for k in new_keywords.split(",") if k.strip()],
                            "location": new_location,
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
            if trace_url:
                st.session_state.trace_urls.append((f"Attempt #{next_attempt} — Retry Search", trace_url))
            st.rerun()

# --- Step 3: Final Workflow Output ---
if st.session_state.final_result:
    st.divider()
    done_col, done_btn_col = st.columns([3, 1])
    done_col.subheader("3. Workflow Completed")
    if st.session_state.trace_urls:
        _, final_url = st.session_state.trace_urls[-1]
        done_btn_col.link_button("View Trace in LangSmith", final_url)

    st.success(st.session_state.final_result["action_summary"])
    for job in st.session_state.final_result["approved_jobs"]:
        st.markdown(f"- **{job['title']}** at **{job['company']}** ([Link]({job['url']}))")