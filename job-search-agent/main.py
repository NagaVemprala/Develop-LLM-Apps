import uuid
from fastapi import FastAPI, HTTPException
from langgraph.types import Command

from schemas import (
    JobSearchRequest,
    StartSearchResponse,
    ResumeSearchRequest,
    FinalSearchResponse,
)
from graph import job_search_graph

app = FastAPI(title="HITL LangGraph Job Search Agent", version="1.0.0")

@app.post("/search-jobs/start", response_model=StartSearchResponse)
async def start_search(request: JobSearchRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "criteria": request,
        "search_query": "",
        "raw_results": "",
        "structured_jobs": [],
        "approved_jobs": [],
        "action_summary": "",
    }

    try:
        # Runs until it hits interrupt() inside human_review
        result = await job_search_graph.ainvoke(initial_state, config=config)
        state_snapshot = await job_search_graph.aget_state(config)

        status = "awaiting_approval" if state_snapshot.next else "completed"
        return StartSearchResponse(
            thread_id=thread_id,
            status=status,
            extracted_jobs=result.get("structured_jobs", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-jobs/resume", response_model=FinalSearchResponse)
async def resume_search(request: ResumeSearchRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    # Verify the thread exists and is actually paused
    state_snapshot = await job_search_graph.aget_state(config)
    if not state_snapshot.values:
        raise HTTPException(status_code=404, detail="Invalid or expired thread_id.")
    if not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Workflow is already completed.")

    try:
        # Resume from interrupt() by passing human input via Command
        final_state = await job_search_graph.ainvoke(
            Command(resume={"approved_indices": request.approved_indices}),
            config=config,
        )
        return FinalSearchResponse(
            thread_id=request.thread_id,
            status="completed",
            approved_jobs=final_state.get("approved_jobs", []),
            action_summary=final_state.get("action_summary", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))