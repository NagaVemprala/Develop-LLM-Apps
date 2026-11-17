from pydantic import BaseModel, Field

class JobSearchRequest(BaseModel):
    role: str = Field(..., example="AI Engineer")
    location: str = Field(default="Remote", example="In-person")
    experience_level: str | None = Field(default=None, example="Senior")
    keywords: list[str] = Field(default_factory=list, example=["Python", "LangGraph", "FastAPI"])
    target_boards: list[str] = Field(
        default_factory=lambda: [
            "greenhouse.io",
            "lever.co",
            "ashbyhq.com",
            "linkedin.com/jobs",
            "indeed.com",
        ],
        description="Domains to prioritize in Tavily search. Pass [] to search the entire web.",
    )

class JobListing(BaseModel):
    title: str = Field(..., description="Exact job title")
    company: str = Field(..., description="Hiring company name")
    location: str = Field(..., description="Job location or Remote status")
    url: str = Field(..., description="Direct application or posting URL")
    summary: str = Field(..., description="1-2 sentence summary of key responsibilities and stack")

class JobListingBatch(BaseModel):
    jobs: list[JobListing] = Field(default_factory=list)

class JobSearchResponse(BaseModel):
    query_used: str
    total_found: int
    results: list[JobListing]

class StartSearchResponse(BaseModel):
    thread_id: str
    status: str  # "awaiting_approval" or "completed"
    extracted_jobs: list[JobListing]

class ResumeSearchRequest(BaseModel):
    thread_id: str
    approved_indices: list[int] = Field(
        ...,
        example=[0, 2],
        description="0-based indices of the jobs you want to approve for next steps.",
    )

class FinalSearchResponse(BaseModel):
    thread_id: str
    status: str
    approved_jobs: list[JobListing]
    action_summary: str