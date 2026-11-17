import re
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator

# Patterns that indicate a generic search/index page rather than a single job posting
GENERIC_SEARCH_URL_PATTERNS = [
    r"/jobs/search",
    r"/jobs/remote-",
    r"/jobs/[^/]+-jobs/?$",
    r"/search\?",
    r"/Jobs/",
    r"builtin\.com/jobs/remote",
    r"builtinaustin\.com/jobs/remote",
    r"builtinchicago\.org/jobs/remote",
    r"simplyhired\.com/search",
    r"ziprecruiter\.com/Jobs/",
    r"indeed\.com/q-",
    r"indeed\.com/jobs\?",
    r"linkedin\.com/jobs/[^v]",  # Allows /jobs/view/... while blocking /jobs/remote-...
]

def is_specific_job_url(url: str) -> bool:
    """Returns True only if the URL points to an individual job posting."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    # Reject bare homepages
    if parsed.path in ("", "/"):
        return False
    for pattern in GENERIC_SEARCH_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    return True


class JobSearchRequest(BaseModel):
    role: str = Field(..., example="AI Engineer")
    location: str = Field(default="Remote", example="Remote")
    experience_level: str | None = Field(default=None, example="Senior")
    keywords: list[str] = Field(default_factory=list, example=["Python", "LangGraph", "FastAPI"])
    target_boards: list[str] = Field(
        default_factory=lambda: [
            "boards.greenhouse.io",
            "job-boards.greenhouse.io",
            "jobs.lever.co",
            "jobs.ashbyhq.com",
            "myworkdayjobs.com",
            "icims.com",
            "jobs.smartrecruiters.com",
            "linkedin.com/jobs/view",
        ],
    )


class JobListing(BaseModel):
    title: str
    company: str
    location: str
    url: str
    summary: str
    match_score: int = Field(
        default=5,
        ge=1,
        le=10,
        description="1-10 alignment score against the user's role, location, experience level, and keywords.",
    )
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="Subset of user keywords (and closely related skills) explicitly found in the job snippet.",
    )

    @field_validator("url")
    @classmethod
    def validate_direct_job_url(cls, v: str) -> str:
        if not is_specific_job_url(v):
            raise ValueError(f"Generic search or non-specific job URL rejected: {v}")
        return v


class JobListingBatch(BaseModel):
    jobs: list[JobListing] = Field(default_factory=list)


class EnrichedJobDetails(BaseModel):
    url: str
    key_responsibilities: list[str] = Field(default_factory=list)
    required_qualifications: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    compensation_and_benefits: str | None = Field(
        default=None, description="Salary range, equity, or benefits if listed on the page"
    )
    full_description_markdown: str = Field(
        default="", description="Clean summary or extracted text of the full posting"
    )


class EnrichedJobBatch(BaseModel):
    enriched_jobs: list[EnrichedJobDetails] = Field(default_factory=list)