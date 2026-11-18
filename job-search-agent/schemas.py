import re
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator

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
    if parsed.path in ("", "/"):
        return False
    for pattern in GENERIC_SEARCH_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    return True


class JobSearchRequest(BaseModel):
    role: str = Field(..., example="Business Analyst")
    location: str = Field(default="Remote", example="Remote")
    experience_level: str | None = Field(default=None, example="Entry Level")
    keywords: list[str] = Field(default_factory=list, example=["aws", "sql", "excel"])
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
    source_type: str = Field(
        default="ATS",
        description="'ATS' for direct applicant tracking systems, 'LinkedIn' for LinkedIn postings.",
    )
    match_score: int = Field(default=5, ge=1, le=10)
    matched_keywords: list[str] = Field(default_factory=list)

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
    compensation_and_benefits: str | None = Field(default=None)
    full_description_markdown: str = Field(default="")


class EnrichedJobBatch(BaseModel):
    enriched_jobs: list[EnrichedJobDetails] = Field(default_factory=list)