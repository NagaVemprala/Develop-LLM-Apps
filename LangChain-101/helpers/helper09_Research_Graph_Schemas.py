from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# 1. Graph State definition
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    run_web_search_or_rewrite: str  # "yes" or "no"
    retry_count: int

# 2. Schema for Document Relevance Grading
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# 3. Schema for Hallucination / Groundedness Checking
class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# 4. Schema for Answer Quality / Question Relevance Checking
class GradeAnswer(BaseModel):
    """Binary score to evaluate if answer directly addresses the question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )