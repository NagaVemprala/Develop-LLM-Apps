from typing import List
from pydantic import BaseModel, Field

class ModernTool(BaseModel):
    tool_name: str = Field(description="Name of the software, library, or platform (e.g., Tableau, Python/Pandas, Power BI).")
    relevance: str = Field(description="Why this specific tool is relevant to business and management students.")

class Activity(BaseModel):
    step_number: int = Field(description="Sequential order of the activity.")
    title: str = Field(description="Name of the hands-on lab, discussion, or exercise.")
    instruction: str = Field(description="Step-by-step description of what students must complete.")
    duration_minutes: int = Field(description="Estimated in-class or asynchronous completion time.")

class Assignment(BaseModel):
    title: str = Field(description="Title of the assignment or deliverable.")
    deliverable: str = Field(description="Format of submission (e.g., executive memo, GitHub repo, slide deck).")
    rubric_focus: List[str] = Field(description="2-3 core grading criteria items.")

class SyllabusBlueprint(BaseModel):
    course_title: str = Field(description="Formal, polished academic title for the course.")
    target_audience: str = Field(description="Primary level: Undergraduate, MBA, or MS Analytics.")
    core_topics: List[str] = Field(description="Key theoretical and applied concepts covered across modules.")
    recommended_tools: List[ModernTool] = Field(description="Tools and technologies integrated into the coursework.")
    planned_activities: List[Activity] = Field(description="Scaffolded classroom activities or lab walk-throughs.")
    assignments: List[Assignment] = Field(description="Major assessment milestones.")