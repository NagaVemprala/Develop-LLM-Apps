import json
from langchain_core.prompts import ChatPromptTemplate

from models import get_chat_model
from helpers.helper01_Syllabus_Schemas import SyllabusBlueprint

SYSTEM_PROMPT = """You are an expert curriculum designer and instructional technologist specializing in business, analytics, and operations coursework.

Your task is to take informal, unstructured faculty course notes and produce a rigorous, structured curriculum blueprint.

Ensure all outputs strictly adhere to the requested schema. Pay particular attention to:
1. Recommending industry-standard tools appropriate for business graduates.
2. Formulating scaffolded, actionable hands-on activities.
3. Defining assignments with realistic executive or analytical deliverables."""

def build_syllabus_pipeline(provider: str = "openai", model_name: str = ""):
    # 1. Instantiate the model
    llm = get_chat_model(provider=provider, model_name=model_name, temperature=0.1)

    # 2. Bind structured output via Pydantic
    structured_llm = llm.with_structured_output(SyllabusBlueprint)

    # 3. Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Transform the following unstructured notes into a structured syllabus blueprint:\n\n{raw_notes}")
    ])

    # 4. Compose via LangChain Expression Language (LCEL)
    chain = prompt | structured_llm
    return chain

def main():
    print("=== Smart Syllabus Builder (Phase 1: Foundations) ===")
    
    # Example messy input simulating notes from an instructor
    sample_notes = """
    Planning an elective for business grads on Decision Modeling and Optimization. 
    Students know basic Excel, but I want to push them into Python and linear programming.
    Topics: LP formulation, sensitivity analysis, integer programming, transport problems, maybe some simulation.
    Need tools: definitely Python, maybe PuLP or SciPy, Excel Solver for comparison, maybe Streamlit to make apps.
    Want them to do some lab sessions where they solve production mix problems step by step.
    For grading: one midterm case study on supply chain network design and a final project where they build a small web tool.
    """

    print("\n[Input Notes]:")
    print(sample_notes.strip())
    print("\nProcessing with LangChain + Structured Output...")

    pipeline = build_syllabus_pipeline(provider="openai", model_name="gpt-5.4-nano")
    
    # Invocation
    result: SyllabusBlueprint = pipeline.invoke({"raw_notes": sample_notes})

    print("\n[Structured Syllabus Output (Validated Pydantic -> JSON)]:\n")
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()