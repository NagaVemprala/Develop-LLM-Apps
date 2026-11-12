# main.py
import json
from langchain_core.prompts import ChatPromptTemplate
from models import get_chat_model
from schemas import SyllabusBlueprint
from helpers.helper02_token_guard import check_context_budget, ContextWindowExceededError

SYSTEM_PROMPT = """You are an expert curriculum designer and instructional technologist specializing in business, analytics, and operations coursework.

Your task is to take informal, unstructured faculty course notes and produce a rigorous, structured curriculum blueprint.

Ensure all outputs strictly adhere to the requested schema."""

def run_monitored_syllabus_pipeline(raw_notes: str, provider: str = "openai", model_name: str = "gpt-4o-mini"):
    llm = get_chat_model(provider=provider, model_name=model_name, temperature=0.1)

    # Setting include_raw=True returns a dict: {"raw": AIMessage, "parsed": SyllabusBlueprint, "parsing_error": ...}
    structured_llm = llm.with_structured_output(SyllabusBlueprint, include_raw=True)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Transform the following unstructured notes into a structured syllabus blueprint:\n\n{raw_notes}")
    ])

    # 1. Format messages without invoking the LLM yet
    formatted_messages = prompt_template.format_messages(raw_notes=raw_notes)

    # 2. Pre-flight context budget check
    try:
        report = check_context_budget(
            model=llm,
            messages=formatted_messages,
            model_name=model_name,
            max_output_tokens=4096,
            safety_margin=500
        )
        print("=== [Pre-Flight Token Budget Check] ===")
        print(f"Model: {model_name}")
        print(f"Prompt Tokens: {report.prompt_tokens}")
        print(f"Projected Total (Prompt + Max Output + Margin): {report.total_projected_tokens}")
        print(f"Context Headroom Remaining: {report.remaining_headroom} / {report.max_context_limit}\n")
    except ContextWindowExceededError as e:
        print(f"Execution aborted prior to API call: {e}")
        return

    # 3. Execution (Network Call)
    response = structured_llm.invoke(formatted_messages)

    parsed_result: SyllabusBlueprint = response["parsed"]
    raw_message = response["raw"]

    # 4. Post-Execution Telemetry from usage_metadata
    usage = getattr(raw_message, "usage_metadata", None)

    print("=== [Post-Execution Usage Telemetry] ===")
    if usage:
        print(f"Actual Input Tokens:  {usage.get('input_tokens')}")
        print(f"Actual Output Tokens: {usage.get('output_tokens')}")
        print(f"Total Tokens Billed:  {usage.get('total_tokens')}")
    else:
        # Fallback for providers that embed usage inside response_metadata
        resp_meta = getattr(raw_message, "response_metadata", {})
        token_usage = resp_meta.get("token_usage", {})
        print(f"Raw Token Usage: {token_usage}")

    print("\n=== [Parsed Syllabus Output] ===")
    print(json.dumps(parsed_result.model_dump(), indent=2))

    from langchain_community.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        result = structured_llm.invoke(formatted_messages)
        print(f"Prompt Tokens:    {cb.prompt_tokens}")
        print(f"Completion Tokens:{cb.completion_tokens}")
        print(f"Total Tokens:     {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost:.6f}")

if __name__ == "__main__":
    sample_notes = """
    Planning an elective for business grads on Decision Modeling and Optimization. 
    Students know basic Excel, but I want to push them into Python and linear programming.
    Topics: LP formulation, sensitivity analysis, integer programming, transport problems, maybe some simulation.
    Need tools: Python, PuLP, Excel Solver, Streamlit.
    Lab sessions solving product mix problems.
    Midterm case study and a final web app project.
    """
    run_monitored_syllabus_pipeline(sample_notes)