from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from helper06_Research_Graph_Schemas import GraphState, GradeHallucinations, GradeAnswer

load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

def decide_to_generate(state: GraphState) -> str:
    """Decides whether to generate an answer or rewrite the query first."""
    run_rewrite = state["run_web_search_or_rewrite"]
    retries = state.get("retry_count", 0)

    # Prevent infinite query rewriting loops
    if run_rewrite == "yes" and retries < 2:
        return "rewrite_query"
    return "generate"

def grade_generation_groundedness(state: GraphState) -> str:
    """Evaluates hallucination and question relevance."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    context_str = "\n\n".join([d.page_content for d in documents])

    # 1. Check Hallucination (Grounding)
    hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade whether the answer is strictly grounded in the given context facts ('yes' or 'no')."),
        ("human", "Context:\n{context}\n\nGenerated Answer:\n{generation}")
    ])
    hallucination_chain = hallucination_prompt | hallucination_grader
    grounded_eval = hallucination_chain.invoke({"context": context_str, "generation": generation})

    if grounded_eval.binary_score == "no":
        print("--- [DECISION: Answer is hallucinated / not grounded. Re-generating...] ---")
        return "regenerate"

    # 2. Check Question Fulfillment
    answer_grader = llm.with_structured_output(GradeAnswer)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade whether the answer addresses the question directly ('yes' or 'no')."),
        ("human", "Question:\n{question}\n\nGenerated Answer:\n{generation}")
    ])
    answer_chain = answer_prompt | answer_grader
    addressed_eval = answer_chain.invoke({"question": question, "generation": generation})

    if addressed_eval.binary_score == "yes":
        print("--- [DECISION: Answer is grounded and addresses query. Done.] ---")
        return "useful"
    else:
        print("--- [DECISION: Answer fails to address the question. Rewriting...] ---")
        return "not_useful"