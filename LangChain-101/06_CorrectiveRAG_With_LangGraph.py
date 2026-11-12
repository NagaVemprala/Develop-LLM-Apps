# main.py
from langgraph.graph import StateGraph, START, END
from helpers.helper09_Research_Graph_Schemas import GraphState
from helpers.helper07_Research_Graph_Nodes import retrieve, grade_documents, rewrite_query, generate
from helpers.helper08_Research_Conditional_Edges import decide_to_generate, grade_generation_groundedness

def build_crag_workflow():
    workflow = StateGraph(GraphState)

    # 1. Define Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)

    # 2. Connect Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Branch: decide whether to rewrite query or generate
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate"
        }
    )
    workflow.add_edge("rewrite_query", "grade_documents")

    # Branch: check hallucinations and question relevance
    workflow.add_conditional_edges(
        "generate",
        grade_generation_groundedness,
        {
            "useful": END,
            "regenerate": "generate",
            "not_useful": "rewrite_query"
        }
    )

    # Compile the runnable graph
    app = workflow.compile()
    return app

def main():
    crag_app = build_crag_workflow()

    print("=== Corrective RAG (CRAG) with LangGraph Active ===")
    
    # Test with a question that requires precise empirical grounding
    sample_query = "What role do negative sentiments and emotional valence play in customer dissatisfaction regarding product defects versus design?"

    initial_state: GraphState = {
        "question": sample_query,
        "documents": [],
        "generation": "",
        "run_web_search_or_rewrite": "no",
        "retry_count": 0
    }

    # Stream execution steps to visualize routing
    print(f"\nUser Query: {sample_query}\n")
    for output in crag_app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"✔ Completed Step: [{node_name}]")
            if "run_web_search_or_rewrite" in state_update:
                print(f"   -> Rewrite flagged: {state_update['run_web_search_or_rewrite']}")
            if "question" in state_update and node_name == "rewrite_query":
                print(f"   -> Reformulated Query: {state_update['question']}")

    # Final generated payload
    final_state = crag_app.invoke(initial_state)
    print("\n" + "=" * 80)
    print("FINAL VALIDATED GENERATION")
    print("=" * 80)
    print(final_state["generation"])
    print("=" * 80)

if __name__ == "__main__":
    main()