# main.py
import json
from pathlib import Path
from helpers.helper05_vectorDB_PDF import build_vector_store, DATA_DIR, DB_DIR
from helpers.helper06_RAG_Retriever_Chain import get_research_rag_chain, ResearchSynthesis

def display_synthesis(result: ResearchSynthesis):
    print("\n" + "=" * 70)
    print("RESEARCH SYNTHESIS OVERVIEW")
    print("=" * 70)
    print(result.summary)

    print("\n" + "=" * 70)
    print("EMPIRICAL EVIDENCE TABLE")
    print("=" * 70)
    for idx, ev in enumerate(result.evidence_points, 1):
        pages = ", ".join(map(str, ev.page_numbers)) if ev.page_numbers else "N/A"
        print(f"\n[{idx}] Paper: {ev.paper_title} (Page {pages})")
        if ev.dataset_context:
            print(f"    - Platform / Data: {ev.dataset_context}")
        if ev.methodology:
            print(f"    - Methodology:     {ev.methodology}")
        print(f"    - Finding:         {ev.key_finding}")

    if result.research_gaps:
        print("\n" + "=" * 70)
        print("IDENTIFIED GAPS / CONFLICTING FINDINGS")
        print("=" * 70)
        for gap in result.research_gaps:
            print(f"  • {gap}")
    print("\n" + "-" * 70)

def main():
    if not Path(DB_DIR).exists() or not any(Path(DB_DIR).iterdir()):
        print(f"Vector database not found. Please ensure your papers are in '{DATA_DIR}'.")
        choice = input("Build index now? (y/n): ").strip().lower()
        if choice == "y":
            build_vector_store()
        else:
            return

    rag_chain = get_research_rag_chain()
    print("\n=== E-Commerce Review Literature RAG Ready ===")
    print("Example Queries:")
    print(" 1. How does review length and sentiment polarity impact review helpfulness?")
    print(" 2. What deep learning approaches are used for aspect-based sentiment extraction?")
    print(" 3. What features are most predictive of fraudulent or incentivized customer reviews?")
    print("\nType your query below ('exit' to stop):\n")

    while True:
        try:
            query = input("Research Query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break

            print("\nSearching literature chunks and synthesizing...")
            synthesis: ResearchSynthesis = rag_chain.invoke(query)
            display_synthesis(synthesis)

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()