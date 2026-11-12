# main.py
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from helpers.helper03_vector_store import build_vector_store, get_retriever
from helpers.helper04_single_course_retriever import build_conversational_rag_chain
import os

# In-memory dictionary mapping session_id -> ChatMessageHistory
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def main():
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print("Initial vector database not found. Creating...")
        build_vector_store(db_path)

    retriever = get_retriever(db_path, k=2)
    rag_chain = build_conversational_rag_chain(retriever)

    # Wrap the RAG pipeline with session state management
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    session_id = "student_session_001"
    config = {"configurable": {"session_id": session_id}}

    print("=== OTM Course Assistant Initialized ===")
    print("Type your questions below ('exit' or 'quit' to stop).\n")

    while True:
        try:
            user_input = input("Student: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Session ended.")
                break

            response = conversational_chain.invoke({"input": user_input}, config=config)

            print(f"\nAssistant: {response['answer']}\n")
            
            # Print citations from metadata
            sources = {doc.metadata.get("source", "unknown") for doc in response.get("context", [])}
            print(f"[Sources Consulted: {', '.join(sources)}]\n" + "-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted.")
            break

if __name__ == "__main__":
    main()