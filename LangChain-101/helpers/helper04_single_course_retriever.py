# rag_chain.py
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

def build_conversational_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 1. Prompt to reformulate query if chat history exists
    rephrase_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if "
        "needed and otherwise return it as is."
    )
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=rephrase_prompt
    )

    # 2. Prompt for final answer synthesis using retrieved context
    qa_system_prompt = (
        "You are an academic course assistant for OTM course. "
        "Answer the question using ONLY the following retrieved context. "
        "Cite the specific document source and section if available. "
        "If you do not know the answer based on the context, state clearly that "
        "it is not addressed in the course syllabus.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Full retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain