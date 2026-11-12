# nodes.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from helper06_Research_Graph_Schemas import (
    GraphState,
    GradeDocuments,
    GradeHallucinations,
    GradeAnswer
)

load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

# Connect to existing Chroma database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

### NODE 1: Retrieve
def retrieve(state: GraphState) -> dict:
    """Retrieves documents from vectorstore based on state question."""
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "retry_count": state.get("retry_count", 0)
    }

### NODE 2: Grade Documents
def grade_documents(state: GraphState) -> dict:
    """Evaluates whether retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state["documents"]
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system_prompt = (
        "You are an academic reviewer grading relevance of retrieved context to a query. "
        "If the document contains keyword(s) or semantic meaning related to the question, "
        "grade it as relevant ('yes'). Otherwise, grade it as not relevant ('no')."
    )
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document:\n\n{document}\n\nUser Question: {question}")
    ])
    grader_chain = grade_prompt | structured_llm_grader

    filtered_docs = []
    run_rewrite = "no"

    for doc in documents:
        grade = grader_chain.invoke({"document": doc.page_content, "question": question})
        if grade.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            run_rewrite = "yes"  # Mark for query rewrite if any doc was irrelevant or noisy

    # If no relevant documents were found, flag for query transformation
    if not filtered_docs:
        run_rewrite = "yes"

    return {
        "documents": filtered_docs if filtered_docs else documents,
        "run_web_search_or_rewrite": run_rewrite
    }

### NODE 3: Rewrite Query
def rewrite_query(state: GraphState) -> dict:
    """Transforms the query to optimize for dense vector retrieval."""
    question = state["question"]
    
    system_prompt = (
        "You are an academic query reformulator. Look at the input query and formulate "
        "an improved, search-optimized query that targets the underlying technical and empirical constructs."
    )
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Initial Query: {question}\n\nFormulate an improved search query:")
    ])
    
    chain = rephrase_prompt | llm
    better_question = chain.invoke({"question": question}).content.strip()

    # Increment retry counter
    retries = state.get("retry_count", 0) + 1
    
    # Re-retrieve with new query
    re_retrieved_docs = retriever.invoke(better_question)
    
    return {
        "question": better_question,
        "documents": re_retrieved_docs,
        "retry_count": retries,
        "run_web_search_or_rewrite": "no"
    }

### NODE 4: Generate Answer
def generate(state: GraphState) -> dict:
    """Generates an answer using the filtered document context."""
    question = state["question"]
    documents = state["documents"]

    context_str = "\n\n".join(
        [f"[Source: {d.metadata.get('file_name', 'Doc')} (Page {d.metadata.get('page', 0)+1})]\n{d.page_content}" 
         for d in documents]
    )

    system_prompt = (
        "You are an academic researcher assistant. Answer the user question using ONLY the provided "
        "literature context. Always cite exact paper filenames and page numbers. "
        "Do not extrapolate or assume facts not present in the context.\n\n"
        "Context:\n{context}"
    )
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}")
    ])
    
    chain = gen_prompt | llm
    generation = chain.invoke({"context": context_str, "question": question}).content
    return {"generation": generation}