from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Schema for structured research extraction
class PaperEvidence(BaseModel):
    paper_title: str = Field(description="Name of the cited paper or file.")
    page_numbers: List[int] = Field(description="Page numbers where the evidence appears.")
    dataset_context: Optional[str] = Field(description="Platform/dataset analyzed (e.g., Amazon Electronics, Yelp reviews, TripAdvisor).")
    methodology: Optional[str] = Field(description="Algorithm or analytical method (e.g., RoBERTa aspect extraction, econometric panel regression, LDA).")
    key_finding: str = Field(description="Specific finding or quantitative coefficient related to the user's research question.")

class ResearchSynthesis(BaseModel):
    summary: str = Field(description="Synthesized overview answering the query across the literature.")
    evidence_points: List[PaperEvidence] = Field(description="Detailed per-paper empirical evidence.")
    research_gaps: List[str] = Field(description="Contradictions, limitations, or open research gaps identified across the papers.")

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("file_name", "Unknown Paper")
        page = doc.metadata.get("page", 0) + 1  # 1-indexed page
        formatted.append(f"--- [Paper: {source} | Page: {page}] ---\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_research_rag_chain(db_dir: str = "./chroma_db", k: int = 5):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    # Retrieve using Maximal Marginal Relevance (MMR) to prevent redundant chunks from a single paper
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 15, "lambda_mult": 0.7}
    )

    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.0)
    structured_llm = llm.with_structured_output(ResearchSynthesis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an academic research assistant specializing in e-commerce, consumer behavior, "
         "and text analytics of customer reviews.\n\n"
         "Analyze the provided literature context below and synthesize findings for the user query. "
         "Extract precise datasets, analytical methodologies, and empirical results. Do not extrapolate "
         "beyond the provided context."),
        ("human", "Research Question: {query}\n\nLiterature Context:\n{context}")
    ])

    chain = (
        {"context": retriever | format_docs_with_sources, "query": RunnablePassthrough()}
        | prompt
        | structured_llm
    )
    return chain