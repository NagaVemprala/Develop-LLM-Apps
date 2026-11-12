# store.py
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Sample business course policy document
SAMPLE_SYLLABUS_DOCS = [
    Document(
        page_content="""
        Course: OTM 680 - Applied Optimization & Decision Analytics
        Instructor: Dr. Vemprala | Office: Franz Hall 320
        Grading Policy:
        - Midterm Case Study: 25%
        - Final Project (Streamlit Optimization Dashboard): 35%
        - Weekly Hands-on Labs (PuLP & Excel Solver): 25%
        - Professionalism & Active Participation: 15%
        """,
        metadata={"source": "syllabus_overview.txt", "section": "grading"}
    ),
    Document(
        page_content="""
        Late Submission Policy:
        Assignments submitted within 24 hours of the deadline incur an automatic 15% deduction.
        Submissions beyond 24 hours receive zero credit unless an official medical or university
        accommodation is approved prior to the due date.
        
        Academic Integrity & AI Policy:
        Generative AI tools (ChatGPT, Claude, Copilot) are encouraged for debugging Python code and
        ideation. However, all submitted mathematical formulations and written analytical memos must be
        the student's original work. Direct copy-pasting of AI text without attribution constitutes a
        violation of academic honesty.
        """,
        metadata={"source": "course_policies.txt", "section": "integrity_and_deadlines"}
    ),
    Document(
        page_content="""
        Final Project Milestones:
        - Week 5: Topic Proposal & Business Case Definition (Submit 1-page executive summary).
        - Week 9: Mathematical Model Formulation & Baseline PuLP Script.
        - Week 14: Final Streamlit Application Deployment & 10-minute Executive Presentation.
        Late proposals are not accepted due to the compressed sprint schedule.
        """,
        metadata={"source": "final_project_guide.txt", "section": "milestones"}
    )
]

def build_vector_store(persist_directory: str = "./chroma_db", docs: list[Document] = SAMPLE_SYLLABUS_DOCS) -> Chroma:
    """Chunks documents, computes OpenAI embeddings, and writes to Chroma."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def get_retriever(persist_directory: str = "./chroma_db", k: int = 2):
    """Loads an existing Chroma database and returns a vector retriever."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

if __name__ == "__main__":
    print("Ingesting text and generating vector database...")
    build_vector_store()
    print("Vector database successfully built in ./chroma_db")