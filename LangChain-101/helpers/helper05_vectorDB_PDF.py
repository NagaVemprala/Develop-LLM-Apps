import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("./Data/papers")
DB_DIR = "./chroma_db"

def load_and_chunk_papers(pdf_dir: Path) -> List[Document]:
    """Reads all PDF research papers and splits them into semantically rich chunks."""
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created '{pdf_dir}'. Please drop your e-commerce research PDFs in the created folder.")
        return []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'.")
        return []

    documents = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        raw_pages = loader.load()
        
        # Enrich document metadata with clean filename for citation
        for page in raw_pages:
            page.metadata["paper_title"] = pdf_path.stem
            page.metadata["file_name"] = pdf_path.name
        documents.extend(raw_pages)

    # 800-token chunks with 100-token overlap preserve full methodology and hypothesis blocks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Extracted {len(documents)} pages -> {len(chunks)} searchable chunks.")
    return chunks

def build_vector_store(persist_dir: str = DB_DIR) -> Chroma:
    """Builds and persists Chroma index with text-embedding-3-small embeddings."""
    chunks = load_and_chunk_papers(DATA_DIR)
    if not chunks:
        raise ValueError("No documents processed. Add PDFs into data/papers/ first.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector store successfully indexed in '{persist_dir}'.")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()