import hashlib
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("./Data/papers")
DB_DIR = "./chroma_db"

def get_existing_indexed_files(vectorstore: Chroma) -> Set[str]:
    """Inspects the vector store to see which file names are already present."""
    # Chroma allows fetching metadata directly from the collection
    existing_records = vectorstore.get(include=["metadatas"])
    metadatas = existing_records.get("metadatas", [])
    
    indexed_files = {m.get("file_name") for m in metadatas if m and "file_name" in m}
    return indexed_files

def generate_chunk_id(doc: Document) -> str:
    """
    Creates a deterministic unique ID based on file name, page, and content hash.
    Ensures safe, idempotent re-runs without duplicate embeddings.
    """
    source = doc.metadata.get("file_name", "unknown")
    page = doc.metadata.get("page", 0)
    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
    return f"{source}_p{page}_{content_hash}"

def process_and_add_new_documents():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 1. Connect to the existing Chroma database
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    # 2. Identify which files are already in the database
    already_indexed = get_existing_indexed_files(vectorstore)
    print(f"Already indexed files ({len(already_indexed)}): {already_indexed}")

    # 3. Detect new, unprocessed PDFs
    all_pdfs = list(DATA_DIR.glob("*.pdf"))
    new_pdfs = [p for p in all_pdfs if p.name not in already_indexed]

    if not new_pdfs:
        print("No new PDF files found to ingest. Database is up to date.")
        return

    print(f"Found {len(new_pdfs)} new file(s) to add: {[p.name for p in new_pdfs]}")

    # 4. Load and tag only the new documents
    raw_pages: List[Document] = []
    for pdf_path in new_pdfs:
        print(f"Loading new file: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for p in pages:
            p.metadata["file_name"] = pdf_path.name
            p.metadata["paper_slug"] = pdf_path.stem
        raw_pages.extend(pages)

    # 5. Chunk the newly loaded pages
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "]
    )
    new_chunks = splitter.split_documents(raw_pages)

    # 6. Generate IDs for deduplication
    chunk_ids = [generate_chunk_id(chunk) for chunk in new_chunks]

    # 7. Append only the new chunks to the existing database
    print(f"Embedding and adding {len(new_chunks)} chunks to Chroma...")
    vectorstore.add_documents(documents=new_chunks, ids=chunk_ids)

    print("Incremental ingestion complete! New documents are now live in the RAG pipeline.")

if __name__ == "__main__":
    process_and_add_new_documents()