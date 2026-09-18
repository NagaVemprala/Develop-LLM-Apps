"""
01_research_papers_RAG.py
=========================

A self-contained Retrieval-Augmented Generation (RAG) app for research papers.

WHAT IT DOES (the five steps of RAG)
------------------------------------
  1. LOAD      Reads every PDF inside ./Input_Files, page by page (pypdf).
  2. SPLIT     Cuts each page into smaller, overlapping text chunks.
  3. EMBED     Converts every chunk into a vector using OpenAI embeddings.
  4. STORE     Saves vectors + text + metadata in a persistent Chroma vector
               database inside ./Output_VectorDB (created on first run).
  5. RETRIEVE  For a question, finds the most similar chunks and asks the LLM
     + GENERATE to answer using ONLY those chunks (grounded answers).

CONVERSATION MEMORY (why the chatbot stays on topic)
----------------------------------------------------
The 'chat' command keeps the conversation, so follow-ups such as "and what
about its limitations?" still make sense:
  * HISTORY        Every previous question and answer is replayed to the LLM
                   as normal chat turns (the last RAG_HISTORY_TURNS of them).
  * QUERY          A follow-up makes a poor search string ("what about that?"),
    REWRITING      so the LLM first rewrites it into a standalone question
                   using the history; that rewritten question is what gets
                   searched in the vector database, while the answer is still
                   written from the original question plus the full context.
The one-shot 'ask' command stays stateless, and 'clear' inside the chat
forgets the conversation so you can start a fresh topic.

Nothing is indexed twice: every chunk gets a stable id built from
(file name + page + chunk number), so re-running 'build' updates the existing
vectors instead of duplicating them.

QUICK START
-----------
  1. Drop the PDFs into the "Input_Files" folder.
  2. Make sure OPENAI_API_KEY is set in the .env file at the repository root.
  3. Install the extra packages:
         pip install chromadb langchain-chroma langchain-text-splitters pypdf
  4. Build the vector database:
         python 01_research_papers_RAG.py build
  5. Ask questions:
         python 01_research_papers_RAG.py ask "What is the main contribution?"
         However, the limitation of this RAG application is that the peper containing the
         maximum similarity with this question would be returned. Therefore, it is 
         recommended to run the code as below:  
         python 01_research_papers_RAG.py ask "What is the main contribution?" --all-papers
         python 01_research_papers_RAG.py chat      (interactive loop)

MULTI-PAPER / COMPARISON MODE
-----------------------------
A plain top-k search can return every chunk from a single paper, leaving the
other papers invisible to the model. When the question looks comparative
("compare the methodologies of my papers", "which paper solves X?") or when
--all-papers is given, the search is instead run once PER PDF and the results
are merged, so every paper is represented. Inside the chat, typing 'all'
toggles that mode on and off by hand.

COMMANDS
--------
  build [--reset]   Ingest the PDFs into Output_VectorDB.
  stats             Show how many chunks are stored, per file.
  search "..."      Show the raw chunks retrieved for a question (no LLM call).
                    Add --all-papers to inspect the per-paper fan-out.
  ask "..."         Retrieve relevant chunks and generate a grounded answer.
                    Add --all-papers to force comparison mode.
  chat              Interactive question/answer loop WITH conversation memory
                    (default when no command). Inside the loop: 'clear' starts a
                    fresh topic, 'history' shows what is remembered, 'all'
                    toggles comparison mode, 'exit' quits.
"""

# ============================================================================
# HOW THIS PROGRAM BEHAVES WHEN Input_Files CONTAINS SEVERAL PDF PAPERS
# ============================================================================
#
# WHAT HAPPENS TODAY (the simple version)
# ---------------------------------------
#   1. Every PDF in Input_Files is read page by page. Each page remembers which
#      file it came from and its page number (metadata: source, page).
#   2. Each page is cut into chunks, and ALL chunks from ALL papers are poured
#      into ONE Chroma collection (name: research_papers). The papers are mixed
#      into a single pool - the database has no idea that "paper 3" exists as a
#      whole document.
#   3. A question is turned into a vector and the k closest chunks are pulled
#      out of that pool. Nothing says "take a chunk from every paper": the
#      winners are simply the k most similar chunks, wherever they come from.
#   4. The answer is written from those chunks only, and each chunk is printed
#      with its file name and page, so a statement can always be traced back.
#
# WHAT ALREADY WORKS WELL WITH MANY PAPERS
# ----------------------------------------
#   * Tracing: every answer lists [file, page] for the chunks it used, so you
#     can tell which paper a statement came from.
#   * One paper at a time: ask about a specific paper and follow it up; the
#     conversation memory (see CONVERSATION MEMORY above) keeps you on it.
#   * Inventory: 'stats' shows how many chunks each PDF contributed, which
#     proves whether a file was ingested at all.
#   * Checking retrieval: 'search' prints the chunks a question would use,
#     before any answer is written.
#
# DRAWBACKS ONCE MANY PAPERS ARE ADDED
# ------------------------------------
#   1. No per-paper coverage. The top-k chunks can all come from one or two
#      papers. Measured on the single sample paper, the question "what
#      methodology does the paper use?" returned pages 3, 3, 2, 18 - two of
#      those four chunks were from the same page. With eight papers, four
#      chunks can easily all belong to paper #1, and the other seven papers
#      never reach the model at all.
#   2. Silent gaps. The model only sees the chunks it was handed, so it cannot
#      warn you that "paper 7 was not considered". Missing coverage looks
#      exactly like a confident, complete answer.
#   3. Comparison questions are unreliable. "Compare the methodologies of all
#      my papers" is answered from whichever few papers happened to match the
#      wording - not from every paper in the folder.
#   4. "Which paper solves X?" is a wording lottery. Retrieval matches the
#      wording of your QUESTION, not the truth about which paper addresses X,
#      so the best paper can be missed if it uses different vocabulary.
#   5. Chunks compete with each other. The abstract, introduction and
#      conclusion of one paper say similar things, so they crowd the other
#      papers out of the fixed k slots.
#   6. No document-level record. The database stores only (source, page,
#      chunk) per chunk - there is no title, year, abstract, problem statement
#      or methodology summary to search or filter on.
#   7. Themes spread across pages. "Problem solved" or "methodology" is usually
#      spread over several pages, so a single chunk shows only a fragment.
#
# WHAT CAN BE DONE TO IMPROVE IT
# ------------------------------
#   A. Name the PDFs meaningfully (free - do it today). The file name is what
#      appears in every citation, so "2021_Prova_RFMVDA_ecommerce.pdf" is far
#      more useful than "1_Paper1.pdf".
#   B. Per-paper fan-out retrieval. IMPLEMENTED (see retrieve_per_paper).
#      For comparison questions the search is run once PER FILE (Chroma filters
#      by source file), a couple of chunks are taken from each paper and the
#      answer is written from the union. This guarantees that every paper is
#      represented instead of hoping the top-k covers them. It switches on
#      automatically when the question sounds comparative, with --all-papers on
#      'search'/'ask'/'chat', or by typing 'all' inside the chat. The remaining
#      drawback is that a paper may simply not mention the topic: fan-out makes
#      sure the model SEES every paper and can say so, which is why option C
#      (a summary per paper) is still worth adding for "what is this paper
#      about?" style questions.
#   C. Build a "paper map" while ingesting. One LLM call per PDF extracts the
#      problem solved, methodology, dataset, results and limitations, stored in
#      a second collection (or a small JSON file). Questions like "which paper
#      solves X?" or "compare the methods" are then answered from the map, and
#      only afterwards drilled into the chunks. This is the real fix for
#      cross-paper questions, at the cost of one extra LLM call per PDF.
#   D. Enrich the metadata. Copy title/authors/year (from the PDF metadata or
#      the first page) into every chunk, so citations can read "(Smith et al.,
#      2021)" and retrieval can be filtered ("only 2023 papers").
#   E. Reduce redundancy. Use MMR (maximum marginal relevance) or a larger k so
#      near-duplicate chunks do not fill the slots. Helpful, but it does not
#      guarantee that every paper is covered.
#   F. Demand honest citation discipline. Ask the model to say explicitly when
#      a paper it was given does not contain the answer.
#
# HOW TO SEE THE CURRENT BEHAVIOUR YOURSELF
# -----------------------------------------
#   python 01_research_papers_RAG.py stats                  -> chunks per PDF
#   python 01_research_papers_RAG.py search "your question" -k 8
#       -> look at which files and pages come back. If they all belong to one
#          or two papers, that is drawback 1 in action.
# ============================================================================

# ============================================================
# 0. IMPORTS AND CONFIGURATION
# ============================================================

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Windows consoles sometimes default to a legacy encoding; make sure answer
# text (which can contain smart quotes) prints cleanly instead of as garbage.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Folder layout --------------------------------------------------------
# Paths are anchored to this file, so the script works no matter which
# directory you run it from.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input_Files"        # knowledge base: the source PDFs
VECTOR_DIR = BASE_DIR / "Output_VectorDB"   # Chroma persistence folder

# --- Environment (.env) ---------------------------------------------------
# Load the repository-root .env first, then an optional project-local .env.
# Already-set variables are never overwritten, so the root file is the
# primary source and a local file can only add missing keys.
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# --- Tunable settings (override any of them with environment variables) ---
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "research_papers")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-5.4-nano")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
BATCH_SIZE = 100                            # chunks embedded per API round-trip
HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "6"))   # remembered exchanges

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using "
    "ONLY the provided context taken from their PDF documents. "
    "If the context does not contain the answer, say clearly that the answer "
    "was not found in the documents instead of guessing. "
    "Earlier turns of the conversation are included so you can tell what the "
    "user is referring to (words like 'it', 'that paper' or 'the second one'), "
    "but never state facts that are not in the provided context. "
    "Keep the answer concise and, at the end, list which sources you used in "
    "the form [file name, page number]."
)

# Prompt used to turn a follow-up into a question that can be searched alone.
CONDENSE_PROMPT = (
    "You rewrite follow-up questions into standalone questions. "
    "You are given the conversation so far and the user's latest message. "
    "Rewrite that latest message so that it can be understood on its own and "
    "used as a search query against a document database: replace pronouns and "
    "vague references with the concrete subject they point to. "
    "If the message is already standalone, return it unchanged. "
    "Do not answer the question, do not explain, and do not add quotes: "
    "return only the rewritten question."
)

# --- Multi-paper (comparison) settings ------------------------------------
PER_PAPER_K = int(os.getenv("RAG_PER_PAPER_K", "2"))   # chunks taken per paper
AUTO_COMPARE = os.getenv("RAG_AUTO_COMPARE", "1").lower() not in {"0", "false", "no"}

# Words that usually mean "please look at ALL my papers, not just the few that
# happen to be most similar". Matching one of these switches on the per-paper
# fan-out automatically; --all-papers forces it and 'all' toggles it in chat.
COMPARE_KEYWORDS = (
    "compare", "comparison", "contrast", "versus", " vs ",
    "all papers", "all the papers", "all documents", "all pdfs",
    "each paper", "every paper", "the papers", "these papers",
    "across papers", "across the papers", "which paper",
    "differences between", "overview of the papers",
)

# Used instead of SYSTEM_PROMPT when several papers are in the context: the
# model must cover every paper it was given and admit the ones that say nothing
# about the question, which is what makes cross-paper answers trustworthy.
COMPARE_SYSTEM_PROMPT = (
    "You are a careful research assistant. The context below contains passages "
    "from SEVERAL different papers; each passage is labelled with the file it "
    "came from and its page. Answer the user's question by comparing them. "
    "Cover every paper that appears in the context: say what each one "
    "contributes, and state explicitly when a paper does not address the "
    "question at all. Never state a fact that is not in the context, and never "
    "make claims about a paper that is missing from it. Keep the answer "
    "organised per paper and finish with the sources used in the form "
    "[file name, page number]."
)


def banner(title: str) -> None:
    """Print a simple visual separator so the console output is easy to read."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_api_key() -> None:
    """Fail early with a friendly message when the OpenAI key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set.\n"
            "Add a line like OPENAI_API_KEY=sk-... to the .env file in the "
            "repository root and try again."
        )


# ============================================================
# 1. LOAD: READ THE PDF FILES FROM Input_Files
# ============================================================

def list_pdf_files() -> list[Path]:
    """Return every PDF inside Input_Files (searched recursively), sorted."""
    if not INPUT_DIR.exists():
        sys.exit(f"The input folder does not exist yet: {INPUT_DIR}")

    pdf_files = sorted(path for path in INPUT_DIR.rglob("*.pdf") if path.is_file())
    if not pdf_files:
        sys.exit(f"No PDF files found in {INPUT_DIR}. Add some and retry.")
    return pdf_files


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """Extract the text of one PDF, one Document per page.

    Metadata is attached to every page so that later we can always tell the
    LLM (and ourselves) exactly which file and page an answer came from.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            # Scanned/image-only pages have no extractable text layer.
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": page_number},
            )
        )
    return pages


# ============================================================
# 2. SPLIT: TURN PAGES INTO OVERLAPPING CHUNKS
# ============================================================

def build_chunks() -> tuple[list[Document], list[str]]:
    """Read every PDF and split it into chunks ready for embedding.

    Why chunk at all? Embedding models and LLMs have limited context, and a
    small, focused chunk produces a much sharper vector than a whole paper.
    The overlap keeps sentences that straddle a boundary in at least one
    chunk, so meaning is not cut in half.

    Returns the chunks plus one stable id per chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    chunk_ids: list[str] = []

    for pdf_path in list_pdf_files():
        pages = load_pdf_pages(pdf_path)
        page_chunk_total = 0

        for page_doc in pages:
            # Split one page at a time so every chunk keeps its page number.
            for index, chunk in enumerate(splitter.split_documents([page_doc]), start=1):
                chunk.metadata["chunk"] = index
                chunks.append(chunk)
                # Stable id => re-running 'build' overwrites instead of duplicating.
                chunk_ids.append(
                    f"{chunk.metadata['source']}::p{chunk.metadata['page']}::c{index}"
                )
                page_chunk_total += 1

        print(f"  {pdf_path.name}: {len(pages)} pages with text -> "
              f"{page_chunk_total} chunks")

    return chunks, chunk_ids


# ============================================================
# 3. VECTOR STORE: EMBEDDINGS + CHROMA
# ============================================================

def get_embeddings() -> OpenAIEmbeddings:
    """Create the embedding model that turns text into vectors."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store(embeddings: OpenAIEmbeddings | None = None) -> Chroma:
    """Open (or create) the persistent Chroma collection.

    Chroma writes its own files into VECTOR_DIR, so the database survives
    between runs: you build it once and query it as often as you like.
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings or get_embeddings(),
        # Cosine distance (0 = identical meaning, 2 = opposite) is the natural
        # metric for OpenAI embeddings, which are already length-normalised.
        collection_metadata={"hnsw:space": "cosine"},
    )


def collection_count(store: Chroma) -> int:
    """Return how many chunks are currently stored in the collection."""
    try:
        return store._collection.count()      # cheap: reads the count only
    except Exception:
        return len(store.get(include=["metadatas"]).get("ids") or [])


# ============================================================
# 4. BUILD: CREATE / REFRESH THE VECTOR DATABASE
# ============================================================

def build_vector_store(reset: bool = False) -> None:
    """Run the LOAD -> SPLIT -> EMBED -> STORE pipeline."""
    require_api_key()

    if reset and VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)
        print(f"Removed the existing vector database at {VECTOR_DIR.name}.")

    banner("STEP 1-2  Reading PDFs and splitting them into chunks")
    chunks, chunk_ids = build_chunks()
    if not chunks:
        sys.exit("No text could be extracted from the PDFs (are they scans?).")
    print(f"  Total chunks to embed: {len(chunks)}")

    banner(f"STEP 3-4  Embedding with '{EMBEDDING_MODEL}' and saving to Chroma")
    store = get_vector_store()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch_docs = chunks[start:start + BATCH_SIZE]
        batch_ids = chunk_ids[start:start + BATCH_SIZE]
        store.add_documents(batch_docs, ids=batch_ids)
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    banner("DONE")
    print(f"  Vector database : {VECTOR_DIR}")
    print(f"  Collection      : {COLLECTION_NAME}")
    print(f"  Stored chunks   : {collection_count(store)}")
    print("\n  Next: python 01_research_papers_RAG.py ask \"your question\"")


# ============================================================
# 5. RETRIEVE + GENERATE: ANSWER QUESTIONS FROM THE KNOWLEDGE BASE
# ============================================================

# Cache of the PDF names stored in the database. It is filled on first use so
# a chat session does not re-read every row of metadata on each question;
# restart the script after a rebuild to refresh it.
_SOURCE_CACHE: list[str] = []


def indexed_sources(store: Chroma | None = None) -> list[str]:
    """Return the sorted file names of the papers currently stored."""
    global _SOURCE_CACHE
    if _SOURCE_CACHE:
        return _SOURCE_CACHE

    store = store or get_vector_store()
    stored = store.get(include=["metadatas"])
    _SOURCE_CACHE = sorted(
        {(metadata or {}).get("source", "unknown")
         for metadata in stored.get("metadatas") or []}
    )
    return _SOURCE_CACHE


def is_comparison_question(text: str) -> bool:
    """True when the wording suggests the user wants all papers compared.

    Deliberately crude: a keyword hit only chooses a retrieval strategy, and
    --all-papers (or 'all' inside the chat) can force it either way.
    """
    lowered = text.lower()
    return any(keyword in lowered for keyword in COMPARE_KEYWORDS)


def retrieve(question: str, top_k: int = TOP_K, all_papers: bool = False,
             per_paper_k: int = PER_PAPER_K) -> list[tuple[Document, float]]:
    """Return the chunks that are semantically closest to the question.

    Normally this is one top_k similarity search across the whole collection.
    With all_papers=True it switches to the per-paper fan-out below, so that
    every paper in the database is represented at least once.
    """
    if all_papers:
        return retrieve_per_paper(question, per_paper_k)

    store = get_vector_store()
    if collection_count(store) == 0:
        sys.exit("The vector database is empty. Run 'build' first.")

    # score = distance (smaller means more similar)
    return store.similarity_search_with_score(question, k=top_k)


def search_one_paper(store: Chroma, vector: list[float], source: str,
                     k: int) -> list[tuple[Document, float]]:
    """Run one vector search restricted to a single PDF.

    Chroma's own query API is used here instead of the LangChain wrapper
    because it accepts a ready-made vector and returns distances, so the
    question is embedded only once for the whole fan-out.
    """
    raw = store._collection.query(
        query_embeddings=[vector],
        n_results=k,
        where={"source": source},           # only this PDF's chunks
        include=["documents", "metadatas", "distances"],
    )
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    return [
        (Document(page_content=text, metadata=meta or {}), float(distance))
        for text, meta, distance in zip(documents, metadatas, distances)
    ]


def retrieve_per_paper(question: str,
                       per_paper_k: int = PER_PAPER_K) -> list[tuple[Document, float]]:
    """Search EACH stored paper separately, then merge what comes back.

    This fixes the biggest multi-paper weakness: a plain top-k search can return
    four chunks from one paper, so the other papers never reach the model. Here
    every file gets its own search (filtered by source file), which guarantees
    coverage no matter how the wording of the question compares with the rest.

    The question is embedded once and the same vector is reused for every
    paper, so a fan-out costs one embedding call plus one local lookup per PDF.
    """
    store = get_vector_store()
    if collection_count(store) == 0:
        sys.exit("The vector database is empty. Run 'build' first.")

    sources = indexed_sources(store)
    vector = get_embeddings().embed_query(question)     # embed once, reuse below

    print(f"\nComparison mode: taking up to {per_paper_k} chunk(s) from each "
          f"of {len(sources)} paper(s).")

    merged: list[tuple[Document, float]] = []
    for source in sources:
        found = search_one_paper(store, vector, source, per_paper_k)
        best = f"{found[0][1]:.3f}" if found else "n/a"
        print(f"  - {source}: {len(found)} chunk(s), best distance {best}")
        merged.extend(found)

    # Closest first, and never the same chunk twice.
    merged.sort(key=lambda pair: pair[1])
    seen: set[tuple] = set()
    unique: list[tuple[Document, float]] = []
    for doc, score in merged:
        key = (doc.metadata.get("source"), doc.metadata.get("page"),
               doc.metadata.get("chunk"))
        if key not in seen:
            seen.add(key)
            unique.append((doc, score))
    return unique


def format_context(results: list[tuple[Document, float]]) -> str:
    """Flatten the retrieved chunks into a single, clearly labelled context."""
    blocks = []
    for position, (doc, _score) in enumerate(results, start=1):
        header = f"[{position}] {doc.metadata.get('source')} "                 f"(page {doc.metadata.get('page')}, chunk {doc.metadata.get('chunk')})"
        blocks.append(header + "\n" + doc.page_content)
    return "\n\n".join(blocks)


def format_context_by_paper(results: list[tuple[Document, float]]) -> str:
    """Group the retrieved chunks per paper, for comparison questions.

    Compare mode is about WHICH paper says what, so the context is laid out as
    one section per file instead of a single flat list of chunks.
    """
    grouped: dict[str, list[tuple[Document, float]]] = {}
    for doc, score in results:
        name = doc.metadata.get("source", "unknown")
        grouped.setdefault(name, []).append((doc, score))

    sections = []
    for name, items in grouped.items():
        lines = [f"### {name}"]
        for doc, _score in items:
            lines.append(
                f"[page {doc.metadata.get('page')}, chunk "
                f"{doc.metadata.get('chunk')}] {doc.page_content}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def print_sources(results: list[tuple[Document, float]], full_text: bool = False) -> None:
    """Show retrieved chunks with clear separation and optional full text."""
    banner("RETRIEVED SOURCE PASSAGES")
    if not results:
        print("No matching chunks found.")
        return

    for position, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk = doc.metadata.get("chunk", "?")

        print(f"\n[{position}] {source} | Page {page} | Chunk {chunk} | (Distance: {score:.3f})")
        print("-" * 70)

        if full_text:
            # Print the entire chunk cleanly indented
            clean_text = doc.page_content.strip()
            print(clean_text)
        else:
            # Expanded preview (350 characters instead of 90)
            clean_preview = " ".join(doc.page_content.split())[:350]
            print(f"{clean_preview}...")
        print("-" * 70)

def build_llm(temperature: float = 0) -> ChatOpenAI:
    """Create the chat model used for both rewriting and answering.

    Building it once per session (and reusing it) avoids re-creating the
    client on every question.
    """
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


def history_to_messages(history: list[tuple[str, str]]) -> list:
    """Turn stored (question, answer) pairs back into real chat messages.

    Only the plain question and answer text is replayed - not the big CONTEXT
    blocks from earlier turns - so the prompt stays small even in a long chat.
    """
    messages = []
    for previous_question, previous_answer in history:
        messages.append(HumanMessage(content=previous_question))
        messages.append(AIMessage(content=previous_answer))
    return messages


def condense_question(question: str, history: list[tuple[str, str]] | None,
                      llm: ChatOpenAI | None = None) -> str:
    """Rewrite a follow-up into a question that can be searched on its own.

    "and what about its limitations?" is a bad search string: alone it names no
    subject, so the vector search would return almost anything. Given the
    conversation, the LLM turns it into something like "what are the limitations
    of the RFMVDA model?" - that is what we search for.

    With no history this is a no-op, so the first question of a conversation
    costs nothing extra and cannot be distorted by the rewriting step.
    """
    if not history:
        return question

    llm = llm or build_llm()
    messages = [
        SystemMessage(content=CONDENSE_PROMPT),
        *history_to_messages(history),
        HumanMessage(content=question),
    ]
    rewritten = str(llm.invoke(messages).content).strip().strip('"').strip()
    return rewritten or question


def wants_comparison(question: str, search_query: str,
                     forced: bool = False) -> bool:
    """Decide whether this question should look at every paper.

    True when the user asked for it (--all-papers, or 'all' in the chat), or
    when the wording itself sounds comparative and auto-detection is enabled.
    """
    if forced:
        return True
    if not AUTO_COMPARE:
        return False
    return is_comparison_question(question) or is_comparison_question(search_query)


def answer_question(question: str, top_k: int = TOP_K,
                    history: list[tuple[str, str]] | None = None,
                    llm: ChatOpenAI | None = None,
                    all_papers: bool = False,
                    per_paper_k: int = PER_PAPER_K) -> str:
    """Full RAG step: retrieve relevant chunks, then let the LLM answer.

    Pass 'history' - a list of previous (question, answer) pairs - to keep the
    conversation going, so follow-up questions are understood in context.
    Pass all_papers=True (or just ask a comparative question) to collect chunks
    from every PDF instead of only the few best-matching ones.
    """
    llm = llm or build_llm()

    # 1. Make the question searchable on its own (no-op without history).
    search_query = condense_question(question, history, llm)
    if search_query != question:
        print(f"\nFollow-up understood as: {search_query}")

    # 2. Pick the retrieval strategy, then fetch the chunks.
    compare = wants_comparison(question, search_query, all_papers)
    results = retrieve(search_query, top_k, all_papers=compare,
                       per_paper_k=per_paper_k)

    # 3. Lay the context out for the task: per paper when comparing, as one
    #    flat list otherwise, and use the matching system prompt.
    if compare:
        context = format_context_by_paper(results)
        system_prompt = COMPARE_SYSTEM_PROMPT
    else:
        context = format_context(results)
        system_prompt = SYSTEM_PROMPT

    messages = [
        SystemMessage(content=system_prompt),
        *history_to_messages(history or []),
        HumanMessage(
            content=f"CONTEXT FROM THE DOCUMENTS:\n{context}\n\nQUESTION: {question}"
        ),
    ]
    answer = llm.invoke(messages).content
    print_sources(results)
    return answer if isinstance(answer, str) else str(answer)


def show_stats() -> None:
    """Report what is currently stored in the vector database."""
    store = get_vector_store()
    count = collection_count(store)

    banner("VECTOR DATABASE CONTENTS")
    print(f"  Location   : {VECTOR_DIR}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Chunks     : {count}")
    if count == 0:
        print("\n  The database is empty. Run: python 01_research_papers_RAG.py build")
        return

    try:
        stored = store.get(include=["metadatas"])
        per_file = Counter(
            (metadata or {}).get("source", "unknown")
            for metadata in stored.get("metadatas") or []
        )
        print("\n  Chunks per file:")
        for name, amount in sorted(per_file.items()):
            print(f"    - {name}: {amount}")
    except Exception as error:                      # details are optional
        print(f"  (could not list per-file details: {error})")


def interactive_chat(top_k: int = TOP_K, turns: int = HISTORY_TURNS,
                     all_papers: bool = False,
                     per_paper_k: int = PER_PAPER_K) -> None:
    """Conversational REPL: follow-up questions keep the previous context.

    The conversation lives in 'history' as (question, answer) pairs. It is
    replayed to the LLM on every new question and is used to rewrite follow-ups
    into standalone search queries, so the chat behaves like a real assistant.

    Comparison mode (one search per paper instead of a single top-k search) is
    switched on automatically for comparative questions; 'all' forces it on or
    off for the rest of the session.

    Commands inside the loop:
      exit / quit / q   leave the chat
      clear / reset     forget the conversation and start a new topic
      history           show the exchanges that are currently remembered
      all               toggle per-paper (comparison) mode on/off
    """
    count = collection_count(get_vector_store())
    if count == 0:
        sys.exit("The vector database is empty. Run 'build' first.")

    llm = build_llm()                       # one client for the whole session
    history: list[tuple[str, str]] = []
    compare_all = all_papers                # forced by --all-papers, toggled by 'all'

    banner("RESEARCH RAG - INTERACTIVE CHAT")
    print(f"  Knowledge base : {INPUT_DIR.name}  ({count} chunks indexed)")
    print("  Ask a question, then ask follow-ups - the conversation is kept.")
    print(f"  Remembering the last {turns} exchange(s); type 'clear' to forget.")
    print("  Comparison questions automatically search every paper once.")
    print("  Commands: 'clear' (new topic), 'history' (show), 'all' (per-paper "
          "mode), 'exit' (quit).")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        command = question.lower()
        if command in {"exit", "quit", "q"}:
            print("Bye!")
            break
        if command in {"clear", "reset"}:
            history.clear()
            print("Conversation cleared - the next question starts a new topic.")
            continue
        if command == "history":
            if not history:
                print("Nothing remembered yet.")
            else:
                print("Remembered exchanges:")
                for position, (asked, _answered) in enumerate(history, start=1):
                    print(f"  {position}. {asked}")
            continue
        if command in {"all", "compare"}:
            compare_all = not compare_all
            print("Per-paper comparison mode is now "
                  + ("ON - every paper is searched separately."
                     if compare_all else
                     "OFF - only the closest chunks are used."))
            continue
        if not question:
            continue

        try:
            print("\nAnswer:\n")
            answer = answer_question(question, top_k, history, llm,
                                     compare_all, per_paper_k)
            print(answer)

            # Remember this exchange so the next question has the context.
            history.append((question, answer))
            if turns > 0:
                del history[:-turns]        # keep only the newest 'turns' pairs
            else:
                history.clear()             # --turns 0 -> stateless chat
        except Exception as error:          # keep the loop alive
            print(f"Something went wrong: {error}")


# ============================================================
# 6. COMMAND LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    """Define the small command set described at the top of this file."""
    parser = argparse.ArgumentParser(
        description="RAG over the research papers stored in Input_Files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="ingest the PDFs into the vector database")
    build.add_argument("--reset", action="store_true",
                       help="delete Output_VectorDB first, then rebuild from scratch")

    sub.add_parser("stats", help="show what is stored in the vector database")

    # Shared options: every question command can switch to per-paper retrieval.
    def add_paper_options(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--all-papers", action="store_true", dest="all_papers",
                            help="search every paper separately instead of the "
                                 "best-matching chunks (comparison mode)")
        parser.add_argument("--per-paper-k", type=int, default=PER_PAPER_K,
                            dest="per_paper_k",
                            help="how many chunks to take from each paper in "
                                 "comparison mode")

    search = sub.add_parser("search", help="show retrieved chunks, without the LLM")
    search.add_argument("question", help="the question to search for")
    search.add_argument("-k", "--top-k", type=int, default=TOP_K, dest="top_k")
    add_paper_options(search)

    ask = sub.add_parser("ask", help="ask one question and get a grounded answer")
    ask.add_argument("question", help="the question to ask")
    ask.add_argument("-k", "--top-k", type=int, default=TOP_K, dest="top_k")
    add_paper_options(ask)

    chat = sub.add_parser(
        "chat",
        help="interactive question/answer loop with conversation memory")
    chat.add_argument("-k", "--top-k", type=int, default=TOP_K, dest="top_k")
    chat.add_argument("--turns", type=int, default=HISTORY_TURNS, dest="turns",
                      help="how many previous exchanges to remember (0 = none)")
    add_paper_options(chat)

    return parser.parse_args()


def main() -> None:
    print("Are you entering the application?")
    args = parse_args()
    command = args.command or "chat"        # no arguments -> interactive chat
    print(args, command)
    if command == "build":
        build_vector_store(reset=args.reset)
    elif command == "stats":
        show_stats()
    elif command == "search":
        require_api_key()
        print_sources(retrieve(args.question, args.top_k, args.all_papers,
                               args.per_paper_k))
    elif command == "ask":
        require_api_key()
        print("\nAnswer:\n")
        print(answer_question(args.question, args.top_k,
                              all_papers=args.all_papers,
                              per_paper_k=args.per_paper_k))
    else:
        require_api_key()
        interactive_chat(args.top_k, args.turns, args.all_papers,
                         args.per_paper_k)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
