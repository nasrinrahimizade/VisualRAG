import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingestion import load_and_chunk_papers


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "cv_papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Load the HuggingFace embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def build_vector_store(papers_dir: str = "data/papers") -> Chroma:
    """
    Loads PDFs, embeds them, and persists to ChromaDB.
    Only call this once — or when you add new papers.
    """
    print("Building vector store from scratch...")

    # Load and chunk all papers
    chunks = load_and_chunk_papers(papers_dir)

    # Load embedding model
    embeddings = get_embeddings()

    # Create and persist the Chroma vector store
    print(f"\nEmbedding {len(chunks)} chunks into ChromaDB (this may take a few minutes)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector store saved to: {CHROMA_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    Use this after the first build — much faster.
    """
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    count = vector_store._collection.count()
    print(f"Loaded vector store: {count} chunks in collection '{COLLECTION_NAME}'")
    return vector_store


def get_or_build_vector_store(papers_dir: str = "data/papers") -> Chroma:
    """
    Smart loader: builds the vector store if it doesn't exist,
    otherwise loads the existing one from disk.
    """
    chroma_path = Path(CHROMA_DIR)

    if chroma_path.exists() and any(chroma_path.iterdir()):
        print("Existing vector store found. Loading from disk...")
        return load_vector_store()
    else:
        print("No existing vector store found. Building now...")
        return build_vector_store(papers_dir)


def query(vector_store: Chroma, text: str, n_results: int = 5) -> list:
    """
    Query the vector store and return the top-k most similar chunks.

    Args:
        vector_store: The loaded Chroma vector store.
        text: The query string.
        n_results: Number of results to return (default 5).

    Returns:
        A list of LangChain Document objects with content and metadata.
    """
    results = vector_store.similarity_search(text, k=n_results)
    return results


def print_results(results: list):
    """Pretty-print query results for debugging."""
    print(f"\n{'='*60}")
    print(f"Top {len(results)} results:")
    print(f"{'='*60}")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Source : {doc.metadata.get('source', 'Unknown')}")
        print(f"    Page   : {doc.metadata.get('page', '?')}")
        print(f"    Preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    # Build or load the vector store
    vs = get_or_build_vector_store()

    # Run a test query
    test_query = "What methods are used for object detection?"
    print(f"\nRunning test query: '{test_query}'")

    results = query(vs, test_query, n_results=5)
    print_results(results)