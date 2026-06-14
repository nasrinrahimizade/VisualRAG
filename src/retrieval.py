# This file handles converting chunks into vectors and storing/retrieving them.

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from ingestion import load_and_chunk_papers

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "cv_papers"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_chroma_client():
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_vector_store(papers_dir: str = None) -> Chroma:
    if papers_dir is None:
        papers_dir = str(Path(__file__).parent.parent / "data" / "papers")

    print("Building vector store from scratch...")
    chunks = load_and_chunk_papers(papers_dir)
    embeddings = get_embeddings()

    # Use persistent client directly
    client = get_chroma_client()

    # Delete collection if it exists to start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except:
        pass

    print(f"\nEmbedding {len(chunks)} chunks into ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client,  # use persistent client instead of persist_directory
    )

    # Verify it was saved
    count = client.get_collection(COLLECTION_NAME).count()
    print(f"Vector store saved. Verified {count} chunks in ChromaDB.")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    client = get_chroma_client()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )

    count = vector_store._collection.count()
    print(f"Loaded vector store: {count} chunks in '{COLLECTION_NAME}'")
    return vector_store


def get_or_build_vector_store(papers_dir: str = None) -> Chroma:
    client = get_chroma_client()

    # Check if collection exists and has data
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            print(f"Existing vector store found ({count} chunks). Loading...")
            return load_vector_store()
        else:
            print("Collection exists but is empty. Rebuilding...")
            return build_vector_store(papers_dir)
    except:
        print("No existing vector store found. Building now...")
        return build_vector_store(papers_dir)


def query(vector_store: Chroma, text: str, n_results: int = 5, source_filter: str = None) -> list:
    if source_filter:
        results = vector_store.max_marginal_relevance_search(
            text, k=n_results, filter={"source": source_filter}
        )
    else:
        results = vector_store.max_marginal_relevance_search(text, k=n_results)
    return results


def print_results(results: list):
    print(f"\n{'='*60}")
    print(f"Top {len(results)} results:")
    print(f"{'='*60}")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Source : {doc.metadata.get('source', 'Unknown')}")
        print(f"    Page   : {doc.metadata.get('page', '?')}")
        print(f"    Preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    vs = get_or_build_vector_store()
    results = query(vs, "PhysTwin tracking deformable objects")
    print_results(results)