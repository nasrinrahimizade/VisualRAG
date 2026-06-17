import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def load_and_chunk_papers(papers_dir: str = "data/papers") -> list:
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    pdf_files = sorted(papers_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {papers_dir}")

    print(f"Found {len(pdf_files)} PDF(s). Loading...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            for chunk in chunks:
                chunk.page_content = clean_chunk_text(chunk.page_content)
                chunk.metadata["source"] = pdf_path.name
                chunk.metadata["page"] = chunk.metadata.get("page", 0)

            # Filter out chunks that became too short after cleaning
            chunks = [c for c in chunks if len(c.page_content) > 100]
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  WARNING: Could not process {pdf_path.name}: {e}")
            continue

    print(f"\nDone. Total chunks: {len(all_chunks)}")
    return all_chunks


def clean_chunk_text(text: str) -> str:
    """Remove LaTeX encoding and other noise from PDF chunks."""
    # Remove LaTeX encoded strings like <latexit sha1_base64="...">...</latexit>
    text = re.sub(r'<latexit[^>]*>.*?</latexit>', '', text, flags=re.DOTALL)
    # Remove leftover base64 garbage
    text = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()


if __name__ == "__main__":
    chunks = load_and_chunk_papers()
    sample = chunks[0]
    print("\n--- Sample Chunk ---")
    print(f"Source : {sample.metadata['source']}")
    print(f"Page   : {sample.metadata['page']}")
    print(f"Preview: {sample.page_content[:300]}")