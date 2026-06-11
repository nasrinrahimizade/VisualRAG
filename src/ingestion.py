import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
                chunk.metadata["source"] = pdf_path.name
                chunk.metadata["page"] = chunk.metadata.get("page", 0)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  WARNING: Could not process {pdf_path.name}: {e}")
            continue

    print(f"\nDone. Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_papers()
    sample = chunks[0]
    print("\n--- Sample Chunk ---")
    print(f"Source : {sample.metadata['source']}")
    print(f"Page   : {sample.metadata['page']}")
    print(f"Preview: {sample.page_content[:300]}")