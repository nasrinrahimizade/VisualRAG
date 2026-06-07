import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_papers(papers_dir: str = "data/papers") -> list:
    """
    Loads all PDFs from the given directory, splits them into chunks,
    and attaches metadata to each chunk.

    Args:
        papers_dir: Path to the folder containing your PDF papers.

    Returns:
        A list of LangChain Document objects (chunks), each with metadata.
    """

    papers_path = Path(papers_dir)

    if not papers_path.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    # Collect all PDF files in the directory
    pdf_files = sorted(papers_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {papers_dir}")

    print(f"Found {len(pdf_files)} PDF(s). Loading...")

    # Set up the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")

        try:
            # Load the PDF — returns one Document per page
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Split the pages into chunks
            chunks = splitter.split_documents(pages)

            # Attach/clean metadata on every chunk
            for chunk in chunks:
                chunk.metadata["source"] = pdf_path.name
                chunk.metadata["filepath"] = str(pdf_path)
                # PyPDFLoader already adds "page", but we make it explicit
                chunk.metadata["page"] = chunk.metadata.get("page", 0)

            all_chunks.extend(chunks)

        except Exception as e:
            print(f"  WARNING: Could not process {pdf_path.name}: {e}")
            continue

    print(f"\nDone. Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_papers()

    # Quick sanity check
    sample = chunks[0]
    print("\n--- Sample Chunk ---")
    print(f"Source : {sample.metadata['source']}")
    print(f"Page   : {sample.metadata['page']}")
    print(f"Length : {len(sample.page_content)} characters")
    print(f"Preview: {sample.page_content[:300]}")