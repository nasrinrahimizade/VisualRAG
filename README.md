# VisualRAG — Multimodal Research Assistant for Computer Vision Papers

A RAG-powered research assistant that answers questions grounded in a personal collection of computer vision and robotics papers. Every answer is sourced directly from your corpus with citations to specific papers and page numbers — no hallucination.

> 🚧 Image input support (BLIP-2) coming soon.

---

## What it does

- Ask a natural language question about computer vision or robotics
- The system retrieves the most relevant chunks from your paper collection
- An LLM generates a grounded answer citing specific papers and page numbers
- If the answer isn't in the corpus, the system says so

---

## Architecture

```
User Question
      │
      ▼
 Embedding Model (all-mpnet-base-v2)
      │
      ▼
 ChromaDB Vector Store (23 papers, ~1591 chunks)
      │
      ▼
 Retrieved Top-5 Chunks
      │
      ▼
 Groq LLM (Llama 3.3 70B)
      │
      ▼
 Answer + Citations (paper + page number)
```

---

## Stack

| Component | Tool |
|---|---|
| PDF loading & chunking | LangChain + PyPDFLoader |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Vector store | ChromaDB (local, persistent) |
| LLM | Llama 3.3 70B via Groq API (free) |
| Orchestration | LangChain LCEL |
| Interface | Streamlit (coming soon) |

---

## Example

**Q:** What is the mAP of Fast YOLO on PASCAL VOC 2007?

**A:** According to Yolo.pdf, Fast YOLO achieves a mAP of 52.7% at 155 FPS on PASCAL VOC 2007, making it the fastest object detector on record while being twice as accurate as any other real-time detector at the time.

**Sources:** `Yolo.pdf — Page 4`

<!-- Add screenshot here -->

---

## Retrieval Evaluation

Tested on 10 questions with known answers from the corpus. **Score: 8/10.**

| Question | Result |
|---|---|
| mAP of Fast YOLO on PASCAL VOC 2007? | ✓ |
| PhysTwin core representation? | ✓ |
| HOPE-Net evaluation dataset? | ✓ |
| HOI4D main contribution? | ✓ |
| HandOccNet occlusion handling? | ✓ |
| AffordPose main task? | ✓ |
| HOIDiffusion data generation? | ✓ |
| SHOWMe benchmark method? |✗ |
| RealSense sensor discussion? | ✓ |
| HO-3D dataset backbone? | ✗ |

---

## Setup

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/yourusername/visualrag.git
cd visualrag

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Place your PDF papers in `data/papers/`, then build the vector store:

```bash
python src/retrieval.py
```

This runs once and takes 10–20 minutes. After that it loads from disk in seconds.

Test the pipeline:

```bash
python src/rag_chain.py
```

---

## Project Structure

```
visualrag/
├── data/papers/         ← PDF papers
├── notebooks/           ← development and evaluation notebooks
├── src/
│   ├── ingestion.py     ← PDF loading and chunking
│   ├── retrieval.py     ← ChromaDB vector store
│   └── rag_chain.py     ← LLM + RAG chain
├── chroma_db/           ← auto-generated, gitignored
└── .env                 ← API keys, gitignored
```

---

## Known Limitations

- Scanned PDFs with no text layer will not be indexed
- LaTeX-heavy figure captions can produce garbled chunks
- Specific factual queries retrieve better than broad conceptual ones

