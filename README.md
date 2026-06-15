# VisualRAG — Multimodal Research Assistant for Computer Vision Papers

A RAG-powered research assistant that answers questions grounded in a personal collection of computer vision and robotics papers. Supports both **text queries** and **image queries** — upload an image and get back relevant papers and answers. Every answer is cited with paper filename and page number.

---

## What it does

- Ask a natural language question about computer vision or robotics
- **Or upload an image** — a robot, a depth map, a segmentation result — and get relevant papers back
- The system retrieves the most relevant chunks from your paper collection
- An LLM generates a grounded answer citing specific papers and page numbers
- If the answer isn't in the corpus, the system says so — no hallucination

---

## Architecture

```
User Input
    │
    ├── Text query ──────────────────────────────────┐
    │                                                 │
    └── Image → ViT-GPT2 captioner                   │
                generates text description            │
                        │                            │
                        ▼                            ▼
               Enriched query: "Computer vision research related to: {caption}"
                                         │
                                         ▼
                        Embedding Model (all-mpnet-base-v2)
                                         │
                                         ▼
                        ChromaDB Vector Store
                        (23 papers, ~1591 chunks)
                        MMR retrieval (k=4, fetch_k=12)
                                         │
                                         ▼
                        Groq LLM (Llama 3.1 8B)
                        grounded prompt — no hallucination
                                         │
                                         ▼
                        Answer + Citations
                        (paper filename + page number)
```

---

## Stack

| Component | Tool |
|---|---|
| PDF loading & chunking | LangChain + PyPDFLoader |
| Text splitting | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| Embeddings | HuggingFace `sentence-transformers/all-mpnet-base-v2` |
| Vector store | ChromaDB (local, persistent) |
| Retrieval strategy | MMR — Maximum Marginal Relevance |
| Image captioning | BLIP-2 (`Salesforce/blip2-opt-2.7b`) |
| LLM | Llama 3.1 8B via Groq API (free) |
| Orchestration | LangChain LCEL |
| Interface | Streamlit (coming soon) |

---

## Paper Corpus (23 papers)

The corpus focuses on hand-object interaction, pose estimation, and object detection:

- Hand pose estimation and reconstruction surveys
- Hand-object interaction datasets (HOI4D, AffordPose, H2O, SHOWMe)
- 3D reconstruction methods (PhysTwin, MagicHOI, HOIDiffusion)
- Object detection baselines (YOLO)
- Depth camera documentation (Intel RealSense D400)

---

## Example — Text Query

**Q:** What is the mAP of Fast YOLO on PASCAL VOC 2007?

**A:** According to Yolo.pdf, Fast YOLO achieves a mAP of 52.7% at 155 FPS on PASCAL VOC 2007, making it the fastest object detector on record while being twice as accurate as any other real-time detector at the time.

**Sources:** `Yolo.pdf — Page 4`

---

## Example — Image Query

**Image:** Hand gesture photos

**Generated caption:** "a series of images showing different types of hand gestures"

**Top source:** `A survey of deep learning methods and datasets for hand pose.pdf`

**A:** According to the survey paper, hand gesture recognition is a topic of active interest in the hand pose estimation literature...

<!-- Add screenshot here -->

---

## Evaluation

### Text Retrieval (10 questions with known answers)

**Score: 8/10**

| Question | Result |
|---|---|
| mAP of Fast YOLO on PASCAL VOC 2007? | ✓ |
| PhysTwin core representation? | ✓ |
| HOPE-Net evaluation dataset? | ✓ |
| HOI4D main contribution? | ✓ |
| HandOccNet occlusion handling? | ✓ |
| AffordPose main task? | ✓ |
| HOIDiffusion data generation? | ✓ |
| SHOWMe benchmark method? | ✓ |
| RealSense sensor discussion? | ✗ |
| HO-3D dataset backbone? | ✗ |

### Image Retrieval (4 test images)

**Score: 2/4**

| Image | Caption Quality | Retrieval |
|---|---|---|
| Robot arm grasping | ✓ Good caption | ✗ Corpus lacks industrial robotics papers |
| Hand gestures | ✓ Good caption | ✓ Correct papers retrieved |
| Segmentation mask | ✓ Good caption | ✓ Correct papers retrieved |
| RealSense camera | ✗ Misidentified as thermometer | ✗ Wrong retrieval |

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
HF_HOME=E:\huggingface_cache   # optional: redirect model cache away from C: drive
```

Place your PDF papers in `data/papers/`, then build the vector store (one-time, ~15 min):

```bash
python src/retrieval.py
```

Test the text pipeline:

```bash
python src/rag_chain.py
```

---

## Project Structure

```
visualrag/
├── data/
│   ├── papers/              ← PDF papers
│   └── test_images/         ← test images for image query evaluation
├── notebooks/
│   ├── 01_retrieval_pipeline.ipynb   ← text RAG testing and evaluation
│   └── 02_image_input.ipynb          ← image query testing
├── src/
│   ├── ingestion.py         ← PDF loading and chunking
│   ├── retrieval.py         ← ChromaDB vector store
│   ├── rag_chain.py         ← LLM + RAG chain + image query
│   └── image_captioner.py   ← BLIP-2 image captioning
├── chroma_db/               ← auto-generated, gitignored
└── .env                     ← API keys, gitignored
```

---

## Known Limitations

- Scanned PDFs with no text layer will not be indexed
- LaTeX-heavy figure captions produce garbled chunks
- Image captioning quality depends on the model — technical equipment (depth cameras, sensors) is sometimes misidentified
- Image retrieval works best when the corpus contains papers directly related to what's in the image
- Specific factual queries retrieve better than broad conceptual ones

