import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval import get_or_build_vector_store
from rag_chain import build_rag_chain, ask
from image_captioner import load_captioner, caption_image

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="VisualRAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }

    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 8px 0px !important;
    }

    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 16px !important;
        color: #1a1a1a !important;
    }

    .stButton button {
        background-color: #f7f7f8 !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
    }

    .stButton button:hover {
        background-color: #ebebeb !important;
    }

    .attach-btn button {
        background-color: #f0f0f0 !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        padding: 0 !important;
        font-size: 1.2em !important;
        border: 1px solid #e5e5e5 !important;
    }

    .attach-btn button:hover {
        background-color: #e0e0e0 !important;
    }

    [data-testid="stExpander"] {
        background-color: #f7f7f8 !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #f7f7f8 !important;
        border: 1px dashed #e5e5e5 !important;
        border-radius: 8px !important;
    }

    hr { border-color: #e5e5e5 !important; }
    .stCaption { color: #6e6e80 !important; }

    .source-card {
        background-color: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85em;
    }

    .source-title {
        color: #10a37f;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .source-preview {
        color: #6e6e80;
        font-size: 0.9em;
    }

    .welcome-container {
        text-align: center;
        padding: 40px 20px;
    }

    .welcome-title {
        font-size: 2em;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        font-size: 1em;
        color: #6e6e80;
        margin-bottom: 30px;
    }

    /* Remove default container border from st.container(height=...) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models once and cache ──────────────────────────────
@st.cache_resource
def load_all():
    vs = get_or_build_vector_store()
    chain, retriever = build_rag_chain(vs)
    processor, model = load_captioner()
    return chain, retriever, processor, model


with st.spinner("Loading models..."):
    chain, retriever, processor, model = load_all()


# ── Session state ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 VisualRAG")
    st.markdown("Research assistant grounded in computer vision papers.")

    st.divider()

    if st.button("✏️ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.show_uploader = False
        st.session_state.pending_image = None
        st.rerun()

    st.divider()

    st.markdown("**Corpus**")
    st.caption("23 papers — hand-object interaction, pose estimation, object detection")

    st.divider()

    st.markdown("**Stack**")
    st.caption("Embeddings: all-mpnet-base-v2")
    st.caption("LLM: Llama 3.1 8B · Groq")
    st.caption("Vision: BLIP-2")
    st.caption("Store: ChromaDB · MMR retrieval")

    st.divider()
    st.caption(f"Messages: {len(st.session_state.chat_history)}")


# ── Top bar ─────────────────────────────────────────────────
_, top_col = st.columns([6, 1])
with top_col:
    if st.button("✏️ New Chat", key="top_new_chat"):
        st.session_state.chat_history = []
        st.session_state.show_uploader = False
        st.session_state.pending_image = None
        st.rerun()


# ── Scrollable chat container ────────────────────────────────
chat_container = st.container(height=580)

with chat_container:

    # Welcome screen
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">VisualRAG</div>
            <div class="welcome-subtitle">
                Ask questions about computer vision papers,<br>
                or attach an image using the + button below.
            </div>
        </div>
        """, unsafe_allow_html=True)

        suggestions = [
            "What is the mAP of Fast YOLO on PASCAL VOC 2007?",
            "What tracking method does PhysTwin use?",
            "How does HandOccNet handle occlusion?",
            "What is the main contribution of HOI4D dataset?",
        ]

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(
                    suggestion,
                    use_container_width=True,
                    key=f"suggestion_{i}"
                ):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": suggestion,
                        "image": None,
                        "image_name": None,
                    })
                    with st.spinner("Searching papers..."):
                        result = ask(chain, retriever, suggestion)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "image": None,
                        "image_name": None,
                    })
                    st.rerun()

    # Chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):

            # Image thumbnail if present
            if message.get("image") is not None:
                st.image(message["image"], width=200)

            # Text content
            if message.get("content"):
                st.write(message["content"])

            # Sources for assistant
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">📄 {src['source']} — Page {src['page']}</div>
                            <div class="source-preview">{src['preview']}...</div>
                        </div>
                        """, unsafe_allow_html=True)


# ── Image preview above input ────────────────────────────────
if st.session_state.pending_image is not None:
    prev_col1, prev_col2, prev_col3 = st.columns([1, 4, 1])
    with prev_col1:
        st.image(st.session_state.pending_image, width=80)
    with prev_col2:
        st.caption(f"📎 {st.session_state.pending_image.name}")
    with prev_col3:
        if st.button("✕", help="Remove image"):
            st.session_state.pending_image = None
            st.rerun()


# ── File uploader (shown when + clicked) ────────────────────
if st.session_state.show_uploader:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_uploader"
    )
    if uploaded is not None:
        st.session_state.pending_image = uploaded
        st.session_state.show_uploader = False
        st.rerun()


# ── Bottom input row ─────────────────────────────────────────
input_col1, input_col2 = st.columns([1, 20])

with input_col1:
    st.markdown('<div class="attach-btn">', unsafe_allow_html=True)
    if st.button("＋", help="Attach an image", key="attach_btn"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with input_col2:
    question = st.chat_input("Ask something about your papers...")


# ── Handle submission ────────────────────────────────────────
if question:
    has_image = st.session_state.pending_image is not None

    if has_image:
        image_bytes = st.session_state.pending_image.getvalue()
        image_name = st.session_state.pending_image.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(image_name).suffix
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        with st.spinner("Reading image..."):
            cap = caption_image(tmp_path, processor, model)

        os.unlink(tmp_path)

        query = f"""
Image caption (auto-generated): {cap}
User says about the image: {question}
Find research papers related to both of the above.
"""
        st.session_state.pending_image = None

        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "image": image_bytes,
            "image_name": image_name,
        })

    else:
        history_context = ""
        if st.session_state.chat_history:
            recent = st.session_state.chat_history[-6:]
            exchanges = []
            for msg in recent:
                if msg["role"] == "user":
                    exchanges.append(f"Previous question: {msg['content']}")
                elif msg["role"] == "assistant":
                    exchanges.append(
                        f"Previous answer summary: {msg['content'][:150]}"
                    )
            history_context = "\n".join(exchanges)

        query = (
            f"{history_context}\n\nNew question: {question}"
            if history_context
            else question
        )

        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "image": None,
            "image_name": None,
        })

    with st.spinner("Searching papers..."):
        result = ask(chain, retriever, query)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "image": None,
        "image_name": None,
    })

    st.rerun()