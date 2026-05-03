import streamlit as st
import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

## Load the Nvidia Api key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_EMD_KEY'] = os.getenv('NVIDIA_EMD_KEY')

## NVIDIA NIM Inferencing
llm = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key=os.getenv('NVIDIA_API_KEY'),
  temperature=1,
  top_p=1,
  max_completion_tokens=4096,
)

FAISS_INDEX_PATH = "faiss_index"

def get_embeddings():
    return NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        api_key=os.getenv('NVIDIA_EMD_KEY'),
        truncate="NONE"
    )

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = get_embeddings()

        if os.path.exists(FAISS_INDEX_PATH):
            # ── Load from disk — no re-embedding needed ──
            st.session_state.vectors = FAISS.load_local(
                FAISS_INDEX_PATH,
                st.session_state.embeddings,
                allow_dangerous_deserialization=True,
            )
            st.session_state.final_docs = []          # already persisted, count unavailable
            st.session_state._faiss_source = "loaded"
        else:
            # ── Build from PDFs and persist ──
            st.session_state.loader = PyPDFDirectoryLoader('us_census')
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_docs = st.session_state.splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
            st.session_state.vectors.save_local(FAISS_INDEX_PATH)
            st.session_state._faiss_source = "built"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NVIDIA NIM · RAG Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 960px !important;
}

/* ── Hero header ── */
.hero {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 0.4rem;
}
.hero-badge {
    background: linear-gradient(135deg, #76b900 0%, #4e8000 100%);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 2px;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    line-height: 1 !important;
    margin: 0 !important;
}
.hero-title span { color: #76b900; }
.hero-sub {
    font-size: 0.8rem;
    color: #5a5a72;
    letter-spacing: 0.04em;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #1a1a28;
    padding-bottom: 1.5rem;
}

/* ── Divider ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #3a3a55;
    margin-bottom: 0.6rem;
}

/* ── Status pill ── */
.status-wrap { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 2rem; }
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #2a2a3e;
    flex-shrink: 0;
}
.status-dot.ready { background: #76b900; box-shadow: 0 0 8px #76b90088; }
.status-text { font-size: 0.72rem; color: #4a4a62; }
.status-text.ready { color: #76b900; }

/* ── Text input ── */
[data-testid="stTextInput"] > div > div {
    background: #0f0f1a !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 4px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="stTextInput"] > div > div:focus-within {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 3px #76b90018 !important;
}
[data-testid="stTextInput"] label {
    color: #4a4a62 !important;
    font-size: 0.7rem !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #1e1e30 !important;
    color: #6a6a88 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    border-radius: 3px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stButton"] > button:hover {
    background: #76b900 !important;
    border-color: #76b900 !important;
    color: #0a0a0f !important;
    box-shadow: 0 4px 20px #76b90030 !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(1px) !important;
}

/* ── Answer card ── */
.answer-card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-left: 3px solid #76b900;
    border-radius: 4px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    font-size: 0.87rem;
    line-height: 1.75;
    color: #c8c8e0;
    position: relative;
}
.answer-card::before {
    content: 'RESPONSE';
    position: absolute;
    top: -0.6rem;
    left: 1.2rem;
    background: #0a0a0f;
    padding: 0 0.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: #76b900;
}

/* ── Timing badge ── */
.timing-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #0f0f1a;
    border: 1px solid #1a1a2e;
    border-radius: 2px;
    padding: 0.3rem 0.75rem;
    font-size: 0.68rem;
    color: #3a3a55;
    margin-top: 0.8rem;
    font-family: 'DM Mono', monospace;
}
.timing-badge span { color: #76b900; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d0d1a !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 4px !important;
    margin-top: 1rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #3a3a55 !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: #76b900 !important; }

/* ── Chunk cards ── */
.chunk-card {
    background: #0a0a12;
    border: 1px solid #15152a;
    border-radius: 3px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #6a6a88;
}
.chunk-num {
    font-family: 'Syne', sans-serif;
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: #2e2e48;
    margin-bottom: 0.5rem;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    color: #4a4a62 !important;
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Success / Info messages ── */
[data-testid="stAlert"] {
    background: #0d120d !important;
    border: 1px solid #1a2e1a !important;
    border-left: 3px solid #76b900 !important;
    border-radius: 3px !important;
    color: #76b900 !important;
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ NIM</div>
    <h1 class="hero-title">NVIDIA <span>RAG</span> Demo</h1>
</div>
<p class="hero-sub">Retrieval-Augmented Generation · US Census Documents · FAISS Vector Store</p>
""", unsafe_allow_html=True)

# ── Vector store status ────────────────────────────────────────────────────────
is_ready = "vectors" in st.session_state
index_on_disk = os.path.exists(FAISS_INDEX_PATH)

if is_ready:
    source = st.session_state.get("_faiss_source", "built")
    n_chunks = len(st.session_state.get("final_docs", []))
    if source == "loaded":
        status_label = "Vector store loaded from disk"
    else:
        status_label = f"Vector store built &middot; {n_chunks} chunks &middot; saved to disk"
elif index_on_disk:
    status_label = "Saved index found on disk &mdash; click Embed to load it"
else:
    status_label = "Vector store not initialised"

st.markdown(f"""
<div class="status-wrap">
    <div class="status-dot {'ready' if is_ready else 'disk' if index_on_disk else ''}"></div>
    <span class="status-text {'ready' if is_ready else ''}">
        {status_label}
    </span>
</div>
""", unsafe_allow_html=True)

# ── Embed button ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Step 01 · Initialise</p>', unsafe_allow_html=True)

if st.button("⬡  Embed Documents"):
    spinner_msg = "Loading FAISS index from disk…" if os.path.exists(FAISS_INDEX_PATH) else "Loading PDFs and building FAISS index…"
    with st.spinner(spinner_msg):
        vector_embedding()
    source = st.session_state.get("_faiss_source", "built")
    if source == "loaded":
        st.success("Index loaded from local disk — no re-embedding needed.")
    else:
        st.success(f"Vector store built and saved to `{FAISS_INDEX_PATH}/`. Future runs will load from disk.")
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Query input ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Step 02 · Ask a Question</p>', unsafe_allow_html=True)

prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}
""")

prompt1 = st.text_input(
    "Query",
    placeholder="e.g. What is the median household income in California?",
    label_visibility="collapsed",
)

# ── Answer ─────────────────────────────────────────────────────────────────────
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please embed the documents first using the button above.")
    else:
        with st.spinner("Retrieving context and generating response…"):
            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            elapsed = time.process_time() - start

        # Answer card
        st.markdown(f"""
        <div class="answer-card">
            {response['answer']}
        </div>
        <div class="timing-badge">⏱ inference time <span>{elapsed:.3f}s</span></div>
        """, unsafe_allow_html=True)

        # Source chunks
        with st.expander("▸  Retrieved Source Chunks"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-num">CHUNK {i + 1:02d}</div>
                    {doc.page_content}
                </div>
                """, unsafe_allow_html=True)