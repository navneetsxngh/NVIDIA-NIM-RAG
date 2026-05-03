# ⚡ NVIDIA NIM RAG Demo

A **Retrieval-Augmented Generation (RAG)** application powered by NVIDIA NIM inference endpoints, LangChain, and Streamlit. Ask questions against your own PDF documents — answers are grounded in retrieved context, not hallucinated.

---

## How It Works

```
PDFs → Chunking → NVIDIA Embeddings → FAISS Index (persisted locally)
                                              ↓
              User Query → Retriever → Top-K Chunks → NVIDIA LLM → Answer
```

1. PDFs are loaded and split into overlapping chunks
2. Each chunk is embedded using NVIDIA's `llama-3.2-nemoretriever-300m-embed-v1` model
3. Embeddings are stored in a local FAISS index (saved to disk — rebuilt only once)
4. At query time, the top relevant chunks are retrieved and passed as context to the LLM
5. `openai/gpt-oss-20b` via NVIDIA NIM generates a grounded answer

---

## Features

- **Persistent FAISS index** — embeddings are saved locally with `save_local()` and reloaded on subsequent runs, eliminating redundant API calls
- **NVIDIA NIM inference** — fast, scalable LLM and embedding endpoints
- **Source transparency** — every answer expands to show the exact retrieved chunks it was based on
- **Inference timing** — response latency displayed on every query
- **Clean Streamlit UI** — dark-themed, status-aware interface

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | NVIDIA NIM · `openai/gpt-oss-20b` |
| Embeddings | NVIDIA NIM · `llama-3.2-nemoretriever-300m-embed-v1` |
| Vector Store | FAISS (local persistence) |
| Orchestration | LangChain |
| Document Loader | LangChain `PyPDFDirectoryLoader` |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/nvidia-nim-rag-demo.git
cd nvidia-nim-rag-demo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_EMD_KEY=your_nvidia_embedding_key_here
```

> Get your API keys from [build.nvidia.com](https://build.nvidia.com)

### 4. Add your PDFs

Place your PDF documents inside a folder named `us_census/` (or update the path in `app.py`):

```
nvidia-nim-rag-demo/
├── us_census/
│   ├── document1.pdf
│   └── document2.pdf
├── app.py
├── .env
└── README.md
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. Click **Embed Documents** on first run — this builds the FAISS index and saves it to `faiss_index/` on disk
2. On all subsequent runs, the index loads from disk instantly (no re-embedding)
3. Type any question in the query box and hit Enter
4. Expand **Retrieved Source Chunks** to inspect the context behind every answer

---

## Project Structure

```
nvidia-nim-rag-demo/
├── app.py               # Main Streamlit application
├── faiss_index/         # Auto-generated — persisted FAISS vector store
│   ├── index.faiss
│   └── index.pkl
├── us_census/           # Your PDF documents go here
├── .env                 # API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## .gitignore

Make sure to exclude secrets and large generated files:

```
.env
faiss_index/
__pycache__/
*.pyc
```

---

## Requirements

```
streamlit
langchain
langchain-nvidia-ai-endpoints
langchain-community
langchain-text-splitters
langchain-core
faiss-cpu
python-dotenv
pypdf
```

---

## License

MIT