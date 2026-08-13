# HybridRAG — Production-Grade RAG Chatbot

Upload any document and get precise, grounded answers powered by a full hybrid retrieval pipeline — not just basic semantic search.

![HybridRAG Demo](demo.png)

---

## Why HybridRAG is Different

Most RAG demos use a single embedding model for retrieval. HybridRAG combines four techniques in sequence:

- **Semantic search** — finds conceptually similar chunks via embeddings
- **BM25 keyword search** — catches exact matches that semantic search misses (codes, names, specific terms)
- **RRF fusion** — merges both ranked lists into one without score-scale conflicts
- **Cross-encoder reranking** — re-scores the fused candidates with query+chunk interaction for final precision

---

## Pipeline Architecture

**Online phase (runs per query):**

```
User Query
    │
    ▼
Input Guardrail       ← regex-based prompt injection detection
    │
    ▼
Greeting Check        ← intercepts greetings before pipeline runs
    │
    ▼
Semantic Search ──┐
                  ├── parallel retrieval, top-20 each
Keyword Search ───┘   (all-MiniLM-L6-v2 + BM25Okapi)
    │
    ▼
RRF Fusion            ← reciprocal rank fusion merges both ranked lists
    │
    ▼
Cross-Encoder Rerank  ← ms-marco-MiniLM-L-6-v2 rescores top-20, returns top-5
    │
    ▼
Answer Generation     ← llama-3.3-70b-versatile via Groq, strict context-only prompt
    │
    ▼
Output Guardrail      ← PII and safety check on generated answer
    │
    ▼
Faithfulness Eval     ← sentence-level semantic similarity score, logged per response
    │
    ▼
Response + Logging    ← answer returned, latency + faithfulness + guardrail triggers logged
```

**Offline phase (runs once at document upload):**

```
Document Upload
    │
    ▼
Format Detection      ← PDF / TXT / DOCX (max 20MB enforced)
    │
    ▼
Text Extraction       ← PyMuPDF (PDF) / python-docx (DOCX) / built-in (TXT)
    │
    ▼
Semantic Chunking     ← sentence embeddings + cosine similarity boundary detection
    │
    ▼
LanceDB Indexing ─────── chunk embeddings stored persistently on disk
    │
BM25 Indexing ────────── tokenized chunks indexed for keyword search
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Document parsing | PyMuPDF, python-docx |
| Chunking | Semantic chunking (embedding-based topic boundaries) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Keyword search | BM25Okapi (`rank-bm25`) |
| Vector storage | LanceDB (persistent, unlike FAISS) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Evaluation | Sentence-level semantic faithfulness scoring |
| Guardrails | Regex-based input/output safety checks |
| Logging | Per-response latency, faithfulness score, guardrail triggers |
| API Backend | FastAPI |
| Gradio UI | Gradio `ChatInterface` |
| Custom UI | Vanilla HTML / CSS / JS |
| Testing | pytest (9 tests) |
| Deployment | Docker |

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/mohitkrishna21/HybridRAG.git
cd HybridRAG
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Running the App

**Option A — Gradio UI (standalone)**

```bash
python app.py
```

Opens at `http://localhost:7860`

**Option B — FastAPI + Custom HTML UI**

```bash
python -m uvicorn fastapi_backend:app --reload
```

Opens at `http://localhost:8000`

**Option C — Docker (recommended)**

```bash
docker build -t hybridrag .
docker run -p 8000:8000 --env-file .env -v huggingface_cache:/root/.cache/huggingface hybridrag
```

Opens at `http://localhost:8000`. No local Python setup required — just Docker and a `.env` file with your `GROQ_API_KEY`. The `-v` flag caches HuggingFace model weights so they are not re-downloaded on every container restart.

---

## Running Tests

```bash
pytest tests/ -v
```

9 tests covering input/output guardrails, cosine similarity, document loading, and faithfulness evaluation.

---

## Supported Formats

PDF · TXT · DOCX · Max 20MB

---

## Key Design Decisions

**LanceDB over FAISS** — LanceDB persists to disk automatically. FAISS is in-memory only — data is lost on every server restart.

**Hybrid search over semantic-only** — BM25 catches exact keyword matches (product codes, names, specific terms) that semantic embeddings miss. Pure semantic search would fail on queries like "SKU-AXP-3918".

**RRF fusion over direct score combination** — BM25 and cosine similarity scores are on incomparable scales. RRF merges by rank position instead of raw scores, making it scale-agnostic.

**Cross-encoder reranking** — bi-encoders embed query and document separately, losing query-document interaction nuance. The cross-encoder processes query + chunk together for final precision, run only on the small fused candidate list to keep latency practical.

**Reranker cached at startup** — loading the cross-encoder fresh per request added ~18 seconds of latency. Caching at startup reduced this to ~2-3 seconds.

**Temperature 0.2** — lower temperature keeps the LLM closer to retrieved context, reducing hallucination in grounded Q&A tasks.

---

## Known Limitations

**Server state persists until restart** — the loaded document index is stored in server memory. Uploading a new document replaces the previous index; the chat clears automatically.

**Single document per session** — multi-document retrieval across a knowledge base is not currently supported.

**CPU-only inference** — the embedding and reranking models run on CPU. On large documents (100+ pages), upload processing may take 30-60 seconds.

**Groq API dependency** — answer generation requires an active Groq API key. If Groq is unavailable, retrieval still works but generation fails.

---

## Future Work

- AWS EC2 deployment with Docker for persistent hosting and real uptime
- Multi-document knowledge base support
- Agentic layer — dynamic decision between document search and web search
- RAGAS-style evaluation harness for systematic pipeline quality tracking
- PostgreSQL vector extension as alternative to LanceDB for managed deployments

---

## License

MIT
