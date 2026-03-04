# Archival RAG System

A Retrieval-Augmented Generation (RAG) system for ingesting archival document corpora (JSONL format) and querying them with a hybrid retrieval pipeline backed by an LLM.

## What This Project Does

The system allows you to:
1. **Ingest archival corpora** — Stream `.jsonl` files, chunk documents, embed with BGE, and store in a FAISS vector index
2. **Retrieve documents** — Three retrieval modes: Dense (semantic), Sparse (BM25 keyword), and Hybrid (RRF fusion)
3. **Query with an LLM** — Retrieved context is passed to LLaMA 3.1 70B to generate an answer


## Architecture

```
User Query
    │
    ├─► Dense Retriever      (FAISS vector search via BAAI/bge-small-en-v1.5)
    │
    ├─► Sparse Retriever     (BM25 via rank_bm25 — cached index from docstore)
    │
    └─► Hybrid Retriever     (Reciprocal Rank Fusion of Dense + Sparse)
            │
            └─► LLM (LLaMA 3.1 70B via OpenRouter) → Answer
```

Each stage is tracked as a **named Inngest step**, visible individually in the Inngest dashboard.

## Project Structure

```
.
├── main_jsonl_chat.py          # FastAPI + Inngest functions (ingest & query)
├── src/
│   ├── retrievers.py           # DenseRetriever, SparseRetriever, HybridRetriever
│   ├── evaluation.py           # nDCG, MRR, Recall, stratified metrics, significance tests
│   ├── ingest_corpus_jsonl.py  # JSONL streaming, chunking, embeddings
│   ├── faiss_storage.py        # FAISS vector store (persistent, disk-backed)
│   ├── generation.py           # LLM call helpers
│   ├── bm25_retriever.py       # Low-level BM25 utilities
│   └── custom_types.py         # Pydantic models
├── data/
│   └── corpus2.jsonl           # Archival document corpus
├── storage/
│   ├── faiss.index             # FAISS index (auto-created on ingest)
│   ├── docstore.jsonl          # Chunk payloads with metadata
│   ├── ids.json                # Vector IDs
│   └── bm25_index.pkl          # Cached BM25 index (auto-created on first query)
└── pyproject.toml
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
# or with uv:
uv sync
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your key from [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Run the Server

**Terminal 1** — FastAPI server:
```bash
uv run uvicorn main_jsonl_chat:app
```

**Terminal 2** — Inngest dev server:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

Open the Inngest UI at `http://localhost:8288` to trigger and monitor events.

## Inngest Events

### Ingest a Corpus

```json
{
  "name": "app/rag_ingest_nls_corpus",
  "data": {
    "jsonl_path": "data/corpus2.jsonl",
    "source_id": "nls-corpus-v2"
  }
}
```

### Query the Corpus

```json
{
  "name": "app/rag_query_nls_corpus",
  "data": {
    "question": "Who fought at the Battle of Bara?",
    "top_k": 5,
    "retrieval_mode": "hybrid"
  }
}
```

`retrieval_mode` options: `"dense"` | `"sparse"` | `"hybrid"` (default)

**Response fields:**
| Field | Description |
|---|---|
| `answer` | LLM-generated answer |
| `sources` | Source document IDs used |
| `num_contexts` | Number of retrieved chunks |

### Inngest Pipeline Steps

Each query run is broken into 4 tracked steps:

| Step | Name in UI | What it does |
|---|---|---|
| 1 | `retrieval-dense` | FAISS vector search |
| 2 | `retrieval-sparse` | BM25 keyword search (cached index) |
| 3 | `retrieval-hybrid-merge` | RRF fusion of both results |
| 4 | `llm-answer` | LLM answer generation |

## Evaluation

Use `src/evaluation.py` to benchmark retrieval quality against ground-truth relevance judgements (`qrels`):

```python
from src.evaluation import batch_retriever_to_results, compute_stratified_metrics
from src.retrievers import HybridRetriever

queries = {"q1": "battles in Kordofan", "q2": "Mahdi uprising"}
qrels   = {"q1": {"doc_001": 1}, "q2": {"doc_007": 2}}
query_metadata = {"q1": {"ocr_quality_tier": "low"}, "q2": {"ocr_quality_tier": "high"}}

results = batch_retriever_to_results(queries, HybridRetriever(), top_k=10)
report  = compute_stratified_metrics(results, qrels, query_metadata)
```

**Available metrics:** nDCG@k, MRR, Recall@k — overall and stratified by OCR quality tier (`high` / `medium` / `low`).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `EMB_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMB_DIM` | `384` | Embedding dimension |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `DEFAULT_MODEL` | `meta-llama/llama-3.1-70b-instruct` | LLM for answer generation |
