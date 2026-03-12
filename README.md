# Archival RAG System

A Retrieval-Augmented Generation (RAG) system for ingesting archival document corpora (JSONL format) and querying them with a hybrid retrieval pipeline backed by an LLM.


## Corpus Statistics

> **397 documents** spanning **638 years** (1299–1937) across **2 collections**.

### Collections

| Collection | Documents |
|---|---|
| `africa_and_new_imperialism` | 238 (59.9%) |
| `indiaraj` | 159 (40.1%) |

### OCR Quality

| Tier | Count | Share |
|---|---|---|
| 🟢 High (≥ 0.9) | 191 | 48.1% |
| 🟡 Medium (0.7–0.9) | 136 | 34.3% |
| 🔴 Low (< 0.7) | 70 | 17.6% |

Estimated OCR confidence scores: **mean 0.967**, median 0.979, std 0.038, range 0.606–0.996.

### Document Length (words)

| Statistic | Value |
|---|---|
| Mean | 38,859 |
| Median | 22,078 |
| Min / Max | 22 / 560,676 |
| P25 / P75 | 4,543 / 51,924 |
| P90 / P95 / P99 | 90,024 / 121,374 / 311,350 |


### Temporal Distribution

Documents span **19 decades** with the following concentrations:

| Period | Documents |
|---|---|
| Medieval (1290s–1310s) | 14 |
| Late 18th century (1760s–1790s) | 102 |
| 19th century (1800s–1890s) | 262 |
| Early 20th century (1920s–1930s) | 4 |
| **Peak decade** | 1860s (81 docs) |

## Total Number of Chunks

> **121,722 chunks** stored across all indexes (verified 2026-03-05).

### Storage Files

| File | Size | Purpose |
|---|---|---|
| `storage/ids.json` | ~4.7 MB | List of all 121,722 chunk IDs |
| `storage/docstore.jsonl` | ~154 MB | One chunk (text + metadata) per line — 121,722 lines |
| `storage/faiss.index` | ~187 MB | Dense FAISS vector index over all chunks |
| `storage/bm25_index.pkl` | ~139 MB | BM25 sparse index over all chunks |

**Total storage footprint: ~485 MB**

> All four storage artifacts are in sync at **121,722 chunks**.

## What This Project Does

The system allows you to:
1. **Ingest archival corpora** — Stream `.jsonl` files, chunk documents, embed with BGE, and store in a FAISS vector index
2. **Retrieve documents** — Three retrieval modes: Dense (semantic), Sparse (BM25 keyword), and Hybrid (RRF fusion)
3. **Query with an LLM** — Retrieved context is passed to LLaMA 3.1 70B via OpenRouter to generate answers.
4. **Evaluate Performance** — A dedicated evaluation suite to measure retriever accuracy and LLM answer quality.


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
            ├─► [Optional] Compressor (RECOMP) → Improves context efficiency
            │
            └─► LLM (LLaMA 3.1 70B via OpenRouter) → Answer
```

## Evaluation Framework

The project includes a robust evaluation module to measure both search and generation quality.

### Supported Metrics
- **Retriever Metrics**:
  - **MRR** (Mean Reciprocal Rank): Measures rank quality of the first correct document.
  - **Recall@K**: Measures if the ground truth document is within the top $K$ results.
  - **nDCG**: Measures ranking efficiency and position sensitivity.
- **Generation Metrics (Ragas)**:
  - **Faithfulness**: Is the answer derived solely from the provided context?
  - **Answer Relevancy**: Does the answer directly address the user query?
  - **Context Precision/Recall**: Quality and completeness of the retrieved context.

### Running Evaluation
To run a full evaluation on the `rag_questions.json` dataset:
```bash
python evaluation/evaluate.py
```

### Evaluation Artifacts
- **Detailed Dataset**: `data/rag_dataset/rag_dataset.csv` — Inspect every query, context, and generated answer.
- **Summary Report**: `results/evaluation_results_[timestamp].json` — Quantitative summary of all scores.

Each stage is tracked as a **named Inngest step**, visible individually in the Inngest dashboard.

## Project Structure

```
.
├── main_jsonl_chat.py          # FastAPI + Inngest functions (ingest & query)
├── evaluation/
│   ├── evaluate.py             # Main evaluation entry point
│   ├── metrics.py              # Mathematical IR metric implementations
│   └── trigger_evaluation.py   # Inngest trigger for async evaluation
├── src/
│   ├── retrievers.py           # DenseRetriever, SparseRetriever, HybridRetriever
│   ├── ingest_corpus_jsonl.py  # JSONL streaming, chunking, embeddings
│   ├── faiss_storage.py        # FAISS vector store (persistent, disk-backed)
│   ├── generation.py           # LLM call helpers
│   └── custom_types.py         # Pydantic models
    └── compressor.py           # [Planned] RECOMP context compression
├── data/
│   ├── corpus2.jsonl           # Archival document corpus
│   └── rag_dataset/            # Exported evaluation datasets (CSV)
├── results/                    # Exported evaluation summary reports (JSON)
├── storage/                    # Persistent index storage (FAISS, BM25)
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
  "data": {
    "jsonl_path": "data/corpus2.jsonl",
    "source_id": "nls-corpus-v2"
  }
}
```

### Query the Corpus

```json
{
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

## Configuration

| Variable | Default | Description |
|---|---|---|
| `EMB_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMB_DIM` | `384` | Embedding dimension |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `DEFAULT_MODEL` | `meta-llama/llama-3.1-70b-instruct` | LLM for answer generation |
