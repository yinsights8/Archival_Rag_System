# Archival RAG System

A Retrieval-Augmented Generation (RAG) system for ingesting PDF documents and querying them using a large language model.

## What This Project Does

The Archival RAG System allows you to:
1. **Ingest PDF documents** - Upload PDFs which are then chunked and converted into vector embeddings
2. **Query your documents** - Ask questions and get answers based on the content of your ingested PDFs

The system uses:
- **FAISS** for vector similarity search
- **LangChain** for document processing and LLM orchestration
- **BGE Small (BAAI/bge-small-en-v1.5)** for text embeddings
- **LLaMA 3.1 70B** (via OpenRouter) for generating answers
- **Inngest** for event-driven workflow management

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Run the Server

```bash

terminal 1:
uv run uvicorn main_jsonl_chat:app

terminal 2:
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

```

The API will be available at `http://localhost:8000`.

## API Usage

### Ingest a PDF

Send an event to Inngest to ingest a PDF:

```python
from inngest import Inngest

inngest_client = Inngest(app_id="archival_rag_system")

inngest_client.send_event(
    name="app/rag ingest jsonl",
    data={
        "pdf_path": "/path/to/your/document.pdf",
        "source_id": "my-document"
    }
)
```

### Query a PDF

Send a query event:

```python
inngest_client.send_event(
    name="app/rag_query_pdf",
    data={
        "question": "What is this document about?",
        "top_k": 5
    }
)
```

The response will include:
- `answer` - The LLM's answer
- `sources` - List of source documents used
- `num_contexts` - Number of context chunks retrieved

## Project Structure

```
.
├── main.py                 # FastAPI app and Inngest functions
├── src/
│   ├── ingest_pdf.py       # PDF loading, chunking, embeddings
│   ├── faiss_storage.py    # FAISS vector store
│   └── custom_types.py     # Pydantic models
├── storage/                # Vector store files
│   ├── faiss.index         # FAISS index
│   ├── docstore.jsonl      # Document payloads
│   └── ids.json            # Vector IDs
└── pyproject.toml          # Dependencies
```

## Configuration

Key settings in `src/ingest_pdf.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMB_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMB_DIM` | `384` | Embedding dimension |
| `CHUNK_SIZE` | `512` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |

In `main.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `meta-llama/llama-3.1-70b-instruct` | LLM model |
