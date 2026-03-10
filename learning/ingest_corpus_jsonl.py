import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
from custom_types import RAGChunkAndSrc

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMB_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMB_DIM = 384
CHUNK_SIZE=512
CHUNK_OVERLAP=200

embed_model = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME)

# ---------- JSONL loading ----------

def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream a JSONL file safely (one JSON object per line).
    Uses orjson if available, otherwise falls back to stdlib json.
    """
    try:
        import orjson  # type: ignore
        loads = orjson.loads
        decode_error = orjson.JSONDecodeError
    except Exception:
        import json
        loads = lambda b: json.loads(b.decode("utf-8"))
        decode_error = ValueError

    with open(path, "rb") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object, got {type(obj)}")
                yield obj
            except decode_error as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}") from e


# ---------- Chunking (RecursiveCharacterTextSplitter) ----------

def _get_splitter(chunk_size: int, overlap: int):
    """
    Uses langchain-text-splitters if installed.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # separators=["\n\n", "\n", " ", ""],  # paragraph -> line -> word -> char
        length_function=len,
    )


def build_payloads(rec: Dict[str, Any], chunks: List[str]) -> List[Dict[str, Any]]:
    """
    Preserve your earlier metadata fields per chunk.
    """
    doc_id = str(rec.get("doc_id", ""))
    title = str(rec.get("title", ""))
    collection = str(rec.get("collection", ""))
    source_dir = str(rec.get("source_dir", ""))


    # Inngest-level source_id as a stable fallback.
    # citation_source = source_dir or doc_id or source_id

    payloads: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        payloads.append(
            {
                "text": ch,
                "source": source_dir or doc_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "title": title,
                "collection": collection,
                "date": rec.get("date", ""),
                "date_numeric": rec.get("date_numeric", None),
                "ocr_quality": rec.get("ocr_quality", None),
                "ocr_quality_tier": rec.get("ocr_quality_tier", None),
            }
        )
    return payloads


def load_and_chunk_jsonl(
    jsonl_path: str,
    source_id: Optional[str] = None,
    *,
    chunk_size: int = 900,
    overlap: int = 150,
    min_chars: int = 200,
    min_ocr_quality: float = 0.0,
    max_docs: Optional[int] = None,
) -> RAGChunkAndSrc:
    """
    Loads a corpus.jsonl, cleans OCR text, chunks per record using RecursiveCharacterTextSplitter,
    and returns a list of chunk dicts with both text + metadata.

    Output format:
      RAGChunkAndSrc(
        source_id=...,
        chunks=[
          {"text": "...", "metadata": {...}},
          ...
        ]
      )
    """
    # if source_id is None:
    #     source_id = jsonl_path

    splitter = _get_splitter(chunk_size=chunk_size, overlap=overlap)

    out_chunks: List[Dict[str, Any]] = []
    docs_seen = 0

    for rec in stream_jsonl(jsonl_path):
        # Optional OCR quality filter
        oq = rec.get("ocr_quality", None)
        if oq is not None:
            try:
                if float(oq) < float(min_ocr_quality):
                    continue
            except Exception:
                pass

        text = rec.get("text", "")
        if not text:
            continue

        # Chunk with boundaries when possible
        chunks = splitter.split_text(text)
        # Enforce minimum chunk size
        chunks = [p.strip() for p in chunks if len(p.strip()) >= min_chars]
        if not chunks:
            continue

        payloads = build_payloads(rec, chunks)

        # Return chunks in a common "text + metadata" structure
        for p in payloads:
            out_chunks.append(
                {
                    "text": p["text"],
                    "metadata": {k: v for k, v in p.items() if k != "text"},
                }
            )

        # docs_seen += 1
        # if max_docs is not None and docs_seen >= max_docs:
        #     break

    return out_chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embed_model.embed_documents(texts)
