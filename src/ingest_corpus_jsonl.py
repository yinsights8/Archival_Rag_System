import re
from typing import Any, Dict, Iterator, List, Tuple

import orjson
from tqdm import tqdm

from faiss_storage import FaissStorage


def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}") from e


def clean_ocr_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150, min_chars: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        ch = text[start:end].strip()
        if len(ch) >= min_chars:
            chunks.append(ch)
        if end == len(text):
            break
    return chunks


def build_payloads(rec: Dict[str, Any], chunks: List[str]) -> List[Dict[str, Any]]:
    doc_id = str(rec.get("doc_id", ""))
    title = str(rec.get("title", ""))
    collection = str(rec.get("collection", ""))
    source_dir = str(rec.get("source_dir", ""))

    payloads = []
    for i, ch in enumerate(chunks):
        payloads.append({
            "text": ch,
            "source": source_dir or doc_id,     # what you want to show as citation/source
            "doc_id": doc_id,
            "chunk_index": i,
            "title": title,
            "collection": collection,
            "date": rec.get("date", ""),
            "date_numeric": rec.get("date_numeric", None),
            "ocr_quality": rec.get("ocr_quality", None),
            "ocr_quality_tier": rec.get("ocr_quality_tier", None),
        })
    return payloads


class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # matches cosine via inner product
        )
        return [v.tolist() for v in vecs]


def ingest(
    jsonl_path: str,
    faiss_dir: str = "./faiss_store",
    chunk_size: int = 900,
    overlap: int = 150,
    batch_size: int = 128,
    min_ocr_quality: float = 0.0,  # set e.g. 0.85 to skip noisy docs
) -> None:
    embedder = LocalEmbedder()
    store = FaissStorage(dir_path=faiss_dir, dim=embedder.dim, normalize=True)

    ids_buf: List[int] = []
    texts_buf: List[str] = []
    payloads_buf: List[Dict[str, Any]] = []

    def flush():
        if not texts_buf:
            return
        vectors = embedder.embed(texts_buf)
        store.upsert(ids_buf, vectors, payloads_buf)
        ids_buf.clear(); texts_buf.clear(); payloads_buf.clear()

    next_id = 0  # local numeric id for FAISS (required)

    for rec in tqdm(stream_jsonl(jsonl_path), desc="Ingest corpus.jsonl"):
        # Optional OCR quality filter
        oq = rec.get("ocr_quality", None)
        if oq is not None:
            try:
                if float(oq) < float(min_ocr_quality):
                    continue
            except Exception:
                pass

        text = clean_ocr_text(rec.get("text", ""))
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        payloads = build_payloads(rec, chunks)
        for ch, payload in zip(chunks, payloads):
            ids_buf.append(next_id)
            texts_buf.append(ch)
            payloads_buf.append(payload)
            next_id += 1

            if len(texts_buf) >= batch_size:
                flush()

    flush()
    print(f"Done. FAISS store at: {faiss_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", default="./faiss_store")
    p.add_argument("--chunk_size", type=int, default=900)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--min_ocr_quality", type=float, default=0.0)
    args = p.parse_args()

    ingest(
        jsonl_path=args.jsonl,
        faiss_dir=args.out,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        min_ocr_quality=args.min_ocr_quality,
    )