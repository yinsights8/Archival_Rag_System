import os
import pickle
from typing import List, Dict, Any, Optional

# Faiss and embeddings
from src.faiss_storage import FaissStorage
from src.ingest_corpus_jsonl import embed_texts

# BM25
try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False


class DenseRetriever:
    """Wrapper for FaissStorage dense retrieval."""
    def __init__(self, faiss_store: Optional[FaissStorage] = None):
        self.store = faiss_store or FaissStorage()

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Convert query to vector and search Faiss."""
        query_vec = embed_texts([query])[0]
        return self.store.search(query_vec, top_k)


class SparseRetriever:
    """BM25 Retriever that loads data from the Faiss docstore and caches its index."""
    def __init__(self, faiss_store: Optional[FaissStorage] = None):
        if not RANK_BM25_AVAILABLE:
            raise ImportError("Please run `pip install rank_bm25 numpy` to use this class.")
            
        self.store = faiss_store or FaissStorage()
        self.cache_path = os.path.join(self.store.dir_path, "bm25_index.pkl")
        self.bm25 = None
        
        # Parallel arrays to reconstruct payload details
        self.texts: List[str] = []
        self.payloads: List[Dict[str, Any]] = []
        
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Loads BM25 index from cache if exists, otherwise builds from docstore."""
        if not os.path.exists(self.store.docstore_path):
            raise FileNotFoundError("Docstore not found. Please ingest data first.")
            
        # 1. Read all payloads from docstore to build local reference
        # Faiss docstore format: {"id": "...", "payload": {...}}
        import orjson
        with open(self.store.docstore_path, "rb") as f:
            for line in f:
                obj = orjson.loads(line)
                payload = obj.get("payload", {})
                self.texts.append(payload.get("text", ""))
                self.payloads.append(payload)
                
        # 2. Check for cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                return
            except Exception as e:
                print(f"Warning: Failed to load BM25 cache ({e}). Rebuilding...")
                
        # 3. Build index
        print("Building BM25 index from docstore...")
        tokenized_corpus = [doc.lower().split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 4. Save cache
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.bm25, f)
        print("BM25 index built and cached.")

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search BM25 index."""
        if not self.bm25:
            return {"contexts": [], "scores": [], "sources": [], "metadatas": []}
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        contexts = []
        result_scores = []
        sources = set()
        metadatas = []
        
        for idx in top_indices:
            if scores[idx] > 0:
                payload = self.payloads[idx]
                
                contexts.append(self.texts[idx])
                result_scores.append(float(scores[idx]))
                
                source = payload.get("source", payload.get("doc_id", ""))
                if source:
                    sources.add(str(source))
                    
                metadatas.append(payload)
                
        return {
            "contexts": contexts,
            "scores": result_scores,
            "sources": list(sources),
            "metadatas": metadatas
        }


class HybridRetriever:
    """Combines Dense and Sparse retrievers using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, faiss_store: Optional[FaissStorage] = None, rrf_k: int = 60):
        self.faiss_store = faiss_store or FaissStorage()
        self.dense = DenseRetriever(self.faiss_store)
        self.sparse = SparseRetriever(self.faiss_store)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search both dense and sparse indices and merge results with RRF.
        Standard RRF formula: Score = 1 / (k + rank)
        """
        # We query more from each to ensure good overlap
        fetch_k = max(top_k * 2, 20)
        
        dense_results = self.dense.search(query, top_k=fetch_k)
        sparse_results = self.sparse.search(query, top_k=fetch_k)
        
        # We use a unique key for each item, usually the chunk text itself if no ID is returned
        # FAISS search returns contexts, scores, sources, metadatas
        
        rrf_scores = {}
        item_details = {} # Map text -> full details (contexts, source, metadata)
        
        # 1. Process Dense Results
        for rank, text in enumerate(dense_results["contexts"]):
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                item_details[text] = {
                    "source": list(dense_results["sources"])[0] if dense_results["sources"] else "", 
                    "metadata": dense_results["metadatas"][rank]
                }
            rrf_scores[text] += 1.0 / (self.rrf_k + rank + 1)
            
        # 2. Process Sparse Results
        for rank, text in enumerate(sparse_results["contexts"]):
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                item_details[text] = {
                    "source": list(sparse_results["sources"])[0] if sparse_results["sources"] else "", 
                    "metadata": sparse_results["metadatas"][rank]
                }
            rrf_scores[text] += 1.0 / (self.rrf_k + rank + 1)
            
        # 3. Sort by RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 4. Format output
        contexts = []
        scores = []
        sources = set()
        metadatas = []
        
        for text, score in sorted_items:
            contexts.append(text)
            scores.append(score)
            
            details = item_details[text]
            if details["source"]:
                sources.add(details["source"])
            metadatas.append(details["metadata"])
            
        return {
            "contexts": contexts,
            "scores": scores,
            "sources": list(sources),
            "metadatas": metadatas
        }
