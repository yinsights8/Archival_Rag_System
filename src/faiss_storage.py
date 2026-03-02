import os
from typing import List, Optional, Any, Dict, Tuple
import faiss 
import numpy as np
import orjson


class FaissStorage:
    """
    FaissStorage is a storage class that uses Faiss to store and 
    retrieve embeddings.
    """
    def __init__(self, dir_path = "./storage", dim: int=384, normalize: bool = True):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        self.dim = dim
        self.normalize = normalize

        self.index_path = os.path.join(self.dir_path, "faiss.index")
        self.docstore_path = os.path.join(self.dir_path, "docstore.jsonl")  # append-only
        self.ids_path = os.path.join(self.dir_path, "ids.json")             # JSON list of string ids

        # In-memory mapping: position -> external id
        # self.id_map: List[str] = []

        self._ids: List[str] = []
        self.id_map: List[str] = []

        # load the existing index and ids if present
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.ids_path, "rb") as f:
                self._ids = orjson.loads(f.read())
        else:
            # Use cosine similarity by normalizing vectors and using inner product
            self.index = faiss.IndexFlatIP(self.dim)

            # Create empty docstore file if not present
            if not os.path.exists(self.docstore_path):
                open(self.docstore_path, "ab").close()

    
    def _as_float32_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        """
        Convert a list of vectors to a float32 numpy matrix and normalize if needed.
        why : Faiss requires float32 matrix. and it should have 2D shape (N, dim).
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (N, {self.dim}), got {arr.shape}")
        if self.normalize:
            faiss.normalize_L2(arr)
        return arr


    def upsert(self, ids: List[int], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """
        Upsert vectors into the Faiss index and docstore.
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ds, vectors, payloads must have same length")
        
        arr = self._as_float32_matrix(vectors)

        start_pos = len(self._ids)
        self.index.add(arr)

        # Append payloads in the same order as vectors were added
        with open(self.docstore_path, "ab") as f:
            for i in range(len(ids)):
                row = {
                    "id": ids[i],
                    "payload": payloads[i]
                }
                f.write(orjson.dumps(row))
                f.write(b"\n")

        self._ids.extend([str(x) for x in ids])

        self._persist()
        # Optional: return start_pos for debugging
        # return start_pos


    def _persist(self) -> None:
        """
        Persist the Faiss index and ids to disk.
        FAISS search returns positions, not your original IDs
        this mapping is needed after restarting, otherwise we can't relate search results back to the original items
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "wb") as f:
            f.write(orjson.dumps(self._ids))

    def _get_payloads_by_positions(self, positions: List[int]) -> List[Optional[Dict[str, Any]]]:
        """
        Retrieve payloads from the JSONL docstore using FAISS vector positions.
        Reads the docstore.jsonl file sequentially and returns payloads aligned with the
        provided positions list.
        """
        
        want = set(p for p in positions if p >= 0)
        if not want:
            return [None for _ in positions]    

        found: Dict[int, Dict[str, Any]] = {}
        with open(self.docstore_path, "rb") as f:
            for pos, line in enumerate(f):
                if pos in want:
                    obj = orjson.loads(line)
                    found[pos] = obj.get("payload", {})
                    if len(found) == len(want):
                        break

        return [found.get(p) for p in positions]
        
    
    def search(self, query_vector: List[float], top_k:int=5) -> Dict:
        """
        Search for the top_k most similar vectors to the query_vector.
        """
        q = np.asarray([query_vector], dtype=np.float32)
        if q.shape[1] != self.dim:
            raise ValueError(f"Expected query_vector dim {self.dim}, got {q.shape[1]}")
        if self.normalize:
            faiss.normalize_L2(q)

        scores, idx = self.index.search(q, top_k)
        positions = idx[0].tolist()
        raw_scores = scores[0].tolist()          # cosine similarity per position
        payloads = self._get_payloads_by_positions(positions)

        contexts: List[str] = []
        result_scores: List[float] = []
        sources = set()
        metadatas: List[Dict[str, Any]] = []
        
        for ps, score, payload in zip(positions, raw_scores, payloads):
            if not payload:
                continue
            
            text = payload.get("text", "")
            source = payload.get("source", payload.get("doc_id", ""))  # flexible
            if text:
                contexts.append(text)
                result_scores.append(score)
                if source:
                    sources.add(str(source))
                metadatas.append(payload)

        return {"contexts": contexts, "scores": result_scores, "sources": list(sources), "metadatas": metadatas}