import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class ChromaStorage:
    """
    ChromaStorage using chromadb to store and retrieve embeddings.
    """
    def __init__(self, dir_path: str = "./storage/chroma", collection_name: str = "archival_rag"):
        self.client = chromadb.PersistentClient(path=dir_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """
        Upsert vectors and metadata into Chroma.
        """
        # Chroma expects ids as strings
        str_ids = [str(i) for i in ids]
        
        self.collection.upsert(
            ids=str_ids,
            embeddings=vectors,
            metadatas=payloads
        )

    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """
        Search for the top_k most similar vectors.
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        # Reformatting to match the FaissStorage output structure
        contexts: List[str] = []
        result_scores: List[float] = []
        sources = set()
        metadatas: List[Dict[str, Any]] = []

        # results['documents'][0] might be empty if not stored, 
        # but usually we store text in metadata or documents.
        # Let's check metadata for 'text' as per FaissStorage pattern.
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                score = results['distances'][0][i] if results['distances'] else 0.0
                
                text = metadata.get("text", "")
                source = metadata.get("source", metadata.get("doc_id", ""))
                
                if text:
                    contexts.append(text)
                    result_scores.append(score)
                    if source:
                        sources.add(str(source))
                    metadatas.append(metadata)

        return {
            "contexts": contexts, 
            "scores": result_scores, 
            "sources": list(sources), 
            "metadatas": metadatas
        }
