from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    LANGCHAIN_BM25_AVAILABLE = True
except ImportError:
    LANGCHAIN_BM25_AVAILABLE = False


def build_bm25_retriever_langchain(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, k: int = 4):
    """
    Builds a LangChain BM25Retriever.
    Run `pip install langchain-community rank_bm25` if not installed.
    """
    if not LANGCHAIN_BM25_AVAILABLE:
        raise ImportError("Please install langchain-community and rank_bm25 to use this function.")
        
    docs = []
    for i, text in enumerate(texts):
        meta = metadatas[i] if metadatas else {}
        docs.append(Document(page_content=text, metadata=meta))
        
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


class SimpleBM25Retriever:
    """
    A standalone BM25 retriever without Langchain, directly using rank_bm25.
    Must have `rank_bm25` and `numpy` installed.
    """
    def __init__(self, corpus_texts: List[str], payloads: Optional[List[Dict[str, Any]]] = None):
        if not RANK_BM25_AVAILABLE:
            raise ImportError("Please run `pip install rank_bm25 numpy` to use this class.")
            
        self.corpus_texts = corpus_texts
        self.payloads = payloads if payloads else [{} for _ in corpus_texts]
        
        # Simple whitespace tokenization 
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search for the top_k most similar documents to the query.
        Returns the same format as FaissStorage.search() for compatibility with the rest of your app.
        """
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
                
                contexts.append(self.corpus_texts[idx])
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
