import math
from typing import List

def calculate_mrr(ground_truths: List[str], retrieved_contexts: List[str]) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR).
    
    Args:
        ground_truths: A list of relevant context IDs or text strings.
        retrieved_contexts: A list of retrieved context IDs or text strings ordered by rank.
        
    Returns:
        float: The MRR score.
    """
    for rank, context in enumerate(retrieved_contexts):
        if context in ground_truths:
            return 1.0 / (rank + 1)
    return 0.0

def calculate_recall_at_k(ground_truths: List[str], retrieved_contexts: List[str], k: int) -> float:
    """
    Calculates Recall@k.
    
    Args:
        ground_truths: A list of relevant context IDs or text strings.
        retrieved_contexts: A list of retrieved context IDs or text strings ordered by rank.
        k: The cut-off rank.
        
    Returns:
        float: The Recall@k score.
    """
    retrieved_k = retrieved_contexts[:k]
    relevant_retrieved = set(retrieved_k).intersection(set(ground_truths))
    
    if not ground_truths:
        return 0.0
        
    return len(relevant_retrieved) / len(ground_truths)

def calculate_ndcg(ground_truths: List[str], retrieved_contexts: List[str], k: int = None) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG).
    Assuming binary relevance (1 if in ground_truths, 0 otherwise).
    
    Args:
        ground_truths: A list of relevant context IDs or text strings.
        retrieved_contexts: A list of retrieved context IDs or text strings ordered by rank.
        k: Optional cut-off rank. If None, considers all retrieved_contexts.
        
    Returns:
        float: The nDCG score.
    """
    if k is not None:
        retrieved_contexts = retrieved_contexts[:k]
        
    dcg = 0.0
    for i, context in enumerate(retrieved_contexts):
        if context in ground_truths:
            # Binary relevance (rel_i = 1)
            dcg += 1.0 / math.log2(i + 2) # i is 0-indexed, so rank is i+1, Formula: log2(rank + 1) -> log2(i+2)
            
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    num_relevant = min(len(ground_truths), len(retrieved_contexts))
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def calculate_precision_at_k(ground_truths: List[str], retrieved_contexts: List[str], k: int) -> float:
    """
    Calculates Precision@k.
    
    Args:
        ground_truths: A list of relevant context IDs or text strings.
        retrieved_contexts: A list of retrieved context IDs or text strings ordered by rank.
        k: The cut-off rank.
        
    Returns:
        float: The Precision@k score.
    """
    if k == 0:
        return 0.0
        
    retrieved_k = retrieved_contexts[:k]
    relevant_retrieved = set(retrieved_k).intersection(set(ground_truths))
    
    return len(relevant_retrieved) / k
