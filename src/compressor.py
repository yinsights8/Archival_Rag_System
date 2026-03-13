import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

class RECOMPCompressor:
    """
    Implements RECOMP (RECTified COMpression and Prepend) for RAG context compression.
    Supports both extractive and abstractive modes.
    """

    def __init__(self, mode: str = "extractive", device: str = None):
        """
        Initialize the compressor.
        
        Args:
            mode: 'extractive' or 'abstractive'.
            device: 'cuda' or 'cpu'. If None, auto-detects.
        """
        self.mode = mode.lower()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.mode == "extractive":
            self.model_name = "fangyuan/nq_extractive_compressor"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        elif self.mode == "abstractive":
            self.model_name = "fangyuan/nq_abstractive_compressor"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        else:
            raise ValueError("Mode must be 'extractive' or 'abstractive'")

        print(f"Initialized RECOMPCompressor in {self.mode} mode on {self.device}")

    def compress(self, query: str, contexts: List[str], top_n: int = 5) -> str:
        """
        Compress retrieved contexts based on the query.
        
        Args:
            query: The user query.
            contexts: List of retrieved context strings.
            top_n: Number of sentences to keep (for extractive) or max output length (for abstractive logic).
            
        Returns:
            str: Compressed context.
        """
        if not contexts:
            return ""

        if self.mode == "extractive":
            return self._compress_extractive(query, contexts, top_n)
        else:
            return self._compress_abstractive(query, contexts)

    def _compress_extractive(self, query: str, contexts: List[str], top_n: int) -> str:
        """
        Scores sentences and selects the most relevant ones.
        """
        # For simplicity, we split contexts into sentences and score them
        # RECOMP models are often trained on specific query-document formats
        sentences = []
        for ctx in contexts:
            # Simple sentence splitting, could be improved with nltk or spacy
            sentences.extend([s.strip() for s in ctx.split('.') if s.strip()])

        if not sentences:
            return "\n".join(contexts)

        # Tokenize pairs: (query, sentence)
        inputs = self.tokenizer(
            [query] * len(sentences),
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            # Assuming binary classification or regression score
            scores = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]

        # Sort sentences by score
        scored_sentences = sorted(zip(scores.tolist(), sentences), key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored_sentences[:top_n]]

        return "\n".join(top_sentences)

    def _compress_abstractive(self, query: str, contexts: List[str]) -> str:
        """
        Generates a concise summary based on query and contexts.
        """
        # Format for abstractive RECOMP usually involves prepending query
        combined_context = " ".join(contexts)
        input_text = f"Question: {query} Context: {combined_context}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
