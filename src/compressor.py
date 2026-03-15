import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
# from visualization.tracker import tracker
from langsmith import traceable, Client
from src.config import get_config
from pathlib import Path
import json

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_ABSTRACTIVE_PROMPT_PATH = PROMPTS_DIR / "recomp_prompt.json"

class RECOMPCompressor:
    """
    Implements RECOMP (RECTified COMpression and Prepend) for RAG context compression.
    Supports both extractive and abstractive modes.
    """

    def __init__(self, mode: str = None, device: str = None):
        """
        Initialize the compressor.
        
        Args:
            mode: 'extractive' or 'abstractive'. If None, loads from config.
            device: 'cuda' or 'cpu'. If None, auto-detects.
        """
        config_dict = get_config()
        comp_config = config_dict.get("compression", {})
        self.mode = (mode or comp_config.get("mode", "extractive")).lower()
        self.top_n = config_dict.get("retrieval", {}).get("top_k", 5)
        self.hub_handle = comp_config.get("hub_handle")

        # Local fallback prompt template
        if DEFAULT_ABSTRACTIVE_PROMPT_PATH.exists():
            try:
                with open(DEFAULT_ABSTRACTIVE_PROMPT_PATH, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
                    self.local_prompt_template = prompt_data.get("template", "")
            except Exception as e:
                print(f"Error loading JSON compressor prompt: {e}")
                self.local_prompt_template = "Question: {query} Context: {context}"
        else:
            self.local_prompt_template = "Question: {query} Context: {context}"
            print(f"Warning: Compressor prompt file not found at {DEFAULT_ABSTRACTIVE_PROMPT_PATH}. Using emergency fallback.")

        self.prompt_template = self._load_prompt()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.mode == "extractive":
            self.model_name = comp_config.get("extractive_model", "fangyuan/nq_extractive_compressor")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        elif self.mode == "abstractive":
            self.model_name = comp_config.get("abstractive_model", "fangyuan/nq_abstractive_compressor")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        else:
            raise ValueError("Mode must be 'extractive' or 'abstractive'")

        print(f"Initialized RECOMPCompressor in {self.mode} mode on {self.device}")

    def _load_prompt(self) -> str:
        """Loads prompt from LangSmith Hub if handle exists, else uses local fallback."""
        if not self.hub_handle or self.mode != "abstractive":
            return self.local_prompt_template

        try:
            client = Client()
            hub_prompt = client.pull_prompt(self.hub_handle)
            
            # Extract template string from Hub object
            if hasattr(hub_prompt, "template"):
                return hub_prompt.template
            elif hasattr(hub_prompt, "messages"):
                for msg in hub_prompt.messages:
                    if hasattr(msg, "prompt") and hasattr(msg.prompt, "template"):
                        return msg.prompt.template
                    elif hasattr(msg, "content"):
                        return msg.content
            
            print(f"Info: Using local compressor prompt as Hub object structure was unexpected for {self.hub_handle}")
            return self.local_prompt_template
            
        except Exception as e:
            print(f"Warning: Failed to pull compressor prompt from Hub ({self.hub_handle}): {e}")
            return self.local_prompt_template

    @traceable(run_type="retriever")
    def compress(self, query: str, contexts: List[str], top_n: int = None) -> str:
        """
        Compress retrieved contexts based on the query.
        
        Args:
            query: The user query.
            contexts: List of retrieved context strings.
            top_n: Number of sentences to keep (for extractive). Defaults to self.top_n.
        """
        if top_n is None:
            top_n = self.top_n
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
        # Format for abstractive RECOMP using dynamic template
        combined_context = " ".join(contexts)
        input_text = self.prompt_template.format(
            query=query,
            context=combined_context
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
