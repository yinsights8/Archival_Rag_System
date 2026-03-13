import os
import re
import json
import yaml
from typing import List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.compressor import RECOMPCompressor

load_dotenv()

# Load centralized config if available
CONFIG_PATH = "config.yaml"
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {}
    return {}

_config = load_config()
_gen_config = _config.get("generation", {})


# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model for query generation
DEFAULT_MODEL = _gen_config.get("llm_model", "meta-llama/llama-3.1-70b-instruct")
DEFAULT_TEMP = _gen_config.get("temperature", 0.1)
DEFAULT_BASE_URL = _gen_config.get("base_url", "https://openrouter.ai/api/v1")


def get_llm_client(model: str = None, temperature: float = None) -> ChatOpenAI:
    """Create OpenRouter client."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to .env file.\n"
            "Get a key at https://openrouter.ai/keys"
        )
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        openai_api_base=DEFAULT_BASE_URL,
        openai_api_key=OPENROUTER_API_KEY,
        temperature=temperature if temperature is not None else DEFAULT_TEMP,
    )


class RAGGenerator:
    """Handles LLM answer generation from retrieved contexts."""

    def __init__(self, llm: ChatOpenAI = None, compressor: RECOMPCompressor = None):
        self.llm = llm or get_llm_client()
        self.compressor = compressor
        self.prompt_template = """You are a historical research assistant analyzing digitized archival documents.

                        The following passages are from historical documents that may contain OCR errors 
                        (e.g., character substitutions, missing words, garbled text).

                        CONTEXT:
                        {context}

                        QUESTION: {question}

                        Instructions:
                        1. Answer the question based ONLY on the provided context
                        2. If the context doesn't contain enough information, say "INSUFFICIENT_CONTEXT"
                        3. Be aware that OCR errors may affect readability
                        4. Rate your confidence in the answer from 0 to 100

                        Respond in this exact JSON format:
                        {{
                            "answer": "your answer here",
                            "confidence": 85,
                            "reasoning": "brief explanation of how you derived the answer",
                            "ocr_issues_noted": "any OCR errors you noticed that affected comprehension"
                        }}"""

    def generate(self, question: str, contexts: List[str], sources: List[str] = None) -> dict:
        """
        Generate a structured answer given a question and retrieved contexts.
        
        Args:
            question: The user question.
            contexts: List of retrieved context strings.
            sources: List of source identifiers.
            
        Returns:
            dict: The generated JSON response containing answer, confidence, reasoning, and ocr_issues_noted.
        """
        if self.compressor:
            context_text = self.compressor.compress(question, contexts)
        else:
            context_text = "\n\n---\n\n".join(contexts)
        
        prompt = self.prompt_template.format(

            context=context_text,
            question=question
        )
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse JSON response
            if "```" in content:
                # Find the first JSON block or the first block that looks like JSON
                blocks = content.split("```")
                for block in blocks:
                    block_stripped = block.strip()
                    if block_stripped.startswith("json"):
                        content = block_stripped[4:].strip()
                        break
                    elif block_stripped.startswith("{"):
                        content = block_stripped
                        break
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract key fields with regex
                answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', content)
                conf_match = re.search(r'"confidence"\s*:\s*(\d+)', content)
                reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
                ocr_match = re.search(r'"ocr_issues_noted"\s*:\s*"([^"]*)"', content)
                
                result = {
                    "answer": answer_match.group(1) if answer_match else content[:200],
                    "confidence": int(conf_match.group(1)) if conf_match else 50,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Failed to parse structured response",
                    "ocr_issues_noted": ocr_match.group(1) if ocr_match else "",
                }
            
            # Attach model name for reference
            result["model"] = self.llm.model_name
            return result
            
        except Exception as e:
            return {
                "answer": f"ERROR: {str(e)}",
                "confidence": 0,
                "reasoning": "",
                "ocr_issues_noted": "",
                "model": getattr(self.llm, "model_name", "unknown"),
                "error": str(e),
            }


