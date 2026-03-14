import os
import re
import json
import yaml
import textwrap
from typing import List
import openai
from dotenv import load_dotenv
from src.compressor import RECOMPCompressor
from langsmith import traceable, Client
from langsmith.wrappers import wrap_openai
from pathlib import Path
from src import config

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

load_dotenv()

_config = config.get_config()
_gen_config = _config.get("generation", {})


# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model for query generation
DEFAULT_MODEL = _gen_config.get("llm_model")
DEFAULT_TEMP = _gen_config.get("temperature")
DEFAULT_BASE_URL = _gen_config.get("base_url")


def get_llm_client(model: str = None, temperature: float = None) -> openai.OpenAI:
    """Create OpenRouter client using native OpenAI library."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to .env file.\n"
            "Get a key at https://openrouter.ai/keys"
        )
    client = openai.OpenAI(
        base_url=DEFAULT_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    # Wrap for LangSmith tracing
    return wrap_openai(client)


class RAGGenerator:
    """Handles LLM answer generation from retrieved contexts."""

    def __init__(self, llm: openai.OpenAI = None, compressor: RECOMPCompressor = None):
        self.llm = llm or get_llm_client()
        self.model_name = _gen_config.get("llm_model", DEFAULT_MODEL)
        self.compressor = compressor
        self.hub_handle = _gen_config.get("hub_handle")
        
        # Local fallback prompt template from external file
        if DEFAULT_PROMPT_PATH.exists():
            self.local_prompt_template = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()
        else:
            # Hardcoded emergency fallback if file is missing
            self.local_prompt_template = "Context: {context}\nQuestion: {question}"
            print(f"Warning: Prompt file not found at {DEFAULT_PROMPT_PATH}. Using emergency fallback.")

        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Loads prompt from LangSmith Hub if handle exists, else uses local fallback."""
        if not self.hub_handle:
            return self.local_prompt_template

        try:
            client = Client()
            hub_prompt = client.pull_prompt(self.hub_handle)
            
            # Extract template string from Hub object (handles PromptTemplate, ChatPromptTemplate, etc.)
            if hasattr(hub_prompt, "template"):
                return hub_prompt.template
            elif hasattr(hub_prompt, "messages"):
                # For ChatPromptTemplate, we usually want the user message content
                for msg in hub_prompt.messages:
                    if hasattr(msg, "prompt") and hasattr(msg.prompt, "template"):
                        return msg.prompt.template
                    elif hasattr(msg, "content"):
                        return msg.content
            
            print(f"Info: Using local prompt as Hub object structure was unexpected for {self.hub_handle}")
            return self.local_prompt_template
            
        except Exception as e:
            print(f"Warning: Failed to pull prompt from Hub ({self.hub_handle}): {e}")
            print("Falling back to local prompt template.")
            return self.local_prompt_template

    @traceable
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
        # print(f"DEBUG: Number of input contexts: {len(contexts)}")
        if self.compressor:
            context_text = self.compressor.compress(question, contexts)
        else:
            context_text = "\n\n---\n\n".join(contexts)
        
        if not context_text:
            print(f"Warning: Empty context for question: {question[:50]}...")
            
        prompt = self.prompt_template.format(
            context=context_text,
            question=question
        )
        
        # print(f"DEBUG: Final Prompt length: {len(prompt)}")
        # print(f"DEBUG: Context length in prompt: {len(context_text)}")
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=DEFAULT_TEMP,
            )
            content = response.choices[0].message.content.strip()
            
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
            result["model"] = self.model_name
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


