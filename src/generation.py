import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model for query generation
DEFAULT_MODEL = "meta-llama/llama-3.1-70b-instruct"


def get_llm_client() -> ChatOpenAI:
    """Create OpenRouter client."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to .env file.\n"
            "Get a key at https://openrouter.ai/keys"
        )
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=OPENROUTER_API_KEY,
    )


class RAGGenerator:
    """Handles LLM answer generation from retrieved contexts."""

    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or get_llm_client()
        self.prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the context provided.
            If the answer is not in the context, say "I don't know".
            
            Question: {question}
            Context: {context}
            
            Answer:
            Sources: {sources}
            """
        )
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def generate(self, question: str, contexts: List[str], sources: List[str] = None) -> str:
        """
        Generate an answer given a question and retrieved contexts.
        
        Args:
            question: The user question.
            contexts: List of retrieved context strings.
            sources: List of source identifiers.
            
        Returns:
            str: The generated answer.
        """
        context_text = "\n\n".join(f"- {c}" for c in contexts)
        sources_text = ", ".join(sources) if sources else "N/A"
        return self.chain.invoke({
            "context": context_text,
            "question": question,
            "sources": sources_text,
        })
