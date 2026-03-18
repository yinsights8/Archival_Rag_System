import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.generation_local import LocalRAGGenerator
    print("Successfully imported LocalRAGGenerator.")
except ImportError as e:
    print(f"Error importing LocalRAGGenerator: {e}")
    sys.exit(1)

def test_local_generation():
    print("\n--- Testing LocalRAGGenerator ---")
    
    # Initialize generator
    # Note: This will attempt to connect to the base_url in config.yaml
    generator = LocalRAGGenerator()
    
    question = "What is the capital of France?"
    contexts = ["Paris is the capital of France.", "Lyon is a city in France."]
    
    print(f"Question: {question}")
    print(f"Contexts: {contexts}")
    
    try:
        print("Sending request to local LLM...")
        result = generator.generate(question, contexts)
        
        print("\n--- Result ---")
        print(f"Answer: {result.get('answer')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Reasoning: {result.get('reasoning')}")
        print(f"Model: {result.get('model')}")
        print(f"Usage: {result.get('usage')}")
        
    except Exception as e:
        print(f"\nCaught expected or unexpected error: {e}")
        print("Note: This is expected if no local LLM (LM Studio/Ollama) is running.")

if __name__ == "__main__":
    test_local_generation()
