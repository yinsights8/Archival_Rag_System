import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.retrievers import DenseRetriever, SparseRetriever, HybridRetriever
from src.compressor import RECOMPCompressor
from src.generation import RAGGenerator
from src.config import get_config
from langsmith import traceable
from datasets import Dataset
from langchain_core.tracers import LangChainTracer

load_dotenv()
tracer = LangChainTracer(project_name=os.getenv("LANGSMITH_PROJECT"))

@traceable(run_type="chain")
def live_qa_session(query: str, retriever: DenseRetriever, generator: RAGGenerator, config_dict: dict, top_k: int = 10, evaluator_mode: bool = False):
    """Orchestrates the RAG flow for a single query."""
    print(f"\n[QUERY]: {query}")
    
    # 1. Retrieve
    print(f"--- Retrieving top-{top_k} contexts ---")
    retrieval_res = retriever.search(query, top_k=top_k)
    contexts = retrieval_res.get("contexts", [])
    sources = retrieval_res.get("sources", [])
    
    if not contexts:
        print("No contexts found in the index.")
        return
    
    # 2. Generate
    print("--- Generating answer ---")
    response = generator.generate(query, contexts, sources)
    
    answer = response.get("answer", "No answer generated.")
    confidence = response.get("confidence", 0)
    reasoning = response.get("reasoning", "")
    
    print(f"\n[ANSWER]: {answer}")
    print(f"[CONFIDENCE]: {confidence}/100")
    print(f"[REASONING]: {reasoning}")
    
    # 3. Live Evaluation (Ragas)
    if evaluator_mode:
        if not contexts:
            print("\n[SKIP EVALUATION]: No context documents retrieved.")
            return

        print("\n--- Running Live Evaluation (Ragas) ---")
        try:
            from ragas import evaluate
            from ragas.metrics import Faithfulness, AnswerRelevancy
            from ragas.llms import llm_factory
            from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
            from openai import OpenAI

            # Setup Ragas LLM/Embeddings from config
            gen_config = config_dict.get("generation", {})
            ret_config = config_dict.get("retrieval", {})
            
            # Use a slightly more "robust" model for EVALUATION if current model might be buggy with Ragas
            # (Ragas often uses internal system prompts)
            eval_model = generator.model_name
            if "gemma-3" in eval_model:
                # Gemma-3 might reject "Developer Instructions" (system prompts) used by Ragas
                # Try fallback or explicitly inform the user
                pass

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            openai_client = OpenAI(
                base_url=gen_config.get("base_url", "https://openrouter.ai/api/v1"), 
                api_key=openrouter_key
            )
            ragas_llm = llm_factory(eval_model, client=openai_client)
            ragas_embeddings = RagasHFEmbeddings(model=ret_config.get("embedding_model", "BAAI/bge-small-en-v1.5"))
            
            # Patch embeddings methods
            if not hasattr(ragas_embeddings, "embed_query"):
                ragas_embeddings.embed_query = ragas_embeddings.embed_text
            if not hasattr(ragas_embeddings, "embed_documents"):
                ragas_embeddings.embed_documents = ragas_embeddings.embed_texts

            # Prepare dataset for Ragas
            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [""] # No ground truth for live QA
            }
            ragas_dataset = Dataset.from_dict(data)
            
            # Evaluate (Faithfulness and AnswerRelevancy don't strictly require ground truth)
            results = evaluate(
                ragas_dataset,
                metrics=[
                    Faithfulness(llm=ragas_llm),
                    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                ],
                raise_exceptions=False,
                callbacks=[tracer]
            )
            
            if results and len(results) > 0:
                print("\n[LIVE EVALUATION RESULTS]:")
                for metric, score in results.items():
                    print(f"  - {metric}: {score:.4f}")
            else:
                print("\n[EVALUATION FAILED]: Ragas returned no results. This might be due to model compatibility issues (e.g. system prompt rejection).")
                
        except Exception as e:
            print(f"Live evaluation failed: {str(e)}")
            if "Developer instruction" in str(e):
                print("Tip: The selected model (Gemma-3?) may not allow system prompts used by Ragas. Try using a Llama model.")

def main():
    parser = argparse.ArgumentParser(description="Live Archival RAG Interaction")
    parser.add_argument("--eval", action="store_true", help="Enable live Ragas evaluation")
    parser.add_argument("--no-compress", action="store_true", help="Disable context compression")
    parser.add_argument("--top-k", type=int, help="Override top-k retrieval")
    args = parser.parse_args()

    config_dict = get_config()
    retrieval_top_k = args.top_k or config_dict.get("retrieval", {}).get("top_k", 10)
    
    print("--- Initializing Archival RAG System ---")
    retriever = HybridRetriever()
    
    compressor_mode = config_dict.get("compression", {}).get("mode", "extractive")
    if args.no_compress:
        compressor = None
        print("Compression: DISABLED")
    else:
        compressor = RECOMPCompressor(mode=compressor_mode)
        print(f"Compression: ENABLED ({compressor_mode})")
        
    generator = RAGGenerator(compressor=compressor)
    
    print("\nSystem ready. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break
                
            live_qa_session(query, retriever, generator, config_dict=config_dict, top_k=retrieval_top_k, evaluator_mode=args.eval)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

    print("\nExiting. Trace available in LangSmith (if enabled).")

if __name__ == "__main__":
    main()
