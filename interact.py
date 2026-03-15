import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent))

from src.retrievers import DenseRetriever
from src.compressor import RECOMPCompressor
from src.generation import RAGGenerator
from src.config import get_config
from langsmith import traceable, get_current_run_tree
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
    
    # check the model response cost usage 
    usage = response.get("usage", {})
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = usage.get("cost")
        cost_str = f" | [COST]: ${cost:.6f}" if cost is not None else ""
        print(f"[USAGE]: {prompt_tokens} prompt + {completion_tokens} completion = {usage.get('total_tokens', 0)} tokens{cost_str}")
        
        # Log to LangSmith metadata for this session run
        rt = get_current_run_tree()
        if rt:
            rt.metadata.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get('total_tokens', 0),
                "cost": cost
            })
    
    # 3. Live Evaluation (Ragas)
    if evaluator_mode:
        if not contexts:
            print("\n[SKIP EVALUATION]: No context documents retrieved.")
            return

        print("\n--- Running Live Evaluation (Ragas) ---")
        try:
            from ragas import evaluate
            from ragas.llms import llm_factory
            from ragas.metrics import faithfulness, answer_relevancy
            from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
            from openai import OpenAI

            # Setup Ragas LLM/Embeddings from config
            gen_config = config_dict.get("generation", {})
            ret_config = config_dict.get("retrieval", {})
            

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            openai_client = OpenAI(
                base_url=gen_config.get("base_url", "https://openrouter.ai/api/v1"), 
                api_key=openrouter_key
            )
            eval_model = gen_config.get("llm_model", "no_model_selected")
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
            
            
            # Setup metrics with the LLM/Embeddings
            faithfulness.llm = ragas_llm
            answer_relevancy.llm = ragas_llm
            answer_relevancy.embeddings = ragas_embeddings
            
            results = evaluate(
                ragas_dataset,
                metrics=[faithfulness, answer_relevancy],
                raise_exceptions=False
            )
            
            if results and len(results) > 0:
                print("\n[LIVE EVALUATION RESULTS]:")
                for metric, score in results.items():
                    print(f"  - {metric}: {score:.4f}")
            else:
                print("\n[EVALUATION FAILED]: Ragas returned no results. This might be due to model compatibility issues (e.g. system prompt rejection).")
                
        except Exception as e:
            # import traceback
            # traceback.print_exc()
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
    retriever = DenseRetriever()
    
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
