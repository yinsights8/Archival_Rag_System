import os
from typing import List, Dict, Any

from dataclasses import dataclass
from datasets import Dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_mrr, calculate_recall_at_k, calculate_ndcg
from src.retrievers import DenseRetriever, SparseRetriever, HybridRetriever
from src.faiss_storage import FaissStorage
from dotenv import load_dotenv
load_dotenv()

@dataclass
class QAPair:
    query: str
    ground_truth_answer: str
    ground_truth_contexts: List[str] # could be subset of text or document IDs
    ground_truth_doc_ids: List[str] # The IDs of the documents that contain the answer

class RAGEvaluator:
    def __init__(self, faiss_store_path: str = None):
        if faiss_store_path:
            self.store = FaissStorage(dir_path=faiss_store_path)
        else:
            self.store = FaissStorage()
            
        self.dense = DenseRetriever(self.store)
        self.sparse = SparseRetriever(self.store)
        self.hybrid = HybridRetriever(self.store)
        
    def evaluate_retrievers(self, dataset: List[QAPair], top_k: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the dense, sparse, and hybrid retrievers using standard IR metrics.
        """
        results = {
            "dense": {"mrr": 0.0, f"recall@{top_k}": 0.0, f"ndcg@{top_k}": 0.0},
            "sparse": {"mrr": 0.0, f"recall@{top_k}": 0.0, f"ndcg@{top_k}": 0.0},
            "hybrid": {"mrr": 0.0, f"recall@{top_k}": 0.0, f"ndcg@{top_k}": 0.0},
        }
        
        # Save the evaluation dataset to disk
        import json as _json
        os.makedirs("data/queries", exist_ok=True)
        with open("data/queries/rag_questions_eval.json", "w", encoding="utf-8") as f:
            _json.dump(
                [{"query": qa.query, "ground_truth_answer": qa.ground_truth_answer,
                  "ground_truth_contexts": qa.ground_truth_contexts,
                  "ground_truth_doc_ids": qa.ground_truth_doc_ids} for qa in dataset],
                f, indent=2, ensure_ascii=False
            )
        print(f"Saved evaluation dataset to data/queries/rag_questions_eval.json")

        if not dataset:
            return results
            
        for qa in dataset:
            # 1. Fetch from Retrievers
            dense_res = self.dense.search(qa.query, top_k=top_k)
            sparse_res = self.sparse.search(qa.query, top_k=top_k)
            hybrid_res = self.hybrid.search(qa.query, top_k=top_k)
            
            # 2. Extract retrieved document IDs (using metadata source/doc_id for order)
            def extract_doc_ids(res):
                # Retrieve from metadata to keep the rank order
                ids = []
                for meta in res.get("metadatas", []):
                    # fallback to "doc_id" if "source" is not available, then empty string
                    ids.append(str(meta.get("doc_id", meta.get("source", ""))))
                return ids
                
            dense_ids = extract_doc_ids(dense_res)
            sparse_ids = extract_doc_ids(sparse_res)
            hybrid_ids = extract_doc_ids(hybrid_res)
            
            # print(f"DEBUG - Ground Truth IDs: {qa.ground_truth_doc_ids}")
            # print(f"DEBUG - Dense Retrieved: {dense_ids}")
            
            # 3. Calculate metrics for each retriever
            for name, r_ids in [("dense", dense_ids), ("sparse", sparse_ids), ("hybrid", hybrid_ids)]:
                # It's better to match by document ID to handle exact matches
                results[name]["mrr"] += calculate_mrr(qa.ground_truth_doc_ids, r_ids)
                results[name][f"recall@{top_k}"] += calculate_recall_at_k(qa.ground_truth_doc_ids, r_ids, top_k)
                results[name][f"ndcg@{top_k}"] += calculate_ndcg(qa.ground_truth_doc_ids, r_ids, top_k)
                
        # 4. Average results
        num_samples = len(dataset)
        for name in results:
            for metric in results[name]:
                results[name][metric] /= num_samples
                
        return results

    def evaluate_generation_with_ragas(self, dataset: List[QAPair], top_k: int = 5, retriever_type: str = "hybrid") -> Dict[str, float]:
        """
        Evaluates generator outputs using ragas metrics.
        Requires OpenAI API key to be set in environment.
        """
        try:
            from ragas import evaluate
            # Note: Importing from top-level ragas.metrics is deprecated in 0.4.x but 
            # importing from collections causes TypeError in evaluate() as metrics 
            # are not recognized as Metric instances in 0.4.3.
            from ragas.metrics import (
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
            )
            from ragas.llms import llm_factory
            from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
        except ImportError:
            print("Please install ragas: pip install ragas")
            return {}
        
        # Create an OpenAI client pointed at OpenRouter
        from openai import OpenAI
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            print("OPENROUTER_API_KEY not set. Cannot run Ragas evaluation.")
            return {}
        
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
        
        ragas_llm = llm_factory(
            "meta-llama/llama-3.1-70b-instruct",
            client=openai_client,
        )
        ragas_embeddings = RagasHFEmbeddings(model="BAAI/bge-small-en-v1.5")
        # Monkey-patch missing Langchain-style methods that Ragas metrics expect
        if not hasattr(ragas_embeddings, "embed_query"):
            ragas_embeddings.embed_query = ragas_embeddings.embed_text
        if not hasattr(ragas_embeddings, "embed_documents"):
            ragas_embeddings.embed_documents = ragas_embeddings.embed_texts
            
        retriever_map = {
            "dense": self.dense,
            "sparse": self.sparse,
            "hybrid": self.hybrid
        }
        
        retriever = retriever_map.get(retriever_type, self.hybrid)
        
        # Prepare lists for Ragas dataset
        queries = []
        ground_truths = []
        answers = []
        contexts = []
        
        print("Generating answers for Ragas evaluation...")
        from src.generation import RAGGenerator
        generator = RAGGenerator()
        
        for qa in dataset:
            queries.append(qa.query)
            # Ragas expects ground_truth as a plain string
            ground_truths.append(qa.ground_truth_answer)
            
            # Retrieve context
            res = retriever.search(qa.query, top_k=top_k)
            retrieved_contexts = res.get("contexts", [])
            contexts.append(retrieved_contexts)
            
            # Generate answer using RAGGenerator
            try:
                sources = res.get("sources", [])
                answer = generator.generate(qa.query, retrieved_contexts, sources)
                answers.append(answer)
            except Exception as e:
                print(f"  Warning: Generation failed for query '{qa.query[:50]}...': {e}")
                answers.append("Generation failed.")
        
        data = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        

        ragas_dataset = Dataset.from_dict(data)
        # Save the dataset to disk
        os.makedirs("data/rag_dataset", exist_ok=True)
        ragas_dataset.save_to_disk("data/rag_dataset")
        # Save as CSV for easier inspection
        ragas_dataset.to_csv("data/rag_dataset/rag_dataset.csv")
        print("Saved Ragas evaluation dataset to data/rag_dataset/rag_dataset.csv")


        
        print("Running Ragas metrics...")
        result = evaluate(
            ragas_dataset,
            metrics=[
                ContextPrecision(llm=ragas_llm),
                Faithfulness(llm=ragas_llm),
                AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                ContextRecall(llm=ragas_llm),
            ],
            raise_exceptions=False
        )
        
        return result


if __name__ == "__main__":
    import json
    import argparse
    from datetime import datetime
    
    print("--- Running Evaluation on JSON dataset ---")
    
    parser = argparse.ArgumentParser(description="Run evaluation testing.")
    parser.add_argument("--queries-file", type=str, default="data/queries/rag_questions.json")
    args = parser.parse_args()

    # Load questions from the JSON file
    try:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            rag_questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {args.queries_file}")
        sys.exit(1)
        
    eval_data = []
    for q in rag_questions:
        if q.get("status") in ("verified", "partially_verified", "generated"):
            eval_data.append(QAPair(
                query=q["question"],
                ground_truth_answer=q.get("answer", ""),
                ground_truth_contexts=[],
                ground_truth_doc_ids=[q.get("doc_id", "")]
            ))
            
    if not eval_data:
        print(f"No valid evaluation questions found in {args.queries_file}.")
        sys.exit(1)
        
    print(f"Loaded {len(eval_data)} questions from {args.queries_file}")
    
    evaluator = RAGEvaluator()
    
    # Store all results for export
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "queries_file": args.queries_file,
        "num_questions": len(eval_data),
        "retriever_evaluation": {},
        "generation_evaluation": {}
    }
    
    # Check if docstore exists, if not, skip
    if not os.path.exists(evaluator.store.docstore_path):
         print("Warning: Index/Docstore not found. Please ingest data before running the evaluation. Skipping evaluation logic.")
    else:
        print("\n1. Evaluating Retrievers...")
        retriever_results = evaluator.evaluate_retrievers(eval_data, top_k=5)
        all_results["retriever_evaluation"] = retriever_results
        
        for r_name, metrics in retriever_results.items():
            print(f"- {r_name.capitalize()} Retriever:")
            for m_name, score in metrics.items():
                print(f"  - {m_name}: {score:.4f}")
                
        print("\n2. Evaluating Generation (Requires OpenAI API Key and Cost)...")
        # To avoid unexpected costs during a dummy run, we check for API key
        if os.getenv("OPENROUTER_API_KEY"):
            ragas_results = evaluator.evaluate_generation_with_ragas(eval_data, top_k=5)
            print("Ragas Results:", ragas_results)
            
            # handle both EvaluationResult object and dictionary
            # Ragas 0.4.x has summarized scores in _scores_dict (repr shows this)
            # result.scores is a list of per-row scores, which caused the crash
            if hasattr(ragas_results, "_scores_dict"):
                scores_dict = ragas_results._scores_dict
            elif hasattr(ragas_results, "items"):
                scores_dict = ragas_results
            else:
                # fallback: try to convert to dict or use as is
                try:
                    scores_dict = dict(ragas_results)
                except (TypeError, ValueError):
                    scores_dict = ragas_results
            
            all_results["generation_evaluation"] = {
                "ragas_results": {k: float(v) if not isinstance(v, (dict, list)) else v for k, v in scores_dict.items()},
                "retriever_type": "hybrid" # default used in method
            }
        else:
            print("Skipping Ragas: No OPENROUTER_API_KEY found.")
            all_results["generation_evaluation"] = {"status": "skipped", "reason": "No OPENROUTER_API_KEY"}

        # Export results to JSON
        os.makedirs("results", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"results/evaluation_results_{timestamp_str}.json"
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
            
        print(f"\n--- Evaluation Complete ---")
        print(f"Results exported to: {export_path}")
