import json
import argparse
import sys
import os

# Add root project dir to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inngest

def main():
    parser = argparse.ArgumentParser(description="Trigger RAG Evaluation Pipeline via Inngest")
    parser.add_argument("--queries-file", type=str, default="data/queries/rag_questions.json", 
                        help="Path to the generated rag_questions.json file")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Number of documents to retrieve per query (default: 5)")
    parser.add_argument("--run-generation", action="store_true", 
                        help="Run LLM generation evaluation (warning: requires OpenAI Key and has API costs)")
    
    args = parser.parse_args()
    
    # Load questions from the output of `create_queries.py`
    try:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            rag_questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {args.queries_file}")
        print("Please run `python src/create_queries.py --corpus data/corpus.jsonl` first to generate evaluation data.")
        return
        
    eval_data = []
    
    # Filter for active questions and map to the format expected by the evaluation endpoint
    for q in rag_questions:
        if q.get("status") in ("verified", "partially_verified", "generated"):
            eval_data.append({
                "query": q["question"],
                "ground_truth_answer": q.get("answer", ""),
                "ground_truth_doc_ids": [q.get("doc_id", "")]
            })
            
    if not eval_data:
        print(f"No valid evaluation questions found in {args.queries_file}.")
        return
        
    print(f"Loaded {len(eval_data)} questions for evaluation.")
    
    # Initialize the Inngest client to trigger the event
    client = inngest.Inngest(
        app_id="archival_rag_system",
        serializer=inngest.PydanticSerializer()
    )
    
    event = inngest.Event(
        name="app/rag_evaluate_nls_corpus",
        data={
            "eval_data": eval_data,
            "top_k": args.top_k,
            "run_generation": args.run_generation
        }
    )
    
    print("Triggering evaluation event...")
    
    try:
        # Trigger the event synchronously 
        client.send_sync(event)
        print("\nEvaluation event triggered successfully!")
        print("Check your Inngest dev server dashboard to monitor the metrics execution.")
    except Exception as e:
        print(f"Failed to trigger Inngest event: {e}")
        print("Make sure your Inngest dev server is running (`npx inngest-cli dev`).")

if __name__ == "__main__":
    main()
