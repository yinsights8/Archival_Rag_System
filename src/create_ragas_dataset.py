import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrievers import DenseRetriever
from src.generation import RAGGenerator
from src.compressor import RECOMPCompressor
from src.config import get_config

"""
run script : uv run src/create_ragas_dataset.py --queries-file data/queries/rag_questions_eval.json --output data/ragas/my_ragas_dataset_ext.csv --use-compressor  --compressor-mode extractive
run script : uv run src/create_ragas_dataset.py --queries-file data/queries/rag_questions_eval.json --output data/my_ragas_dataset.csv
"""


def generate_dataset(queries_file: str, output_csv: str = "data/ragas_dataset.csv", use_compressor: bool = False, compressor_mode: str = "extractive"):
    """
    Reads questions and ground truths from queries_file, runs the RAG pipeline,
    and saves the required format for Ragas evaluation.
    """
    print(f"Loading queries from: {queries_file}")
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize RAG components
    print("Initializing RAG Pipeline Components...")
    retriever = DenseRetriever()
    
    compressor = None
    if use_compressor:
        compressor = RECOMPCompressor(mode=compressor_mode)
        
    generator = RAGGenerator(compressor=compressor)
    
    # Ragas Dataset Lists
    questions = []
    ground_truths = []
    sys_answers = []
    sys_contexts = []
    
    print(f"Generating answers for {len(data)} questions...")
    for item in tqdm(data):
        # We only want to evaluate questions that are generated or verified
        if item.get("status") in ["rejected"]:
            continue
            
        question = item.get("query", "")
        ground_truth = item.get("ground_truth_answer", "")
        
        # 1. Retrieve
        retrieval_res = retriever.search(question, top_k=20)
        contexts = retrieval_res.get("contexts", [])
        sources = retrieval_res.get("sources", [])
        
        # 2. Generate
        try:
            response = generator.generate(question, contexts, sources)
            answer = response.get("answer", "")
        except Exception as e:
            print(f"Failed to generate answer for: {question[:30]}... Error: {e}")
            answer = "Generation Failed"
            
        # Append to dataset
        questions.append(question)
        ground_truths.append(ground_truth)
        sys_answers.append(answer)
        sys_contexts.append(contexts)
        
    # Compile
    dataset_dict = {
        "question": questions,
        "answer": sys_answers,
        "contexts": sys_contexts,
        "ground_truth": ground_truths
    }
    
    ragas_dataset = Dataset.from_dict(dataset_dict)
    
    # Save Results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    ragas_dataset.to_csv(output_csv)
    
    print(f"\nSuccessfully generated and saved Ragas dataset to {output_csv}")
    print(f"Total evaluated items: {len(questions)}")
    
    return ragas_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-file", type=str, default="data/queries/rag_questions_eval.json")
    parser.add_argument("--output", type=str, default="data/ragas_dataset.csv")
    parser.add_argument("--use-compressor", action="store_true")
    parser.add_argument("--compressor-mode", type=str, default="extractive")
    args = parser.parse_args()
    
    generate_dataset(
        queries_file=args.queries_file, 
        output_csv=args.output,
        use_compressor=args.use_compressor,
        compressor_mode=args.compressor_mode
    )
