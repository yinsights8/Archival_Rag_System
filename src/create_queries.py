"""
Generate and manage evaluation queries for retrieval and RAG experiments.

Uses an LLM (via OpenRouter) to generate queries from document metadata,
with support for human-in-the-loop verification by domain specialists.

Produces:
- 50 retrieval queries with relevance judgments (BEIR format)
- 30 RAG verification questions with ground truth answers

Usage:
    python scripts/create_queries.py --corpus data/corpus.jsonl --output data/queries/
    python scripts/create_queries.py --verify data/queries/  # Human verification mode
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model for query generation
DEFAULT_MODEL = "meta-llama/llama-3.1-70b-instruct"


def get_client() -> OpenAI:
    """Create OpenRouter client."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to .env file.\n"
            "Get a key at https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


def sample_documents_for_queries(corpus_path: Path, n_retrieval: int = 50,
                                  n_rag: int = 30, seed: int = 42) -> tuple[list, list]:
    """Sample documents from corpus for query generation.
    
    For retrieval queries: diverse sample across tiers and topics.
    For RAG queries: prefer documents with enough text for QA.
    """
    random.seed(seed)
    
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    
    # Filter out very short documents
    viable = [r for r in records if r.get("word_count", 0) >= 100]
    
    if len(viable) < n_retrieval + n_rag:
        print(f"Warning: Only {len(viable)} viable documents (need {n_retrieval + n_rag})")
    
    # Stratified sample for diversity
    random.shuffle(viable)
    
    # For retrieval: spread across OCR tiers
    retrieval_docs = []
    by_tier = {}
    for r in viable:
        tier = r.get("ocr_quality_tier", "unknown")
        by_tier.setdefault(tier, []).append(r)
    
    per_tier = max(1, n_retrieval // max(1, len(by_tier)))
    for tier, docs in by_tier.items():
        retrieval_docs.extend(docs[:per_tier])
    
    # Fill remaining
    used_ids = {d["doc_id"] for d in retrieval_docs}
    remaining = [r for r in viable if r["doc_id"] not in used_ids]
    retrieval_docs.extend(remaining[:n_retrieval - len(retrieval_docs)])
    retrieval_docs = retrieval_docs[:n_retrieval]
    
    # For RAG: prefer longer documents with higher OCR quality
    rag_candidates = sorted(
        [r for r in viable if r["doc_id"] not in used_ids],
        key=lambda r: (r.get("ocr_quality") or 0) * min(r.get("word_count", 0), 5000),
        reverse=True,
    )
    rag_docs = rag_candidates[:n_rag]
    
    return retrieval_docs, rag_docs


def generate_retrieval_queries(docs: list[dict], client: OpenAI,
                                model: str = DEFAULT_MODEL) -> list[dict]:
    """Generate factoid retrieval queries from document samples using LLM."""
    queries = []
    
    for i, doc in enumerate(docs):
        text_preview = doc.get("text", "")[:2000]  # First 2K chars for context
        title = doc.get("title", "Unknown")
        date = doc.get("date", "Unknown")
        collection = doc.get("collection", "Unknown")
        language = doc.get("language", "Unknown")

        
        prompt = f"""You are a historical research specialist. Based on this digitized archival document, 
                    generate ONE specific factoid query that a researcher might use to find this document.

                    Document Title: {title}
                    Date: {date}
                    Collection: {collection}
                    Language: {language}
                    Text excerpt (may contain OCR errors):
                    ---
                    {text_preview}
                    ---

                    Requirements:
                    1. The query should be a natural language question a historian would ask
                    2. It should be specific enough that this document is relevant
                    3. It should relate to verifiable historical facts, events, people, or places
                    4. Do NOT reference OCR errors or document quality
                    5. Vary query types: who/what/when/where/how questions

                    Respond with ONLY a JSON object:
                    {{
                        "query": "your question here",
                        "query_type": "factoid|topical|comparative",
                        "expected_answer": "brief expected answer from the document",
                        "difficulty": "easy|medium|hard"
                    }}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            
            content = response.choices[0].message.content.strip()
            # Parse JSON from response (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            query_data = json.loads(content)
            query_data["doc_id"] = doc["doc_id"]
            query_data["ocr_quality_tier"] = doc.get("ocr_quality_tier", "unknown")
            query_data["ocr_quality"] = doc.get("ocr_quality")
            query_data["query_id"] = f"Q{i+1:03d}"
            query_data["verified_by"] = []  # For human verification
            query_data["status"] = "generated"  # generated → verified → approved
            queries.append(query_data)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{len(docs)} retrieval queries")
                
        except Exception as e:
            print(f"  Error generating query for {doc['doc_id']}: {e}")
            continue
    
    return queries


def generate_rag_questions(docs: list[dict], client: OpenAI,
                           model: str = DEFAULT_MODEL) -> list[dict]:
    """Generate verification questions with ground truth for RAG evaluation."""
    questions = []
    
    for i, doc in enumerate(docs):
        text_preview = doc.get("text", "")[:3000]  # More context for RAG
        title = doc.get("title", "Unknown")
        
        prompt = f"""You are creating a factual verification dataset for testing RAG systems on historical archives.

                    Based on this archival document, create ONE question-answer pair where:
                    1. The answer is explicitly stated in the text
                    2. The answer is a verifiable fact (date, name, place, number, event)
                    3. The question is natural and would be asked by a researcher
                    4. Provide the exact text span that contains the answer (as evidence)

                    Document Title: {title}
                    Text (may contain OCR errors):
                    ---
                    {text_preview}
                    ---

                    Respond with ONLY a JSON object:
                    {{
                        "question": "your question",
                        "answer": "the factual answer",
                        "evidence": "exact text span from the document supporting the answer",
                        "answer_type": "date|person|place|number|event|concept",
                        "difficulty": "easy|medium|hard",
                        "notes": "any relevant context about OCR quality affecting this Q&A"
                    }}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=400,
            )
            
            content = response.choices[0].message.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            qa_data = json.loads(content)
            qa_data["doc_id"] = doc["doc_id"]
            qa_data["ocr_quality_tier"] = doc.get("ocr_quality_tier", "unknown")
            qa_data["ocr_quality"] = doc.get("ocr_quality")
            qa_data["question_id"] = f"RAG{i+1:03d}"
            qa_data["verified_by"] = []
            qa_data["status"] = "generated"
            questions.append(qa_data)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{len(docs)} RAG questions")
                
        except Exception as e:
            print(f"  Error generating question for {doc['doc_id']}: {e}")
            continue
    
    return questions


def save_queries(queries: list[dict], rag_questions: list[dict], output_dir: Path,
                 skip_template: bool = False):
    """Save queries in multiple formats for different tools.
    
    Args:
        skip_template: If True, do not overwrite verification_template.json.
                       Used when called from verify mode to preserve rejection records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ─── Retrieval queries ───
    
    # BEIR-format queries.jsonl
    with open(output_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({
                "_id": q["query_id"],
                "text": q["query"],
                "metadata": {
                    "type": q.get("query_type", ""),
                    "difficulty": q.get("difficulty", ""),
                    "status": q["status"],
                }
            }) + "\n")
    
    # BEIR-format qrels.tsv (query-document relevance)
    with open(output_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in queries:
            f.write(f"{q['query_id']}\t{q['doc_id']}\t1\n")
    
    # Full query details (for verification and analysis)
    with open(output_dir / "queries_full.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    # ─── RAG questions ───
    with open(output_dir / "rag_questions.json", "w", encoding="utf-8") as f:
        json.dump(rag_questions, f, indent=2, ensure_ascii=False)
    
    # ─── Human verification template ───
    if not skip_template:
        verification = {
            "instructions": (
                "Please verify each query/question below. For each item:\n"
                "1. Check if the query is reasonable and answerable\n"
                "2. Check if the expected answer is correct\n"
                "3. Add your name to 'verified_by' list\n"
                "4. Set status to 'verified' (acceptable) or 'rejected' (needs revision)\n"
                "5. Add notes in 'reviewer_notes' if needed\n\n"
                "Two domain specialists should verify each item independently."
            ),
            "retrieval_queries": queries,
            "rag_questions": rag_questions,
        }
        
        with open(output_dir / "verification_template.json", "w", encoding="utf-8") as f:
            json.dump(verification, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Saved queries to: {output_dir}")
    print(f"  Retrieval queries: {len(queries)}")
    print(f"  RAG questions: {len(rag_questions)}")
    print(f"\n  Files:")
    print(f"    queries.jsonl          - BEIR-format retrieval queries")
    print(f"    qrels.tsv              - Query relevance judgments")
    print(f"    queries_full.json      - Full query details")
    print(f"    rag_questions.json     - RAG verification questions")
    print(f"    verification_template.json - Human verification template")
    print(f"\n  → Send verification_template.json to 2 domain specialists")
    print(f"{'='*60}")


def verify_queries(query_dir: Path):
    """Interactive human verification mode for generated queries."""
    template_path = query_dir / "verification_template.json"
    
    if not template_path.exists():
        print(f"No verification template found at {template_path}")
        return
    
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(data["instructions"])
    print(f"\n{'='*60}")
    
    reviewer_name = input("Enter your name (reviewer): ").strip()
    if not reviewer_name:
        print("Name required.")
        return
    
    # Verify retrieval queries
    print(f"\n--- Retrieval Queries ({len(data['retrieval_queries'])}) ---\n")
    for q in data["retrieval_queries"]:
        if reviewer_name in q.get("verified_by", []):
            continue
        
        print(f"\n[{q['query_id']}] OCR Tier: {q.get('ocr_quality_tier', '?')}")
        print(f"  Query: {q['query']}")
        print(f"  Expected: {q.get('expected_answer', 'N/A')}")
        print(f"  Type: {q.get('query_type', '?')} | Difficulty: {q.get('difficulty', '?')}")
        # Show prior reviewer decisions
        if q.get("verified_by"):
            print(f"  Prior reviews: {q['verified_by']} → status={q.get('status', '?')}")
        if q.get("reviewer_notes"):
            for note in q["reviewer_notes"]:
                print(f"  ⚠ Note: {note}")
        
        action = input("  (a)ccept / (r)eject / (s)kip / (q)uit: ").strip().lower()
        
        if action == "q":
            break
        elif action == "s":
            continue
        elif action == "a":
            q["verified_by"].append(reviewer_name)
            q.setdefault("decisions", {})[reviewer_name] = "accept"
            # Only set to verified if no one rejected
            if q.get("decisions", {}).get(reviewer_name) == "accept" and \
               all(d == "accept" for d in q.get("decisions", {}).values()):
                q["status"] = "verified" if len(q["verified_by"]) >= 2 else "partially_verified"
            else:
                q["status"] = "disputed"
        elif action == "r":
            q["verified_by"].append(reviewer_name)
            q.setdefault("decisions", {})[reviewer_name] = "reject"
            q["status"] = "rejected"
            notes = input("  Reason for rejection: ").strip()
            if not notes:
                notes = "no reason given"
            q.setdefault("reviewer_notes", []).append(f"{reviewer_name}: {notes}")
    
    # Verify RAG questions
    print(f"\n--- RAG Questions ({len(data['rag_questions'])}) ---\n")
    for q in data["rag_questions"]:
        if reviewer_name in q.get("verified_by", []):
            continue
        
        print(f"\n[{q['question_id']}] OCR Tier: {q.get('ocr_quality_tier', '?')}")
        print(f"  Question: {q['question']}")
        print(f"  Answer: {q.get('answer', 'N/A')}")
        print(f"  Evidence: {q.get('evidence', 'N/A')}")
        print(f"  Type: {q.get('answer_type', '?')} | Difficulty: {q.get('difficulty', '?')}")
        if q.get("notes"):
            print(f"  Notes: {q['notes']}")
        # Show prior reviewer decisions
        if q.get("verified_by"):
            print(f"  Prior reviews: {q['verified_by']} → status={q.get('status', '?')}")
        if q.get("reviewer_notes"):
            for note in q["reviewer_notes"]:
                print(f"  ⚠ Note: {note}")
        
        action = input("  (a)ccept / (r)eject / (s)kip / (q)uit: ").strip().lower()
        
        if action == "q":
            break
        elif action == "s":
            continue
        elif action == "a":
            q["verified_by"].append(reviewer_name)
            q.setdefault("decisions", {})[reviewer_name] = "accept"
            if all(d == "accept" for d in q.get("decisions", {}).values()):
                q["status"] = "verified" if len(q["verified_by"]) >= 2 else "partially_verified"
            else:
                q["status"] = "disputed"
        elif action == "r":
            q["verified_by"].append(reviewer_name)
            q.setdefault("decisions", {})[reviewer_name] = "reject"
            q["status"] = "rejected"
            notes = input("  Reason for rejection: ").strip()
            if not notes:
                notes = "no reason given"
            q.setdefault("reviewer_notes", []).append(f"{reviewer_name}: {notes}")
    
    # Save updated template
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Update individual files
    approved_queries = [q for q in data["retrieval_queries"] if q.get("status") in ("verified", "partially_verified")]
    approved_rag = [q for q in data["rag_questions"] if q.get("status") in ("verified", "partially_verified")]
    save_queries(approved_queries, approved_rag, query_dir, skip_template=True)
    
    ret_verified = sum(1 for q in data["retrieval_queries"] if "verified" in q.get("status", ""))
    ret_rejected = sum(1 for q in data["retrieval_queries"] if q.get("status") == "rejected")
    rag_verified = sum(1 for q in data["rag_questions"] if "verified" in q.get("status", ""))
    rag_rejected = sum(1 for q in data["rag_questions"] if q.get("status") == "rejected")
    print(f"\nVerification summary:")
    print(f"  Retrieval: {ret_verified} verified, {ret_rejected} rejected")
    print(f"  RAG: {rag_verified} verified, {rag_rejected} rejected")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation queries")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl",
                       help="Path to corpus JSONL file")
    parser.add_argument("--output", type=str, default="data/queries",
                       help="Output directory for queries")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"OpenRouter model for generation (default: {DEFAULT_MODEL})")
    parser.add_argument("--n-retrieval", type=int, default=50,
                       help="Number of retrieval queries to generate")
    parser.add_argument("--n-rag", type=int, default=30,
                       help="Number of RAG questions to generate")
    parser.add_argument("--verify", type=str, default=None,
                       help="Enter verification mode for queries in this directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.verify:
        verify_queries(Path(args.verify))
        return
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("Run build_corpus.py first.")
        sys.exit(1)
    
    print("Sampling documents for query generation...")
    retrieval_docs, rag_docs = sample_documents_for_queries(
        corpus_path, args.n_retrieval, args.n_rag, args.seed
    )
    
    print(f"Selected {len(retrieval_docs)} docs for retrieval queries")
    print(f"Selected {len(rag_docs)} docs for RAG questions")
    
    client = get_client()
    
    print("\nGenerating retrieval queries...")
    queries = generate_retrieval_queries(retrieval_docs, client, args.model)
    
    print("\nGenerating RAG verification questions...")
    rag_questions = generate_rag_questions(rag_docs, client, args.model)
    
    save_queries(queries, rag_questions, Path(args.output))


if __name__ == "__main__":
    main()
