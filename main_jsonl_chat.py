import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from src.ingest_corpus_jsonl import load_and_chunk_jsonl, embed_texts
from src.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAQQueryResult, RAGEvalResult
from src.faiss_storage import FaissStorage
from src.generation import RAGGenerator
import uuid
import os

from dotenv import load_dotenv


load_dotenv()





# Create an Inngest client
inngest_client = inngest.Inngest(
    app_id="archival_rag_system",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)



# Create an Inngest function
@inngest_client.create_function(
    fn_id="RAG: Ingest NLS Corpus",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="app/rag_ingest_nls_corpus"),
)
async def rag_ingest_nls_corpus(ctx: inngest.Context) -> str:
    # Step 1 & 2: load the jsonl file, chunk the text, and store it into the vec store
    def _process_corpus(ctx: inngest.Context) -> RAGUpsertResult:
        jsonl_path = ctx.event.data["jsonl_path"]
        source_id = ctx.event.data.get("source_id", jsonl_path)

        # Stream records, clean, chunk per record (keep metadata!)
        chunks = load_and_chunk_jsonl(jsonl_path)
        
        texts = [c["text"] for c in chunks]
        vecs = embed_texts(texts)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        
        payloads = []
        for i, c in enumerate(chunks):
            payload = {"source": source_id, "text": c["text"]}
            payload.update(c.get("metadata", {}))
            payloads.append(payload)
            
        FaissStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))
        
    upsert_result = await ctx.step.run("process-corpus", lambda: _process_corpus(ctx), output_type=RAGUpsertResult)
    return upsert_result.model_dump_json()
    # return chunk_and_src


@inngest_client.create_function(
    fn_id="RAG: Query NLS Corpus",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="app/rag_query_nls_corpus")
)
async def rag_query_nls_corpus(ctx: inngest.Context) -> str:
    from src.retrievers import DenseRetriever, SparseRetriever, HybridRetriever
    from src.custom_types import RAGSearchResult

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    retrieval_mode = ctx.event.data.get("retrieval_mode", "hybrid")  # "dense" | "sparse" | "hybrid"

    store = FaissStorage()

    # ── Step 1: Dense retrieval (FAISS vector search) ──────────────────────────
    def _dense_search() -> dict:
        return DenseRetriever(store).search(question, top_k=top_k)

    dense_result = await ctx.step.run("retrieval-dense", _dense_search)

    # ── Step 2: Sparse retrieval (BM25 keyword search) ─────────────────────────
    def _sparse_search() -> dict:
        return SparseRetriever(store).search(question, top_k=top_k)

    sparse_result = await ctx.step.run("retrieval-sparse", _sparse_search)

    # ── Step 3: Hybrid merge (Reciprocal Rank Fusion) ──────────────────────────
    def _hybrid_merge() -> RAGSearchResult:
        if retrieval_mode == "dense":
            found = dense_result
        elif retrieval_mode == "sparse":
            found = sparse_result
        else:
            found = HybridRetriever(store).search(question, top_k=top_k)
        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
            metadatas=found["metadatas"],
        )

    found = await ctx.step.run("retrieval-hybrid-merge", _hybrid_merge, output_type=RAGSearchResult)


    # ── Step 4: LLM answer generation ──────────────────────────────────────────
    generator = RAGGenerator()

    gen_result = await ctx.step.run(
        "llm-answer",
        lambda: generator.generate(question, found.contexts, found.sources),
    )

    return RAQQueryResult(
        answer=gen_result.get("answer", ""),
        sources=found.sources,
        num_contexts=len(found.contexts),
        confidence=gen_result.get("confidence", 0),
        reasoning=gen_result.get("reasoning", ""),
        ocr_issues_noted=gen_result.get("ocr_issues_noted", "")
    ).model_dump_json()


@inngest_client.create_function(
    fn_id="RAG: Evaluate NLS Corpus",
    trigger=inngest.TriggerEvent(event="app/rag_evaluate_nls_corpus")
)
async def rag_evaluate_nls_corpus(ctx: inngest.Context) -> str:
    from evaluation.evaluate import RAGEvaluator, QAPair
    from src.custom_types import RAGEvalResult

    eval_data = ctx.event.data["eval_data"]
    top_k = int(ctx.event.data.get("top_k", 5))
    run_generation = ctx.event.data.get("run_generation", False)

    dataset = []
    for item in eval_data:
        dataset.append(QAPair(
            query=item["query"],
            ground_truth_answer=item.get("ground_truth_answer", ""),
            ground_truth_contexts=item.get("ground_truth_contexts", []),
            ground_truth_doc_ids=item.get("ground_truth_doc_ids", [])
        ))

    def _evaluate() -> dict:
        evaluator = RAGEvaluator()
        retriever_results = evaluator.evaluate_retrievers(dataset, top_k=top_k)
        
        generator_results = {}
        if run_generation:
            generator_results = evaluator.evaluate_generation_with_ragas(dataset, top_k=top_k)
            
        return {
            "retriever_metrics": retriever_results,
            "generator_metrics": generator_results
        }
    
    eval_result = await ctx.step.run("evaluate-corpus", _evaluate)
    
    return RAGEvalResult(
        retriever_metrics=eval_result["retriever_metrics"],
        generator_metrics=eval_result["generator_metrics"]
    ).model_dump_json()

app = FastAPI(port=8000)

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest_nls_corpus, rag_query_nls_corpus, rag_evaluate_nls_corpus])