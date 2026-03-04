import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from src.ingest_corpus_jsonl import load_and_chunk_jsonl, embed_texts
from src.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAQQueryResult
from src.faiss_storage import FaissStorage
import uuid
import os

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



# define the model 
def get_client() -> ChatOpenAI:
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
    context_text = "\n\n".join(f"- {c}" for c in found.contexts)

    llm = get_client()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context provided.
        If the answer is not in the context, say "I don't know".
        
        Question: {question}
        Context: {context}
        
        Answer:
        Sources: {sources}
        """
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    answer = await ctx.step.run(
        "llm-answer",
        lambda: chain.invoke({"context": context_text, "question": question}),
    )

    return RAQQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    ).model_dump_json()

app = FastAPI(port=8000)

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest_nls_corpus, rag_query_nls_corpus])