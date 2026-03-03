import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from src.ingest_pdf import load_and_chunk_pdf, embed_texts
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
    fn_id="rag ingest jsonl",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="app/rag ingest jsonl"),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> str:
    # Step 1: load the pdf file from the path and chunk it
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
        
    
    # Step 2: chunk the text and store it into the vec store
    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        FaissStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))
        
    chunk_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    upsert_result = await ctx.step.run("upsert", lambda: _upsert(chunk_and_src), output_type=RAGUpsertResult)
    return upsert_result.model_dump_json()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="app/rag_query_pdf")
)
async def rag_query_pdf(ctx: inngest.Context) -> str:
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]  # 1. convert the question into vector
        store = FaissStorage()
        found = store.search(query_vec, top_k)  # 2. search the top_k most similar vectors

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
            metadatas=found["metadatas"]
        )
        
    question = ctx.event.data["question"]   # get the question from inngest frontend
    top_k = int(ctx.event.data.get("top_k", 5))  # get the top_k from inngest frontend

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # Build a single context string from retrieved chunks
    context_text = "\n\n".join(f"- {c}" for c in found.contexts)

    # Define the LLM chain
    llm = get_client()
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context provided.
        If the answer is not in the context, say "I don't know".
        
        Question: {question}
        Context: {context}
        
        Answer:
        """
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Generate and track this as an Inngest step
    answer = await ctx.step.run(
        "llm-answer",
        lambda: chain.invoke({"context": context_text, "question": question}),
    )

    return RAQQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    ).model_dump_json()

app = FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf])