import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from src.ingest_pdf import load_and_chunk_pdf, embed_texts
from src.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult
from src.faiss_storage import FaissStorage
import uuid

# Create an Inngest client
inngest_client = inngest.Inngest(
    app_id="archival_rag_system",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)



# Create an Inngest function
@inngest_client.create_function(
    fn_id="RAG: ingest PDF",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="app/rag_ingest_pdf"),
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


# @inngest_client.create_function(
#     fn_id="RAG: Query PDF",
#     # Event that triggers this function
#     trigger=inngest.TriggerEvent(event="app/rag_query_pdf")
# )
# async def rag_query_pdf(ctx: inngest.Context) -> str:
    


app = FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])