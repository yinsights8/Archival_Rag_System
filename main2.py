import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api



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
async def rag_ingest_jsonl(ctx: inngest.Context) -> str:
    ctx.logger.info("hello world !")
    return {"message":"hello world !"}




app = FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [welcome_fun])