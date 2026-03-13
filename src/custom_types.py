import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[dict]
    source_id: str = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]


class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
    confidence: int = 0
    reasoning: str = ""
    ocr_issues_noted: str = ""


class RAGEvalResult(pydantic.BaseModel):
    retriever_metrics: dict
    generator_metrics: dict = {}