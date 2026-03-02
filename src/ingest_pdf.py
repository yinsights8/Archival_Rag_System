from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


EMB_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMB_DIM = 384
CHUNK_SIZE=512
CHUNK_OVERLAP=200

embed_model = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# load the pdf file from the path and chunk it
def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    text = loader.load()
    text = [d.page_content for d in text if getattr(d, "page_content", None)]

    chunks = []
    for t in text:
        chunks.extend(splitter.split_text(t))
    
    return chunks


# embed the text using huggingface embedding model
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = embed_model.embed_documents(texts)
    return response