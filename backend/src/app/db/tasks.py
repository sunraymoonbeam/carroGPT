# tasks.py
import uuid
from functools import lru_cache
from typing import List, Tuple

from qdrant_client import QdrantClient

from ..core.config import settings
from .utils import (
    process_docx,
    process_pdf_llamaparse,
    process_url,
)


# lru_cache ensures that we only create one instance of QdrantClient,
# which is reused for all subsequent calls to get_qdrant_client_sync.
@lru_cache(maxsize=1)
def get_qdrant_client_sync() -> QdrantClient:
    """
    Returns a cached QdrantClient instance for synchronous upserts.
    """
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_KEY)


# Ren Hwa: we shifted to a fully synchronous function for document processing
# Context: uploading of documents and processing was taking really too long
# Should just accept the files and return the client a 202 Accepted response
# FastAPI background tasks will handle the processing (which is just a wrapper around Starlette's BackgroundTasks)
# If async: the function would be called in the same event loop as the FastAPI request,
# which would block the request until processing is done, defeating the purpose of async.
# Sync however would be put in a seperate thread, allowing the FastAPI main app to continue processing other requests.


def process_documents(
    collection_name: str,
    files: List[Tuple[str, bytes]],
    urls: List[str],
) -> None:
    """
    Background task: write each (filename, raw_bytes) to temp file,parse into chunks, and upsert all to Qdrant.

    Args:
        collection_name (str): Name of the Qdrant collection to upsert into.
        files (List[Tuple[str, bytes]]): List of tuples containing filename and raw bytes.
        urls (List[str]): List of URLs to process.
    """
    texts, metadata, ids = [], [], []

    # Process in-memory files
    for filename, raw_bytes in files:
        lowered = filename.lower()

        if lowered.endswith(".pdf"):
            chunks = process_pdf_llamaparse(filename, raw_bytes)

        elif lowered.endswith(".docx"):
            chunks = process_docx(filename, raw_bytes)

        else:
            print(f"Unsupported file type for {filename}. Skipping.")
            continue

        # Collect chunk outputs
        for text, payload in chunks:
            texts.append(text)
            metadata.append(payload)
            ids.append(str(uuid.uuid4()))

    # Process URLs
    for url in urls:
        for text, payload in process_url(url):
            texts.append(text)
            metadata.append(payload)
            ids.append(str(uuid.uuid4()))

    # Upsert all text chunks along with their ids and metadata to Qdrant
    print(
        f"### Upserting {len(texts)} chunks into Qdrant collection '{collection_name} ### '"
    )
    if texts:
        qdrant_client = get_qdrant_client_sync()
        qdrant_client.add(
            collection_name=collection_name,
            documents=texts,
            metadata=metadata,
            ids=ids,
        )
