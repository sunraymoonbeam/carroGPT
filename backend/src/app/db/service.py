from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import CountResult as HttpCountResult
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import CollectionInfo

from ..db.exceptions import (
    CollectionNotFoundError,
)

# from ..db.utils import (
#     process_docx_file,
#     process_pdf_llamaparse_file # process_pdf_unstructured
#     process_url,
# )


async def list_collections(qdrant_client: AsyncQdrantClient) -> List[str]:
    """
    Retrieve a list of all existing Qdrant collection names.

    Args:
        qdrant_client (AsyncQdrantClient): Qdrant client instance.

    Returns:
        List[str]: A list of collection names.
    """
    response = await qdrant_client.get_collections()
    return [collection.name for collection in response.collections]


async def delete_collection(
    qdrant_client: AsyncQdrantClient, collection_name: str
) -> bool:
    """
    Delete a Qdrant collection and all its contents.

    Args:
        qdrant_client (AsyncQdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection to delete.

    Returns:
        bool: True if deletion succeeded.

    Raises:
        CollectionNotFoundError: If the collection does not exist.
    """
    exists = await qdrant_client.collection_exists(collection_name=collection_name)
    if not exists:
        raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

    await qdrant_client.delete_collection(collection_name=collection_name)
    return True


async def get_collection_details(
    qdrant_client: AsyncQdrantClient, collection_name: str
) -> Dict[str, Any]:
    """
    Retrieve detailed info for a single collection, handling both FastEmbed
    and manually created collections.

    Args:
        qdrant_client (AsyncQdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection.

    Returns:
        Dict[str, Any]: Metadata keys: name, vector_size, distance, num_vectors,
        collection_type, vector_fields, sample_documents, status, segments_count,
        vectors_count, indexed_vectors_count.

    Raises:
        CollectionNotFoundError: If the collection does not exist.
    """
    exists = await qdrant_client.collection_exists(collection_name=collection_name)
    if not exists:
        raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

    coll_info: CollectionInfo = await qdrant_client.get_collection(
        collection_name=collection_name
    )
    vectors_config = coll_info.config.params.vectors
    vector_info = _extract_vector_info(vectors_config)

    count_result: HttpCountResult = await qdrant_client.count(
        collection_name=collection_name, exact=True
    )
    num_points = count_result.count

    sample_docs = await _get_sample_documents(qdrant_client, collection_name, limit=3)

    return {
        "name": collection_name,
        "num_vectors": num_points,
        "vector_size": vector_info["vector_size"],
        "segments_count": coll_info.segments_count,
        "distance": vector_info["distance"],
        "collection_type": vector_info["collection_type"],
        "vector_fields": vector_info["vector_fields"],
        "status": str(coll_info.status),
        "vectors_count": coll_info.vectors_count,
        "indexed_vectors_count": coll_info.indexed_vectors_count,
        "sample_documents": sample_docs,
    }


def _extract_vector_info(vectors_config: Any) -> Dict[str, Any]:
    """
    Extract vector information from different collection types.

    FastEmbed collections have unnamed vector config (direct VectorParams).
    Manual collections have named vector fields (a dict of VectorParams).

    Args:
        vectors_config (Any): The `vectors` attribute from CollectionInfo.config.params.

    Returns:
        Dict[str, Any]: {vector_size, distance, collection_type, vector_fields}.
    """
    try:
        # FastEmbed (unnamed) format:
        if hasattr(vectors_config, "size") and hasattr(vectors_config, "distance"):
            return {
                "vector_size": vectors_config.size,
                "distance": _get_distance_metric(vectors_config.distance),
                "collection_type": "fastembed",
                "vector_fields": ["default"],
            }

        # Manual (named) format:
        elif isinstance(vectors_config, dict):
            if not vectors_config:
                raise ValueError("Empty vectors configuration")

            first_field_name = next(iter(vectors_config.keys()))
            first_field_config = vectors_config[first_field_name]
            return {
                "vector_size": first_field_config.size,
                "distance": _get_distance_metric(first_field_config.distance),
                "collection_type": "manual",
                "vector_fields": list(vectors_config.keys()),
            }

        # Unknown:
        else:
            return {
                "vector_size": None,
                "distance": None,
                "collection_type": "unknown",
                "vector_fields": [],
            }
    except Exception:
        return {
            "vector_size": None,
            "distance": None,
            "collection_type": "unknown",
            "vector_fields": [],
        }


def _get_distance_metric(distance_obj: Any) -> Optional[str]:
    """
    Extract distance metric from various distance object types.

    Args:
        distance_obj (Any): Could be an enum or string representation.

    Returns:
        Optional[str]: The distance metric as a string, or None on failure.
    """
    try:
        if hasattr(distance_obj, "value"):
            return str(distance_obj.value)
        return str(distance_obj)
    except Exception:
        return None


async def _get_sample_documents(
    qdrant_client: AsyncQdrantClient, collection_name: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get a few sample documents from the collection for preview.

    Args:
        qdrant_client (AsyncQdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection.
        limit (int): Number of points to fetch (default: 3).

    Returns:
        List[Dict[str, Any]]: Each dict has {id, payload, text_preview, source, chunk_index}.
    """

    # Qdrant scroll API to fetch sample documents, Returns all points in a page-by-page manner.
    # https://api.qdrant.tech/api-reference/points/scroll-points
    resp = await qdrant_client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points = resp[0]

    # Build sample documents from points
    sample_documents: List[Dict[str, Any]] = []
    for point in points:
        sample: Dict[str, Any] = {
            "id": str(point.id),
            "text": point.payload.get("text"),
            "source": point.payload.get("source", "Unknown"),
            "file_type": point.payload.get("file_type", "unknown"),
            "page_number": point.payload.get("page_number"),
            "upload_date": point.payload.get("upload_date"),
        }
        sample_documents.append(sample)

    return sample_documents


# Ren Hwa: we shifted to a fully synchronous function for document processing
# Context: uploading of documents and processing was taking really too long
# Should just accept the files and return the client a 202 Accepted response
# FastAPI background tasks will handle the processing (which is just a wrapper around Starlette's BackgroundTasks)
# If async: the function would be called in the same event loop as the FastAPI request,
# which would block the request until processing is done, defeating the purpose of async.
# Sync however would be put in a seperate thread, allowing the FastAPI main app to continue processing other requests.

# async def upload_documents_to_collection(
#     qdrant_client: AsyncQdrantClient,
#     collection_name: str,
#     files: List[UploadFile],
#     urls: List[str],
# ) -> List[str]:
#     """
#     Process PDF/DOCX UploadFiles and web URLs, split them into text chunks
#     (LangChain), and upsert into Qdrant using FastEmbed. This will auto-create
#     the collection if it does not exist.

#     Args:
#         qdrant_client (AsyncQdrantClient): Qdrant client instance.
#         collection_name (str): Target collection name.
#         files (List[UploadFile]): List of UploadFile objects (.pdf, .docx).
#         urls (List[str]): List of URL strings.

#     Returns:
#         List[str]: Newly generated Qdrant point IDs (UUID strings).

#     Raises:
#         DocumentProcessingError: On loader/upsert failure.
#     """
#     texts: List[str] = []
#     metadata: List[Dict[str, Any]] = []
#     ids: List[str] = []

#     # Process file uploads:
#     for upload in files:
#         filename = upload.filename
#         lowered = filename.lower()
#         try:
#             if lowered.endswith(".pdf"):
#                 processed = process_pdf_llamaparse_file(upload)
#             elif lowered.endswith(".docx"):
#                 processed = process_docx_file(upload)
#             else:
#                 raise DocumentProcessingError(f"Unsupported file type: '{filename}'")

#         except DocumentProcessingError:
#             raise

#         except Exception as exc:
#             raise DocumentProcessingError(f"Error processing '{filename}': {exc}")

#         for chunk_text, payload in processed:
#             texts.append(chunk_text)
#             metadata.append(payload)
#             ids.append(str(uuid.uuid4()))  # Generate new UUIDs for each chunk

#     # Process URLs:
#     for url in urls:
#         try:
#             processed = process_url(url)
#         except DocumentProcessingError:
#             raise
#         except Exception as exc:
#             raise DocumentProcessingError(f"Error processing URL '{url}': {exc}")

#         for chunk_text, payload in processed:
#             texts.append(chunk_text)
#             metadata.append(payload)
#             ids.append(str(uuid.uuid4()))

#     if not texts:
#         return []

#     try:
#         await qdrant_client.add(
#             collection_name=collection_name,
#             documents=texts,
#             metadata=metadata,
#             ids=ids,
#         )
#     except Exception as exc:
#         raise DocumentProcessingError(f"AsyncQdrantClient.add upsert error: {exc}")

#     return ids


async def search_in_collection(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    top_k: int = 5,
    qdrant_filter: Optional[dict] = None,  # Ren Hwa: to be implemented later
) -> List[ScoredPoint]:
    """
    Perform a semantic search by passing raw `query_text` to Qdrant’s FastEmbed API,
    returning the top_k most similar points (optionally filtered by payload).

    Args:
        qdrant_client (AsyncQdrantClient): Qdrant client instance.
        collection_name (str): Target collection name.
        query_text (str): Raw text to embed & search.
        top_k (int): Number of nearest neighbors to return.
        qdrant_filter (dict, optional): A Qdrant filter JSON.

    Returns:
        List[ScoredPoint]: Each has id, score, and payload.

    Raises:
        CollectionNotFoundError: If the collection does not exist.
    """
    exists = await qdrant_client.collection_exists(collection_name=collection_name)
    if not exists:
        raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

    results = await qdrant_client.query(
        collection_name=collection_name,
        query_text=query_text,
        limit=top_k,
        # query_filter=qdrant_filter,
    )
    return results
