import logging
from typing import List, Optional, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from ..core.config import settings
from ..db import service
from ..db.exceptions import (
    CollectionNotFoundError,
    DocumentProcessingError,
)
from ..db.schemas import (
    CollectionDetailResponse,
    DeleteCollectionResponse,
    ListCollectionsResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .tasks import process_documents

router = APIRouter(prefix="/collections", tags=["collections"])
logger = logging.getLogger(__name__)


async def get_qdrant_client() -> AsyncQdrantClient:
    """
    Dependency that returns an AsyncQdrantClient instance.
    """
    return AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_KEY)


# --------------------------------------------------------------------
# GET COLLECTIONS
# --------------------------------------------------------------------
@router.get(
    "/",
    response_model=ListCollectionsResponse,
    status_code=status.HTTP_200_OK,
)
async def list_collections(
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
) -> ListCollectionsResponse:
    """
    List all existing Qdrant collections.

    Returns:
        ListCollectionsResponse: Contains a list of collection names.
    """
    try:
        names = await service.list_collections(qdrant_client=qdrant)
        logger.info(f"Retreived {len(names)} collections")
        return ListCollectionsResponse(collections=names)

    except Exception as exc:
        logger.error("Error listing collections: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {exc}",
        )


@router.get(
    "/{collection_name}",
    response_model=CollectionDetailResponse,
    status_code=status.HTTP_200_OK,
)
async def get_collection_detail(
    collection_name: str = Path(
        ...,
        description="Name of the collection to retrieve details for.",
        min_length=1,
    ),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
) -> CollectionDetailResponse:
    """
    Get detailed information about a specific collection.

    Args:
        collection_name (str): The name of the collection.
        qdrant (AsyncQdrantClient): Injected Qdrant client.

    Returns:
        CollectionDetailResponse: Detailed metadata and a few sample documents.
    """
    try:
        details = await service.get_collection_details(
            qdrant_client=qdrant, collection_name=collection_name
        )
        logger.info(f"Retrieved details for collection '{collection_name}'")
        return CollectionDetailResponse(**details)

    except CollectionNotFoundError as exc:
        logger.warning("Collection not found: %s", collection_name)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    except Exception as exc:
        logger.error(
            "Unexpected error getting details for '%s': %s",
            collection_name,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve collection details: {exc}",
        )


# --------------------------------------------------------------------
# UPLOAD / CREATE COLLECTION
# --------------------------------------------------------------------
@router.post(
    "/{collection_name}/documents",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    collection_name: str,
    files: Optional[List[UploadFile]] = File(None),
    urls: Optional[List[str]] = Form(None),
):
    """
    Upload (upsert) new documents into the given collection. The collection
    will be auto-created if it does not already exist.
    You must supply at least one file or one URL.

    Args:
        background_tasks (BackgroundTasks): FastAPI background task manager.
        collection_name (str): Target collection name.
        files (Optional[List[UploadFile]]): List of `.pdf`/`.docx` uploads (optional).
        urls (Optional[List[str]]): List of web URLs to scrape & process (optional).

    Returns:
        JSONResponse: Status message indicating acceptance of the upload.
    """
    # eagerly read all files while handles are open
    # (Ren Hwa: if not eagerly read, the file handles will be closed by the time the background task runs)
    # this leads to an I.O error when trying to read the file in the background task
    buffered: List[Tuple[str, bytes]] = []
    for f in files or []:
        data = await f.read()
        buffered.append((f.filename, data))  # need the filename for source metadata

    # schedule background task for processing of the documents with [(filename, raw_bytes)] + URLs
    background_tasks.add_task(
        process_documents,
        collection_name,
        buffered,
        urls or [],
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": (
                f"Accepted {len(buffered)} files "
                f"and {len(urls or [])} URLs for background processing."
            )
        },
    )


# @router.post(
#     "/{collection_name}/documents",
#     response_model=UploadDocumentsResponse,
#     status_code=status.HTTP_201_CREATED,
# )
# async def upload_documents(
#     collection_name: str = Path(
#         ...,
#         description="Name of the collection to upload documents into (auto-created if needed).",
#         min_length=1,
#     ),
#     files: Optional[List[UploadFile]] = File(
#         None,
#         description="Zero or more PDF/DOCX files to upload (multipart/form-data).",
#     ),
#     urls: Optional[List[HttpUrl]] = Form(
#         None,
#         description="Zero or more URLs to fetch and process into text chunks.",
#     ),
#     qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
# ) -> UploadDocumentsResponse:
#     """
#     Upload (upsert) new documents into the given collection. The collection
#     will be auto-created if it does not already exist.

#     You must supply at least one file or one URL.

#     Args:
#         collection_name (str): Target collection name.
#         files (List[UploadFile]): List of `.pdf`/`.docx` uploads (optional).
#         urls (List[HttpUrl]): List of web URLs to scrape & process (optional).
#         qdrant (AsyncQdrantClient): Injected Qdrant client.

#     Returns:
#         UploadDocumentsResponse: Number of items uploaded, their IDs, and a message.
#     """
#     if (not files or len(files) == 0) and (not urls or len(urls) == 0):
#         logger.warning("Upload attempt without files or URLs for '%s'", collection_name)
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Must supply at least one PDF/DOCX file or one URL.",
#         )

#     try:
#         new_ids = await service.upload_documents_to_collection(
#             qdrant_client=qdrant,
#             collection_name=collection_name,
#             files=files or [],
#             urls=[str(u) for u in (urls or [])],
#         )
#         logger.info(
#             "Uploaded %d document chunks into '%s'", len(new_ids), collection_name
#         )
#         return UploadDocumentsResponse(
#             uploaded_ids=new_ids,
#             uploaded_count=len(new_ids),
#             message=f"Uploaded {len(new_ids)} document chunks into '{collection_name}'.",
#         )

#     except DocumentProcessingError as exc:
#         logger.error(
#             "Document processing error for '%s': %s",
#             collection_name,
#             exc,
#             exc_info=True,
#         )
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
#         )

#     except Exception as exc:
#         logger.error(
#             "Unexpected error uploading documents into '%s': %s",
#             collection_name,
#             exc,
#             exc_info=True,
#         )
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Unexpected error uploading documents: {exc}",
#         )


# --------------------------------------------------------------------
# DELETE COLLECTION
# --------------------------------------------------------------------
@router.delete(
    "/{collection_name}",
    response_model=DeleteCollectionResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_collection(
    collection_name: str = Path(
        ...,
        description="Name of the collection to delete.",
        min_length=1,
    ),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
) -> DeleteCollectionResponse:
    """
    Delete an existing collection and all its contents.

    Args:
        collection_name (str): The name of the collection to delete.
        qdrant (AsyncQdrantClient): Injected Qdrant client.

    Returns:
        DeleteCollectionResponse: {name, deleted, message}
    """
    try:
        deleted = await service.delete_collection(
            qdrant_client=qdrant, collection_name=collection_name
        )
        logger.info("Deleted collection '%s'", collection_name)
        return DeleteCollectionResponse(
            name=collection_name,
            deleted=deleted,
            message=f"Collection '{collection_name}' deleted successfully.",
        )

    except CollectionNotFoundError as exc:
        logger.warning("Collection not found (delete request): %s", collection_name)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    except Exception as exc:
        logger.error(
            "Error deleting collection '%s': %s", collection_name, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection '{collection_name}': {exc}",
        )


# --------------------------------------------------------------------
# SEARCH COLLECTION
# --------------------------------------------------------------------
@router.post(
    "/{collection_name}/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
)
async def search_collection(
    collection_name: str = Path(
        ...,
        description="Name of the collection to search against.",
        min_length=1,
    ),
    body: SearchRequest = Body(...),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    """
    Run a semantic search in the given collection.

    Args:
        collection_name (str): Target collection name.
        body (SearchRequest): {query: str, top_k: int}
        qdrant (AsyncQdrantClient): Injected Qdrant client.

    Returns:
        SearchResponse: A list of hits with id, score, page_content, and source.
    """
    try:
        hits = await service.search_in_collection(
            qdrant_client=qdrant,
            collection_name=collection_name,
            query_text=body.query,
            top_k=body.top_k,
        )

    except CollectionNotFoundError as exc:
        logger.warning("Search on non-existent collection '%s'", collection_name)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    except DocumentProcessingError as exc:
        logger.error(
            "Search processing error for '%s': %s", collection_name, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    except Exception as exc:
        logger.error(
            "Unexpected search error for '%s': %s", collection_name, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed for '{collection_name}': {exc}",
        )

    response_items: List[SearchHit] = []
    for hit in hits:
        response_items.append(
            SearchHit(
                id=str(hit.id),
                score=hit.score,
                text=hit.document,
                source=hit.metadata.get("source", "unknown"),
                file_type=hit.metadata.get("file_type", "unknown"),
                page_number=hit.metadata.get("page_number"),
                upload_date=hit.metadata.get("upload_date"),
            )
        )

    logger.info(
        "Returned %d hits for query '%s' in '%s'",
        len(response_items),
        body.query,
        collection_name,
    )
    return SearchResponse(results=response_items)
