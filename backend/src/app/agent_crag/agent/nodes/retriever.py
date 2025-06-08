"""Pull FAQ snippets from Qdrant with dynamic collection selection."""

import logging
from typing import Any, Dict

from langchain_core.documents import Document

from ....db.router import get_qdrant_client
from ....db.service import search_in_collection
from ..state import State

logger = logging.getLogger(__name__)


async def retrieve_documents(state: State) -> Dict[str, Any]:
    """
    Perform a vector/text search in Qdrant for the user's question.
    Uses the collection_name from state instead of hardcoded value.

    Returns:
        documents       : List[Document] from the specified collection
        api_status      : "success", "partial_failure", or "failure"
        error_message   : Optional[str]
    """
    print("---RETRIEVE DOCUMENTS---")

    try:
        # Get collection name from state
        collection_name = state.get("collection_name")
        print(f"Searching in collection: {collection_name}")

        # Query Qdrant for relevant documents
        client = await get_qdrant_client()
        hits = await search_in_collection(
            qdrant_client=client,
            collection_name=collection_name,
            query_text=state["question"],
            top_k=5,
        )

        # Build Document objects from the hits
        documents = []
        for hit in hits:
            documents.append(
                Document(
                    page_content=hit.document,
                    metadata={
                        "id": hit.id,
                        "score": hit.score,
                        "source": hit.metadata.get("source", "unknown"),
                        "file_type": hit.metadata.get("file_type", "text"),
                        "page_number": hit.metadata.get("page_number"),
                    },
                )
            )
        print(f"Retrieved {len(documents)} snippet(s) from {collection_name}")
        return {
            "documents": documents,
            "api_status": "success" if documents else "partial_failure",
            "error_message": None,
        }

    except Exception as exc:
        logger.error("Error retrieving documents: %s", exc, exc_info=True)
        print(f"Retrieval error: {exc}")
        return {"documents": [], "api_status": "failure", "error_message": str(exc)}
