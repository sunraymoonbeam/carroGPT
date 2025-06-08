import json
import logging

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from ...core.config import get_settings
from ...db.router import get_qdrant_client
from ...db.service import search_in_collection

settings = get_settings()
logger = logging.getLogger(__name__)


@tool
async def retrieve_carro_documents(
    question: str, collection_name: str = "carro_test"
) -> str:
    """
    Search Carro's FAQ database for relevant information.

    Args:
        question (str): The user's question to search for.
        collection_name (str): The name of the Qdrant collection to search in.

    Returns:
        str: A JSON string containing the retrieval results, each containing fields: id, text, score, source, file_type, page_number.
    """
    try:
        print(f"---RETRIEVING FROM COLLECTION: {collection_name}---")
        client = await get_qdrant_client()
        hits = await search_in_collection(
            qdrant_client=client,
            collection_name=collection_name,
            query_text=question,
            top_k=3,
        )

        docs = []
        for hit in hits:
            docs.append(
                {
                    "id": hit.id,
                    "text": hit.document,
                    "score": hit.score,
                    "source": hit.metadata.get("source", "Unknown"),
                    "file_type": hit.metadata.get("file_type"),
                    "page_number": hit.metadata.get("page_number"),
                }
            )
        return json.dumps(docs)

    except Exception as exc:
        logger.error("Error retrieving documents: %s", exc, exc_info=True)
        return json.dumps([])


@tool
def search_web(query: str) -> str:
    """
    Search the web for current automotive information.

    Args:
        query (str): The user's query to search for.

    Returns:
        str: A JSON string containing the search results, each with title, content, and URL.
    """
    try:
        tavily = TavilySearchResults(
            max_results=2,
            tavily_api_key=settings.TAVILY_API_KEY,
        )
        results = tavily.invoke(query)

        # Convert TavilySearchResults to a list of dictionaries
        hits = []
        for r in results or []:
            hits.append(
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                }
            )
        return json.dumps(hits)
    except Exception as exc:
        logger.error("Error in web search: %s", exc, exc_info=True)
        return json.dumps([])


tools = [retrieve_carro_documents, search_web]
