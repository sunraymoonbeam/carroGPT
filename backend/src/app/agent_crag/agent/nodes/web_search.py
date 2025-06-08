"""Call Tavily via LangChain tool and convert results to Documents."""

import asyncio
from typing import Any, Dict, List, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from ....core.config import settings
from ..state import State

web_search_tool = TavilySearchResults(
    max_results=3, tavily_api_key=settings.TAVILY_API_KEY
)


async def search_web(state: State) -> Dict[str, Any]:
    """
    Use TavilySearchResults to fetch up-to-date web snippets.

    Returns:
        search_results : List[Document] with page_content and metadata {url, source}
        api_status     : "success", "partial_failure", or "failure"
        error_message  : Optional[str]
    """
    print("---WEB SEARCH---")
    query = state.get("rewritten_query") or state["question"]

    try:
        raw: Union[List[Dict[str, Any]], str] = await asyncio.to_thread(
            web_search_tool.run, query
        )

    except Exception as exc:
        print(f"Web search exception: {exc}")
        return {
            "search_results": [],
            "api_status": "failure",
            "error_message": str(exc),
        }

    documents: List[Document] = []
    if isinstance(raw, list):
        # Each item is a dict: {"content", "url", "title", ...}
        for item in raw:
            snippet = (item.get("content") or item.get("raw_content") or "").strip()
            if snippet:
                documents.append(
                    Document(
                        page_content=snippet,
                        metadata={"url": item.get("url"), "source": "Web"},
                    )
                )
    else:
        # raw is a str → treat as error message or empty
        if raw and raw.strip():
            print(f"Tavily returned error string: {raw.strip()}")
            return {
                "search_results": [],
                "api_status": "failure",
                "error_message": raw.strip(),
            }

    print(f"Tavily returned {len(documents)} snippet(s)")
    return {
        "search_results": documents,
        "api_status": "success" if documents else "partial_failure",
        "error_message": None,
    }
