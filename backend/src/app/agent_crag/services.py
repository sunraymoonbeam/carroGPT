"""
Service layer for Carro ReAct Agent operations.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from .schemas import (
    ChatRequest,
    ClassifierStatus,
    Document,
    EvaluatorStatus,
    GeneratorResponse,
    RetrieverStatus,
    RewriterStatus,
    Sources,
    WebSearchStatus,
)

logger = logging.getLogger(__name__)


async def run_stream_chat(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> StreamingResponse:
    """
    Stream state updates as SSE with conversation memory.

    Args:
        thread_id (str): Conversation thread identifier.
        request (Request): FastAPI request to access the agent graph.
        chat_request (ChatRequest): Incoming chat payload.

    Returns:
        StreamingResponse: SSE stream of node status and final response.
    """
    # Obtain the graph from the request state
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize the state with the user's message and collection name
    # Messages are automatically handled by the graph's add_messages decorator
    initial_state: Dict[str, Any] = {
        "messages": [HumanMessage(content=chat_request.message)],
        "collection_name": chat_request.collection_name or "carro_test",
        "question": chat_request.message.strip(),
        "category": "",
        "needs_web_search": False,
        "documents": [],
        "filtered_documents": [],
        "doc_relevance": 0.0,
        "rewritten_query": "",
        "search_results": [],
        "answer": "",
        "api_status": "pending",
        "error_message": None,
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Generator function to yield state updates as SSE.
        This function processes each step of the graph and yields status updates.
        """
        # Stream through the graph's async steps
        async for step in graph.astream(initial_state, config=config):
            for node, output in step.items():
                print(f"### {node.upper()} OUTPUT ###\n{output}")

                status = output.get("api_status")
                err = output.get("error_message")

                # 1. Classifier
                if node == "classifier":
                    msg = ClassifierStatus(
                        node=node,
                        category=output.get("category", ""),
                        needs_web_search=output.get("needs_web_search", False),
                        api_status=status,
                        error_message=err,
                    )
                    yield f"data: {msg.json()}\n\n"

                # 2. Retriever
                elif node == "retriever":
                    # Build Document list
                    documents: List[Document] = []

                    # Retrieve documents if available
                    for doc in output.get("documents", []) or []:
                        meta = doc.metadata
                        documents.append(
                            Document(
                                text=doc.page_content,
                                id=meta.get("id", "Unknown ID"),
                                score=float(meta.get("score", 0.0)),
                                source=meta.get("source", "Unknown Source"),
                                file_type=meta.get("file_type", "Unknown File Type"),
                                page_number=meta.get("page_number"),
                            )
                        )
                    msg = RetrieverStatus(
                        node=node,
                        documents=documents,
                        document_count=len(documents),
                        api_status=status,
                        error_message=err,
                    )
                    yield f"data: {msg.json()}\n\n"

                # 3. Evaluator
                elif node == "evaluator":
                    filtered = output.get("filtered_documents", []) or []
                    ratio = output.get("doc_relevance", 0.0)
                    msg = EvaluatorStatus(
                        node=node,
                        filtered_count=len(filtered),
                        doc_relevance=ratio,
                        api_status=status,
                        error_message=err,
                    )
                    yield f"data: {msg.json()}\n\n"

                # 4. Query Rewriter
                elif node == "query_rewriter":
                    msg = RewriterStatus(
                        node=node,
                        original_question=output.get("question", ""),
                        rewritten_query=output.get("rewritten_query", ""),
                        api_status=status,
                        error_message=err,
                    )
                    yield f"data: {msg.json()}\n\n"

                # 5. Web Search
                elif node == "web_search":
                    urls = [
                        doc.metadata.get("url", "")
                        for doc in output.get("search_results", []) or []
                    ]
                    msg = WebSearchStatus(
                        node=node,
                        urls=urls,
                        web_result_count=len(urls),
                        api_status=status,
                        error_message=err,
                    )
                    yield f"data: {msg.json()}\n\n"

                # 6. Generator (final response)
                elif node == "generator":
                    # Build Document list from filtered_documents for citing Relevant Sources
                    documents: List[Document] = []
                    for doc in output.get("filtered_documents", []) or []:
                        meta = doc.metadata
                        documents.append(
                            Document(
                                text=doc.page_content,
                                id=meta.get("id", "Unknown ID"),
                                score=float(meta.get("score", 0.0)),
                                source=meta.get("source", "Unknown"),
                                file_type=meta.get("file_type", "Unknown"),
                                page_number=meta.get("page_number"),
                            )
                        )

                    # Build URL list from search_results for citing Relevant Sources
                    urls = [
                        doc.metadata.get("url", "Unknown URL")
                        for doc in output.get("search_results", []) or []
                    ]
                    # Deduplicate URLs (this maintains order)
                    urls = list(dict.fromkeys(url for url in urls if url))

                    # Create Sources object to hold both documents and URLs
                    sources = Sources(
                        documents=documents,
                        urls=urls,
                    )
                    msg = GeneratorResponse(
                        response=output.get("answer", ""),
                        sources=sources,
                        api_status=status,
                        error_message=err,
                    )
                    yield f"data: {msg.json()}\n\n"

        # Signal completion
        yield 'data: {"type":"complete"}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def run_chat(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> GeneratorResponse:
    """
    Handle non-streaming chat by returning a complete response object.

    Args:
        thread_id (str): Conversation thread identifier.
        request (Request): FastAPI request to access the agent graph.
        chat_request (ChatRequest): Incoming chat payload.

    Returns:
        GeneratorResponse: Final chat response with sources.
    """
    # Validate incoming message
    if not chat_request.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    # Obtain the graph from the request state
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize the state for processing
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content=chat_request.message)],
        "collection_name": chat_request.collection_name or "carro_test",
        "question": chat_request.message.strip(),
        "category": "",
        "needs_web_search": False,
        "documents": [],
        "filtered_documents": [],
        "doc_relevance": 0.0,
        "rewritten_query": "",
        "search_results": [],
        "answer": "",
        "api_status": "pending",
        "error_message": None,
    }

    final_output: Optional[Dict[str, Any]] = None
    async for step in graph.astream(state, config=config):
        if "generator" in step:
            final_output = step["generator"]
            break

    if final_output is None:
        logger.error("No response generated in simple_chat_service")
        raise HTTPException(500, "No response generated")

    # Build Document list from filtered_documents for citing Relevant Sources
    documents: List[Document] = []
    for doc in final_output.get("filtered_documents", []) or []:
        meta = doc.metadata
        documents.append(
            Document(
                text=doc.page_content,
                id=meta.get("id", "Unknown ID"),
                score=float(meta.get("score", 0.0)),
                source=meta.get("source", "Unknown Source"),
                file_type=meta.get("file_type", "Unknown File Type"),
                page_number=meta.get("page_number"),
            )
        )

    # Build URL list from search_results for citing Relevant Sources
    urls = [
        doc.metadata.get("url", "Unknown URL")
        for doc in final_output.get("search_results", []) or []
    ]
    # Deduplicate URLs (this maintains order)
    urls = list(dict.fromkeys(url for url in urls if url))

    # Create Sources object to hold both documents and URLs
    sources = Sources(documents=documents, urls=urls)

    return GeneratorResponse(
        response=final_output.get("answer", ""),
        sources=sources,
        api_status=final_output.get("api_status"),
        error_message=final_output.get("error_message"),
    )
