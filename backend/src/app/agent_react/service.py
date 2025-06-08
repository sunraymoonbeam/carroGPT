"""
Service layer for Carro ReAct Agent: streaming and simple endpoints.
Includes tool usage extraction.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import AgentStep, ChatRequest, ChatResponse, StreamingStatus

logger = logging.getLogger(__name__)


def extract_tools_used(events: List[Dict[str, Any]]) -> List[str]:
    """
    Extract list of tools used from graph events.

    Args:
        events (List[dict]): List of event dictionaries from the graph execution.

    Returns:
        List[str]: List of unique tool names used during the conversation.
    """
    tools_used = set()
    for event in events:
        for node_name, node_output in event.items():
            if node_name == "tools" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, "name") and message.name:
                        tools_used.add(message.name)
    return list(tools_used)


async def run_stream_chat(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> StreamingResponse:
    """
    Stream ReAct agent responses as SSE, showing tool calls and results.

    Args:
        thread_id (str): Conversation thread identifier.
        request (Request): FastAPI request to access react_graph.
        chat_request (ChatRequest): Incoming chat payload.

    Returns:
        StreamingResponse: SSE stream of AgentStep and StreamingStatus messages.
    """
    # Obtain the graph from the request state
    graph = request.app.state.react_graph
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        events_log: List[Dict[str, Any]] = []
        try:
            # Prepend system message enforcing collection usage
            initial_messages = [
                SystemMessage(
                    content=(
                        f"Current collection_name: `{chat_request.collection_name}`.\n\n"
                        "Whenever you call `retrieve_carro_documents`, pass the above collection_name field exactly."
                    )
                ),
                HumanMessage(content=chat_request.message),
            ]
            initial_state: Dict[str, Any] = {
                "messages": initial_messages,
                "collection_name": chat_request.collection_name,
            }

            # Stream graph execution
            async for event in graph.astream(initial_state, config=config):
                events_log.append(event)
                for node_name, node_output in event.items():
                    # Agent thinking or making tool calls
                    if node_name == "agent":
                        messages = node_output.get("messages", [])

                        if messages:
                            last = messages[-1]
                            # If tool calls present, emit tool_call step
                            if hasattr(last, "tool_calls") and last.tool_calls:
                                tool_names = [tc["name"] for tc in last.tool_calls]
                                step = AgentStep(
                                    step_type="tool_call",
                                    content=f"Using tools: {', '.join(tool_names)}",
                                    tool_calls=[
                                        {
                                            "name": tc["name"],
                                            "arguments": tc["args"],
                                            "id": tc["id"],
                                        }
                                        for tc in last.tool_calls
                                    ],
                                )
                                yield f"data: {step.json()}\n\n"
                            else:
                                # Agent's final reply
                                step = AgentStep(
                                    step_type="final_response",
                                    content=last.content,
                                )
                                yield f"data: {step.json()}\n\n"

                    # Tool execution results
                    elif node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "content"):
                                step = AgentStep(
                                    step_type="tool_result",
                                    content=msg.content,
                                    metadata={"tool_name": msg.name},
                                )
                                yield f"data: {step.json()}\n\n"

            # Completion signal with tools used
            status = StreamingStatus(
                type="complete",
                message="Conversation completed",
                data={"tools_used": extract_tools_used(events_log)},
            )
            yield f"data: {status.json()}\n\n"

        except Exception as e:
            logger.error(f"Error in event_generator: {e}", exc_info=True)
            error_status = StreamingStatus(
                type="error",
                message=f"An error occurred: {str(e)}",
            )
            yield f"data: {error_status.json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


async def run_chat(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> ChatResponse:
    """
    Execute the ReAct agent non-streaming, returning the final ChatResponse.

    Args:
        thread_id (str): Conversation thread identifier.
        request (Request): FastAPI request to access react_graph.
        chat_request (ChatRequest): Incoming chat payload.

    Returns:
        ChatResponse: Contains final response text, conversation_id, and tools_used.
    """
    # Obtain the graph from the request state
    graph = request.app.state.react_graph
    config = {"configurable": {"thread_id": thread_id}}

    events_log: List[Dict[str, Any]] = []

    # Build initial state
    # Prepend system message enforcing collection usage
    initial_messages = [
        SystemMessage(
            content=(
                f"Current collection_name: `{chat_request.collection_name}`.\n\n"
                "Whenever you call `retrieve_carro_documents`, pass the above collection_name field exactly."
            )
        ),
        HumanMessage(content=chat_request.message),
    ]
    initial_state: Dict[str, Any] = {
        "messages": initial_messages,
        "collection_name": chat_request.collection_name,
    }

    # Execute graph
    async for event in graph.astream(initial_state, config=config):
        events_log.append(event)
        # Capture final agent response
        if "agent" in event:
            messages = event["agent"].get("messages", [])
            if messages and not hasattr(messages[-1], "tool_calls"):
                final = messages[-1].content

    # Return ChatResponse
    if not messages:
        raise HTTPException(status_code=500, detail="No final response generated")

    return ChatResponse(
        response=final,
        conversation_id=thread_id,
        tools_used=extract_tools_used(events_log),
    )
