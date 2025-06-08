"""
FastAPI router for Carro ReAct Agent.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .schemas import ChatRequest, ChatResponse
from .service import run_chat, run_stream_chat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/carro-react-agent", tags=["carro-react-agent"])


@router.post("/chat/{thread_id}")
async def stream_chat(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> StreamingResponse:
    """
    Stream ReAct agent responses as Server-Sent Events.
    Shows agent thinking, tool calls, and results in real-time.

    Args:
        thread_id (str): Unique identifier for the conversation thread.
        request (Request): FastAPI request object.
        chat_request (ChatRequest): Request body containing the user's message.

    Returns:
        StreamingResponse: Server-Sent Events stream of the chat response.

    Raises:
        HTTPException: If the message is empty or if an internal error occurs.
    """
    # Validate incoming message
    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        return await run_stream_chat(
            thread_id=thread_id, request=request, chat_request=chat_request
        )
    except Exception as e:
        logger.error(f"Error in stream_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/chat/simple/{thread_id}")
async def chat_simple(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> ChatResponse:
    """
    Simple non-streaming chat endpoint.
    Returns the final response after all agent steps complete.

    Args:
        thread_id (str): Unique identifier for the conversation thread.
        request (Request): FastAPI request object.
        chat_request (ChatRequest): Request body containing the user's message.

    Returns:
        ChatResponse: Final response from the agent after processing the request.

    Raises:
        HTTPException: If the message is empty or if an internal error occurs.
    """
    # Validate incoming message
    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        return await run_chat(
            thread_id=thread_id, request=request, chat_request=chat_request
        )
    except Exception as e:
        logger.error(f"Error in simple chat: {e}", exc_info=True)

        # Fall back to generic error response
        return ChatResponse(
            response=(
                "I'm sorry, but I'm experiencing technical difficulties. "
                "Please try again later or contact Carro support."
            ),
            conversation_id=thread_id,
            tools_used=[],
        )
