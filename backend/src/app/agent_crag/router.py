"""
Fixed FastAPI router that properly preserves conversation memory.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .schemas import ChatRequest, GeneratorResponse
from .services import run_chat, run_stream_chat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/carro-agent", tags=["carro-agent"])


@router.post("/chat/{thread_id}")
async def stream_chat(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> StreamingResponse:
    """
    Stream state updates as Server-Sent Events (SSE), with proper conversation memory.
    This endpoint supports dynamic collection selection via the `collection_name` field.

    Args:
        thread_id (str): Unique identifier for the chat thread.
        request (Request): FastAPI request object.
        chat_request (ChatRequest): Request body containing the user's message and optional collection name.

    Returns:
        StreamingResponse: A streaming response that yields ChatResponse objects.
    """
    # Validate incoming message
    if not chat_request.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    try:
        return await run_stream_chat(
            thread_id=thread_id, request=request, chat_request=chat_request
        )
    except Exception as e:
        logger.error(f"Error in stream_chat: {e}")
        raise HTTPException(500, f"Internal Server Error, error {e}")


@router.post("/chat/simple/{thread_id}")
async def chat_simple(
    thread_id: str,
    request: Request,
    chat_request: ChatRequest,
) -> GeneratorResponse:
    """
    Non-streaming endpoint returning a single ChatResponse (Final Node).

    Args:
        thread_id (str): Unique identifier for the chat thread.
        request (Request): FastAPI request object.
        chat_request (ChatRequest): Request body containing the user's message and optional collection name.

    Returns:
        GeneratorResponse: A single response containing the bot's answer and conversation memory.
    """
    # Validate incoming message
    if not chat_request.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    try:
        return await run_chat(
            thread_id=thread_id, request=request, chat_request=chat_request
        )
    except Exception as e:
        logger.error(f"Error in chat_simple: {e}")
        raise HTTPException(500, f"Internal Server Error, error {e}")
