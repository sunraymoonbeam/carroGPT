"""Pydantic schemas for Carro ReAct Agent API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoints.

    Attributes:
        message (str): User's message/question.
        collection_name (Optional[str]): Collection to search in for FAQ retrieval.
    """

    message: str = Field(..., description="User's message/question")
    collection_name: Optional[str] = Field(
        default="carro_test", description="Collection to search in for FAQ retrieval"
    )


class ChatResponse(BaseModel):
    """Response model for simple chat endpoint.

    Attributes:
        response (str): Agent's final response.
        conversation_id (str): Unique identifier for the conversation thread.
        tools_used (List[str]): List of tools used in this conversation.
    """

    response: str = Field(..., description="Agent's final response")
    conversation_id: str = Field(..., description="Thread/conversation ID")
    tools_used: List[str] = Field(
        default=[], description="List of tools used in this conversation"
    )


class ToolCall(BaseModel):
    """Model for tool call information

    Attributes:
        name (str): Name of the tool called.
        arguments (dict): Arguments passed to the tool.
        id (str): Unique identifier for the tool call.
    """

    name: str
    arguments: dict
    id: str


class AgentStep(BaseModel):
    """Model for individual agent steps in streaming response.

    Attributes:
        step_type (str): Type of step: 'agent_thinking', 'tool_call', 'tool_result', 'final_response'.
        content (str): Content of the step.
        tool_calls (Optional[List[ToolCall]]): Tool calls made by the agent in this step.
        metadata (Optional[dict]): Additional metadata for the step.
    """

    step_type: str = Field(
        ...,
        description="Type of step: 'agent_thinking', 'tool_call', 'tool_result', 'final_response'",
    )
    content: str = Field(..., description="Content of the step")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls made by agent"
    )
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class StreamingStatus(BaseModel):
    """Status update for streaming responses.

    Attributes:
        type (str): Type of status update.
        message (str): Status message.
        data (Optional[dict]): Additional data related to the status.
    """

    type: str = Field(..., description="Status type")
    message: str = Field(..., description="Status message")
    data: Optional[dict] = Field(default=None, description="Additional data")
