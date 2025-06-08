"""State management for Carro ReAct Agent."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """
    State for the Carro ReAct Agent.

    Attributes:
        messages (Annotated[list[BaseMessage], add_messages]):
            Conversation history messages, merged automatically.

        collection_name (str):
            Dynamic collection selection for Qdrant queries.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    collection_name: str
