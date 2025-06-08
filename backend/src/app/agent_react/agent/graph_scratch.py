"""ReAct Agent Graph Implementation (from scratch)."""

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from ...core.config import get_settings
from .prompts import SYSTEM_PROMPT
from .state import State
from .tools import tools

settings = get_settings()


def create_agent_node(llm_with_tools):
    """Create the agent node that decides on actions."""

    def agent(state: State):
        messages = state["messages"]

        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent


def create_tool_router():
    """Create conditional edge function for routing."""

    def tool_router(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]

        # If the last message has tool calls, go to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    return tool_router


def create_graph() -> StateGraph:
    """Create and compile the Carro ReAct agent graph."""

    # Initialize OpenAI LLM with tools
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective model, you can change to gpt-4o if needed
        temperature=0,
        streaming=True,  # Enable streaming for better UX
        api_key=settings.OPENAI_API_KEY,
    )
    llm_with_tools = llm.bind_tools(tools)

    # Create nodes
    agent_node = create_agent_node(llm_with_tools)
    tool_node = ToolNode(tools)
    tool_router = create_tool_router()

    # Build graph
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tool_router, ["tools", "__end__"])
    builder.add_edge("tools", "agent")

    # Compile graph with memory saver
    graph = builder.compile(checkpointer=MemorySaver())

    return graph
