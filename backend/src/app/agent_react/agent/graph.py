# graph.py

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from ...core.config import get_settings
from .prompts import SYSTEM_PROMPT
from .tools import tools

settings = get_settings()

# Ren Hwa: creating a ReAct agent graph using langgraph's create_react_agent is as easy as PIE!
# No need to define nodes or edges manually, just provide the model, tools, and prompt.
# create_react_agent handles the rest, including the agent node and tool routing logic.
# Based on paper “ReAct: Synergizing Reasoning and Acting in Language Models” (https://arxiv.org/abs/2210.03629)


def create_graph():
    """Create the ReAct agent graph."""

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True,
        api_key=settings.OPENAI_API_KEY,
    )

    # Build a ReAct agent with create_react_agent.  No separate .compile(...) call needed.
    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )

    return graph
