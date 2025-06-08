"""Rewrite the user question for improved web search."""

from typing import Dict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ....core.config import settings
from ..state import State
from .prompts import query_rewriter_prompt


class Rewritten(BaseModel):
    """Return a concise, search-engine-friendly version of the user’s question.

    Attributes:
        rewritten_query (str): The rewritten query optimized for search engines.
    """

    rewritten_query: str = Field(description="Search-engine-friendly text")


llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
).with_structured_output(Rewritten)


chain = query_rewriter_prompt | llm


def rewrite_query(state: State) -> Dict[str, str]:
    """
    Generate a 'rewritten_query' from the original state['question'].
    Update state on error.

    Args:
        state (State): The current state containing the user question.

    Returns:
        Dict[str, str]: A dictionary with the original question, rewritten query,
                        API status, and error message (if any).
    """
    print("---REWRITE QUERY---")

    try:
        out = chain.invoke({"question": state["question"]})
        rewritten = out.rewritten_query
        print(f"Original: {state["question"]} -> Rewritten: {rewritten}")
        return {
            "question": state["question"],
            "rewritten_query": rewritten,
            "api_status": "success",
        }

    except Exception as exc:
        print(f"   Rewrite error: {exc}")
        return {
            "rewritten_query": state["question"],
            "api_status": "failure",
            "error_message": str(exc),
        }
