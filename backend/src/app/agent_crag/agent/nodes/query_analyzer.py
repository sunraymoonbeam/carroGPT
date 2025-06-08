"""Classify a user question into a Carro domain category."""

from typing import Dict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ....core.config import settings
from ..state import State
from .prompts import query_analyzer_prompt


class QueryClassification(BaseModel):
    """Structured output from GPT-4o-mini.
    Classifies user queries into Carro service categories and flags real-time data needs.

    Attributes:
        category (str): One of the predefined categories.
        needs_realtime_data (bool): True if the query requires real-time data (e.g., pricing, inventory).
    """

    category: str = Field(
        description=(
            "One of: greeting, used_car_inquiry, pricing_finance, "
            "seller_services, after_sales, terms_conditions, irrelevant"
        )
    )
    needs_realtime_data: bool = Field(
        description="True if question needs real-time info (pricing / inventory)."
    )


llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
).with_structured_output(QueryClassification)


chain = query_analyzer_prompt | llm


def classify_query(state: State) -> Dict:
    """
    Invoke the LLM classifier and update state with category and flags.

    Args:
        state (State): The current state containing the user question.

    Returns:
        Dict: A dictionary with the classified category, whether web search is needed,
              and API status.
    """
    print("---CLASSIFY QUERY---")

    question = state["question"]
    try:
        response = chain.invoke({"question": question})
        return {
            "category": response.category,
            "needs_web_search": response.needs_realtime_data,
            "api_status": "success",
        }

    except Exception as exc:
        # On failure, default to a safe category
        return {
            "category": "used_car_inquiry",
            "needs_web_search": False,
            "api_status": "partial_failure",
            "error_message": str(exc),
        }
