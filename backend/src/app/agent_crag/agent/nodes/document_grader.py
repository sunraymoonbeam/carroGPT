"""Grade each FAQ snippet for relevance and compute doc_relevance."""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ....core.config import settings
from ..state import State
from .prompts import grading_prompt

logger = logging.getLogger(__name__)


class BinaryGrade(BaseModel):
    """Return 'yes' or 'no' depending on snippet relevance.

    Attributes:
        binary_score (str): 'yes' or 'no' indicating relevance.
    """

    binary_score: str = Field(description="'yes' or 'no'")


llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
).with_structured_output(BinaryGrade)

grading_chain = grading_prompt | llm


def evaluate_documents(state: State) -> Dict[str, Any]:
    """
    Filter out irrelevant FAQ documents and compute doc_relevance.

    Args:
         state (State): The current state containing conversation history, question,
                       category, and any retrieved documents.

    Returns:
        filtered_documents : List[Document] graded “yes”
        doc_relevance      : float in [0.0–1.0]
        api_status         : "success" or "partial_failure" or "failure"
        error_message      : Optional[str]
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    docs: List[Document] = state.get("documents", [])

    if not docs:
        print("No document chunks found. Will require web search.")
        return {
            "filtered_documents": [],
            "doc_relevance": 0.0,
            "api_status": "partial_failure",
        }

    filtered: List[Document] = []
    for idx, doc in enumerate(docs, start=1):
        try:
            grade = grading_chain.invoke(
                {"question": question, "snippet": doc.page_content}
            )
            if grade.binary_score.lower() == "yes":
                print(f"Doc {idx}: ---GRADE: DOCUMENT RELEVANT---")
                filtered.append(doc)

            else:
                print(f"Doc {idx}: ---GRADE: DOCUMENT NOT RELEVANT---")

        except Exception as exc:
            logger.error("Error grading document: %s", exc, exc_info=True)
            print(f"Doc {idx}: ---GRADE: ERROR TREATED AS NOT RELEVANT---")
            continue

    relevance_ratio = len(filtered) / len(docs)
    print(f"---RELEVANCE RATIO = {relevance_ratio:.2f}---")

    return {
        "filtered_documents": filtered,
        "doc_relevance": relevance_ratio,
        "api_status": "success" if filtered else "partial_failure",
        "error_message": None if filtered else "No relevant documents found",
    }
