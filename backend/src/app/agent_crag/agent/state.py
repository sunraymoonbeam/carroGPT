"""CRAG Agent Workflow State Definition.
This module defines the shared state for the Carro CRAG Agent workflow,
including conversation memory and various processing stages."""

from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Shared state for the Carro chatbot workflow with proper conversation memory.

    Attributes:
        messages (Annotated[List[BaseMessage], add_messages]):
            Conversation history - add_messages handles merging automatically.
        ----------------------------------------------------------------------
        question (str):
            Original user question.
        category (str):
            Intent label: greeting, used_car_inquiry, pricing_finance,
            seller_services, after_sales, terms_conditions, or irrelevant.
        needs_web_search (bool):
            Set to True if the classifier decided this query requires live data.
        ----------------------------------------------------------------------
        collection_name (str):
            Name of the Qdrant collection to search against.
        documents (List[Document]):
            Raw FAQ snippets pulled from Qdrant.
        ----------------------------------------------------------------------
        filtered_documents (List[Document]):
            Subset of 'documents' graded as relevant.
        doc_relevance (float):
            Fraction (0.0–1.0) of FAQ docs graded as relevant.
        ----------------------------------------------------------------------
        rewritten_query (str):
            Transformed query for web search (empty if not used).
        ----------------------------------------------------------------------
        search_results (Optional[List[Document]]):
            List of web result Documents, if a web search was performed.
        ----------------------------------------------------------------------
        answer (str):
            Bot's final answer text.
        ----------------------------------------------------------------------
        api_status (str):
            Current status of the node: 'pending', 'success', 'partial_failure', or 'failure'.
        error_message (Optional[str]):
            Error message if the node encountered a failure, otherwise None.
    """

    messages: Annotated[List[BaseMessage], add_messages]

    # Query Analyzer
    question: str
    category: str
    needs_web_search: bool

    # Retriever
    collection_name: str
    documents: List[Document]

    # Document Grader
    filtered_documents: List[Document]
    doc_relevance: float

    # Query Rewriter
    rewritten_query: str

    # Web Search
    search_results: Optional[List[Document]]

    # Response Generator
    answer: str

    # Node Status
    api_status: str
    error_message: Optional[str]
