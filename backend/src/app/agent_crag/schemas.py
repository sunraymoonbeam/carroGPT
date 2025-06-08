from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming chat payload.

    Attributes:
        message (str): Customer question.
        collection_name (str): Which Qdrant collection to search against.
    """

    message: str = Field(..., description="Customer question")
    collection_name: str = Field(
        ..., description="Which Qdrant collection to search against"
    )


# Base status model
class BaseStatus(BaseModel):
    """
    Base schema for a status update from any node.

    Attributes:
        type (Literal["status"]): Always 'status'.
        node (str): Name of the node producing this status.
        api_status (Literal["pending", "success", "partial_failure", "failure"]): Status of the node.
        error_message (Optional[str]): Error message if the node encountered a failure.
    """

    type: Literal["status"] = Field("status", description="Always 'status'")
    node: str = Field(..., description="Name of the node producing this status")
    api_status: Literal["pending", "success", "partial_failure", "failure"] = Field(
        ..., description="Status of the node"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if the node encountered a failure"
    )


# --------------------------------------------------------------------
# Query Analyzer Node Status
# --------------------------------------------------------------------
class ClassifierStatus(BaseStatus):
    """
    Status schema for the 'classifier' node.

    Attributes:
        category (str): Predicted category label.
        needs_web_search (bool): True if real-time data is required.
    """

    category: str = Field(..., description="Predicted category label")
    needs_web_search: bool = Field(
        ..., description="True if real-time data is required"
    )


# --------------------------------------------------------------------
# Retriever Node Status
# --------------------------------------------------------------------
class Document(BaseModel):
    """
    Representation of a single document chunk hit from Qdrant.

    Attributes:
        id (str): Point ID in Qdrant.
        score (float): Relevance score returned by Qdrant.
        text (str): Full text of the snippet.
        source (str): Source label, usually the name of the file.
        file_type (Optional[str]): Type of the file (e.g., 'pdf', 'docx').
        page_number (Optional[int]): Page number in the source document.
    """

    id: str = Field(..., description="Point ID in Qdrant")
    text: str = Field(..., description="Full text of the snippet")
    score: float = Field(..., description="Relevance score returned by Qdrant")
    source: str = Field(..., description="Source label, usually the name of the file")
    file_type: Optional[str] = Field(
        None, description="Type of the file (e.g., 'pdf', 'docx')"
    )
    page_number: Optional[int] = Field(
        None, description="Page number in the source document"
    )


class RetrieverStatus(BaseStatus):
    """
    Status schema for the 'retriever' node.

    Attributes:
        document_count (int): Number of FAQ snippets retrieved.
        documents (List[DocumentInfo]): List of detailed FAQ snippet info.
    """

    document_count: int = Field(..., description="Number of FAQ snippets retrieved")
    documents: List[Document] = Field(
        ...,
        description="List of detailed FAQ snippet info (page_content, score, source, file_type, page_number)",
    )


# --------------------------------------------------------------------
# Document Grader Node Status
# --------------------------------------------------------------------
class EvaluatorStatus(BaseStatus):
    """
    Status schema for the 'evaluator' node.

    Attributes:
        filtered_count (int): Number of relevant snippets.
        doc_relevance (float): Ratio (0.0–1.0) of snippets graded relevant.
    """

    filtered_count: int = Field(..., description="Number of relevant snippets")
    doc_relevance: float = Field(
        ..., description="Ratio (0.0–1.0) of snippets graded relevant"
    )


# --------------------------------------------------------------------
# Query Rewriter Node Status
# --------------------------------------------------------------------
class RewriterStatus(BaseStatus):
    """
    Status schema for the 'query_rewriter' node.

    Attributes:
        original_question (str): The original user question.
        rewritten_query (str): Rewritten query for web search.
    """

    original_question: str = Field(..., description="The original user question")
    rewritten_query: str = Field(..., description="Rewritten query for web search")


# --------------------------------------------------------------------
# 7. Web Search Node Status
# --------------------------------------------------------------------
class WebSearchStatus(BaseStatus):
    """
    Status schema for the 'web_search' node.

    Attributes:
        web_result_count (int): Number of web results found.
        urls (List[str]): List of URLs from web search results.
    """

    web_result_count: int = Field(..., description="Number of web results found")
    urls: List[str] = Field(..., description="List of URLs from web search results")


# --------------------------------------------------------------------
# 9. Generator (final response) Node Status
# --------------------------------------------------------------------
class Sources(BaseModel):
    """
    Container holding both relevant document chunks info and web-search URLs.

    Attributes:
        documents (List[DocumentInfo]): List of FAQ snippet objects.
        urls (List[str]): List of web-search result URLs.
    """

    documents: List[Document] = Field(
        ..., description="List of FAQ snippet objects (id, score, content, source)"
    )
    urls: List[str] = Field(..., description="List of web-search result URLs")


class GeneratorResponse(BaseModel):
    """
    Final response from the 'generator' node.

    Attributes:
        type (Literal["response"]): Always 'response'.
        response (str): Final answer text.
        sources (Sources): Container of FAQ docs and web URLs.
        api_status (Literal["success", "partial_failure", "failure"]): Final status of the generator node.
        error_message (Optional[str]): Error message if generation failed.
    """

    type: Literal["response"] = Field("response", description="Always 'response'")
    response: str = Field(..., description="Final answer text")
    sources: Sources = Field(..., description="Container of FAQ docs and web URLs")
    api_status: Literal["success", "partial_failure", "failure"] = Field(
        ..., description="Final status of the generator node"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if generation failed"
    )
