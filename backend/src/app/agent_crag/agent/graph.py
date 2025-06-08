"""CRAG Agent Graph Implementation."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .edges import decide_next_step, should_search_web
from .nodes.document_grader import evaluate_documents
from .nodes.generator import generate_response
from .nodes.query_analyzer import classify_query
from .nodes.query_rewriter import rewrite_query
from .nodes.retriever import retrieve_documents
from .nodes.web_search import search_web
from .state import State


def create_graph():
    """
    Create the full graph for the Carro chatbot agent.
    Flow:
    1. classify_query
    2. If greeting/irrelevant: generator
       Else: retriever
    3. evaluate_documents (filter & set needs_web_search)
    4. If needs_web_search: rewrite_query → web_search → generator
       Else: generator
    """

    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("classifier", classify_query)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("evaluator", evaluate_documents)
    workflow.add_node("query_rewriter", rewrite_query)
    workflow.add_node("web_search", search_web)
    workflow.add_node("generator", generate_response)

    # Start → classifier
    workflow.add_edge(START, "classifier")

    # classifier → (via decide_next_step) → retriever or generator
    workflow.add_conditional_edges(
        "classifier",
        decide_next_step,
        {"retriever": "retriever", "generator": "generator"},
    )

    # retriever → evaluator
    workflow.add_edge("retriever", "evaluator")

    # evaluator → (via should_search_web) → query_rewriter or generator
    workflow.add_conditional_edges(
        "evaluator",
        should_search_web,
        {"query_rewriter": "query_rewriter", "generator": "generator"},
    )

    # query_rewriter → web_search → generator
    workflow.add_edge("query_rewriter", "web_search")
    workflow.add_edge("web_search", "generator")

    # generator → END
    workflow.add_edge("generator", END)

    # Compile graph with memory saver
    graph = workflow.compile(checkpointer=MemorySaver())
    return graph
