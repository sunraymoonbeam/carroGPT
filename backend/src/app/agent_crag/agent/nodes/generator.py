from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ....core.config import settings
from ..state import State
from .prompts import generator_prompt

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY)

parser = StrOutputParser()

response_chain = generator_prompt | llm | parser


def generate_response(state: State) -> Dict[str, Any]:
    """
    Generate a response based on the current state of the conversation.

    1. Builds `history` by walking through state["messages"] (BaseMessage list).
    2. Feeds {history} + other contexts into the prompt.
    3. Returns both "messages": [AIMessage(...)] so LangGraph can merge it.

    Args:
        state (State): The current state containing conversation history, question,
                       category, and any retrieved documents.

    Returns:
        Dict[str, Any]: A dictionary containing the generated answer, filtered documents,
                        search results, API status, error message (if any), and messages.=
    """
    print("---GENERATE RESPONSE---")

    # Extract question, category, and documents from state
    question = state["question"]
    category = state.get("category", "unknown")
    documents = state.get("filtered_documents", [])
    web_sources = state.get("search_results") or []

    # Build FAQ and web-context strings
    if documents:
        doc_context = "\n\n".join(doc.page_content for doc in documents[:4])
    else:
        doc_context = "No relevant FAQ information available."

    if web_sources:
        web_context = "\n\n".join(doc.page_content[:300] for doc in web_sources[:3])
    else:
        web_context = "No current web search information available."

    # Build `history` by serializing state["messages"]
    # Each BaseMessage has a .content and a type (HumanMessage or AIMessage).
    history_pieces = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history_pieces.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_pieces.append(f"Assistant: {msg.content}")
        else:
            # In case you ever add more BaseMessage types:
            history_pieces.append(f"{type(msg).__name__}: {msg.content}")

    # Join with newlines (or blank lines if you prefer)
    history_str = (
        "\n".join(history_pieces) if history_pieces else "No prior conversation."
    )

    try:
        answer = response_chain.invoke(
            {
                "history": history_str,
                "question": question,
                "category": category,
                "doc_context": doc_context,
                "web_context": web_context,
            }
        )

        # Wrap the model’s reply in an AIMessage so add_messages merges it
        ai_msg = AIMessage(content=answer)

        return {
            "answer": answer,
            "filtered_documents": documents,
            "search_results": web_sources,
            "api_status": "success",
            "error_message": None,
            "messages": [ai_msg],  # LangGraph appends the AI turn with add_messages
        }

    except Exception as exc:
        # On failure, return a fallback answer and still append as AIMessage
        fallback_texts = {
            "greeting": "Hello! Welcome to Carro. How can I assist you today?",
            "used_car_inquiry": (
                "For used car details and inventory, please visit our website "
                "or let me know what model you are interested in."
            ),
            "pricing_finance": (
                "Pricing and financing can vary. Please contact our finance team "
                "or visit carro.sg/finance for a personalized quote."
            ),
            "seller_services": (
                "To sell your car, please go to carro.sg/sell. Enter your details "
                "for a free valuation or let me know your vehicle model."
            ),
            "after_sales": (
                "For after-sales services, Carro Protect provides warranties and "
                "inspections. Check carro.sg/protect for more details."
            ),
            "terms_conditions": (
                "You can review Carro’s Terms & Conditions on our website under Terms "
                "of Use. Which specific section are you looking for?"
            ),
            "irrelevant": (
                "I’m here to help with Carro-related inquiries like buying, selling, "
                "financing, or after-sales. How can I assist?"
            ),
        }
        fallback_answer = fallback_texts.get(
            category,
            "I’m sorry, but I’m having trouble right now. "
            "Please try again later or contact Carro support.",
        )
        ai_msg = AIMessage(content=fallback_answer)

        return {
            "answer": fallback_answer,
            "filtered_documents": documents,
            "search_results": web_sources,
            "api_status": "failure",
            "error_message": str(exc),
            "messages": [ai_msg],
        }
