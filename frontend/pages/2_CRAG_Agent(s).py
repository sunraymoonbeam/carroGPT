import uuid

import pandas as pd
import requests
import streamlit as st
from api.agent_crag import stream_chat
from utils.session import init_session_state, render_sidebar

# Page Configuration
st.set_page_config(
    page_title="corrective_rag_chat",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Initialize session state and sidebar
    init_session_state()
    render_sidebar()

    # Initialize chat‐specific session keys, thread_id is used to track memory
    ss = st.session_state
    ss.setdefault("crag_thread_id", str(uuid.uuid4()))
    ss.setdefault("crag_messages", [])

    if not ss.collection_name:
        st.warning(
            "⚠️ Please select or create a Knowledge Base Collection in the sidebar "
            "to begin chatting; CarroGPT uses it to ground its answers in your documents."
        )
        return

    # Header
    st.header("Chat with Corrective RAG Agent(s)")
    st.write(
        "This chatbot uses a corrective RAG pipeline to answer your questions. "
        "You’ll see each stage, **query analysis**, **retrieval**, "
        "**document evaluation**, **rewriting of query**, **web search**, and **final generation**, as it happens. "
        "Feel free to ask anything related to Carro’s services."
    )

    # Render past conversation
    for msg in ss.crag_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input for the chat
    user_input = st.chat_input(
        placeholder="e.g. What sources of personal data collection does Carro collect from me? Or How do I get carro certified?."
    )
    if not user_input:
        return

    ss.crag_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Placeholder for streaming status
    status_placeholder = st.empty()
    status_placeholder.markdown("🔎 **Node: Classifier** – Analyzing query...")

    # Containers to collect the execution trace (pipeline steps) and final sources for citation
    execution_trace: list[str] = []
    final_answer = ""
    documents: list[dict] = []
    web_source_urls: list[str] = []

    # Stream the CRAG pipeline via SSE
    try:
        for chunk in stream_chat(
            thread_id=ss.crag_thread_id,
            collection_name=ss.collection_name,
            message=user_input,
        ):
            msg_type = chunk.get("type")

            # --------------------------------------------------------------------
            # 1. CLASSIFIER NODE
            # --------------------------------------------------------------------
            if msg_type == "status" and chunk.get("node") == "classifier":
                category = chunk.get("category", "")
                needs_web = chunk.get("needs_web_search", False)
                op_msg = (
                    f"**Node: Classifier** → Question categorized as '{category}', "
                    f"needs_web_search: {needs_web}"
                )
                execution_trace.append(op_msg)
                status_placeholder.markdown(f"📝 {op_msg}")

            # --------------------------------------------------------------------
            # 2. RETRIEVER NODE
            # --------------------------------------------------------------------
            elif msg_type == "status" and chunk.get("node") == "retriever":
                doc_count = chunk.get("document_count", 0)
                op_msg = (
                    f"**Node: Retriever** → Retrieved {doc_count} document snippet(s) "
                    f"from '{ss.collection_name}'"
                )
                execution_trace.append(op_msg)
                status_placeholder.markdown(f"📚 {op_msg}")

            # --------------------------------------------------------------------
            # 3. DOCUMENT EVALUATOR NODE
            # --------------------------------------------------------------------
            elif msg_type == "status" and chunk.get("node") == "evaluator":
                filtered_count = chunk.get("filtered_count", 0)
                relevance = chunk.get("doc_relevance", 0.0)
                op_msg = (
                    f"**Node: Document Grader** → Filtered {filtered_count} relevant document(s), "
                    f"relevance ratio: {relevance:.2f}"
                )
                execution_trace.append(op_msg)
                status_placeholder.markdown(f"💯 {op_msg}")

            # --------------------------------------------------------------------
            # 4.QUERY REWRITTER NODE
            # --------------------------------------------------------------------
            elif msg_type == "status" and chunk.get("node") == "query_rewriter":
                orig_q = chunk.get("original_question", "")
                rewritten_q = chunk.get("rewritten_query", "")
                op_msg = f"**Node: Query Rewriter** → Original: '{orig_q}' → Rewritten: '{rewritten_q}'"
                execution_trace.append(op_msg)
                status_placeholder.markdown(f"✏️ {op_msg}")

            # --------------------------------------------------------------------
            # 5. WEB SEARCH NODE
            # --------------------------------------------------------------------
            elif msg_type == "status" and chunk.get("node") == "web_search":
                web_count = chunk.get("web_result_count", 0)
                op_msg = f"**Node: Web Search** → Found {web_count} web result(s)"
                execution_trace.append(op_msg)
                status_placeholder.markdown(f"🌐 {op_msg}")

            # --------------------------------------------------------------------
            # 6. GENERATOR (FINAL RESPONSE)
            # --------------------------------------------------------------------
            elif msg_type == "response":
                final_answer = chunk.get("response", "")
                sources = chunk.get("sources", {}) or {}
                documents = sources.get("documents", []) or []
                web_source_urls = sources.get("urls", []) or []

                op_msg = (
                    f"**Node: Generator** → Generated final answer "
                    f"with {len(documents)} document snippet(s) and "
                    f"{len(web_source_urls)} web URL(s)"
                )
                execution_trace.append(op_msg)

                status_placeholder.empty()
                with st.chat_message("assistant"):
                    st.write(final_answer)

            # ── 7. COMPLETE SIGNAL ──────────────────────────────────────────────────
            elif msg_type == "complete":
                break

    except requests.exceptions.Timeout as e:
        status_placeholder.empty()
        with st.chat_message("assistant"):
            st.write("⚠️ Network timeout. Please try again.")
        st.error(f"Timeout error: {e}")
        return

    except Exception as e:
        status_placeholder.empty()
        with st.chat_message("assistant"):
            st.write("❌ An unexpected error occurred. Please try again.")
        st.error(f"Error details: {e}")
        return

    # Store the assistant’s final message in session state
    ss.crag_messages.append({"role": "assistant", "content": final_answer})

    # Show execution trace and sources in separate expanders
    if execution_trace:
        with st.expander("Graph Execution Trace", expanded=False):
            for idx, step in enumerate(execution_trace, start=1):
                st.write(f"{idx}. {step}")

    # Relevant Sources Section
    if documents or web_source_urls:
        with st.expander("Retrieved Documents & Sources", expanded=False):
            if documents:
                st.subheader("Relevant Document Chunks")
                for idx, doc in enumerate(documents, start=1):
                    st.markdown(f"### Document Source: #{idx}")

                    meta_df = pd.DataFrame(
                        [
                            {
                                "ID": doc["id"],
                                "Score": f"{doc['score']:.2f}",
                                "Source": doc["source"],
                                "File Type": doc.get("file_type", "N/A"),
                                "Page": doc.get("page_number", "N/A"),
                            }
                        ]
                    )
                    st.table(meta_df)

                    full_text = doc.get("text", "")
                    snippet = (
                        full_text
                        if len(full_text) <= 200
                        else full_text[:200].rsplit(" ", 1)[0] + "…"
                    )
                    st.write(snippet)

                    if len(full_text) > 200:
                        # use HTML <details> instead of a nested st.expander
                        details_html = f"""
    <details>
    <summary style="cursor: pointer; color: #1f77b4;">Show full document text</summary>
    <div style="margin-top: 0.5em;">{full_text}</div>
    </details>
    """
                        st.markdown(details_html, unsafe_allow_html=True)

                    st.markdown("---")

            if web_source_urls:
                st.subheader("Web Source URLs")
                for url in web_source_urls:
                    st.markdown(f"- {url}")


if __name__ == "__main__":
    main()
