import json
import uuid

import pandas as pd
import requests
import streamlit as st
from api.agent_react import stream_react_chat
from utils.session import init_session_state, render_sidebar

# Page Configuration
st.set_page_config(
    page_title="react_chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Initialize session state and sidebar
    init_session_state()
    render_sidebar()

    # Initialize chat‐specific session keys, thread_id is used to track memory
    ss = st.session_state

    ss.setdefault("react_thread_id", str(uuid.uuid4()))
    ss.setdefault("react_messages", [])

    if not ss.collection_name:
        st.warning(
            "⚠️ Please select or create a Knowledge Base Collection in the sidebar "
            "to begin chatting; CarroGPT uses it to ground its answers in your documents."
        )
        return

    # Header
    st.header("Chat with ReAct (Reasoning and Acting) Agent")
    st.write(
        """
        This chatbot helps you explore Carro’s services by reasoning 
        step by step, deciding which tools to call (document retrieval, web search, etc.).
        Feel free to ask anything related to Carro’s services.
        """
    )

    # Render past conversation
    for msg in ss.react_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input for the chat
    user_input = st.chat_input(
        placeholder="e.g. How do I sell my car through Carro? Or what are the latest financing rates?"
    )
    if not user_input:
        return

    ss.react_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Placeholder for streaming status
    status_placeholder = st.empty()
    status_placeholder.markdown("🤔 CarroGPT is thinking...")

    # Containers to collect the final sources for citation
    full_response = ""
    tool_outputs: list[dict] = []

    # Stream the ReAct agent's response via SSE
    try:
        for chunk in stream_react_chat(
            thread_id=ss.react_thread_id,
            collection_name=ss.collection_name,
            message=user_input,
        ):
            step_type = chunk.get("step_type", "")
            content = chunk.get("content", "")
            tool_calls = chunk.get("tool_calls", [])

            # Tool invocation
            if step_type == "tool_call":
                invoked = [tc.get("name", "") for tc in tool_calls]
                if "retrieve_carro_documents" in invoked:
                    status_placeholder.markdown(
                        f"📚 Tool: retriever (querying collection '{ss.collection_name}')"
                    )
                elif "search_web" in invoked:
                    status_placeholder.markdown(
                        "🌐 Tool: search_web (performing web lookup)"
                    )
                else:
                    status_placeholder.markdown("🔧 Tool: Invoking external tool...")

                if content and "Using tools" not in content:
                    status_placeholder.markdown(f"Agent thought: {content}")

            # Tool result
            elif step_type == "tool_result":
                tool_name = chunk.get("metadata", {}).get("tool_name", "")
                try:
                    data = json.loads(content)
                except ValueError:
                    data = content
                tool_outputs.append({"tool": tool_name, "data": data})
                status_placeholder.markdown(f"Result: {tool_name} completed.")

            # Final response
            elif step_type == "final_response":
                status_placeholder.empty()
                with st.chat_message("assistant"):
                    st.write(content)
                full_response = content

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
    ss.react_messages.append({"role": "assistant", "content": full_response})

    # Relevant Sources Section
    if tool_outputs:
        with st.expander("Retrieved Documents & Sources", expanded=False):
            for entry in tool_outputs:
                tool = entry["tool"]
                data = entry["data"]

                if tool == "retrieve_carro_documents" and isinstance(data, list):
                    st.subheader("Retrieved Documents")
                    for idx, doc in enumerate(data, start=1):
                        st.subheader(f"Document #{idx}")
                        # Metadata dataframe
                        df = pd.DataFrame(
                            [
                                {
                                    "id": doc["id"],
                                    "score": float(doc["score"]),
                                    "source": doc["source"],
                                    "file_type": doc["file_type"],
                                    "page_number": doc["page_number"],
                                }
                            ]
                        )
                        st.dataframe(df)

                        # Document text snippet + HTML details for full text
                        full_text = doc.get("text", "")
                        snippet = (
                            full_text
                            if len(full_text) <= 200
                            else full_text[:200].rsplit(" ", 1)[0] + "…"
                        )
                        st.write(snippet)

                        if len(full_text) > 200:
                            details_html = f"""
    <details>
    <summary style="cursor: pointer; color: #1f77b4; font-weight: bold;">
        Show full document text
    </summary>
    <div style="margin-top: 0.5em; white-space: pre-wrap;">{full_text}</div>
    </details>
    """
                            st.markdown(details_html, unsafe_allow_html=True)

                        st.divider()

                elif tool == "search_web" and isinstance(data, list):
                    for hit in data:
                        st.subheader(hit.get("title", ""))
                        st.markdown(f"[Source]({hit.get('url', '')})")
                        st.write(hit.get("content", ""))
                        st.divider()

                else:
                    st.write(data)


if __name__ == "__main__":
    main()
