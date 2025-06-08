import requests
import streamlit as st
from api.collections import list_collections


def init_session_state():
    """Initialize all required session‐state variables if they don't exist."""
    ss = st.session_state
    ss.setdefault("collection_name", None)
    ss.setdefault("show_delete_dialog", False)
    ss.setdefault("delete_target", "")
    ss.setdefault("upload_url_fields", 1)
    ss.setdefault("create_url_fields", 1)
    ss.setdefault("just_created", False)
    ss.setdefault("search_query", "")
    ss.setdefault("search_results", [])


def render_sidebar():
    """Render the sidebar with a (cached) collection selector."""
    st.sidebar.title("Carro Knowledge Base")
    try:

        @st.cache_data(show_spinner=False, ttl=30)
        def _cached_list_collections() -> list[str]:
            return list_collections()

        all_cols = _cached_list_collections()
    except requests.HTTPError as err:
        st.sidebar.error(f"Failed to fetch collections ({err.response.status_code})")
        all_cols = []
    except Exception:
        st.sidebar.error("Unexpected error while fetching collections")
        all_cols = []

    ss = st.session_state

    # If nothing selected yet but we have collections, pick the first one
    if not ss.get("collection_name") and all_cols:
        ss.collection_name = all_cols[0]

    options = [""] + all_cols
    current = ss.collection_name if ss.collection_name in all_cols else ""
    idx = options.index(current)

    selected = st.sidebar.selectbox(
        "Select a collection:",
        options=options,
        index=idx,
        format_func=lambda x: x or "Select a collection...",
        key="sidebar_select",
    )

    if selected != ss.collection_name:
        ss.collection_name = selected
        # Reset any dependent state
        ss.show_delete_dialog = False
        ss.search_query = ""
        ss.search_results = []
        ss.upload_url_fields = 1
        ss.create_url_fields = 1

    # Feedback
    if ss.collection_name:
        st.sidebar.success(f"Current: {ss.collection_name}")
    else:
        if all_cols:
            st.sidebar.warning("No collection selected")
        else:
            st.sidebar.info("No collections found. Create one below.")

    st.sidebar.divider()
