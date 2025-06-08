"""Collections management page for the Carro RAG application.

Users can view details of an existing collection (and upload more documents),
create a new collection (with mandatory initial files or URLs), and delete collections
(with password confirmation). Includes PDF previews for any selected PDF files.
"""

import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from api.collections import (
    delete_collection,
    get_collection_details,
    list_collections,
    search_collection,
    upload_documents,
)
from streamlit_pdf_viewer import pdf_viewer  # pip install streamlit-pdf-viewer
from utils.session import init_session_state, render_sidebar

# Page Configuration
st.set_page_config(
    page_title="Collections",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password123")
COLLECTION_NAME_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,253}[A-Za-z0-9]$")


@st.cache_data(ttl=500)
def get_collections_list() -> List[str]:
    """Get list of all collections with caching."""
    try:
        return list_collections()
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_collection_details_cached(collection_name: str) -> Optional[Dict[str, Any]]:
    """Get collection details with caching."""
    try:
        return get_collection_details(collection_name)
    except Exception:
        return None


def safe_pdf_preview(file: Any, width: str = "100%", height: int = 600) -> None:
    """Safely preview a PDF file with proper error handling."""
    try:
        if not hasattr(file, "read") or not hasattr(file, "name"):
            st.warning("File is no longer accessible for preview.")
            return
        pdf_bytes = file.read()
        if not pdf_bytes:
            st.warning(f"'{file.name}' appears to be empty.")
            return
        with st.expander(f"Preview: {file.name}", expanded=False):
            pdf_viewer(input=pdf_bytes, width=width, height=height, annotations=[])
        file.seek(0)
    except Exception as e:
        st.warning(f"Unable to preview '{file.name}': {e}")
        try:
            file.seek(0)
        except Exception:
            pass


def handle_upload_documents(
    collection_name: str, files: List[Any], urls: List[str]
) -> bool:
    """Handle document upload with progress tracking and error reporting."""
    total_items = len(files) + len(urls)
    successes = 0
    errors: List[str] = []

    if total_items == 0:
        st.warning("No files or URLs to upload.")
        return False

    with st.spinner("Uploading documents for processing..."):
        progress = st.progress(0.0, text="Preparing uploads...")

        # Upload files
        for idx, f in enumerate(files, start=1):
            try:
                f.seek(0)
                progress.progress(
                    (idx - 1) / total_items,
                    text=f"Uploading '{f.name}' ({idx}/{total_items})...",
                )
                upload_documents(collection_name, files=[f], urls=[])
                successes += 1
            except requests.HTTPError as e:
                errors.append(f"File '{f.name}': {e.response.text}")
            except Exception as e:
                errors.append(f"File '{f.name}': {e}")

        # Upload URLs
        for jdx, url in enumerate(urls, start=len(files) + 1):
            try:
                progress.progress(
                    (jdx - 1) / total_items,
                    text=f"Processing URL '{url}' ({jdx}/{total_items})...",
                )
                upload_documents(collection_name, files=[], urls=[url])
                successes += 1
            except requests.HTTPError as e:
                errors.append(f"URL '{url}': {e.response.text}")
            except Exception as e:
                errors.append(f"URL '{url}': {e}")

        # Complete progress
        progress.progress(1.0, text="Upload complete!")
        progress.empty()

    if successes:
        st.toast(f"Uploaded {successes}/{total_items} items successfully!", icon="✅")
    if errors:
        st.error(f"{len(errors)} errors occurred:")
        for err in errors:
            st.caption(f"• {err}")

    return successes > 0


def render_current_collection_tab() -> None:
    ss = st.session_state

    if not ss.collection_name:
        st.warning("Please select or create a collection to view its details.")
        return

    collection_name = ss.collection_name
    details = get_collection_details_cached(collection_name)

    if not details:
        st.error(
            f"Could not retrieve details for '{collection_name}'. It may have been deleted."
        )
        return

    col_left, col_right = st.columns([2, 3], gap="large")
    with col_left:
        st.subheader(f"Collection: `{collection_name}`")
        st.caption("Overview of your vector collection.")

        c1, c2 = st.columns(2)
        c1.metric(
            label="Number of Vectors",
            value=f"{details.get('num_vectors', 0):,}",
            help="Total embeddings stored in this collection.",
        )
        c2.metric(
            label="Segments Count",
            value=details.get("segments_count", "N/A"),
            help="How many index segments Qdrant has created.",
        )

        c3, c4 = st.columns(2)
        c3.metric(
            label="Vector Size",
            value=details.get("vector_size", "N/A"),
            help="Dimensionality of each vector embedding.",
        )
        c4.metric(
            label="Distance Metric",
            value=details.get("distance", "N/A").capitalize(),
            help="The similarity function used.",
        )

    with col_right:
        with st.expander("Advanced Details", expanded=False):
            adv = {
                k: v
                for k, v in details.items()
                if k
                not in [
                    "name",
                    "num_vectors",
                    "segments_count",
                    "vector_size",
                    "distance",
                    "sample_documents",
                ]
            }
            st.json(adv)

        with st.expander("Sample Documents", expanded=True):
            samples = details.get("sample_documents", [])
            if samples:
                df_samples = pd.DataFrame(
                    [
                        {
                            "Document ID": s.get("id"),
                            "Text": s.get("text"),
                            "Source": s.get("source"),
                            "File Type": s.get("file_type"),
                            "Page Number": s.get("page_number"),
                            "Upload Date": s.get("upload_date"),
                        }
                        for s in samples[:3]
                    ]
                )
                st.dataframe(df_samples, use_container_width=True)
            else:
                st.info("No sample documents available.")

    st.divider()
    st.subheader("Upload Additional Documents")
    st.caption(
        "Upload more PDF/DOCX files or enter URLs (one per line) to add to this collection."
    )

    with st.form("upload_more_docs_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            label="Upload PDF or DOCX files:",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"upload_files_{collection_name}",
        )

        if uploaded_files:
            with st.expander("Preview Selected PDFs", expanded=False):
                for f in uploaded_files:
                    if f.type == "application/pdf":
                        safe_pdf_preview(f)
                    else:
                        st.info(f"Cannot preview '{f.name}' (not a PDF).")

        urls_text = st.text_area(
            label="URLs (one per line):",
            placeholder="https://example.com/doc1\nhttps://example.com/doc2",
            key=f"upload_urls_{collection_name}",
        )
        urls = [u.strip() for u in urls_text.split("\n") if u.strip()]

        submitted = st.form_submit_button("Upload Documents", type="primary")
        if submitted:
            if not uploaded_files and not urls:
                st.warning("Please provide at least one file or URL.")
            else:
                success = handle_upload_documents(collection_name, uploaded_files, urls)
                if success:
                    # Clear caches after successful upload
                    get_collection_details_cached.clear()
                    get_collections_list.clear()
                    st.cache_data.clear()
                    st.toast(
                        f"Uploaded to '{collection_name}' successfully!",
                        icon="✅",
                    )
                    st.rerun()

    st.markdown("---")
    st.subheader("Search Collection")
    st.caption("Run a semantic search against this collection.")

    if "search_query" not in ss:
        ss.search_query = ""
    if "search_results" not in ss:
        ss.search_results = []

    with st.form("search_form"):
        search_query = st.text_input(
            label="Enter search query:",
            value=ss.search_query,
            key="search_input",
        )
        top_k = st.slider(
            label="Number of results (top_k):",
            min_value=1,
            max_value=10,
            value=5,
            key="search_topk",
        )
        search_submitted = st.form_submit_button("Search", type="primary")

        if search_submitted:
            if not search_query.strip():
                st.warning("Please enter a query to search.")
                ss.search_results = []
            else:
                ss.search_query = search_query
                with st.spinner("Searching..."):
                    try:
                        hits = search_collection(
                            collection_name, search_query.strip(), top_k
                        )
                        ss.search_results = hits
                    except requests.HTTPError as e:
                        st.error(f"Search failed: {e.response.text}")
                        ss.search_results = []
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                        ss.search_results = []
                st.rerun()

    if ss.search_results:
        df_rows = [
            {
                "Document ID": hit.get("id", ""),
                "Text": hit.get("text", ""),
                "Score": hit.get("score", 0.0),
                "Source": hit.get("source", ""),
                "File Type": hit.get("file_type", ""),
                "Page Number": hit.get("page_number", "N/A"),
            }
            for hit in ss.search_results
        ]
        st.markdown("---")
        st.subheader("Search Results")
        st.dataframe(pd.DataFrame(df_rows), use_container_width=True)


def render_create_collection_tab() -> None:
    ss = st.session_state

    st.subheader("Create New Collection")
    st.caption(
        "Enter a unique name (letters, digits, underscores, hyphens; max 255 chars). "
        "Then upload one or more PDF/DOCX files or paste URLs (one per line)."
    )

    with st.form("create_collection_form", clear_on_submit=True):
        new_name = st.text_input(
            label="Collection Name",
            key="new_collection_name_input",
            help="Allowed: A–Z, a–z, 0–9, underscore, hyphen; max length 255; must start/end with letter or digit.",
        ).strip()

        create_files = st.file_uploader(
            "Upload PDF or DOCX files:",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="create_files_input",
        )
        if create_files:
            with st.expander("Preview Selected PDFs", expanded=False):
                for f in create_files:
                    if f.type == "application/pdf":
                        safe_pdf_preview(f)
                    else:
                        st.info(f"Cannot preview '{f.name}' (not a PDF).")

        create_urls_text = st.text_area(
            label="URLs (one per line):",
            placeholder="https://example.com/doc1\nhttps://example.com/doc2",
            key="create_urls_input",
        )
        create_urls = [u.strip() for u in create_urls_text.split("\n") if u.strip()]

        submitted = st.form_submit_button("Create & Upload", type="primary")

    if submitted:
        if not new_name or not COLLECTION_NAME_REGEX.match(new_name):
            st.error(
                "Invalid name. Use letters, digits, underscores or hyphens (max 255 chars)."
            )
        elif not create_files and not create_urls:
            st.error("Please provide at least one file or URL.")
        else:
            try:
                success = handle_upload_documents(new_name, create_files, create_urls)
                if success:
                    ss.collection_name = new_name
                    # Clear caches after successful creation
                    get_collection_details_cached.clear()
                    get_collections_list.clear()
                    st.cache_data.clear()
                    st.toast(
                        f"Collection '{new_name}' created successfully!",
                        icon="🎉",
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to create collection: {e}")


def render_delete_collection_tab() -> None:
    ss = st.session_state
    st.subheader("Delete Collection")
    st.caption(
        "Choose a collection to delete. "
        "You must confirm with the collection name and admin password."
    )

    all_cols = get_collections_list()

    if not all_cols:
        st.info("No collections available to delete.")
        return

    if "show_delete_dialog" not in ss:
        ss.show_delete_dialog = False
    if "delete_target" not in ss:
        ss.delete_target = None

    choice = st.selectbox(
        "Select collection to delete:",
        options=[""] + all_cols,
        format_func=lambda x: x if x else "Select…",
        key="delete_select_box",
    )

    if st.button("Delete Selected Collection", type="primary"):
        if not choice:
            st.warning("Please select a collection to delete.")
        else:
            ss.delete_target = choice
            ss.show_delete_dialog = True
            st.rerun()


@st.dialog("Confirm Delete Collection", width="small")
def confirm_delete_dialog() -> None:
    ss = st.session_state
    target = ss.delete_target

    st.error(f"You are about to permanently delete '{target}'.")
    st.caption("Re‐enter the collection name and admin password to confirm.")

    pwd = st.text_input("Admin Password:", type="password", key="del_pwd")
    confirm_text = st.text_input(
        "Type Collection Name:", placeholder=target, key="del_confirm_name"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm Delete", type="primary"):
            if pwd != ADMIN_PASSWORD:
                st.error("Incorrect password.")
            elif confirm_text != target:
                st.error("Collection name does not match.")
            else:
                try:
                    delete_collection(target)
                    st.toast(
                        f"🗑️ Collection '{target}' deleted successfully!",
                        icon="✅",
                    )
                    if ss.collection_name == target:
                        ss.collection_name = None
                    # Clear caches after successful deletion
                    get_collection_details_cached.clear()
                    get_collections_list.clear()
                    st.cache_data.clear()
                    ss.show_delete_dialog = False
                    st.rerun()
                except requests.HTTPError as e:
                    st.error(f"Deletion failed: {e.response.text}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    with c2:
        if st.button("Cancel"):
            ss.show_delete_dialog = False
            st.rerun()


def main() -> None:
    init_session_state()
    render_sidebar()
    ss = st.session_state

    st.title("Qdrant Collections")
    st.caption("Create, view, upload to, and delete your Qdrant document collections.")

    tab1, tab2, tab3 = st.tabs(
        ["Current Collection", "Create New Collection", "Delete Collection"]
    )
    with tab1:
        render_current_collection_tab()

    with tab2:
        render_create_collection_tab()

    with tab3:
        render_delete_collection_tab()

    if ss.get("show_delete_dialog", False):
        confirm_delete_dialog()

    # Refresh button at the bottom
    st.markdown("---")
    if st.button(
        "🔄 Refresh",
        type="secondary",
        help="Uploaded files are being processed in the background, so it may take a moment for new documents to appear.",
    ):
        get_collection_details_cached.clear()
        get_collections_list.clear()
        st.cache_data.clear()
        st.toast("Collections data refreshed!", icon="🔄")
        st.rerun()


if __name__ == "__main__":
    main()
