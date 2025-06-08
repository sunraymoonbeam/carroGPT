import re
import tempfile
from datetime import datetime
from typing import List, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from llama_parse import LlamaParse  # LlamaParse API client

from ..core.config import get_settings
from .exceptions import DocumentProcessingError


# Ren Hwa: wanted to try Unstructured library, but it is very heavy and not recommended for production use
# Stumbled upon LLAMA Parse API which is apparently really good for parsing PDFs
# However, LlamaParse is a paid service, and it does not provide chunking out of the box,
# so we might need to use LangChain's RecursiveCharacterTextSplitter for that.
def process_pdf_llamaparse(filename: str, raw_bytes: bytes) -> List[Tuple[str, dict]]:
    """
    Write raw_bytes to temp PDF, parse via LlamaParse API, clean pages and return (text_chunk, payload).

    Args:
        filename (str): Name of the PDF file.
        raw_bytes (bytes): Raw bytes of the PDF file.

    Returns:
        List[Tuple[str, dict]]: List of tuples containing cleaned text and metadata payloads.

    Raises:
        DocumentProcessingError: If any error occurs during processing.
    """
    settings = get_settings()

    # Write pdf bytes to temp PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        tmp_path = tmp.name

    # Parse via LlamaParse, each document is one page
    try:
        parser = LlamaParse(
            api_key=settings.LLAMA_PARSE_API_KEY,
            result_type="text",
            verbose=False,
        )
        documents = parser.load_data(tmp_path)
    except Exception as e:
        raise DocumentProcessingError(f"LlamaParse failed for '{filename}': {e}")

    # Ren Hwa: Please experiment with this
    # Clean & build payloads
    def _clean_page(text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            line = re.sub(r" {2,}", " ", line).strip()
            cleaned.append(line)
        return "\n".join(cleaned)

    # Build payloads
    processed: List[Tuple[str, dict]] = []
    for page_num, doc in enumerate(documents, start=1):
        raw_text = getattr(doc, "text", "") or ""
        page_text = _clean_page(raw_text)
        payload = {
            "text": page_text,
            "source": filename,
            "file_type": "pdf",
            "page_number": page_num,
            "upload_date": datetime.now().isoformat(),
        }
        processed.append((page_text, payload))

    return processed


def process_pdf(filename: str, raw_bytes: bytes) -> List[Tuple[str, dict]]:
    """
    Write raw_bytes to temp PDF, load & split via PyPDFLoader, return list of (text_chunk, payload).

    Args:
        filename (str): Name of the PDF file.
        raw_bytes (bytes): Raw bytes of the PDF file.

    Returns:
        List[Tuple[str, dict]]: List of tuples containing text chunks and metadata payloads.

    Raises:
        DocumentProcessingError: If any error occurs during processing.
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        tmp_path = tmp.name

    # Load and split
    try:
        loader = PyPDFLoader(tmp_path)
        docs: List[Document] = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", ". ", "\n", " ", ""],  # Custom separators
        )
        split_docs = splitter.split_documents(docs)
    except Exception as e:
        raise DocumentProcessingError(f"PDF processing failed for '{filename}': {e}")

    # Build payloads
    processed: List[Tuple[str, dict]] = []
    for chunk in split_docs:
        text = chunk.page_content
        meta = chunk.metadata or {}
        page_number = meta.get("page") or meta.get("page_number")
        payload = {
            "text": text,
            "source": filename,
            "file_type": "pdf",
            "page_number": page_number,
            "upload_date": datetime.now().isoformat(),
        }
        processed.append((text, payload))

    return processed


def process_docx(filename: str, raw_bytes: bytes) -> List[Tuple[str, dict]]:
    """
    Write raw_bytes to temp DOCX, load & split via Docx2txtLoader, return list of (text_chunk, payload).
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        tmp_path = tmp.name

    # Load and split
    try:
        loader = Docx2txtLoader(tmp_path)
        docs: List[Document] = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        split_docs = splitter.split_documents(docs)
    except Exception as e:
        raise DocumentProcessingError(f"DOCX processing failed for '{filename}': {e}")

    # Build payloads
    processed: List[Tuple[str, dict]] = []
    for chunk in split_docs:
        text = chunk.page_content
        payload = {
            "text": text,
            "source": filename,
            "file_type": "docx",
            "upload_date": datetime.now().isoformat(),
        }
        processed.append((text, payload))

    return processed


def process_url(url: str) -> List[Tuple[str, dict]]:
    """
    Fetch via WebBaseLoader, split and return (text, payload).
    """
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        split_docs = splitter.split_documents(docs)
    except Exception as e:
        raise DocumentProcessingError(f"URL processing failed for '{url}': {e}")

    processed: List[Tuple[str, dict]] = []
    for chunk in split_docs:
        text = chunk.page_content
        payload = {
            "text": text,
            "source": url,
            "file_type": "url",
            "upload_date": datetime.now().isoformat(),
        }
        processed.append((text, payload))

    return processed


# # -------------------------------------------------------------------------
# # PDF via PyPDFLoader + LangChain splitter
# # -------------------------------------------------------------------------
# def process_pdf(upload_file: UploadFile) -> List[Tuple[str, dict]]:
#     filename = upload_file.filename

#     # Read bytes
#     try:
#         raw_bytes = upload_file.file.read()
#     except Exception as e:
#         raise DocumentProcessingError(f"Failed to read PDF bytes for '{filename}': {e}")

#     # Load pages
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#             tmp.write(raw_bytes)
#             tmp.flush()
#             loader = PyPDFLoader(tmp.name)
#             docs: List[Document] = loader.load()
#     except Exception as e:
#         raise DocumentProcessingError(f"PyPDFLoader failed for '{filename}': {e}")

#     # Split into chunks
#     try:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=150,
#             separators=["\n\n", ". ", "\n", " ", ""],
#         )
#         split_docs = splitter.split_documents(docs)
#     except Exception as e:
#         raise DocumentProcessingError(
#             f"Failed to split PDF '{filename}' into chunks: {e}"
#         )

#     # Build payloads
#     processed: List[Tuple[str, dict]] = []
#     for chunk in split_docs:
#         text = chunk.page_content
#         meta = chunk.metadata or {}
#         page_number = meta.get("page") or meta.get("page_number")

#         payload = {
#             "text": text,
#             "source": filename,
#             "file_type": "pdf",
#             "page_number": page_number,
#             "upload_date": datetime.now().isoformat(),
#         }
#         processed.append((text, payload))

#     return processed


# # -------------------------------------------------------------------------
# # LlamaParse (API) variant
# # -------------------------------------------------------------------------
# def process_pdf_llamaparse(upload_file: UploadFile) -> List[Tuple[str, dict]]:
#     """
#     Process a PDF via LlamaParse, clean each page's text (collapse multiple spaces
#     but keep line breaks), and return List of (page_text, payload) tuples.
#     """
#     filename = upload_file.filename
#     settings = get_settings()

#     # Read uploaded bytes
#     try:
#         raw_bytes = upload_file.file.read()
#     except Exception as e:
#         raise DocumentProcessingError(f"Failed to read PDF bytes for '{filename}': {e}")

#     # Write to temp file & parse via LlamaParse
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#             tmp.write(raw_bytes)
#             tmp.flush()
#             tmp_path = tmp.name

#         parser = LlamaParse(
#             api_key=settings.LLAMA_PARSE_API_KEY,
#             result_type="text",
#             verbose=False,
#         )
#         # returns one document per page
#         documents = parser.load_data(tmp_path)
#     except Exception as e:
#         raise DocumentProcessingError(f"LlamaParse failed for '{filename}': {e}")

#     # Helper to clean but preserve lines
#     def _clean_page(text: str) -> str:
#         lines = text.splitlines()
#         cleaned = []
#         for line in lines:
#             # collapse 2+ spaces into one
#             line = re.sub(r" {2,}", " ", line)
#             # strip leading/trailing spaces
#             cleaned.append(line.strip())
#         return "\n".join(cleaned)

#     # Build payloads
#     processed: List[Tuple[str, dict]] = []
#     for page_num, doc in enumerate(documents, start=1):
#         raw_text = getattr(doc, "text", "") or ""
#         page_text = _clean_page(raw_text)

#         payload = {
#             "text": page_text,
#             "source": filename,
#             "file_type": "pdf",
#             "page_number": page_num,
#             "upload_date": datetime.now().isoformat(),
#         }
#         processed.append((page_text, payload))

#     return processed


# # -------------------------------------------------------------------------
# # DOCX via Docx2txtLoader + LangChain splitter
# # -------------------------------------------------------------------------
# def process_docx(upload_file: UploadFile) -> List[Tuple[str, dict]]:
#     filename = upload_file.filename

#     # Read bytes
#     try:
#         raw_bytes = upload_file.file.read()
#     except Exception as e:
#         raise DocumentProcessingError(
#             f"Failed to read DOCX bytes for '{filename}': {e}"
#         )

#     # Load document
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
#             tmp.write(raw_bytes)
#             tmp.flush()
#             loader = Docx2txtLoader(tmp.name)
#             docs: List[Document] = loader.load()
#     except Exception as e:
#         raise DocumentProcessingError(f"Docx2txtLoader failed for '{filename}': {e}")

#     # Split into chunks
#     try:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=150,
#             separators=["\n\n", ". ", "\n", " ", ""],
#         )
#         split_docs = splitter.split_documents(docs)
#     except Exception as e:
#         raise DocumentProcessingError(
#             f"Failed to split DOCX '{filename}' into chunks: {e}"
#         )

#     # Build payloads
#     processed: List[Tuple[str, dict]] = []
#     for chunk in split_docs:
#         text = chunk.page_content

#         payload = {
#             "text": text,
#             "source": filename,
#             "file_type": "docx",
#             "upload_date": datetime.now().isoformat(),
#         }
#         processed.append((text, payload))

#     return processed


# def process_url(url: str) -> List[Tuple[str, dict]]:
#     """
#     Fetch a web page via LangChain’s WebBaseLoader, split into chunks,
#     and prepare (text, payload) pairs.
#     """
#     try:
#         loader = WebBaseLoader([url])
#         docs = loader.load()
#     except Exception as e:
#         raise DocumentProcessingError(f"WebBaseLoader failed for URL '{url}': {e}")

#     try:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=150,
#             separators=["\n\n", ". ", "\n", " ", ""],
#         )
#         split_docs = splitter.split_documents(docs)
#         chunk_texts = [chunk.page_content for chunk in split_docs]
#     except Exception as e:
#         raise DocumentProcessingError(f"Failed to split URL '{url}' into chunks: {e}")

#     processed: List[Tuple[str, dict]] = []
#     for chunk in chunk_texts:
#         payload = {
#             "text": chunk,
#             "source": url,
#             "file_type": "url",
#             "upload_date": datetime.now().isoformat(),
#         }
#         processed.append((chunk, payload))

#     return processed


# Ren Hwa: Unstructured library is VERY heavy. Not recommended for production use.
# Ren Hwa: We do not use it for now, but leaving the code here for future reference.
# from unstructured.partition.pdf import partition_pdf  # partition into elements & chunks
# from unstructured.chunking.title import chunk_by_title  # chunk-by-title strategy
# def process_pdf_unstructured(upload_file: UploadFile) -> List[Tuple[str, dict]]:
#     """
#     Process a PDF via Unstructured’s 'by_title' chunking strategy.

#     Returns:
#         List of (chunk_text, payload_dict) tuples.
#     """
#     filename = upload_file.filename

#     # Read bytes from upload
#     try:
#         raw_bytes = upload_file.file.read()
#     except Exception as e:
#         raise DocumentProcessingError(f"Failed to read PDF bytes for '{filename}': {e}")

#     # Write to temp file
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#             tmp.write(raw_bytes)
#             tmp.flush()
#             tmp_path = tmp.name

#         # Partition & chunk in one call
#         elements = partition_pdf(
#             filename=tmp_path,
#             extract_images_in_pdf=True,
#             infer_table_structure=True,
#             chunking_strategy="by_title",
#             max_characters=800,
#             new_after_n_chars=750,
#             combine_text_under_n_chars=0,
#             multipage_sections=True,
#         )
#     except Exception as e:
#         raise DocumentProcessingError(
#             f"Unstructured partitioning failed for '{filename}': {e}"
#         )

#     processed: List[Tuple[str, dict]] = []
#     for idx, element in enumerate(elements):
#         text = element.text
#         meta = getattr(element, "metadata", {})
#         payload = {
#             "source": filename,
#             "source_type": "file",
#             "file_type": "pdf",
#             "chunk_index": idx,
#             "text": text,
#             "element_type": type(element).__name__,
#             "page_number": meta.get("page_number"),
#             "upload_date": datetime.now().isoformat(),
#         }
#         processed.append((text, payload))

#     return processed
