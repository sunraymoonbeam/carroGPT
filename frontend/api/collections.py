# utils/api_collections.py

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

BASE_URL = os.getenv("BACKEND_SERVER_URL", "http://backend:8000")


def list_collections() -> List[str]:
    """
    Fetch all collection names from the backend.

    Raises:
        requests.HTTPError: if the response status is 4xx/5xx.
    """
    endpoint = urljoin(BASE_URL, "/collections/")

    resp = requests.get(endpoint, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    return data.get("collections", [])


def get_collection_details(collection_name: str) -> Optional[Dict[str, Any]]:
    """Fetches detailed information for a specific collection.
    Args:
        collection_name (str): The name of the collection to fetch details for.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing collection details, or None if not found."""
    endpoint = urljoin(BASE_URL, f"/collections/{collection_name}")

    resp = requests.get(endpoint, timeout=10)
    resp.raise_for_status()

    return resp.json()


# def create_collection(collection_name: str) -> Dict[str, Any]:
#     """
#     Create a new, empty collection.

#     Raises:
#         requests.HTTPError: if the response status is 4xx/5xx.
#     """
#     endpoint = urljoin(BASE_URL, "/collections/")
#     payload = {"collection_name": collection_name}
#     resp = requests.post(endpoint, json=payload, timeout=10)
#     resp.raise_for_status()
#     return resp.json()


def upload_documents(
    collection_name: str,
    files: Optional[List[Any]] = None,
    urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Upload one or more files or URLs into a collection. The collection is auto-created
    if it doesn't already exist.

    Args:
        collection_name (str): The name of the collection to upload documents to.
        files (Optional[List[Any]]): List of file-like objects to upload.
        urls (Optional[List[str]]): List of URLs to fetch and index.

    Returns:
        Dict[str, Any]: Response containing details of the uploaded documents.

    Raises:
        requests.HTTPError: if the response status is 4xx/5xx.
    """
    if not (files or urls):
        raise ValueError("Must supply at least one file or URL to upload.")

    endpoint = urljoin(BASE_URL, f"/collections/{collection_name}/documents")
    multipart = []
    data = []

    if files:
        for f in files:
            filename = getattr(f, "name", "upload")
            multipart.append(("files", (filename, f, "application/octet-stream")))

    if urls:
        for u in urls:
            data.append(("urls", u))

    resp = requests.post(endpoint, files=multipart, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


def delete_collection(collection_name: str) -> Dict[str, Any]:
    """
    Delete an existing collection and all its contents.

    Args:
        collection_name (str): The name of the collection to delete.

    Returns:
        Dict[str, Any]: Response confirming deletion of the collection.

    Raises:
        requests.HTTPError: if the response status is 4xx/5xx.
    """
    endpoint = urljoin(BASE_URL, f"/collections/{collection_name}")

    resp = requests.delete(endpoint, timeout=10)
    resp.raise_for_status()

    return resp.json()


def search_collection(
    collection_name: str, query: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Run a semantic search in a given collection.

    Args:
        collection_name (str): The name of the collection to search.
        query (str): The search query string.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict[str, Any]]: List of search results, each containing document metadata.

    Raises:
        requests.HTTPError: if the response status is 4xx/5xx.
    """
    endpoint = urljoin(BASE_URL, f"/collections/{collection_name}/search")
    payload = {"query": query, "top_k": top_k}

    resp = requests.post(endpoint, json=payload, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    return data.get("results", [])
