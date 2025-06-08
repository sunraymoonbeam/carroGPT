# utils/api_agent_react.py

import json
import os
from typing import Any, Dict, Generator

import requests

BASE_URL = os.getenv("BACKEND_SERVER_URL", "http://backend:8000")


def stream_react_chat(
    thread_id: str, collection_name: str, message: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream SSE from Carro-ReAct agent.

    Args:
        thread_id (str): Unique identifier for the chat thread.
        collection_name (str): Name of the collection to query.
        message (str): User's message to send to the agent.

    Yields:
        Dict[str, Any]: Parsed JSON chunks from the server.

    Raises:
        requests.HTTPError: If the server returns a 4xx or 5xx status code.
    """
    endpoint = f"{BASE_URL}/carro-react-agent/chat/{thread_id}"
    payload = {"collection_name": collection_name, "message": message}

    resp = requests.post(endpoint, json=payload, stream=True, timeout=30)
    resp.raise_for_status()

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data:"):
            chunk = line.replace("data:", "", 1).strip()
            try:
                yield json.loads(chunk)
            except json.JSONDecodeError:
                continue


def react_chat(thread_id: str, collection_name: str, message: str) -> Dict[str, Any]:
    """
    Single-shot call to Carro-ReAct agent.

    Args:
        thread_id (str): Unique identifier for the chat thread.
        collection_name (str): Name of the collection to query.
        message (str): User's message to send to the agent.

    Returns:
        Dict[str, Any]: Response from the agent containing the answer and any tool calls.
    """
    endpoint = f"{BASE_URL}/carro-react-agent/chat/simple/{thread_id}"
    payload = {"collection_name": collection_name, "message": message}

    resp = requests.post(endpoint, json=payload, timeout=30)
    resp.raise_for_status()

    return resp.json()
