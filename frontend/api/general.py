import os
import time

import requests
import streamlit as st

BASE_URL = os.getenv("BACKEND_SERVER_URL", "http://backend:8000")
SYSTEM_HEALTH_URL = f"{BASE_URL}/system/health"


def wait_for_backend(timeout=30, interval=2):
    """
    Wait for the backend service to be ready.

    Args:
        url: The URL to check for readiness.
        timeout: Maximum time to wait in seconds.
        interval: Time to wait between checks in seconds.

    Returns:
        True if the backend is ready, otherwise raises an error.
    """
    start = time.time()
    with st.spinner("Waiting for backend to spin up...this could take a few seconds."):
        while time.time() - start < timeout:
            try:
                r = requests.get(SYSTEM_HEALTH_URL, timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(interval)
    st.error("Backend not ready. Exiting application.")
    st.stop()
