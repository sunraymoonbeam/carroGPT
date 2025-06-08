# src/app/main.py

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from .agent_crag.agent.graph import create_graph
from .agent_crag.router import router as agent_router
from .agent_react.agent.graph import create_graph as create_react_graph
from .agent_react.router import router as react_router
from .core.config import get_settings
from .db.router import get_qdrant_client
from .db.router import router as db_router
from .system.router import router as system_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: manage startup and shutdown tasks.

    Startup:
        - Test connectivity to Qdrant (raises on failure).
        - Placeholder for other startup tasks (caches, etc).

    Shutdown:
        - Placeholder for cleanup tasks (closing connections, etc).
    """
    # Startup
    logger.info("Starting application…")

    # Acquire an AsyncQdrantClient via dependency function and attempt connection
    logger.info("Testing Qdrant connectivity…")
    try:
        qdrant_client = await get_qdrant_client()
        await qdrant_client.get_collections()
        logger.success("Connected to Qdrant successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise

    # Initialize the LangGraph agent using the lifespan event
    logger.info("Initializing CRAG Agent…")
    try:
        app.state.graph = create_graph()
        logger.success("CRAG agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CRAG agent: {e}")
        raise

    # Initialize the ReAct agent
    logger.info("Initializing ReAct agent…")
    try:
        app.state.react_graph = create_react_graph()
        logger.success("ReAct agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ReAct agent: {e}")
        raise

    # Yield control back to FastAPI; endpoints are now available
    yield

    # Shutdown
    logger.info("Application shutdown: no additional cleanup required")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    root_path=settings.API_V1_STR,
)

# Include routers
app.include_router(db_router)
app.include_router(system_router)
app.include_router(agent_router)
app.include_router(react_router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # module path for this app
        host="0.0.0.0",
        port=8000,
        reload=True,  # restart on code changes
    )
