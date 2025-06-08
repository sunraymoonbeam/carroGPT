# app/core/config.py
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration.

    Attributes:
        API_V1_STR (str): API version string.
        PROJECT_NAME (str): Name of the project.
        PROJECT_DESCRIPTION (str): Description of the project.
        OPENAI_API_KEY (str): OpenAI API key for accessing OpenAI services.
        QDRANT_URL (str): URL for the Qdrant vector database.
        QDRANT_KEY (str): API key for accessing Qdrant.
        TAVILY_API_KEY (str): API key for Tavily web search service.
        LLAMA_PARSE_API_KEY (str): API key for LlamaParse PDF processing service.
        ALGORITHM (str): Algorithm used for JWT encoding/decoding.
        ACCESS_TOKEN_EXPIRE_MINUTES (int): Token expiration time in minutes.
        BACKEND_CORS_ORIGINS (List[str]): List of allowed CORS origins.
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Carro Bot"
    PROJECT_DESCRIPTION: str = (
        "A customer support bot for Carro, built with FastAPI, Langgraph and Qdrant."
    )

    # OpenAI
    OPENAI_API_KEY: str

    # Quadrant
    QDRANT_URL: str
    QDRANT_KEY: str

    # Tavily for web search
    TAVILY_API_KEY: str

    # LlamaParse for PDF processing
    LLAMA_PARSE_API_KEY: str

    # JWT_SECRET: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    BACKEND_CORS_ORIGINS: List[str] = []

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
