"""
Configuration management for Flexcyz
Loads from environment variables with sensible defaults
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Flexcyz IDE"
    DEBUG: bool = True
    PROJECTS_DIR: str = "./projects"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Redis (for state/memory)
    REDIS_URL: str = "redis://localhost:6379"
    
    # LLM Providers (at least one required)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_HOST: str = "http://localhost:11434"
    
    # Search APIs
    TAVILY_API_KEY: Optional[str] = None
    EXA_API_KEY: Optional[str] = None
    
    # GitHub
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None
    
    # Sandbox
    DOCKER_SOCKET: str = "/var/run/docker.sock"
    SANDBOX_MEMORY_LIMIT: str = "512m"
    SANDBOX_CPU_LIMIT: float = 1.0
    
    # Agent Configuration
    MAX_CONCURRENT_AGENTS: int = 5
    DEFAULT_LLM_MODEL: str = "gpt-4-turbo-preview"
    FALLBACK_MODEL: str = "groq/llama-3.1-70b"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Ensure projects directory exists
os.makedirs(get_settings().PROJECTS_DIR, exist_ok=True)
