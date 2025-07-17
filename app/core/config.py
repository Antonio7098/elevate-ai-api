from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # LLM API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Vector Database Configuration
    vector_store_type: str = "chromadb"  # "pinecone" or "chromadb"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    chroma_persist_directory: str = "./chroma_db"
    
    # Embedding Service Configuration
    embedding_service_type: str = "openai"  # "openai", "google", or "local"
    openai_embedding_model: str = "text-embedding-3-small"
    google_embedding_model: str = "embedding-001"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    
    # Database
    database_url: Optional[str] = None
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# (Debug print removed) 