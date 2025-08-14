from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import validator


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
    openrouter_api_key: Optional[str] = None
    
    # Vector Database Configuration
    vector_store_type: str = "pinecone"  # "pinecone" or "chromadb"
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
    
    # Feature Flags
    use_llm: bool = False  # Toggle real LLM usage in endpoints (env: USE_LLM)
    
    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    max_context_tokens: int = 8000
    batch_size: int = 100
    
    # Caching Configuration
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 1000
    
    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # Security Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    # RAG Configuration
    default_search_results: int = 5
    max_search_results: int = 20
    similarity_threshold: float = 0.7
    
    @validator('vector_store_type')
    def validate_vector_store_type(cls, v):
        if v not in ['pinecone', 'chromadb']:
            raise ValueError('vector_store_type must be either "pinecone" or "chromadb"')
        return v
    
    @validator('embedding_service_type')
    def validate_embedding_service_type(cls, v):
        if v not in ['openai', 'google', 'local']:
            raise ValueError('embedding_service_type must be either "openai", "google", or "local"')
        return v
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('similarity_threshold must be between 0.0 and 1.0')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 