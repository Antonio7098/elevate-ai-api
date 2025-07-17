"""
Service initialization and management for the RAG system.

This module handles the initialization of vector store and embedding services
on application startup.
"""

import logging
from typing import Optional
from app.core.config import settings
from app.core.vector_store import create_vector_store, VectorStore
from app.core.embeddings import initialize_embedding_service, get_embedding_service

logger = logging.getLogger(__name__)

# Global service instances
_vector_store: Optional[VectorStore] = None


async def initialize_services() -> None:
    """Initialize all RAG services on application startup."""
    global _vector_store
    
    try:
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        if settings.embedding_service_type == "openai":
            await initialize_embedding_service(
                service_type="openai",
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model
            )
        elif settings.embedding_service_type == "google":
            await initialize_embedding_service(
                service_type="google",
                api_key=settings.google_api_key,
                model=settings.google_embedding_model
            )
        elif settings.embedding_service_type == "local":
            await initialize_embedding_service(
                service_type="local",
                model_name=settings.local_embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding service type: {settings.embedding_service_type}")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        if settings.vector_store_type == "pinecone":
            if not settings.pinecone_api_key or not settings.pinecone_environment:
                raise ValueError("Pinecone API key and environment are required")
            
            _vector_store = create_vector_store(
                store_type="pinecone",
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
        elif settings.vector_store_type == "chromadb":
            _vector_store = create_vector_store(
                store_type="chromadb",
                persist_directory=settings.chroma_persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store type: {settings.vector_store_type}")
        
        await _vector_store.initialize()
        
        logger.info("All RAG services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG services: {e}")
        raise


async def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized. Call initialize_services() first.")
    return _vector_store


async def shutdown_services() -> None:
    """Shutdown all RAG services."""
    global _vector_store
    
    logger.info("Shutting down RAG services...")
    
    # Clean up vector store
    if _vector_store:
        # Note: Most vector stores don't require explicit shutdown
        # but we can add cleanup logic here if needed
        _vector_store = None
    
    logger.info("RAG services shutdown complete") 