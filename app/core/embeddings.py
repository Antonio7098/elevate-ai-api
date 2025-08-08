"""
Embedding service for the RAG system.

This module provides a unified interface for text embedding operations,
supporting OpenAI, Google, and local sentence-transformers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import openai
import google.generativeai as genai
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding operations."""
    pass


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding service."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.client: Optional[openai.OpenAI] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._dimension = 1536  # Default for text-embedding-3-small
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("OpenAI embedding service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise EmbeddingError(f"OpenAI initialization failed: {e}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string using OpenAI."""
        if not self.client:
            raise EmbeddingError("OpenAI client not initialized")
        
        try:
            def _embed():
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            
            embedding = await asyncio.get_event_loop().run_in_executor(
                self._executor, _embed
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text with OpenAI: {e}")
            raise EmbeddingError(f"OpenAI embedding failed: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings using OpenAI."""
        if not self.client:
            raise EmbeddingError("OpenAI client not initialized")
        
        try:
            def _embed_batch():
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [data.embedding for data in response.data]
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self._executor, _embed_batch
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch with OpenAI: {e}")
            raise EmbeddingError(f"OpenAI batch embedding failed: {e}")
    
    def get_dimension(self) -> int:
        """Get the dimension of OpenAI embeddings."""
        return self._dimension


class GoogleEmbeddingService(EmbeddingService):
    """Google embedding service implementation."""
    
    def __init__(self, api_key: str, model: str = "embedding-001"):
        self.api_key = api_key
        self.model = model
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._dimension = 768  # Default for embedding-001
    
    async def initialize(self) -> None:
        """Initialize Google AI client."""
        try:
            genai.configure(api_key=self.api_key)
            logger.info("Google embedding service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI client: {e}")
            raise EmbeddingError(f"Google AI initialization failed: {e}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string using Google AI."""
        try:
            def _embed():
                # Use the embed_content function directly
                embedding = genai.embed_content(
                    model=f"models/{self.model}",
                    content=text
                )
                return embedding['embedding']
            
            embedding = await asyncio.get_event_loop().run_in_executor(
                self._executor, _embed
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text with Google AI: {e}")
            raise EmbeddingError(f"Google AI embedding failed: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings using Google AI."""
        try:
            def _embed_batch():
                embeddings = []
                for text in texts:
                    embedding = genai.embed_content(
                        model=f"models/{self.model}",
                        content=text
                    )
                    embeddings.append(embedding['embedding'])
                return embeddings
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self._executor, _embed_batch
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch with Google AI: {e}")
            raise EmbeddingError(f"Google AI batch embedding failed: {e}")
    
    def get_dimension(self) -> int:
        """Get the dimension of Google AI embeddings."""
        return self._dimension
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text (alias for embed_text)."""
        return await self.embed_text(text)


class LocalEmbeddingService(EmbeddingService):
    """Local sentence-transformers embedding service implementation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=2)  # Lower for local model
        # Default fallback dimension (align with common prod index size); overridable via env
        import os
        try:
            self._dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        except Exception:
            self._dimension = 768
    
    async def initialize(self) -> None:
        """Initialize sentence-transformers model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            # Fall back to deterministic local embedding without the package
            logger.warning("sentence-transformers not installed; using fallback local embeddings")
            self.model = None
            self._dimension = 384
            return
        
        try:
            def _load_model():
                return SentenceTransformer(self.model_name)
            
            self.model = await asyncio.get_event_loop().run_in_executor(
                self._executor, _load_model
            )
            # If model failed to load, keep fallback mode
            if self.model is not None:
                self._dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Local embedding service initialized with model: {self.model_name}")
            else:
                logger.warning("Failed to load sentence-transformers model; using fallback embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers model: {e}")
            # Do not raise; enable fallback embeddings
            self.model = None
            self._dimension = 384
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string using sentence-transformers."""
        if not self.model:
            # Fallback deterministic embedding
            return self._fallback_embed(text)
        
        try:
            def _embed():
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            embedding = await asyncio.get_event_loop().run_in_executor(
                self._executor, _embed
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text with sentence-transformers: {e}")
            # Fallback deterministic embedding on error
            return self._fallback_embed(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings using sentence-transformers."""
        if not self.model:
            # Fallback deterministic embeddings
            return [self._fallback_embed(text) for text in texts]
        
        try:
            def _embed_batch():
                embeddings = self.model.encode(texts, convert_to_tensor=False)
                if hasattr(embeddings, 'tolist'):
                    return embeddings.tolist()
                return embeddings
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self._executor, _embed_batch
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch with sentence-transformers: {e}")
            # Fallback on error
            return [self._fallback_embed(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get the dimension of sentence-transformers embeddings."""
        return self._dimension
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text (alias for embed_text)."""
        return await self.embed_text(text)

    def _fallback_embed(self, text: str) -> List[float]:
        """Generate a deterministic fallback embedding for offline/dev.

        Uses a seeded pseudo-random vector based on the input text hash.
        """
        # Choose dimension (keep consistent with default local model)
        dim = self._dimension or 384
        # Stable seed from text
        seed = (abs(hash(text)) % (2**32))
        rng = np.random.default_rng(seed)
        vec = rng.normal(loc=0.0, scale=1.0, size=dim)
        # Normalize to unit length to mimic embedding scale
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()


# Factory function to create the appropriate embedding service
def create_embedding_service(
    service_type: str = "openai",
    **kwargs
) -> EmbeddingService:
    """Create an embedding service instance based on configuration."""
    if service_type.lower() == "openai":
        return OpenAIEmbeddingService(
            api_key=kwargs["api_key"],
            model=kwargs.get("model", "text-embedding-3-small")
        )
    elif service_type.lower() == "google":
        return GoogleEmbeddingService(
            api_key=kwargs["api_key"],
            model=kwargs.get("model", "embedding-001")
        )
    elif service_type.lower() == "local":
        return LocalEmbeddingService(
            model_name=kwargs.get("model_name", "all-MiniLM-L6-v2")
        )
    else:
        raise ValueError(f"Unsupported embedding service type: {service_type}")


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        raise EmbeddingError("Embedding service not initialized. Call initialize_embedding_service() first.")
    return _embedding_service


async def initialize_embedding_service(
    service_type: str = "openai",
    **kwargs
) -> None:
    """Initialize the global embedding service."""
    global _embedding_service
    _embedding_service = create_embedding_service(service_type, **kwargs)
    await _embedding_service.initialize()
    logger.info(f"Global embedding service initialized: {service_type}") 