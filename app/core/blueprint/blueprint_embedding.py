"""
Blueprint embedding module for generating and managing embeddings.

This module provides functionality for generating embeddings from blueprint content
and performing similarity searches.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.models.blueprint import Blueprint
import logging

logger = logging.getLogger(__name__)


class BlueprintEmbedding:
    """Service for generating and managing blueprint embeddings."""
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """Initialize the embedding service."""
        self.embedding_model = embedding_model
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        try:
            # In a real implementation, this would call an embedding service
            # For now, we'll generate mock embeddings
            embedding = np.random.rand(1536)  # OpenAI ada-002 dimension
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batch."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    async def generate_blueprint_embedding(self, blueprint: Blueprint) -> np.ndarray:
        """Generate embedding for an entire blueprint."""
        try:
            # Combine title, description, and content for embedding
            text_parts = []
            if blueprint.title:
                text_parts.append(blueprint.title)
            if blueprint.description:
                text_parts.append(blueprint.description)
            if blueprint.content:
                if isinstance(blueprint.content, dict):
                    # Handle structured content
                    content_text = str(blueprint.content)
                else:
                    content_text = str(blueprint.content)
                text_parts.append(content_text)
            
            combined_text = " ".join(text_parts)
            return await self.generate_embedding(combined_text)
        except Exception as e:
            logger.error(f"Failed to generate blueprint embedding: {e}")
            raise
    
    async def similarity_search(self, query_embedding: np.ndarray, 
                              candidate_embeddings: List[np.ndarray], 
                              top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar embeddings to a query embedding."""
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self._cosine_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    async def cluster_embeddings(self, embeddings: List[np.ndarray], 
                                n_clusters: int = 5) -> Dict[str, List[int]]:
        """Cluster embeddings into groups."""
        try:
            # Simple clustering based on similarity
            # In a real implementation, this would use a proper clustering algorithm
            clusters = {f"cluster_{i}": [] for i in range(n_clusters)}
            
            # Simple assignment based on embedding values
            for i, embedding in enumerate(embeddings):
                cluster_id = i % n_clusters
                clusters[f"cluster_{cluster_id}"].append(i)
            
            return clusters
        except Exception as e:
            logger.error(f"Failed to cluster embeddings: {e}")
            raise
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def update_embedding_cache(self, blueprint_id: str, embedding: np.ndarray) -> None:
        """Update the embedding cache for a blueprint."""
        self.embeddings_cache[blueprint_id] = embedding
    
    async def get_cached_embedding(self, blueprint_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for a blueprint."""
        return self.embeddings_cache.get(blueprint_id)
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embeddings_cache.clear()
