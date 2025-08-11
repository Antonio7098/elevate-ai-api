"""
Intelligent caching system for premium cost optimization.
Implements sophisticated caching strategies for responses, embeddings, and context.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import asyncio

@dataclass
class CachedResponse:
    """Cached response with metadata"""
    content: str
    query_hash: str
    user_id: str
    model_used: str
    confidence: float
    cost: float
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int = 3600  # 1 hour default TTL

@dataclass
class CachedEmbedding:
    """Cached embedding with metadata"""
    text: str
    embedding: List[float]
    model: str
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int = 86400  # 24 hours default TTL

@dataclass
class CachedContext:
    """Cached context assembly with metadata"""
    context_key: str
    assembled_context: str
    user_id: str
    mode: str
    sufficiency_score: float
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int = 1800  # 30 minutes default TTL

class SemanticCache:
    """Semantic similarity-based caching"""
    
    def __init__(self):
        self.cache = {}
        self.similarity_threshold = 0.85
        self.max_cache_size = 1000
    
    async def get_similar_response(self, query: str, user_id: str) -> Optional[CachedResponse]:
        """Find semantically similar cached response"""
        try:
            query_hash = self._hash_query(query)
            
            # Check exact match first
            if query_hash in self.cache:
                cached = self.cache[query_hash]
                if self._is_valid_cache_entry(cached):
                    self._update_access(cached)
                    return cached
            
            # Check semantic similarity
            for cached_response in self.cache.values():
                if self._is_valid_cache_entry(cached_response):
                    similarity = self._calculate_semantic_similarity(query, cached_response.content)
                    if similarity >= self.similarity_threshold:
                        self._update_access(cached_response)
                        return cached_response
            
            return None
            
        except Exception as e:
            print(f"Error in semantic cache lookup: {e}")
            return None
    
    async def cache_response(self, query: str, response: CachedResponse):
        """Cache a response"""
        try:
            query_hash = self._hash_query(query)
            
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                self._evict_least_used()
            
            self.cache[query_hash] = response
            
        except Exception as e:
            print(f"Error caching response: {e}")
    
    def _hash_query(self, query: str) -> str:
        """Create hash for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries"""
        try:
            # Simple Jaccard similarity for now
            words1 = set(query1.lower().split())
            words2 = set(query2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _is_valid_cache_entry(self, cached: CachedResponse) -> bool:
        """Check if cache entry is still valid"""
        now = datetime.utcnow()
        return (now - cached.created_at).total_seconds() < cached.ttl_seconds
    
    def _update_access(self, cached: CachedResponse):
        """Update access metadata"""
        cached.accessed_at = datetime.utcnow()
        cached.access_count += 1
    
    def _evict_least_used(self):
        """Evict least used cache entries"""
        if not self.cache:
            return
        
        # Sort by access count and last access time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].accessed_at)
        )
        
        # Remove 10% of least used entries
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            del self.cache[sorted_entries[i][0]]

class ResponseCache:
    """Response caching with TTL and access tracking"""
    
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 500
    
    async def get_response(self, query_hash: str) -> Optional[CachedResponse]:
        """Get cached response by query hash"""
        try:
            if query_hash in self.cache:
                cached = self.cache[query_hash]
                if self._is_valid_cache_entry(cached):
                    self._update_access(cached)
                    return cached
            
            return None
            
        except Exception as e:
            print(f"Error getting cached response: {e}")
            return None
    
    async def cache_response(self, query_hash: str, response: CachedResponse):
        """Cache a response"""
        try:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                self._evict_oldest()
            
            self.cache[query_hash] = response
            
        except Exception as e:
            print(f"Error caching response: {e}")
    
    def _is_valid_cache_entry(self, cached: CachedResponse) -> bool:
        """Check if cache entry is still valid"""
        now = datetime.utcnow()
        return (now - cached.created_at).total_seconds() < cached.ttl_seconds
    
    def _update_access(self, cached: CachedResponse):
        """Update access metadata"""
        cached.accessed_at = datetime.utcnow()
        cached.access_count += 1
    
    def _evict_oldest(self):
        """Evict oldest cache entries"""
        if not self.cache:
            return
        
        # Sort by creation time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove 10% of oldest entries
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            del self.cache[sorted_entries[i][0]]

class EmbeddingCache:
    """Embedding caching for reuse"""
    
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 2000
    
    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding"""
        try:
            cache_key = self._create_cache_key(text, model)
            
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if self._is_valid_cache_entry(cached):
                    self._update_access(cached)
                    return cached.embedding
            
            return None
            
        except Exception as e:
            print(f"Error getting cached embedding: {e}")
            return None
    
    async def cache_embedding(self, text: str, embedding: List[float], model: str):
        """Cache an embedding"""
        try:
            cache_key = self._create_cache_key(text, model)
            
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                self._evict_least_used()
            
            cached_embedding = CachedEmbedding(
                text=text,
                embedding=embedding,
                model=model,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                access_count=1
            )
            
            self.cache[cache_key] = cached_embedding
            
        except Exception as e:
            print(f"Error caching embedding: {e}")
    
    def _create_cache_key(self, text: str, model: str) -> str:
        """Create cache key for embedding"""
        return hashlib.md5(f"{text}:{model}".encode()).hexdigest()
    
    def _is_valid_cache_entry(self, cached: CachedEmbedding) -> bool:
        """Check if cache entry is still valid"""
        now = datetime.utcnow()
        return (now - cached.created_at).total_seconds() < cached.ttl_seconds
    
    def _update_access(self, cached: CachedEmbedding):
        """Update access metadata"""
        cached.accessed_at = datetime.utcnow()
        cached.access_count += 1
    
    def _evict_least_used(self):
        """Evict least used cache entries"""
        if not self.cache:
            return
        
        # Sort by access count and last access time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].accessed_at)
        )
        
        # Remove 10% of least used entries
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            del self.cache[sorted_entries[i][0]]

class ContextCache:
    """Context assembly caching"""
    
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 300
    
    async def get_context(self, context_key: str) -> Optional[CachedContext]:
        """Get cached context assembly"""
        try:
            if context_key in self.cache:
                cached = self.cache[context_key]
                if self._is_valid_cache_entry(cached):
                    self._update_access(cached)
                    return cached
            
            return None
            
        except Exception as e:
            print(f"Error getting cached context: {e}")
            return None
    
    async def cache_context(self, context_key: str, context: CachedContext):
        """Cache a context assembly"""
        try:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                self._evict_oldest()
            
            self.cache[context_key] = context
            
        except Exception as e:
            print(f"Error caching context: {e}")
    
    def _is_valid_cache_entry(self, cached: CachedContext) -> bool:
        """Check if cache entry is still valid"""
        now = datetime.utcnow()
        return (now - cached.created_at).total_seconds() < cached.ttl_seconds
    
    def _update_access(self, cached: CachedContext):
        """Update access metadata"""
        cached.accessed_at = datetime.utcnow()
        cached.access_count += 1
    
    def _evict_oldest(self):
        """Evict oldest cache entries"""
        if not self.cache:
            return
        
        # Sort by creation time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove 10% of oldest entries
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            del self.cache[sorted_entries[i][0]]

class IntelligentCache:
    """Main intelligent caching system"""
    
    def __init__(self):
        self.semantic_cache = SemanticCache()
        self.response_cache = ResponseCache()
        self.embedding_cache = EmbeddingCache()
        self.context_cache = ContextCache()
    
    async def get_or_compute(self, query: str, user_id: str, compute_func, *args, **kwargs) -> Any:
        """Check semantic similarity before computing"""
        try:
            # First check semantic cache
            cached_response = await self.semantic_cache.get_similar_response(query, user_id)
            if cached_response:
                print(f"Semantic cache hit for query: {query[:50]}...")
                return cached_response.content
            
            # Check exact response cache
            query_hash = self.semantic_cache._hash_query(query)
            cached_response = await self.response_cache.get_response(query_hash)
            if cached_response:
                print(f"Exact cache hit for query: {query[:50]}...")
                return cached_response.content
            
            # Compute new response
            print(f"Cache miss, computing response for query: {query[:50]}...")
            response_content = await compute_func(*args, **kwargs)
            
            # Cache the response
            cached_response = CachedResponse(
                content=response_content,
                query_hash=query_hash,
                user_id=user_id,
                model_used="computed",
                confidence=0.8,  # Default confidence
                cost=0.0,  # Will be updated by cost tracker
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                access_count=1
            )
            
            await self.semantic_cache.cache_response(query, cached_response)
            await self.response_cache.cache_response(query_hash, cached_response)
            
            return response_content
            
        except Exception as e:
            print(f"Error in get_or_compute: {e}")
            # Fallback to direct computation
            return await compute_func(*args, **kwargs)
    
    async def cache_embeddings(self, text: str, embedding: List[float], model: str):
        """Cache embeddings for reuse"""
        await self.embedding_cache.cache_embedding(text, embedding, model)
    
    async def get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding"""
        return await self.embedding_cache.get_embedding(text, model)
    
    async def cache_context(self, context_key: str, context: CachedContext):
        """Cache assembled context for similar queries"""
        await self.context_cache.cache_context(context_key, context)
    
    async def get_cached_context(self, context_key: str) -> Optional[CachedContext]:
        """Get cached context"""
        return await self.context_cache.get_context(context_key)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        try:
            return {
                'semantic_cache_size': len(self.semantic_cache.cache),
                'response_cache_size': len(self.response_cache.cache),
                'embedding_cache_size': len(self.embedding_cache.cache),
                'context_cache_size': len(self.context_cache.cache),
                'total_cache_size': (
                    len(self.semantic_cache.cache) +
                    len(self.response_cache.cache) +
                    len(self.embedding_cache.cache) +
                    len(self.context_cache.cache)
                ),
                'cache_hit_rates': {
                    'semantic': self._calculate_hit_rate(self.semantic_cache.cache),
                    'response': self._calculate_hit_rate(self.response_cache.cache),
                    'embedding': self._calculate_hit_rate(self.embedding_cache.cache),
                    'context': self._calculate_hit_rate(self.context_cache.cache)
                }
            }
            
        except Exception as e:
            print(f"Error getting cache statistics: {e}")
            return {}
    
    def _calculate_hit_rate(self, cache: Dict) -> float:
        """Calculate cache hit rate"""
        try:
            if not cache:
                return 0.0
            
            total_accesses = sum(entry.access_count for entry in cache.values())
            return total_accesses / len(cache) if len(cache) > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating hit rate: {e}")
            return 0.0
    
    async def clear_expired_entries(self):
        """Clear expired cache entries"""
        try:
            # Clear expired entries from all caches
            await self._clear_expired_semantic_cache()
            await self._clear_expired_response_cache()
            await self._clear_expired_embedding_cache()
            await self._clear_expired_context_cache()
            
        except Exception as e:
            print(f"Error clearing expired entries: {e}")
    
    async def _clear_expired_semantic_cache(self):
        """Clear expired entries from semantic cache"""
        expired_keys = []
        for key, cached in self.semantic_cache.cache.items():
            if not self.semantic_cache._is_valid_cache_entry(cached):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.semantic_cache.cache[key]
    
    async def _clear_expired_response_cache(self):
        """Clear expired entries from response cache"""
        expired_keys = []
        for key, cached in self.response_cache.cache.items():
            if not self.response_cache._is_valid_cache_entry(cached):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.response_cache.cache[key]
    
    async def _clear_expired_embedding_cache(self):
        """Clear expired entries from embedding cache"""
        expired_keys = []
        for key, cached in self.embedding_cache.cache.items():
            if not self.embedding_cache._is_valid_cache_entry(cached):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.embedding_cache.cache[key]
    
    async def _clear_expired_context_cache(self):
        """Clear expired entries from context cache"""
        expired_keys = []
        for key, cached in self.context_cache.cache.items():
            if not self.context_cache._is_valid_cache_entry(cached):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.context_cache.cache[key]











