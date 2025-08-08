# Sprint 32: Intelligent Caching Service for Performance Optimization

import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class IntelligentCacheService:
    """
    Intelligent caching service for primitive generation and Core API operations.
    
    Features:
    - Content-aware cache keys based on source text and context
    - TTL management with adaptive expiration
    - Cache invalidation strategies
    - Memory and disk-based caching tiers
    - Statistics and monitoring
    """
    
    def __init__(self, max_memory_cache_size: int = 1000, default_ttl_minutes: int = 60):
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        self.max_memory_cache_size = max_memory_cache_size
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        
    async def get_cached_result(
        self, 
        cache_key: str, 
        operation_type: str = "general"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available and not expired."""
        try:
            self.cache_stats["total_requests"] += 1
            
            if cache_key in self.memory_cache:
                cached_entry = self.memory_cache[cache_key]
                
                # Check expiration
                if datetime.utcnow() < cached_entry["expires_at"]:
                    self.cache_stats["hits"] += 1
                    logger.debug(f"Cache hit for key: {cache_key[:50]}...")
                    return cached_entry["data"]
                else:
                    # Expired entry - remove it
                    del self.memory_cache[cache_key]
                    logger.debug(f"Cache expired for key: {cache_key[:50]}...")
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {str(e)}")
            return None
    
    async def store_cached_result(
        self,
        cache_key: str,
        data: Dict[str, Any],
        operation_type: str = "general",
        custom_ttl_minutes: Optional[int] = None
    ) -> bool:
        """Store result in cache with intelligent TTL and eviction."""
        try:
            # Determine TTL based on operation type and data characteristics
            ttl = self._calculate_intelligent_ttl(data, operation_type, custom_ttl_minutes)
            
            # Check cache size and evict if necessary
            await self._manage_cache_size()
            
            # Store the cached entry
            cached_entry = {
                "data": data,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + ttl,
                "operation_type": operation_type,
                "access_count": 0,
                "last_accessed": datetime.utcnow()
            }
            
            self.memory_cache[cache_key] = cached_entry
            logger.debug(f"Cached result for key: {cache_key[:50]}... (TTL: {ttl})")
            return True
            
        except Exception as e:
            logger.error(f"Cache storage error: {str(e)}")
            return False
    
    def generate_cache_key(
        self,
        source_text: str,
        operation_type: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate intelligent cache key based on content and context."""
        try:
            # Create hash components
            hash_components = [
                source_text.strip(),
                operation_type,
            ]
            
            # Add context if provided
            if context:
                # Sort context keys for consistent hashing
                context_str = json.dumps(context, sort_keys=True)
                hash_components.append(context_str)
            
            # Add user preferences if provided
            if user_preferences:
                prefs_str = json.dumps(user_preferences, sort_keys=True)
                hash_components.append(prefs_str)
            
            # Create hash
            content_hash = hashlib.sha256(
                "|".join(hash_components).encode('utf-8')
            ).hexdigest()
            
            return f"{operation_type}:{content_hash[:16]}"
            
        except Exception as e:
            logger.error(f"Cache key generation error: {str(e)}")
            # Fallback to simple hash
            fallback_hash = hashlib.md5(source_text.encode('utf-8')).hexdigest()
            return f"{operation_type}:{fallback_hash[:16]}"
    
    def _calculate_intelligent_ttl(
        self,
        data: Dict[str, Any],
        operation_type: str,
        custom_ttl_minutes: Optional[int]
    ) -> timedelta:
        """Calculate intelligent TTL based on operation type and data characteristics."""
        if custom_ttl_minutes:
            return timedelta(minutes=custom_ttl_minutes)
        
        # Different TTLs for different operation types
        ttl_mappings = {
            "primitive_generation": 120,  # 2 hours - fairly stable
            "question_generation": 90,   # 1.5 hours - moderately stable
            "mastery_criteria": 180,     # 3 hours - very stable
            "answer_evaluation": 30,     # 30 minutes - context-dependent
            "core_api_sync": 15,         # 15 minutes - frequently changing
            "blueprint_extraction": 240  # 4 hours - very stable
        }
        
        base_ttl = ttl_mappings.get(operation_type, 60)  # Default 1 hour
        
        # Adjust TTL based on data complexity
        if isinstance(data, dict):
            # Longer TTL for complex, expensive operations
            data_size = len(str(data))
            if data_size > 10000:  # Large response
                base_ttl = int(base_ttl * 1.5)
            elif data_size < 1000:  # Small response
                base_ttl = int(base_ttl * 0.7)
        
        return timedelta(minutes=base_ttl)
    
    async def _manage_cache_size(self):
        """Manage cache size with intelligent eviction."""
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Evict oldest and least accessed entries
            entries_with_score = []
            
            for key, entry in self.memory_cache.items():
                # Calculate eviction score (lower = more likely to evict)
                age_hours = (datetime.utcnow() - entry["created_at"]).total_seconds() / 3600
                access_score = entry.get("access_count", 0)
                last_access_hours = (datetime.utcnow() - entry["last_accessed"]).total_seconds() / 3600
                
                # Score favors recent access and high access count
                eviction_score = access_score / max(1, age_hours + last_access_hours)
                entries_with_score.append((eviction_score, key))
            
            # Sort by eviction score and remove lowest scoring entries
            entries_with_score.sort()
            evict_count = len(self.memory_cache) - int(self.max_memory_cache_size * 0.8)
            
            for _, key in entries_with_score[:evict_count]:
                del self.memory_cache[key]
                self.cache_stats["evictions"] += 1
            
            logger.debug(f"Evicted {evict_count} cache entries")
    
    async def invalidate_cache(
        self,
        operation_type: Optional[str] = None,
        pattern: Optional[str] = None
    ):
        """Invalidate cache entries based on operation type or pattern."""
        keys_to_remove = []
        
        for key, entry in self.memory_cache.items():
            should_remove = False
            
            if operation_type and entry.get("operation_type") == operation_type:
                should_remove = True
            elif pattern and pattern in key:
                should_remove = True
            
            if should_remove:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["total_requests"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            "hit_rate": hit_rate,
            "total_entries": len(self.memory_cache),
            "memory_usage_percentage": len(self.memory_cache) / self.max_memory_cache_size,
            **self.cache_stats
        }


# Global cache service instance
cache_service = IntelligentCacheService()


def cached_operation(
    operation_type: str,
    ttl_minutes: Optional[int] = None,
    use_user_context: bool = True
):
    """
    Decorator for caching expensive operations.
    
    Args:
        operation_type: Type of operation for cache key generation
        ttl_minutes: Custom TTL in minutes
        use_user_context: Whether to include user context in cache key
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Extract source text and context for cache key
                source_text = ""
                context = {}
                user_preferences = {}
                
                # Try to extract from common argument patterns
                if args:
                    if isinstance(args[0], str):
                        source_text = args[0]
                    elif hasattr(args[0], 'source_text'):
                        source_text = args[0].source_text
                
                # Extract from kwargs
                source_text = kwargs.get('source_text', source_text)
                context = kwargs.get('context', context)
                user_preferences = kwargs.get('user_preferences', user_preferences) if use_user_context else {}
                
                # Generate cache key
                cache_key = cache_service.generate_cache_key(
                    source_text,
                    operation_type,
                    context,
                    user_preferences
                )
                
                # Try to get cached result
                cached_result = await cache_service.get_cached_result(cache_key, operation_type)
                if cached_result:
                    logger.debug(f"Using cached result for {operation_type}")
                    return cached_result
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                if result:
                    await cache_service.store_cached_result(
                        cache_key,
                        result,
                        operation_type,
                        ttl_minutes
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Caching decorator error: {str(e)}")
                # Fall back to executing without caching
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator
