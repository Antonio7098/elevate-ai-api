# Sprint 32: Response Compression and Optimization Service

import gzip
import json
import zlib
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import asyncio

logger = logging.getLogger(__name__)

class ResponseOptimizationService:
    """
    Service for optimizing API responses through compression, caching headers, and data optimization.
    
    Features:
    - Response compression (gzip, deflate)
    - Intelligent response caching headers
    - Data serialization optimization
    - Response size reduction strategies
    - Performance metrics tracking
    """
    
    def __init__(self):
        self.compression_stats = {
            "total_responses": 0,
            "compressed_responses": 0,
            "total_bytes_sent": 0,
            "total_bytes_saved": 0,
            "average_compression_ratio": 0.0
        }
        self.optimization_rules = {
            "min_compression_size": 1024,  # Only compress responses > 1KB
            "max_response_size": 10 * 1024 * 1024,  # 10MB limit
            "enable_etag": True,
            "default_cache_max_age": 3600,  # 1 hour default cache
        }
    
    async def optimize_response(
        self,
        request: Request,
        data: Union[Dict[str, Any], List[Any]],
        operation_type: str = "general",
        cache_max_age: Optional[int] = None,
        enable_compression: bool = True
    ) -> Response:
        """
        Optimize response with compression, caching headers, and data optimization.
        
        Args:
            request: FastAPI request object
            data: Response data to optimize
            operation_type: Type of operation for cache strategy
            cache_max_age: Custom cache max age in seconds
            enable_compression: Whether to enable compression
            
        Returns:
            Optimized FastAPI Response
        """
        try:
            # Optimize data structure
            optimized_data = await self._optimize_data_structure(data, operation_type)
            
            # Serialize to JSON
            json_content = json.dumps(
                optimized_data,
                ensure_ascii=False,
                separators=(',', ':')  # Compact JSON
            )
            
            # Track original size
            original_size = len(json_content.encode('utf-8'))
            self.compression_stats["total_responses"] += 1
            self.compression_stats["total_bytes_sent"] += original_size
            
            # Check if response is too large
            if original_size > self.optimization_rules["max_response_size"]:
                logger.warning(f"Response size {original_size} exceeds limit, truncating")
                optimized_data = await self._truncate_large_response(optimized_data)
                json_content = json.dumps(optimized_data, ensure_ascii=False, separators=(',', ':'))
                original_size = len(json_content.encode('utf-8'))
            
            # Determine compression
            should_compress = (
                enable_compression and
                original_size >= self.optimization_rules["min_compression_size"] and
                self._should_compress_for_client(request)
            )
            
            # Prepare response
            headers = {}
            content = json_content.encode('utf-8')
            
            # Apply compression
            if should_compress:
                content, encoding = await self._compress_content(content, request)
                if encoding:
                    headers["Content-Encoding"] = encoding
                    compressed_size = len(content)
                    compression_ratio = compressed_size / original_size
                    
                    # Update stats
                    self.compression_stats["compressed_responses"] += 1
                    bytes_saved = original_size - compressed_size
                    self.compression_stats["total_bytes_saved"] += bytes_saved
                    self.compression_stats["average_compression_ratio"] = (
                        compression_ratio if self.compression_stats["compressed_responses"] == 1
                        else (self.compression_stats["average_compression_ratio"] + compression_ratio) / 2
                    )
                    
                    logger.debug(f"Compressed response: {original_size} -> {compressed_size} bytes ({compression_ratio:.2%})")
            
            # Add caching headers
            cache_headers = self._generate_cache_headers(operation_type, cache_max_age)
            headers.update(cache_headers)
            
            # Add optimization metadata
            headers.update({
                "X-Original-Size": str(original_size),
                "X-Optimized": "true",
                "X-Operation-Type": operation_type
            })
            
            return Response(
                content=content,
                media_type="application/json",
                headers=headers
            )
            
        except Exception as e:
            logger.error(f"Response optimization error: {str(e)}")
            # Fallback to simple JSON response
            return JSONResponse(content=data)
    
    async def _optimize_data_structure(
        self,
        data: Union[Dict[str, Any], List[Any]],
        operation_type: str
    ) -> Union[Dict[str, Any], List[Any]]:
        """Optimize data structure for smaller payload size."""
        try:
            if isinstance(data, dict):
                return await self._optimize_dict(data, operation_type)
            elif isinstance(data, list):
                return await self._optimize_list(data, operation_type)
            else:
                return data
        except Exception as e:
            logger.error(f"Data structure optimization error: {str(e)}")
            return data
    
    async def _optimize_dict(self, data: Dict[str, Any], operation_type: str) -> Dict[str, Any]:
        """Optimize dictionary structure."""
        optimized = {}
        
        for key, value in data.items():
            # Skip null/empty values for certain operation types
            if operation_type in ["primitive_list", "question_list"] and value in [None, "", [], {}]:
                continue
            
            # Truncate very long strings
            if isinstance(value, str) and len(value) > 5000:
                optimized[key] = value[:4997] + "..."
            elif isinstance(value, dict):
                optimized[key] = await self._optimize_dict(value, operation_type)
            elif isinstance(value, list):
                optimized[key] = await self._optimize_list(value, operation_type)
            else:
                optimized[key] = value
        
        return optimized
    
    async def _optimize_list(self, data: List[Any], operation_type: str) -> List[Any]:
        """Optimize list structure."""
        optimized = []
        
        # Limit list size for certain operations
        max_items = {
            "primitive_list": 100,
            "question_list": 50,
            "search_results": 20
        }.get(operation_type, 200)
        
        limited_data = data[:max_items] if len(data) > max_items else data
        
        for item in limited_data:
            if isinstance(item, dict):
                optimized.append(await self._optimize_dict(item, operation_type))
            elif isinstance(item, list):
                optimized.append(await self._optimize_list(item, operation_type))
            else:
                optimized.append(item)
        
        # Add truncation indicator if items were limited
        if len(data) > max_items:
            optimized.append({
                "_truncated": True,
                "_total_count": len(data),
                "_showing": max_items
            })
        
        return optimized
    
    async def _truncate_large_response(self, data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """Truncate response that's too large."""
        if isinstance(data, dict):
            # Keep essential fields, truncate others
            essential_fields = ["success", "status", "error", "task_id", "primitive_id", "criterion_id"]
            truncated = {}
            
            for field in essential_fields:
                if field in data:
                    truncated[field] = data[field]
            
            truncated["_truncated"] = True
            truncated["_reason"] = "Response too large"
            return truncated
            
        elif isinstance(data, list):
            # Keep first few items
            truncated = data[:10] if len(data) > 10 else data
            truncated.append({
                "_truncated": True,
                "_reason": "Response too large",
                "_total_count": len(data)
            })
            return truncated
        
        return {"_truncated": True, "_reason": "Response too large"}
    
    def _should_compress_for_client(self, request: Request) -> bool:
        """Check if client supports compression."""
        accept_encoding = request.headers.get("accept-encoding", "")
        return "gzip" in accept_encoding or "deflate" in accept_encoding
    
    async def _compress_content(self, content: bytes, request: Request) -> tuple[bytes, Optional[str]]:
        """Compress content based on client capabilities."""
        accept_encoding = request.headers.get("accept-encoding", "").lower()
        
        if "gzip" in accept_encoding:
            try:
                compressed = gzip.compress(content, compresslevel=6)
                return compressed, "gzip"
            except Exception as e:
                logger.error(f"Gzip compression error: {str(e)}")
        
        if "deflate" in accept_encoding:
            try:
                compressed = zlib.compress(content, level=6)
                return compressed, "deflate"
            except Exception as e:
                logger.error(f"Deflate compression error: {str(e)}")
        
        return content, None
    
    def _generate_cache_headers(self, operation_type: str, cache_max_age: Optional[int]) -> Dict[str, str]:
        """Generate appropriate cache headers based on operation type."""
        headers = {}
        
        if not self.optimization_rules["enable_etag"]:
            return headers
        
        # Determine cache strategy by operation type
        cache_strategies = {
            "primitive_generation": {"max_age": 3600, "must_revalidate": True},  # 1 hour
            "question_generation": {"max_age": 1800, "must_revalidate": True},  # 30 minutes
            "mastery_criteria": {"max_age": 7200, "must_revalidate": False},    # 2 hours
            "answer_evaluation": {"max_age": 300, "must_revalidate": True},     # 5 minutes
            "blueprint_extraction": {"max_age": 14400, "must_revalidate": False}, # 4 hours
            "search_results": {"max_age": 600, "must_revalidate": True}         # 10 minutes
        }
        
        strategy = cache_strategies.get(operation_type, {
            "max_age": self.optimization_rules["default_cache_max_age"],
            "must_revalidate": True
        })
        
        max_age = cache_max_age if cache_max_age is not None else strategy["max_age"]
        
        # Build cache control header
        cache_control_parts = [f"max-age={max_age}"]
        
        if strategy.get("must_revalidate"):
            cache_control_parts.append("must-revalidate")
        
        headers["Cache-Control"] = ", ".join(cache_control_parts)
        
        # Add ETag for cache validation
        headers["ETag"] = f'"{operation_type}-{datetime.utcnow().timestamp():.0f}"'
        
        return headers
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get response optimization statistics."""
        total_responses = self.compression_stats["total_responses"]
        
        return {
            "compression_rate": (
                self.compression_stats["compressed_responses"] / total_responses
                if total_responses > 0 else 0
            ),
            "bytes_saved_percentage": (
                self.compression_stats["total_bytes_saved"] / self.compression_stats["total_bytes_sent"]
                if self.compression_stats["total_bytes_sent"] > 0 else 0
            ),
            "average_response_size": (
                self.compression_stats["total_bytes_sent"] / total_responses
                if total_responses > 0 else 0
            ),
            **self.compression_stats
        }
    
    def update_optimization_rules(self, new_rules: Dict[str, Any]):
        """Update optimization rules."""
        self.optimization_rules.update(new_rules)
        logger.info(f"Updated optimization rules: {new_rules}")


# Global response optimization service instance
response_optimizer = ResponseOptimizationService()


async def optimize_api_response(
    request: Request,
    data: Union[Dict[str, Any], List[Any]],
    operation_type: str = "general",
    cache_max_age: Optional[int] = None
) -> Response:
    """
    Helper function for optimizing API responses.
    
    Args:
        request: FastAPI request object
        data: Response data
        operation_type: Type of operation for optimization strategy
        cache_max_age: Custom cache max age in seconds
        
    Returns:
        Optimized response
    """
    return await response_optimizer.optimize_response(
        request, data, operation_type, cache_max_age
    )
