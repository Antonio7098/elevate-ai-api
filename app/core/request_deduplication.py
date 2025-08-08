# Sprint 32: Request Deduplication and Result Reuse Service

import asyncio
import hashlib
import json
from typing import Dict, Any, Optional, Set, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PendingRequest:
    """Represents a pending request with its future and metadata."""
    future: asyncio.Future
    created_at: datetime
    request_count: int
    original_args: tuple
    original_kwargs: dict

class RequestDeduplicationService:
    """
    Service for deduplicating concurrent requests and reusing results.
    
    Features:
    - Detects duplicate requests in flight
    - Reuses results from ongoing operations
    - Manages request batching and coalescing
    - Prevents redundant expensive operations
    """
    
    def __init__(self, max_pending_time_minutes: int = 10):
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.completed_requests: Dict[str, Any] = {}
        self.request_stats = {
            "total_requests": 0,
            "deduplicated_requests": 0,
            "batched_requests": 0,
            "cache_hits": 0
        }
        self.max_pending_time = timedelta(minutes=max_pending_time_minutes)
        self._cleanup_task = None
        # Don't start cleanup task during init - will start when needed
    
    def _start_cleanup_task(self):
        """Start background task for cleaning up expired pending requests."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        except RuntimeError:
            # No event loop running, cleanup task will start when needed
            pass
    
    async def _cleanup_expired_requests(self):
        """Background task to clean up expired pending requests."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                expired_keys = []
                
                for request_key, pending_request in self.pending_requests.items():
                    if current_time - pending_request.created_at > self.max_pending_time:
                        expired_keys.append(request_key)
                        
                        # Cancel the future if it's not done
                        if not pending_request.future.done():
                            pending_request.future.cancel()
                
                # Remove expired requests
                for key in expired_keys:
                    del self.pending_requests[key]
                    logger.debug(f"Cleaned up expired pending request: {key[:50]}...")
                
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")
    
    def generate_request_key(
        self,
        operation_name: str,
        args: tuple,
        kwargs: dict,
        include_dynamic_fields: bool = False
    ) -> str:
        """Generate a unique key for request deduplication."""
        try:
            # Create a stable representation of the request
            key_components = [operation_name]
            
            # Add args
            for arg in args:
                if hasattr(arg, '__dict__'):
                    # For objects, use their dict representation
                    arg_dict = arg.__dict__ if not include_dynamic_fields else arg.__dict__
                    # Remove timestamp-like fields for better deduplication
                    if not include_dynamic_fields:
                        filtered_dict = {
                            k: v for k, v in arg_dict.items() 
                            if not any(time_field in k.lower() for time_field in ['timestamp', 'created_at', 'updated_at'])
                        }
                        key_components.append(json.dumps(filtered_dict, sort_keys=True))
                    else:
                        key_components.append(json.dumps(arg_dict, sort_keys=True))
                else:
                    key_components.append(str(arg))
            
            # Add kwargs (excluding dynamic fields)
            if kwargs:
                filtered_kwargs = kwargs.copy()
                if not include_dynamic_fields:
                    # Remove timestamp and session-specific fields
                    dynamic_fields = ['timestamp', 'session_id', 'request_id', 'user_session']
                    for field in dynamic_fields:
                        filtered_kwargs.pop(field, None)
                
                key_components.append(json.dumps(filtered_kwargs, sort_keys=True))
            
            # Generate hash
            request_signature = "|".join(key_components)
            request_hash = hashlib.sha256(request_signature.encode('utf-8')).hexdigest()
            
            return f"{operation_name}:{request_hash[:16]}"
            
        except Exception as e:
            logger.error(f"Request key generation error: {str(e)}")
            # Fallback to simple hash
            fallback_content = f"{operation_name}:{str(args)}:{str(kwargs)}"
            fallback_hash = hashlib.md5(fallback_content.encode('utf-8')).hexdigest()
            return f"{operation_name}:{fallback_hash[:16]}"
    
    async def deduplicate_request(
        self,
        operation_name: str,
        operation_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with request deduplication.
        
        If an identical request is already in progress, wait for its result.
        Otherwise, execute the operation and share the result with any duplicate requests.
        """
        self.request_stats["total_requests"] += 1
        
        # Generate request key
        request_key = self.generate_request_key(operation_name, args, kwargs)
        
        # Check if request is already pending
        if request_key in self.pending_requests:
            pending_request = self.pending_requests[request_key]
            pending_request.request_count += 1
            self.request_stats["deduplicated_requests"] += 1
            
            logger.debug(f"Deduplicating request: {request_key[:50]}... (count: {pending_request.request_count})")
            
            try:
                # Wait for the existing request to complete
                result = await pending_request.future
                return result
            except Exception as e:
                logger.error(f"Deduplicated request failed: {str(e)}")
                raise e
        
        # Create new pending request
        future = asyncio.Future()
        pending_request = PendingRequest(
            future=future,
            created_at=datetime.utcnow(),
            request_count=1,
            original_args=args,
            original_kwargs=kwargs
        )
        
        self.pending_requests[request_key] = pending_request
        
        try:
            # Execute the operation
            logger.debug(f"Executing new request: {request_key[:50]}...")
            result = await operation_func(*args, **kwargs)
            
            # Set the result for all waiting requests
            if not future.done():
                future.set_result(result)
            
            # Store in completed requests for short-term reuse
            self.completed_requests[request_key] = {
                "result": result,
                "completed_at": datetime.utcnow(),
                "request_count": pending_request.request_count
            }
            
            logger.debug(f"Completed request: {request_key[:50]}... (served {pending_request.request_count} requests)")
            
            return result
            
        except Exception as e:
            # Set the exception for all waiting requests
            if not future.done():
                future.set_exception(e)
            
            logger.error(f"Request execution failed: {request_key[:50]}... - {str(e)}")
            raise e
            
        finally:
            # Remove from pending requests
            self.pending_requests.pop(request_key, None)
    
    async def batch_requests(
        self,
        operation_name: str,
        batch_operation_func: Callable[[list], Awaitable[list]],
        individual_requests: list,
        batch_size: int = 10,
        batch_timeout_seconds: float = 1.0
    ) -> list:
        """
        Batch multiple similar requests for efficient processing.
        
        Collects requests over a short time window and processes them in batches.
        """
        self.request_stats["batched_requests"] += len(individual_requests)
        
        # Simple batching implementation - can be enhanced with more sophisticated batching logic
        results = []
        
        for i in range(0, len(individual_requests), batch_size):
            batch = individual_requests[i:i + batch_size]
            
            try:
                batch_results = await batch_operation_func(batch)
                results.extend(batch_results)
                
                logger.debug(f"Processed batch of {len(batch)} {operation_name} requests")
                
            except Exception as e:
                logger.error(f"Batch processing failed for {operation_name}: {str(e)}")
                # Fallback to individual processing
                for request in batch:
                    try:
                        # This would need to be implemented per use case
                        individual_result = await self._process_individual_fallback(request)
                        results.append(individual_result)
                    except Exception as individual_error:
                        logger.error(f"Individual fallback failed: {str(individual_error)}")
                        results.append(None)  # or appropriate error result
        
        return results
    
    async def _process_individual_fallback(self, request: Any) -> Any:
        """Fallback processing for individual requests when batch fails."""
        # This should be implemented based on specific use cases
        # For now, return None to indicate processing failure
        logger.warning("Individual fallback processing not implemented")
        return None
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get request deduplication statistics."""
        total_requests = self.request_stats["total_requests"]
        deduplication_rate = (
            self.request_stats["deduplicated_requests"] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            "deduplication_rate": deduplication_rate,
            "pending_requests": len(self.pending_requests),
            "completed_requests_cache": len(self.completed_requests),
            **self.request_stats
        }
    
    async def clear_completed_cache(self, max_age_minutes: int = 30):
        """Clear old completed requests from cache."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        keys_to_remove = []
        
        for key, completed_data in self.completed_requests.items():
            if completed_data["completed_at"] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.completed_requests[key]
        
        logger.debug(f"Cleared {len(keys_to_remove)} old completed requests")


# Global deduplication service instance
deduplication_service = RequestDeduplicationService()


def deduplicated_operation(operation_name: str):
    """
    Decorator for automatically deduplicating requests to expensive operations.
    
    Args:
        operation_name: Name of the operation for identification and stats
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await deduplication_service.deduplicate_request(
                operation_name,
                func,
                *args,
                **kwargs
            )
        return wrapper
    return decorator
