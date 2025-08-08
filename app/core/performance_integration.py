# Sprint 32: Performance Optimization Integration Module

from typing import Dict, Any, Optional, Callable, Awaitable
from fastapi import Request, Response
import logging

# Import all performance optimization services
from app.core.caching_service import cache_service, cached_operation
from app.core.request_deduplication import deduplication_service, deduplicated_operation
from app.core.async_processing import async_service, async_operation
from app.core.connection_pooling import connection_pool_manager, resource_manager
from app.core.response_optimization import response_optimizer, optimize_api_response

logger = logging.getLogger(__name__)

class PerformanceIntegrationService:
    """
    Centralized service for integrating all performance optimizations.
    
    Features:
    - Unified performance optimization application
    - Monitoring and metrics collection
    - Configuration management
    - Health checks and diagnostics
    """
    
    def __init__(self):
        self.enabled_optimizations = {
            "caching": True,
            "deduplication": True,
            "async_processing": True,
            "connection_pooling": True,
            "response_optimization": True
        }
        self.performance_metrics = {
            "total_requests_processed": 0,
            "optimizations_applied": 0,
            "performance_gains": {}
        }
    
    async def apply_all_optimizations(
        self,
        request: Request,
        operation_name: str,
        operation_func: Callable[..., Awaitable[Any]],
        *args,
        operation_type: str = "general",
        force_async: bool = False,
        cache_ttl_minutes: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Apply all enabled performance optimizations to an operation.
        
        Args:
            request: FastAPI request object
            operation_name: Name of the operation
            operation_func: Function to execute with optimizations
            operation_type: Type of operation for optimization strategies
            force_async: Whether to force async processing
            cache_ttl_minutes: Custom cache TTL
            
        Returns:
            Optimized operation result
        """
        self.performance_metrics["total_requests_processed"] += 1
        applied_optimizations = []
        
        try:
            # Create optimized operation function
            optimized_func = operation_func
            
            # Apply caching if enabled
            if self.enabled_optimizations["caching"]:
                optimized_func = cached_operation(
                    operation_type=operation_type,
                    ttl_minutes=cache_ttl_minutes
                )(optimized_func)
                applied_optimizations.append("caching")
            
            # Apply request deduplication if enabled
            if self.enabled_optimizations["deduplication"]:
                optimized_func = deduplicated_operation(operation_name)(optimized_func)
                applied_optimizations.append("deduplication")
            
            # Apply async processing if enabled and requested
            if self.enabled_optimizations["async_processing"] and force_async:
                optimized_func = async_operation(
                    operation_name=operation_name,
                    priority=5
                )(optimized_func)
                applied_optimizations.append("async_processing")
            
            # Execute the optimized operation
            result = await optimized_func(*args, **kwargs)
            
            # Track optimization success
            self.performance_metrics["optimizations_applied"] += len(applied_optimizations)
            optimization_key = ",".join(sorted(applied_optimizations))
            self.performance_metrics["performance_gains"][optimization_key] = (
                self.performance_metrics["performance_gains"].get(optimization_key, 0) + 1
            )
            
            logger.debug(f"Applied optimizations to {operation_name}: {applied_optimizations}")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance optimization error for {operation_name}: {str(e)}")
            # Fallback to direct execution
            return await operation_func(*args, **kwargs)
    
    async def create_optimized_response(
        self,
        request: Request,
        data: Any,
        operation_type: str = "general",
        cache_max_age: Optional[int] = None
    ) -> Response:
        """Create optimized response with compression and caching."""
        if self.enabled_optimizations["response_optimization"]:
            return await optimize_api_response(
                request, data, operation_type, cache_max_age
            )
        else:
            from fastapi.responses import JSONResponse
            return JSONResponse(content=data)
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        dashboard = {
            "enabled_optimizations": self.enabled_optimizations,
            "integration_metrics": self.performance_metrics,
            "service_stats": {}
        }
        
        # Collect stats from all services
        try:
            dashboard["service_stats"]["cache"] = cache_service.get_cache_stats()
            dashboard["service_stats"]["deduplication"] = deduplication_service.get_deduplication_stats()
            dashboard["service_stats"]["async_processing"] = async_service.get_service_stats()
            dashboard["service_stats"]["connection_pools"] = connection_pool_manager.get_pool_statistics()
            dashboard["service_stats"]["resource_management"] = resource_manager.get_resource_stats()
            dashboard["service_stats"]["response_optimization"] = response_optimizer.get_optimization_stats()
        except Exception as e:
            logger.error(f"Error collecting service stats: {str(e)}")
            dashboard["service_stats"]["error"] = str(e)
        
        return dashboard
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all performance services."""
        health_results = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": "2024-01-01T00:00:00Z"  # Would use datetime.utcnow().isoformat()
        }
        
        try:
            # Check connection pool health
            pool_health = await connection_pool_manager.health_check()
            health_results["services"]["connection_pools"] = pool_health
            
            # Check cache service
            cache_stats = cache_service.get_cache_stats()
            health_results["services"]["cache"] = {
                "status": "healthy" if cache_stats["hit_rate"] >= 0 else "warning",
                "hit_rate": cache_stats["hit_rate"],
                "total_entries": cache_stats["total_entries"]
            }
            
            # Check async processing
            async_stats = async_service.get_service_stats()
            health_results["services"]["async_processing"] = {
                "status": "healthy",
                "running_tasks": async_stats["running_tasks"],
                "queue_size": async_stats["queue_size"]
            }
            
            # Check deduplication service
            dedup_stats = deduplication_service.get_deduplication_stats()
            health_results["services"]["deduplication"] = {
                "status": "healthy",
                "deduplication_rate": dedup_stats["deduplication_rate"],
                "pending_requests": dedup_stats["pending_requests"]
            }
            
            # Determine overall health
            service_statuses = [
                service.get("status", "unknown")
                for service in health_results["services"].values()
            ]
            
            if all(status == "healthy" for status in service_statuses):
                health_results["overall_status"] = "healthy"
            elif any(status in ["unhealthy", "degraded"] for status in service_statuses):
                health_results["overall_status"] = "degraded"
            else:
                health_results["overall_status"] = "warning"
                
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            health_results["overall_status"] = "unhealthy"
            health_results["error"] = str(e)
        
        return health_results
    
    def configure_optimizations(self, optimization_config: Dict[str, bool]):
        """Configure which optimizations are enabled."""
        for optimization, enabled in optimization_config.items():
            if optimization in self.enabled_optimizations:
                self.enabled_optimizations[optimization] = enabled
                logger.info(f"Set {optimization} optimization to {enabled}")
    
    async def invalidate_all_caches(self):
        """Invalidate all caches across services."""
        try:
            await cache_service.invalidate_cache()
            await deduplication_service.clear_completed_cache()
            logger.info("Invalidated all caches")
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")


# Global performance integration service
performance_service = PerformanceIntegrationService()


# Convenience functions for common optimization patterns

async def optimize_primitive_operation(
    request: Request,
    operation_name: str,
    operation_func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Response:
    """Optimized execution for primitive-related operations."""
    result = await performance_service.apply_all_optimizations(
        request=request,
        operation_name=operation_name,
        operation_func=operation_func,
        operation_type="primitive_generation",
        cache_ttl_minutes=120,  # 2 hours for primitive operations
        *args,
        **kwargs
    )
    
    return await performance_service.create_optimized_response(
        request=request,
        data=result,
        operation_type="primitive_generation",
        cache_max_age=7200  # 2 hours cache
    )


async def optimize_question_operation(
    request: Request,
    operation_name: str,
    operation_func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Response:
    """Optimized execution for question-related operations."""
    result = await performance_service.apply_all_optimizations(
        request=request,
        operation_name=operation_name,
        operation_func=operation_func,
        operation_type="question_generation",
        cache_ttl_minutes=90,  # 1.5 hours for question operations
        *args,
        **kwargs
    )
    
    return await performance_service.create_optimized_response(
        request=request,
        data=result,
        operation_type="question_generation",
        cache_max_age=5400  # 1.5 hours cache
    )


async def optimize_evaluation_operation(
    request: Request,
    operation_name: str,
    operation_func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Response:
    """Optimized execution for answer evaluation operations."""
    result = await performance_service.apply_all_optimizations(
        request=request,
        operation_name=operation_name,
        operation_func=operation_func,
        operation_type="answer_evaluation",
        cache_ttl_minutes=30,  # 30 minutes for evaluation operations
        *args,
        **kwargs
    )
    
    return await performance_service.create_optimized_response(
        request=request,
        data=result,
        operation_type="answer_evaluation",
        cache_max_age=1800  # 30 minutes cache
    )
