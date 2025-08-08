# Sprint 32: Connection Pooling and Resource Management Service

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """
    Manages HTTP connection pools and resource lifecycle for external APIs.
    
    Features:
    - HTTP connection pooling for Core API calls
    - Resource lifecycle management
    - Connection health monitoring
    - Automatic retry and failover
    - Resource usage statistics
    """
    
    def __init__(
        self,
        max_connections: int = 100,
        max_connections_per_host: int = 30,
        connection_timeout: float = 30.0,
        read_timeout: float = 60.0
    ):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        
        # Connection pools for different services
        self.session_pools: Dict[str, aiohttp.ClientSession] = {}
        self.pool_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
            "connection_errors": 0
        }
        
        # Resource monitoring
        self.active_connections = {}
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = timedelta(minutes=5)
    
    async def get_session(self, service_name: str = "default") -> aiohttp.ClientSession:
        """Get or create a connection session for a service."""
        if service_name not in self.session_pools:
            # Create new session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                ttl_dns_cache=300,  # DNS cache for 5 minutes
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.connection_timeout,
                connect=10.0,
                sock_read=self.read_timeout
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "elevate-ai-api/1.0",
                    "Accept": "application/json",
                    "Connection": "keep-alive"
                }
            )
            
            self.session_pools[service_name] = session
            logger.debug(f"Created new connection pool for service: {service_name}")
        
        return self.session_pools[service_name]
    
    @asynccontextmanager
    async def managed_request(
        self,
        service_name: str,
        method: str,
        url: str,
        **kwargs
    ):
        """Context manager for making HTTP requests with connection pooling."""
        session = await self.get_session(service_name)
        self.pool_stats["total_requests"] += 1
        
        max_retries = kwargs.pop('max_retries', 3)
        retry_delay = kwargs.pop('retry_delay', 1.0)
        
        for attempt in range(max_retries + 1):
            try:
                async with session.request(method, url, **kwargs) as response:
                    self.pool_stats["successful_requests"] += 1
                    yield response
                    return
                    
            except aiohttp.ClientError as e:
                self.pool_stats["connection_errors"] += 1
                
                if attempt < max_retries:
                    self.pool_stats["retry_attempts"] += 1
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.pool_stats["failed_requests"] += 1
                    logger.error(f"Request failed after {max_retries + 1} attempts: {str(e)}")
                    raise e
            
            except Exception as e:
                self.pool_stats["failed_requests"] += 1
                logger.error(f"Unexpected error in managed request: {str(e)}")
                raise e
    
    async def make_core_api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make a request to the Core API with connection pooling."""
        # This would be configured with actual Core API base URL
        base_url = "http://localhost:3000/api/v1"  # Placeholder
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        
        kwargs = {"headers": request_headers}
        if data:
            kwargs["json"] = data
        if timeout:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)
        
        async with self.managed_request("core_api", method, url, **kwargs) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Core API error: {error_text}"
                )
            
            return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on connection pools."""
        current_time = datetime.utcnow()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return {"status": "skipped", "reason": "too_recent"}
        
        health_results = {}
        
        for service_name, session in self.session_pools.items():
            try:
                connector = session.connector
                health_results[service_name] = {
                    "status": "healthy",
                    "total_connections": len(connector._conns),
                    "available_connections": connector.limit - len(connector._conns),
                    "limit": connector.limit,
                    "limit_per_host": connector.limit_per_host
                }
            except Exception as e:
                health_results[service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        self.last_health_check = current_time
        
        return {
            "overall_status": "healthy" if all(
                result.get("status") == "healthy" 
                for result in health_results.values()
            ) else "degraded",
            "services": health_results,
            "stats": self.pool_stats,
            "checked_at": current_time.isoformat()
        }
    
    async def cleanup_connections(self):
        """Clean up connection pools."""
        for service_name, session in self.session_pools.items():
            try:
                await session.close()
                logger.debug(f"Closed connection pool for service: {service_name}")
            except Exception as e:
                logger.error(f"Error closing connection pool for {service_name}: {str(e)}")
        
        self.session_pools.clear()
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        pool_info = {}
        
        for service_name, session in self.session_pools.items():
            try:
                connector = session.connector
                pool_info[service_name] = {
                    "active_connections": len(connector._conns),
                    "max_connections": connector.limit,
                    "max_per_host": connector.limit_per_host,
                    "dns_cache_size": len(connector._dns_cache) if hasattr(connector, '_dns_cache') else 0
                }
            except Exception as e:
                pool_info[service_name] = {"error": str(e)}
        
        return {
            "pools": pool_info,
            "request_stats": self.pool_stats,
            "last_health_check": self.last_health_check.isoformat()
        }


class ResourceManager:
    """
    Manages application resources and lifecycle.
    
    Features:
    - Memory usage monitoring
    - Resource cleanup scheduling
    - Performance metrics collection
    - Resource allocation optimization
    """
    
    def __init__(self):
        self.resource_usage = {
            "memory_peak": 0,
            "active_tasks": 0,
            "cache_entries": 0,
            "connection_pools": 0
        }
        self.cleanup_tasks = []
        self.monitoring_active = False
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        monitor_task = asyncio.create_task(self._resource_monitor())
        self.cleanup_tasks.append(monitor_task)
        logger.info("Started resource monitoring")
    
    async def _resource_monitor(self):
        """Background resource monitoring task."""
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await self._check_resource_limits()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self):
        """Collect current resource usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            self.resource_usage["memory_current"] = memory_info.rss / 1024 / 1024  # MB
            self.resource_usage["memory_peak"] = max(
                self.resource_usage["memory_peak"],
                self.resource_usage["memory_current"]
            )
            
            # CPU usage
            self.resource_usage["cpu_percent"] = process.cpu_percent()
            
        except ImportError:
            logger.warning("psutil not available for resource monitoring")
        except Exception as e:
            logger.error(f"Metrics collection error: {str(e)}")
    
    async def _check_resource_limits(self):
        """Check if resource usage exceeds limits and trigger cleanup."""
        memory_limit_mb = 1000  # 1GB limit
        
        current_memory = self.resource_usage.get("memory_current", 0)
        
        if current_memory > memory_limit_mb:
            logger.warning(f"Memory usage high: {current_memory:.1f}MB > {memory_limit_mb}MB")
            await self.trigger_cleanup()
    
    async def trigger_cleanup(self):
        """Trigger resource cleanup across services."""
        logger.info("Triggering resource cleanup")
        
        try:
            # Import services to trigger cleanup
            from app.core.caching_service import cache_service
            from app.core.request_deduplication import deduplication_service
            from app.core.async_processing import async_service
            
            # Clean up caches
            await cache_service.invalidate_cache()
            await deduplication_service.clear_completed_cache()
            
            # Clean up old async tasks
            await async_service._cleanup_old_tasks()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup error: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        
        for task in self.cleanup_tasks:
            if not task.done():
                task.cancel()
        
        self.cleanup_tasks.clear()
        logger.info("Stopped resource monitoring")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        return {
            "usage": self.resource_usage,
            "monitoring_active": self.monitoring_active,
            "cleanup_tasks": len(self.cleanup_tasks)
        }


# Global instances
connection_pool_manager = ConnectionPoolManager()
resource_manager = ResourceManager()


async def startup_resource_management():
    """Initialize resource management on application startup."""
    await resource_manager.start_monitoring()
    logger.info("Resource management initialized")


async def shutdown_resource_management():
    """Clean up resources on application shutdown."""
    await connection_pool_manager.cleanup_connections()
    await resource_manager.stop_monitoring()
    logger.info("Resource management shut down")
