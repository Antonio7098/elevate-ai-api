#!/usr/bin/env python3
"""
Monitoring and Observability Framework for Blueprint Section Operations
Provides comprehensive logging, metrics collection, and health monitoring.
"""

import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class OperationMetrics:
    """Metrics for a single operation"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def finalize(self, success: bool = True, error_message: Optional[str] = None):
        """Finalize the operation metrics"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    operation_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_duration: float
    min_duration: float
    max_duration: float
    p95_duration: float
    p99_duration: float
    success_rate: float
    throughput: float  # operations per second
    last_updated: datetime

@dataclass
class HealthCheck:
    """Health check result"""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    last_check: datetime
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class MetricsCollector:
    """Collects and aggregates operation metrics"""
    
    def __init__(self, max_metrics_history: int = 1000):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_history))
        self.current_metrics: Dict[str, List[OperationMetrics]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def record_operation(self, metrics: OperationMetrics):
        """Record operation metrics"""
        async with self.lock:
            operation_name = metrics.operation_name
            self.current_metrics[operation_name].append(metrics)
            self.metrics_history[operation_name].append(metrics)
    
    async def get_performance_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific operation"""
        async with self.lock:
            if operation_name not in self.metrics_history:
                return None
            
            metrics_list = list(self.metrics_history[operation_name])
            if not metrics_list:
                return None
            
            # Filter completed operations
            completed_metrics = [m for m in metrics_list if m.duration is not None]
            if not completed_metrics:
                return None
            
            durations = [m.duration for m in completed_metrics]
            successful = [m for m in completed_metrics if m.success]
            
            total_ops = len(completed_metrics)
            successful_ops = len(successful)
            failed_ops = total_ops - successful_ops
            
            avg_duration = statistics.mean(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            # Calculate percentiles
            sorted_durations = sorted(durations)
            p95_duration = statistics.quantiles(sorted_durations, n=20)[18] if len(sorted_durations) >= 20 else max_duration
            p99_duration = statistics.quantiles(sorted_durations, n=100)[98] if len(sorted_durations) >= 100 else max_duration
            
            success_rate = successful_ops / total_ops if total_ops > 0 else 0.0
            throughput = 1.0 / avg_duration if avg_duration > 0 else 0.0
            
            return PerformanceMetrics(
                operation_name=operation_name,
                total_operations=total_ops,
                successful_operations=successful_ops,
                failed_operations=failed_ops,
                avg_duration=avg_duration,
                min_duration=min_duration,
                max_duration=max_duration,
                p95_duration=p95_duration,
                p99_duration=p99_duration,
                success_rate=success_rate,
                throughput=throughput,
                last_updated=datetime.now()
            )
    
    async def get_all_performance_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for all operations"""
        async with self.lock:
            all_metrics = {}
            for operation_name in self.metrics_history.keys():
                metrics = await self.get_performance_metrics(operation_name)
                if metrics:
                    all_metrics[operation_name] = metrics
            return all_metrics
    
    async def clear_old_metrics(self, max_age_hours: int = 24):
        """Clear metrics older than specified age"""
        async with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for operation_name in list(self.metrics_history.keys()):
                # Remove old metrics
                self.metrics_history[operation_name] = deque(
                    [m for m in self.metrics_history[operation_name] 
                     if m.start_time > cutoff_time],
                    maxlen=self.metrics_history[operation_name].maxlen
                )
                
                # Remove empty operation entries
                if not self.metrics_history[operation_name]:
                    del self.metrics_history[operation_name]

class HealthMonitor:
    """Monitors health of various services and endpoints"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, List[HealthCheck]] = defaultdict(list)
        self.max_history = 100
    
    def register_health_check(self, service_name: str, check_function: Callable):
        """Register a health check function for a service"""
        self.health_checks[service_name] = check_function
    
    async def run_health_check(self, service_name: str) -> HealthCheck:
        """Run a health check for a specific service"""
        if service_name not in self.health_checks:
            return HealthCheck(
                service_name=service_name,
                status="unhealthy",
                response_time=0.0,
                last_check=datetime.now(),
                error_message="Health check not registered"
            )
        
        start_time = time.time()
        try:
            result = await self.health_checks[service_name]()
            response_time = time.time() - start_time
            
            health_check = HealthCheck(
                service_name=service_name,
                status=result.get("status", "unknown"),
                response_time=response_time,
                last_check=datetime.now(),
                details=result.get("details")
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            health_check = HealthCheck(
                service_name=service_name,
                status="unhealthy",
                response_time=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
        
        # Store in history
        self.health_history[service_name].append(health_check)
        if len(self.health_history[service_name]) > self.max_history:
            self.health_history[service_name] = self.health_history[service_name][-self.max_history:]
        
        return health_check
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run health checks for all registered services"""
        results = {}
        for service_name in self.health_checks.keys():
            results[service_name] = await self.run_health_check(service_name)
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_services = len(self.health_checks)
        healthy_services = 0
        degraded_services = 0
        unhealthy_services = 0
        
        for service_name, checks in self.health_history.items():
            if checks:
                latest_check = checks[-1]
                if latest_check.status == "healthy":
                    healthy_services += 1
                elif latest_check.status == "degraded":
                    degraded_services += 1
                else:
                    unhealthy_services += 1
        
        overall_status = "healthy"
        if unhealthy_services > 0:
            overall_status = "unhealthy"
        elif degraded_services > 0:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "degraded_services": degraded_services,
            "unhealthy_services": unhealthy_services,
            "last_updated": datetime.now().isoformat()
        }

class MonitoringService:
    """Main monitoring service that coordinates metrics collection and health monitoring"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        # Database connectivity check
        self.health_monitor.register_health_check("database", self._check_database_health)
        
        # Vector store health check
        self.health_monitor.register_health_check("vector_store", self._check_vector_store_health)
        
        # Core API connectivity check
        self.health_monitor.register_health_check("core_api", self._check_core_api_health)
        
        # LLM service health check
        self.health_monitor.register_health_check("llm_service", self._check_llm_service_health)
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # This would be implemented based on your database setup
            # For now, return a mock healthy status
            return {
                "status": "healthy",
                "details": {
                    "connection_pool_size": 10,
                    "active_connections": 3,
                    "query_response_time": 0.05
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_vector_store_health(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            # This would check Pinecone/ChromaDB health
            return {
                "status": "healthy",
                "details": {
                    "index_size": 1000,
                    "embedding_dimension": 1536,
                    "last_sync": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_core_api_health(self) -> Dict[str, Any]:
        """Check Core API connectivity"""
        try:
            # This would make an actual HTTP request to Core API
            return {
                "status": "healthy",
                "details": {
                    "endpoint": "http://localhost:3000/health",
                    "response_time": 0.1
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_llm_service_health(self) -> Dict[str, Any]:
        """Check LLM service health"""
        try:
            # This would check OpenAI/Gemini API status
            return {
                "status": "healthy",
                "details": {
                    "available_models": ["gpt-4", "gemini-pro"],
                    "rate_limit_status": "normal",
                    "last_request": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for monitoring operations"""
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            metadata=metadata
        )
        
        try:
            yield metrics
            metrics.finalize(success=True)
        except Exception as e:
            metrics.finalize(success=False, error_message=str(e))
            raise
        finally:
            await self.metrics_collector.record_operation(metrics)
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard"""
        return {
            "performance_metrics": await self.metrics_collector.get_all_performance_metrics(),
            "health_summary": self.health_monitor.get_health_summary(),
            "recent_health_checks": {
                service: checks[-5:] if checks else []
                for service, checks in self.health_monitor.health_history.items()
            },
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "uptime": "24h",  # This would be calculated
                "version": "1.0.0"
            }
        }
    
    async def start_periodic_health_checks(self, interval_seconds: int = 60):
        """Start periodic health checks"""
        while True:
            try:
                await self.health_monitor.run_all_health_checks()
                self.logger.info("Periodic health checks completed")
            except Exception as e:
                self.logger.error(f"Error during periodic health checks: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def start_metrics_cleanup(self, cleanup_interval_hours: int = 1):
        """Start periodic metrics cleanup"""
        while True:
            try:
                await self.metrics_collector.clear_old_metrics(max_age_hours=24)
                self.logger.info("Metrics cleanup completed")
            except Exception as e:
                self.logger.error(f"Error during metrics cleanup: {e}")
            
            await asyncio.sleep(cleanup_interval_hours * 3600)

# Global monitoring service instance
monitoring_service = MonitoringService()

# Convenience functions for easy usage
async def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function for monitoring operations"""
    return monitoring_service.monitor_operation(operation_name, metadata)

def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance"""
    return monitoring_service
