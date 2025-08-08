# Sprint 32: Async Processing Service for Long-Running Operations

import asyncio
import uuid
from typing import Dict, Any, Optional, Callable, Awaitable, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AsyncTask:
    """Represents an asynchronous task."""
    task_id: str
    operation_name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "operation_name": self.operation_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "has_result": self.result is not None,
            "error": self.error,
            "metadata": self.metadata or {}
        }

class AsyncProcessingService:
    """
    Service for managing long-running asynchronous operations.
    
    Features:
    - Background task execution
    - Progress tracking and status monitoring
    - Result storage and retrieval
    - Task cancellation and cleanup
    - Priority-based task queue
    """
    
    def __init__(self, max_concurrent_tasks: int = 10, max_task_history: int = 1000):
        self.tasks: Dict[str, AsyncTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_task_history = max_task_history
        self.current_concurrent_tasks = 0
        self._worker_tasks: List[asyncio.Task] = []
        # Don't start workers during init - will start when needed
        
    def _start_workers(self):
        """Start background worker tasks."""
        try:
            for i in range(self.max_concurrent_tasks):
                worker_task = asyncio.create_task(self._worker())
                self._worker_tasks.append(worker_task)
                logger.debug(f"Started async worker {i+1}")
        except RuntimeError:
            # No event loop running, workers will start when needed
            pass
    
    async def _worker(self):
        """Background worker for processing async tasks."""
        while True:
            try:
                # Get next task from queue
                task_item = await self.task_queue.get()
                task_id, operation_func, args, kwargs = task_item
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                # Update task status
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                self.current_concurrent_tasks += 1
                
                logger.debug(f"Starting async task: {task_id} ({task.operation_name})")
                
                try:
                    # Execute the operation
                    result = await operation_func(*args, **kwargs)
                    
                    # Update task with result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    task.result = result
                    task.progress = 1.0
                    
                    logger.debug(f"Completed async task: {task_id}")
                    
                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.utcnow()
                    logger.debug(f"Cancelled async task: {task_id}")
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.utcnow()
                    task.error = str(e)
                    logger.error(f"Failed async task: {task_id} - {str(e)}")
                
                finally:
                    self.current_concurrent_tasks -= 1
                    
                    # Remove from running tasks
                    self.running_tasks.pop(task_id, None)
                    
                    # Mark queue task as done
                    self.task_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def submit_task(
        self,
        operation_name: str,
        operation_func: Callable[..., Awaitable[Any]],
        *args,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Submit a long-running task for async execution.
        
        Args:
            operation_name: Human-readable name for the operation
            operation_func: Async function to execute
            *args: Arguments for the operation function
            priority: Task priority (1-10, higher = more important)
            metadata: Additional metadata for the task
            **kwargs: Keyword arguments for the operation function
            
        Returns:
            task_id: Unique identifier for tracking the task
        """
        task_id = str(uuid.uuid4())
        
        # Create task record
        task = AsyncTask(
            task_id=task_id,
            operation_name=operation_name,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to queue
        await self.task_queue.put((task_id, operation_func, args, kwargs))
        
        logger.info(f"Submitted async task: {task_id} ({operation_name})")
        
        # Clean up old tasks if needed
        await self._cleanup_old_tasks()
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """Get the status of a specific task."""
        return self.tasks.get(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # Cancel running task
        if task_id in self.running_tasks:
            running_task = self.running_tasks[task_id]
            running_task.cancel()
        
        # Update task status
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        
        logger.info(f"Cancelled async task: {task_id}")
        return True
    
    async def list_tasks(
        self,
        status_filter: Optional[TaskStatus] = None,
        operation_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[AsyncTask]:
        """List tasks with optional filtering."""
        tasks = list(self.tasks.values())
        
        # Apply filters
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        if operation_filter:
            tasks = [t for t in tasks if operation_filter.lower() in t.operation_name.lower()]
        
        # Sort by creation time (newest first) and limit
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]
    
    async def update_task_progress(self, task_id: str, progress: float, metadata: Optional[Dict[str, Any]] = None):
        """Update the progress of a running task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.progress = max(0.0, min(1.0, progress))
            if metadata:
                task.metadata = {**(task.metadata or {}), **metadata}
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks to prevent memory leaks."""
        if len(self.tasks) <= self.max_task_history:
            return
        
        # Sort tasks by completion time
        completed_tasks = [
            (task_id, task) for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            and task.completed_at
        ]
        
        if len(completed_tasks) <= self.max_task_history // 2:
            return
        
        # Keep newest tasks, remove oldest
        completed_tasks.sort(key=lambda x: x[1].completed_at, reverse=True)
        tasks_to_remove = completed_tasks[self.max_task_history // 2:]
        
        for task_id, _ in tasks_to_remove:
            del self.tasks[task_id]
        
        logger.debug(f"Cleaned up {len(tasks_to_remove)} old async tasks")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for task in self.tasks.values() if task.status == status
            )
        
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": self.current_concurrent_tasks,
            "queue_size": self.task_queue.qsize(),
            "max_concurrent": self.max_concurrent_tasks,
            "status_breakdown": status_counts,
            "worker_tasks": len(self._worker_tasks)
        }


# Global async processing service instance
async_service = AsyncProcessingService()


def async_operation(operation_name: str, priority: int = 5):
    """
    Decorator for automatically submitting operations to async processing.
    
    Args:
        operation_name: Name of the operation for tracking
        priority: Task priority (1-10)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Check if this should be processed async based on arguments
            force_async = kwargs.pop('force_async', False)
            return_task_id = kwargs.pop('return_task_id', False)
            
            if force_async or return_task_id:
                task_id = await async_service.submit_task(
                    operation_name,
                    func,
                    *args,
                    priority=priority,
                    **kwargs
                )
                
                if return_task_id:
                    return {"task_id": task_id, "status": "submitted"}
                else:
                    # Wait for completion
                    while True:
                        task = await async_service.get_task_status(task_id)
                        if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                            if task.status == TaskStatus.COMPLETED:
                                return task.result
                            elif task.status == TaskStatus.FAILED:
                                raise Exception(f"Async operation failed: {task.error}")
                            else:
                                raise Exception("Async operation was cancelled")
                        
                        await asyncio.sleep(0.5)  # Poll every 500ms
            else:
                # Execute synchronously
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator
