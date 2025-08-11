"""
Premium API namespace for advanced RAG features.
This module provides the foundation for premium user features including GraphRAG,
multi-agent orchestration, and advanced context assembly.
"""

from fastapi import APIRouter
from .endpoints import premium_router
from .schemas import *
from .middleware import PremiumUserMiddleware

__all__ = [
    "premium_router",
    "PremiumUserMiddleware"
]



