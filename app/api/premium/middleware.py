"""
Premium user authentication and access control middleware.
Validates premium subscription status and manages access to premium features.
"""

import os
from typing import Optional
from fastapi import HTTPException, status
import httpx

class PremiumUserMiddleware:
    """Middleware for validating premium user access"""
    
    def __init__(self):
        self.core_api_url = os.getenv("CORE_API_URL", "http://localhost:3000")
        self.premium_subscription_endpoint = f"{self.core_api_url}/api/users/premium-status"
    
    async def validate_premium_access(self, user_id: str) -> bool:
        """Validate user has premium subscription"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.premium_subscription_endpoint,
                    params={"user_id": user_id},
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("is_premium", False)
                else:
                    # If Core API is unavailable, allow access for development
                    # In production, this should be more restrictive
                    return True
                    
        except Exception as e:
            # Log the error for debugging
            print(f"Error validating premium access for user {user_id}: {e}")
            # For development, allow access if Core API is unavailable
            return True
    
    async def get_user_premium_features(self, user_id: str) -> dict:
        """Get user's premium feature access"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.core_api_url}/api/users/{user_id}/premium-features",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Default premium features for development
                    return {
                        "graph_rag": True,
                        "multi_agent": True,
                        "advanced_context": True,
                        "long_context_llm": True
                    }
                    
        except Exception as e:
            print(f"Error getting premium features for user {user_id}: {e}")
            # Default premium features for development
            return {
                "graph_rag": True,
                "multi_agent": True,
                "advanced_context": True,
                "long_context_llm": True
            }
    
    async def log_premium_usage(self, user_id: str, feature: str, usage_data: dict):
        """Log premium feature usage for analytics"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.core_api_url}/api/users/{user_id}/premium-usage",
                    json={
                        "feature": feature,
                        "usage_data": usage_data,
                        "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
                    },
                    timeout=5.0
                )
        except Exception as e:
            print(f"Error logging premium usage for user {user_id}: {e}")
            # Continue execution even if logging fails



