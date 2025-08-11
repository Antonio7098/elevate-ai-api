"""
Core API client for premium feature integration.
Provides access to Core API data including user memory, learning analytics, and knowledge primitives.
"""

import os
import httpx
from typing import List, Dict, Any, Optional

class CoreAPIClient:
    """Client for interacting with Core API for premium features"""
    
    def __init__(self):
        self.base_url = os.getenv("CORE_API_URL", "http://localhost:3000")
        self.api_key = os.getenv("CORE_API_KEY", "")
    
    async def get_user_memory(self, user_id: str) -> Dict[str, Any]:
        """Get user's memory data from Core API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/users/{user_id}/memory",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Return default memory structure for development
                    return {
                        "cognitiveApproach": "BALANCED",
                        "learningStyle": "VISUAL",
                        "preferredExplanationStyle": "STEP_BY_STEP",
                        "interactionStyle": "COLLABORATIVE"
                    }
                    
        except Exception as e:
            print(f"Error getting user memory for {user_id}: {e}")
            return {
                "cognitiveApproach": "BALANCED",
                "learningStyle": "VISUAL", 
                "preferredExplanationStyle": "STEP_BY_STEP",
                "interactionStyle": "COLLABORATIVE"
            }
    
    async def get_user_learning_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user's learning analytics from Core API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/users/{user_id}/learning-analytics",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Return default analytics for development
                    return {
                        "learningEfficiency": 0.7,
                        "studyTime": 120,
                        "conceptsReviewed": 15,
                        "masteryLevel": "INTERMEDIATE"
                    }
                    
        except Exception as e:
            print(f"Error getting learning analytics for {user_id}: {e}")
            return {
                "learningEfficiency": 0.7,
                "studyTime": 120,
                "conceptsReviewed": 15,
                "masteryLevel": "INTERMEDIATE"
            }
    
    async def get_user_memory_insights(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's memory insights from Core API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/users/{user_id}/memory-insights",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Return default insights for development
                    return [
                        {
                            "type": "LEARNING_PATTERN",
                            "title": "Visual Learner",
                            "content": "User shows preference for visual explanations",
                            "confidenceScore": 0.8
                        }
                    ]
                    
        except Exception as e:
            print(f"Error getting memory insights for {user_id}: {e}")
            return [
                {
                    "type": "LEARNING_PATTERN",
                    "title": "Visual Learner", 
                    "content": "User shows preference for visual explanations",
                    "confidenceScore": 0.8
                }
            ]
    
    async def get_user_learning_paths(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's learning paths from Core API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/users/{user_id}/learning-paths",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Return default learning paths for development
                    return [
                        {
                            "id": "path-1",
                            "title": "Introduction to Machine Learning",
                            "steps": [
                                {"conceptId": "concept-1", "order": 1},
                                {"conceptId": "concept-2", "order": 2}
                            ]
                        }
                    ]
                    
        except Exception as e:
            print(f"Error getting learning paths for {user_id}: {e}")
            return [
                {
                    "id": "path-1",
                    "title": "Introduction to Machine Learning",
                    "steps": [
                        {"conceptId": "concept-1", "order": 1},
                        {"conceptId": "concept-2", "order": 2}
                    ]
                }
            ]
    
    async def get_learning_blueprint(self, blueprint_id: str) -> Dict[str, Any]:
        """Get learning blueprint from Core API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/learning-blueprints/{blueprint_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Return default blueprint for development
                    return {
                        "id": blueprint_id,
                        "title": "Sample Blueprint",
                        "sourceText": "Sample source text",
                        "createdAt": "2024-01-01T00:00:00Z"
                    }
                    
        except Exception as e:
            print(f"Error getting learning blueprint {blueprint_id}: {e}")
            return {
                "id": blueprint_id,
                "title": "Sample Blueprint",
                "sourceText": "Sample source text", 
                "createdAt": "2024-01-01T00:00:00Z"
            }
    
    async def get_knowledge_primitives(self, blueprint_id: str, include_premium_fields: bool = False) -> List[Dict[str, Any]]:
        """Get knowledge primitives from Core API with optional premium fields"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/learning-blueprints/{blueprint_id}/knowledge-primitives",
                    params={"include_premium_fields": include_premium_fields},
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Return default primitives for development
                    return [
                        {
                            "id": "concept-1",
                            "name": "Machine Learning",
                            "description": "A subset of artificial intelligence",
                            "conceptTags": ["AI", "ML"],
                            "complexityScore": 0.7,
                            "isCoreConcept": True,
                            "semanticSimilarityScore": 0.8,
                            "prerequisiteIds": [],
                            "relatedConceptIds": ["concept-2"]
                        },
                        {
                            "id": "concept-2", 
                            "name": "Neural Networks",
                            "description": "Computing systems inspired by biological neural networks",
                            "conceptTags": ["AI", "ML", "Deep Learning"],
                            "complexityScore": 0.9,
                            "isCoreConcept": False,
                            "semanticSimilarityScore": 0.6,
                            "prerequisiteIds": ["concept-1"],
                            "relatedConceptIds": ["concept-1"]
                        }
                    ]
                    
        except Exception as e:
            print(f"Error getting knowledge primitives for {blueprint_id}: {e}")
            return [
                {
                    "id": "concept-1",
                    "name": "Machine Learning",
                    "description": "A subset of artificial intelligence",
                    "conceptTags": ["AI", "ML"],
                    "complexityScore": 0.7,
                    "isCoreConcept": True,
                    "semanticSimilarityScore": 0.8,
                    "prerequisiteIds": [],
                    "relatedConceptIds": ["concept-2"]
                },
                {
                    "id": "concept-2",
                    "name": "Neural Networks", 
                    "description": "Computing systems inspired by biological neural networks",
                    "conceptTags": ["AI", "ML", "Deep Learning"],
                    "complexityScore": 0.9,
                    "isCoreConcept": False,
                    "semanticSimilarityScore": 0.6,
                    "prerequisiteIds": ["concept-1"],
                    "relatedConceptIds": ["concept-1"]
                }
            ]



