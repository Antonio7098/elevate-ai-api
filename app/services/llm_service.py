"""
LLM Service interface for the Note Creation Agent.
Provides a unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os


class LLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    async def call_llm(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            system_message: Optional system message
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass


class GeminiLLMService(LLMService):
    """Gemini-specific LLM service implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        from app.services.gemini_service import GeminiService, GeminiConfig
        
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        config = GeminiConfig(
            api_key=api_key or "mock_key",
            model_name=model,
            temperature=0.3,
            max_tokens=4000
        )
        
        self.gemini_service = GeminiService(config)
    
    async def call_llm(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Call Gemini LLM."""
        try:
            # Combine system message with prompt if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            response = await self.gemini_service.generate_response(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            print(f"Error calling Gemini LLM: {e}")
            # Return fallback response
            return f"Error generating response: {str(e)}. Please try again."


class MockLLMService(LLMService):
    """Mock LLM service for testing."""
    
    async def call_llm(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Return mock response for testing."""
        return f"Mock LLM response to: {prompt[:100]}..."


def create_llm_service(provider: str = "gemini", **kwargs) -> LLMService:
    """
    Factory function to create LLM service instances.
    
    Args:
        provider: LLM provider ("gemini", "mock")
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured LLM service instance
    """
    if provider == "gemini":
        return GeminiLLMService(**kwargs)
    elif provider == "mock":
        return MockLLMService()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
