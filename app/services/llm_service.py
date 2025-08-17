"""
LLM Service interface for the Note Creation Agent.
Provides a unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Continue without dotenv if not available


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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
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
            # Add JSON formatting instruction if prompt asks for JSON
            json_instruction = ""
            if "JSON" in prompt.upper() or "json" in prompt:
                json_instruction = "IMPORTANT: You must respond with ONLY valid JSON. Do not include any other text, explanations, or formatting outside the JSON object. Ensure the JSON is properly formatted and complete."
            
            # Combine system message with prompt if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            # Add JSON instruction at the beginning if needed
            if json_instruction:
                full_prompt = f"{json_instruction}\n\n{full_prompt}"
            
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
        
        # Check what type of response is needed based on the prompt
        if "context for note ID" in prompt:
            # Return mock context data
            return '''{
                "note_section_id": 1,
                "blueprint_section_id": 1,
                "blueprint_id": 1,
                "section_hierarchy": [
                    {"id": 1, "title": "Main Section", "depth": 0},
                    {"id": 1, "title": "Current Section", "depth": 1}
                ],
                "related_notes": [
                    {"id": 2, "title": "Related Note", "content_preview": "This is a related note content..."}
                ],
                "knowledge_primitives": ["concept1", "concept2"],
                "content_version": 2
            }'''
        
        elif "analyze the following note" in prompt:
            # Return mock analysis data
            return '''{
                "content_structure": "Well organized with clear sections",
                "writing_quality": "Good clarity and flow",
                "grammar_issues": "Minor punctuation issues",
                "improvement_areas": ["Add more examples", "Clarify technical terms"],
                "overall_assessment": "Good quality note with room for improvement",
                "context_alignment": "Well aligned with blueprint section",
                "consistency_score": 0.85
            }'''
        
        elif "Create an edit plan" in prompt:
            # Return mock edit plan
            return '''{
                "summary": "Improve clarity and add examples",
                "context_alignment": "Maintains blueprint section consistency",
                "changes": [
                    {
                        "type": "clarity",
                        "description": "Simplify complex explanations",
                        "reason": "Improve readability for learners",
                        "context_impact": "Better alignment with section objectives"
                    }
                ],
                "new_structure": ["introduction", "main_concepts", "examples", "summary"],
                "style_guidelines": ["Use clear language", "Include practical examples"],
                "cross_references": [2, 3]
            }'''
        
        elif "Apply the following edit plan" in prompt:
            # Return mock edited content
            return '''{
                "note_content": "{\\"type\\": \\"doc\\", \\"content\\": [{\\"type\\": \\"paragraph\\", \\"content\\": [{\\"type\\": \\"text\\", \\"text\\": \\"Mock edited note content in BlockNote format\\"}]}]}",
                "plain_text": "Mock edited note content in plain text format",
                "changes_applied": ["Improved clarity", "Added examples"],
                "context_alignment": "Successfully aligned with blueprint section context"
            }'''
        
        elif "Explain the reasoning" in prompt:
            # Return mock reasoning
            return "The edits were made to improve clarity and maintain consistency with the blueprint section context. Changes focused on simplifying complex explanations and adding practical examples that align with the learning objectives."
        
        elif "grammar and style improvements" in prompt:
            # Return mock grammar suggestions
            return '''[
                {
                    "type": "grammar",
                    "description": "Fix comma usage in compound sentences",
                    "suggested_change": "Add comma before 'and' in compound sentences",
                    "confidence": 0.9,
                    "reasoning": "Improves readability and follows standard grammar rules",
                    "context_relevance": "Maintains professional tone for educational content"
                }
            ]'''
        
        elif "clarity and readability improvements" in prompt:
            # Return mock clarity suggestions
            return '''[
                {
                    "type": "clarity",
                    "description": "Simplify technical terminology",
                    "suggested_change": "Replace complex terms with simpler alternatives",
                    "confidence": 0.85,
                    "reasoning": "Makes content more accessible to learners",
                    "context_relevance": "Aligns with section's beginner-friendly approach"
                }
            ]'''
        
        elif "structural and organizational improvements" in prompt:
            # Return mock structure suggestions
            return '''[
                {
                    "type": "structure",
                    "description": "Improve logical flow between sections",
                    "suggested_change": "Add transition sentences between paragraphs",
                    "confidence": 0.8,
                    "reasoning": "Creates better reading flow and comprehension",
                    "context_relevance": "Maintains consistency with section organization"
                }
            ]'''
        
        elif "Convert the following" in prompt:
            # Return mock BlockNote conversion
            return '''{
                "blocknote_content": "{\\"type\\": \\"doc\\", \\"content\\": [{\\"type\\": \\"paragraph\\", \\"content\\": [{\\"type\\": \\"text\\", \\"text\\": \\"Converted content in BlockNote format\\"}]}]}",
                "plain_text": "Converted content in plain text format",
                "conversion_notes": "Successfully converted from plain text to BlockNote format"
            }'''
        
        else:
            # Default mock response
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
