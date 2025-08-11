"""
Gemini service for premium LLM integration.
Provides access to Google's Gemini models for premium features.
"""

import os
from typing import Dict, Any, Optional
import google.generativeai as genai
from ..llm_service import llm_service

class GeminiService:
    """Service for interacting with Google's Gemini models"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for premium features")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize models with current available versions
        self.models = {
            'gemini_1_5_flash': 'gemini-1.5-flash',  # Primary model (closest to 2.5)
            'gemini_1_5_pro': 'gemini-1.5-pro',      # Secondary model
            'gemini_2_0_pro': 'gemini-2.0-pro'       # Future model
        }
        
        # Default model
        self.default_model = 'gemini_1_5_flash'
    
    async def generate(self, prompt: str, model: str = None) -> str:
        """Generate text using Gemini model with fallback to OpenRouter"""
        try:
            model_name = self.models.get(model or self.default_model, self.models[self.default_model])
            
            # Initialize model
            model_instance = genai.GenerativeModel(
                model_name,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,
                }
            )
            
            # Generate response
            response = model_instance.generate_content(prompt)
            
            if not response.text:
                raise Exception("Gemini returned empty response")
            
            return response.text
            
        except Exception as e:
            print(f"Gemini API call failed: {e}. Falling back to OpenRouter...")
            # Fallback to OpenRouter with GLM 4.5 Air
            try:
                return await llm_service.call_openrouter_ai(
                    prompt, 
                    model="z-ai/glm-4.5-air:free",
                    operation="gemini_fallback"
                )
            except Exception as openrouter_error:
                raise Exception(f"Both Gemini and OpenRouter failed. Gemini error: {e}, OpenRouter error: {openrouter_error}")
    
    async def generate_with_context(self, prompt: str, context: str, model: str = None) -> str:
        """Generate text with additional context"""
        try:
            full_prompt = f"""
            Context: {context}
            
            User Request: {prompt}
            
            Please provide a response based on the context above.
            """
            
            return await self.generate(full_prompt, model)
            
        except Exception as e:
            print(f"Error generating with context: {e}")
            raise e
    
    async def generate_structured(self, prompt: str, structure: Dict[str, Any], model: str = None) -> Dict[str, Any]:
        """Generate structured response"""
        try:
            structured_prompt = f"""
            {prompt}
            
            Please respond in the following JSON structure:
            {structure}
            
            Return only valid JSON.
            """
            
            response = await self.generate(structured_prompt, model)
            
            # Try to parse JSON response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"error": "Failed to parse structured response", "raw_response": response}
                
        except Exception as e:
            print(f"Error generating structured response: {e}")
            raise e
    
    async def select_optimal_model(self, context_size: int, complexity: str) -> str:
        """Select optimal model based on context size and complexity"""
        if complexity == "simple" or context_size < 1000:
            return 'gemini_1_5_flash'  # Fast and cost-effective
        elif complexity == "medium" or context_size < 5000:
            return 'gemini_1_5_pro'    # Balanced performance
        else:
            return 'gemini_1_5_pro'    # Best quality for complex tasks
    
    async def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about available models"""
        model_name = model or self.default_model
        return {
            "model": model_name,
            "provider": "google",
            "available": model_name in self.models,
            "fallback": "openrouter_glm_4_5_air"
        }





