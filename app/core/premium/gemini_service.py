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
        
        # Initialize models with current available versions - PRIORITIZE COST-EFFECTIVE
        self.models = {
            'gemini_2_5_flash_lite': 'gemini-2.5-flash-lite',  # NEW: Most cost-effective for routing/summarization
            'gemini_2_5_flash': 'gemini-2.5-flash',            # Best price-performance balance
            'gemini_1_5_flash': 'gemini-1.5-flash',            # Fallback
            'gemini_1_5_pro': 'gemini-1.5-pro',                # Only for complex reasoning
            'gemini_2_0_pro': 'gemini-2.0-pro'                 # Future model
        }
        
        # Default model - MOST COST-EFFECTIVE
        self.default_model = 'gemini_2_5_flash_lite'
    
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
    
    async def select_optimal_model(self, context_size: int, complexity: str, task_type: str = "general") -> str:
        """Select optimal model based on context size, complexity, and task type"""
        try:
            # For context assembly tasks, prioritize cost efficiency
            if task_type in ["routing", "summarization", "classification", "compression"]:
                if context_size < 10000:  # Small contexts
                    return 'gemini_2_5_flash_lite'  # Most cost-effective
                elif context_size < 100000:  # Medium contexts
                    return 'gemini_2_5_flash'  # Good balance
                else:  # Large contexts
                    return 'gemini_1_5_pro'  # Only when necessary
            
            # For other tasks, use standard selection
            elif complexity == "simple" or context_size < 1000:
                return 'gemini_2_5_flash_lite'  # Most cost-effective
            elif complexity == "medium" or context_size < 5000:
                return 'gemini_2_5_flash'  # Good balance
            else:
                return 'gemini_1_5_pro'  # Best quality for complex tasks
                
        except Exception as e:
            print(f"Error selecting optimal model: {e}")
            return 'gemini_2_5_flash_lite'  # Default to most cost-effective
    
    async def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about available models"""
        model_name = model or self.default_model
        return {
            "model": model_name,
            "provider": "google",
            "available": model_name in self.models,
            "fallback": "openrouter_glm_4_5_air",
            "cost_efficiency": "high" if "flash-lite" in model_name else "medium" if "flash" in model_name else "low"
        }

    async def get_cost_estimate(self, model: str, input_chars: int, output_chars: int) -> Dict[str, Any]:
        """Get cost estimate for model usage"""
        try:
            # Rough token estimation (1 token ≈ 4 characters)
            input_tokens = input_chars // 4
            output_tokens = output_chars // 4
            
            # Current Gemini pricing (as of 2024)
            pricing = {
                "gemini-2.5-flash-lite": {"input": 0.000025, "output": 0.0001},  # Most cost-effective
                "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},       # Good balance
                "gemini-1.5-flash": {"input": 0.0005, "output": 0.0003},         # Standard
                "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},           # Premium
            }
            
            model_pricing = pricing.get(model, pricing["gemini-2.5-flash-lite"])
            
            input_cost = (input_tokens / 1000) * model_pricing["input"]
            output_cost = (output_tokens / 1000) * model_pricing["output"]
            total_cost = input_cost + output_cost
            
            return {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "cost_per_1k_tokens": model_pricing["input"] + model_pricing["output"]
            }
            
        except Exception as e:
            print(f"Error calculating cost estimate: {e}")
            return {"error": str(e)}





