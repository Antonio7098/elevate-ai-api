"""
Long-context LLM service for premium users.
Integrates with Gemini models for large context windows.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import json

from .gemini_service import GeminiService

@dataclass
class ContextManager:
    """Manages large context windows"""
    max_tokens: int = 1000000  # 1M tokens for Gemini 1.5 Pro
    chunk_size: int = 100000   # 100K tokens per chunk
    overlap_size: int = 10000  # 10K token overlap
    
    def split_context(self, context: str) -> List[str]:
        """Split large context into manageable chunks"""
        try:
            # Simple token estimation (4 chars per token)
            estimated_tokens = len(context) // 4
            
            if estimated_tokens <= self.max_tokens:
                return [context]
            
            # Split into chunks
            chunks = []
            words = context.split()
            current_chunk = []
            current_tokens = 0
            
            for word in words:
                word_tokens = len(word) // 4 + 1
                
                if current_tokens + word_tokens > self.chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Keep overlap
                        overlap_words = current_chunk[-self.overlap_size//4:]
                        current_chunk = overlap_words + [word]
                        current_tokens = sum(len(w) // 4 + 1 for w in current_chunk)
                    else:
                        current_chunk = [word]
                        current_tokens = word_tokens
                else:
                    current_chunk.append(word)
                    current_tokens += word_tokens
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
            
        except Exception as e:
            print(f"Error splitting context: {e}")
            return [context]
    
    def reassemble_response(self, responses: List[str]) -> str:
        """Reassemble responses from multiple chunks"""
        try:
            # Simple concatenation with overlap removal
            if len(responses) <= 1:
                return responses[0] if responses else ""
            
            reassembled = responses[0]
            
            for i in range(1, len(responses)):
                # Find overlap and remove it
                current_response = responses[i]
                overlap_found = False
                
                # Look for overlap in the last part of reassembled
                for j in range(len(reassembled) - 100, len(reassembled)):
                    if reassembled[j:] in current_response:
                        reassembled += current_response[len(reassembled[j:]):]
                        overlap_found = True
                        break
                
                if not overlap_found:
                    reassembled += "\n\n" + current_response
            
            return reassembled
            
        except Exception as e:
            print(f"Error reassembling response: {e}")
            return "\n\n".join(responses)

@dataclass
class ModelSelector:
    """Selects optimal Gemini model based on context and complexity"""
    
    def select_model(self, context_size: int, complexity: str, cost_preference: str = "balanced") -> str:
        """Select optimal Gemini model"""
        try:
            if context_size > 500_000:
                return 'gemini_1_5_pro'  # Use Pro for very large contexts
            elif complexity == 'high':
                return 'gemini_1_5_pro'  # Use Pro for complex reasoning
            elif cost_preference == 'cost_effective':
                return 'gemini_1_5_flash'  # Use Flash for cost efficiency
            elif context_size > 100_000:
                return 'gemini_1_5_pro'  # Use Pro for large contexts
            else:
                return 'gemini_1_5_flash'  # Use Flash for smaller contexts
                
        except Exception as e:
            print(f"Error selecting model: {e}")
            return 'gemini_1_5_flash'  # Default to Flash

class LongContextLLM:
    """Long-context LLM service for premium users"""
    
    def __init__(self):
        # Start with Gemini for now, flexible for future model discussions
        self.models = {
            'gemini_1_5_pro': Gemini15Pro(),  # 1M token context
            'gemini_1_5_flash': Gemini15Flash(),  # Fast, cost-effective
            'gemini_2_0_pro': Gemini20Pro()  # Future model when available
        }
        self.context_manager = ContextManager()
        self.model_selector = ModelSelector()
    
    async def generate_with_full_context(self, context: str, query: str) -> str:
        """Generate responses with full context window using Gemini"""
        try:
            # Estimate context size
            context_size = len(context) // 4  # Rough token estimation
            
            # Select optimal model
            model_name = self.model_selector.select_model(context_size, "medium")
            model = self.models[model_name]
            
            # Check if context needs splitting
            if context_size > self.context_manager.max_tokens:
                return await self._generate_with_chunking(context, query, model)
            else:
                return await model.generate_with_context(query, context)
                
        except Exception as e:
            print(f"Error generating with full context: {e}")
            return f"Error generating response: {str(e)}"
    
    async def handle_large_documents(self, document: str, query: str) -> Dict[str, Any]:
        """Handle documents larger than standard context windows"""
        try:
            # Split document into chunks
            chunks = self.context_manager.split_context(document)
            
            # Generate responses for each chunk
            responses = []
            for i, chunk in enumerate(chunks):
                response = await self.generate_with_full_context(chunk, query)
                responses.append({
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "response": response
                })
            
            # Reassemble final response
            final_response = self.context_manager.reassemble_response([r["response"] for r in responses])
            
            return {
                "final_response": final_response,
                "chunk_responses": responses,
                "total_chunks": len(chunks),
                "total_context_size": len(document)
            }
            
        except Exception as e:
            print(f"Error handling large document: {e}")
            return {
                "final_response": f"Error processing document: {str(e)}",
                "chunk_responses": [],
                "total_chunks": 0,
                "total_context_size": len(document)
            }
    
    async def select_optimal_model(self, context_size: int, complexity: str) -> str:
        """Select optimal Gemini model based on context size and complexity"""
        try:
            if context_size > 500_000:
                return 'gemini_1_5_pro'  # Use Pro for very large contexts
            elif complexity == 'high':
                return 'gemini_1_5_pro'  # Use Pro for complex reasoning
            else:
                return 'gemini_1_5_flash'  # Use Flash for cost efficiency
                
        except Exception as e:
            print(f"Error selecting optimal model: {e}")
            return 'gemini_1_5_flash'
    
    async def _generate_with_chunking(self, context: str, query: str, model) -> str:
        """Generate response using chunking for large contexts"""
        try:
            # Split context into chunks
            chunks = self.context_manager.split_context(context)
            
            # Generate responses for each chunk
            responses = []
            for chunk in chunks:
                response = await model.generate_with_context(query, chunk)
                responses.append(response)
            
            # Reassemble final response
            return self.context_manager.reassemble_response(responses)
            
        except Exception as e:
            print(f"Error generating with chunking: {e}")
            return f"Error generating response: {str(e)}"

class Gemini15Pro:
    """Gemini 1.5 Pro model wrapper"""
    
    def __init__(self):
        self.gemini_service = GeminiService()
        self.max_tokens = 1000000  # 1M tokens
        self.model_name = 'gemini_1_5_pro'
    
    async def generate_with_context(self, query: str, context: str) -> str:
        """Generate response with context using Gemini 1.5 Pro"""
        try:
            prompt = f"""
            Context: {context}
            
            User Query: {query}
            
            Please provide a comprehensive response based on the context above.
            """
            
            return await self.gemini_service.generate(prompt, self.model_name)
            
        except Exception as e:
            print(f"Error in Gemini 1.5 Pro generation: {e}")
            return f"Error generating response: {str(e)}"

class Gemini15Flash:
    """Gemini 1.5 Flash model wrapper"""
    
    def __init__(self):
        self.gemini_service = GeminiService()
        self.max_tokens = 100000  # 100K tokens
        self.model_name = 'gemini_1_5_flash'
    
    async def generate_with_context(self, query: str, context: str) -> str:
        """Generate response with context using Gemini 1.5 Flash"""
        try:
            prompt = f"""
            Context: {context}
            
            User Query: {query}
            
            Please provide a response based on the context above.
            """
            
            return await self.gemini_service.generate(prompt, self.model_name)
            
        except Exception as e:
            print(f"Error in Gemini 1.5 Flash generation: {e}")
            return f"Error generating response: {str(e)}"

class Gemini20Pro:
    """Gemini 2.0 Pro model wrapper (future)"""
    
    def __init__(self):
        self.gemini_service = GeminiService()
        self.max_tokens = 2000000  # 2M tokens (future)
        self.model_name = 'gemini_2_0_pro'
    
    async def generate_with_context(self, query: str, context: str) -> str:
        """Generate response with context using Gemini 2.0 Pro"""
        try:
            prompt = f"""
            Context: {context}
            
            User Query: {query}
            
            Please provide a comprehensive response based on the context above.
            """
            
            return await self.gemini_service.generate(prompt, self.model_name)
            
        except Exception as e:
            print(f"Error in Gemini 2.0 Pro generation: {e}")
            return f"Error generating response: {str(e)}"
