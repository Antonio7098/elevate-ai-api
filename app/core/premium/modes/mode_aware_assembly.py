"""
Mode-Aware Context Assembly for premium features.
Provides mode-specific retrieval, reranking, and compression strategies.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..context_assembly_agent import CAARequest

class ModeStrategy(ABC):
    """Abstract base class for mode-specific strategies"""
    
    @abstractmethod
    async def apply_retrieval_strategy(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Apply mode-specific retrieval strategy"""
        pass
    
    @abstractmethod
    async def apply_reranking_strategy(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply mode-specific reranking strategy"""
        pass
    
    @abstractmethod
    async def apply_compression_strategy(self, context: str, target_tokens: int) -> str:
        """Apply mode-specific compression strategy"""
        pass

class ChatModeStrategy(ModeStrategy):
    """Strategy for chat mode - emphasis on session memory and user tone"""
    
    async def apply_retrieval_strategy(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Retrieve with emphasis on conversation continuity"""
        # Include conversation history in retrieval
        conversation_history = user_context.get("conversation_history", [])
        session_context = user_context.get("session_context", {})
        
        # Augment query with session context
        augmented_queries = [query]
        
        if conversation_history:
            # Add context from recent conversation
            recent_context = " ".join([msg.get("content", "") for msg in conversation_history[-3:]])
            augmented_queries.append(f"{query} (context: {recent_context})")
        
        if session_context.get("current_topic"):
            augmented_queries.append(f"{query} (topic: {session_context['current_topic']})")
        
        return augmented_queries
    
    async def apply_reranking_strategy(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank with emphasis on conversational relevance"""
        reranked = []
        
        for chunk in chunks:
            # Boost conversational chunks
            conversational_score = 0.1 if any(word in chunk.get("content", "").lower() 
                                            for word in ["example", "let me", "suppose", "imagine"]) else 0.0
            
            # Boost chunks with user-friendly language
            friendly_score = 0.1 if any(word in chunk.get("content", "").lower() 
                                       for word in ["you can", "we can", "here's how", "step by step"]) else 0.0
            
            chunk["chat_score"] = chunk.get("rerank_score", 0.5) + conversational_score + friendly_score
            reranked.append(chunk)
        
        reranked.sort(key=lambda x: x["chat_score"], reverse=True)
        return reranked
    
    async def apply_compression_strategy(self, context: str, target_tokens: int) -> str:
        """Compress with emphasis on conversational flow"""
        # Preserve conversational elements
        words = context.split()
        if len(words) <= target_tokens:
            return context
        
        # Prioritize conversational phrases
        conversational_phrases = ["for example", "let me explain", "here's how", "step by step"]
        preserved_words = []
        
        for phrase in conversational_phrases:
            if phrase in context.lower():
                preserved_words.extend(phrase.split())
        
        # Add remaining words up to target
        remaining_words = [w for w in words if w not in preserved_words]
        preserved_words.extend(remaining_words[:target_tokens - len(preserved_words)])
        
        return " ".join(preserved_words)

class QuizModeStrategy(ModeStrategy):
    """Strategy for quiz mode - canonical facts and distractor sources"""
    
    async def apply_retrieval_strategy(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Retrieve with emphasis on factual accuracy"""
        # Focus on canonical facts and definitions
        augmented_queries = [
            query,
            f"{query} definition",
            f"{query} facts",
            f"{query} key concepts"
        ]
        
        # Add difficulty-based queries
        user_level = user_context.get("analytics", {}).get("masteryLevel", "BEGINNER")
        if user_level == "ADVANCED":
            augmented_queries.append(f"{query} advanced concepts")
        elif user_level == "INTERMEDIATE":
            augmented_queries.append(f"{query} intermediate concepts")
        
        return augmented_queries
    
    async def apply_reranking_strategy(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank with emphasis on factual accuracy and quiz suitability"""
        reranked = []
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            
            # Boost factual chunks
            factual_score = 0.2 if any(word in content for word in ["definition", "is a", "refers to", "consists of"]) else 0.0
            
            # Boost quiz-friendly chunks
            quiz_score = 0.1 if any(word in content for word in ["example", "case", "scenario", "question"]) else 0.0
            
            # Penalize overly complex chunks
            complexity_penalty = -0.1 if len(content.split()) > 50 else 0.0
            
            chunk["quiz_score"] = chunk.get("rerank_score", 0.5) + factual_score + quiz_score + complexity_penalty
            reranked.append(chunk)
        
        reranked.sort(key=lambda x: x["quiz_score"], reverse=True)
        return reranked
    
    async def apply_compression_strategy(self, context: str, target_tokens: int) -> str:
        """Compress with emphasis on key facts and definitions"""
        # Prioritize factual statements
        sentences = context.split(". ")
        factual_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["is", "are", "refers to", "consists of", "defined as"]):
                factual_sentences.append(sentence)
        
        # Combine factual sentences
        compressed = ". ".join(factual_sentences)
        words = compressed.split()
        
        if len(words) <= target_tokens:
            return compressed
        
        return " ".join(words[:target_tokens])

class DeepDiveModeStrategy(ModeStrategy):
    """Strategy for deep-dive mode - full sources and step-by-step blueprints"""
    
    async def apply_retrieval_strategy(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Retrieve comprehensive information for deep learning"""
        # Comprehensive retrieval for deep understanding
        augmented_queries = [
            query,
            f"{query} detailed explanation",
            f"{query} step by step",
            f"{query} complete guide",
            f"{query} comprehensive overview",
            f"{query} background context",
            f"{query} related concepts"
        ]
        
        return augmented_queries
    
    async def apply_reranking_strategy(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank with emphasis on comprehensive coverage"""
        reranked = []
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            
            # Boost comprehensive chunks
            comprehensive_score = 0.2 if any(word in content for word in ["detailed", "comprehensive", "complete", "thorough"]) else 0.0
            
            # Boost step-by-step content
            step_score = 0.15 if any(word in content for word in ["step", "first", "then", "finally", "next"]) else 0.0
            
            # Boost contextual chunks
            context_score = 0.1 if any(word in content for word in ["background", "context", "history", "overview"]) else 0.0
            
            chunk["deep_dive_score"] = chunk.get("rerank_score", 0.5) + comprehensive_score + step_score + context_score
            reranked.append(chunk)
        
        reranked.sort(key=lambda x: x["deep_dive_score"], reverse=True)
        return reranked
    
    async def apply_compression_strategy(self, context: str, target_tokens: int) -> str:
        """Compress while preserving comprehensive coverage"""
        # Preserve detailed explanations
        sentences = context.split(". ")
        detailed_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["detailed", "comprehensive", "step", "explanation", "guide"]):
                detailed_sentences.append(sentence)
        
        # If no detailed sentences found, use all sentences
        if not detailed_sentences:
            detailed_sentences = sentences
        
        # Combine detailed sentences
        compressed = ". ".join(detailed_sentences)
        words = compressed.split()
        
        if len(words) <= target_tokens:
            return compressed
        
        return " ".join(words[:target_tokens])

class WalkThroughModeStrategy(ModeStrategy):
    """Strategy for walk-through mode - progressive disclosure"""
    
    async def apply_retrieval_strategy(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Retrieve with progressive complexity"""
        # Progressive disclosure approach
        user_level = user_context.get("analytics", {}).get("masteryLevel", "BEGINNER")
        
        if user_level == "BEGINNER":
            augmented_queries = [
                f"{query} basics",
                f"{query} introduction",
                f"{query} simple explanation"
            ]
        elif user_level == "INTERMEDIATE":
            augmented_queries = [
                f"{query} intermediate",
                f"{query} practical examples",
                f"{query} common applications"
            ]
        else:  # ADVANCED
            augmented_queries = [
                f"{query} advanced",
                f"{query} complex scenarios",
                f"{query} expert techniques"
            ]
        
        return augmented_queries
    
    async def apply_reranking_strategy(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank with emphasis on progressive learning"""
        reranked = []
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            
            # Boost progressive content
            progressive_score = 0.2 if any(word in content for word in ["first", "then", "next", "finally", "step"]) else 0.0
            
            # Boost practical content
            practical_score = 0.15 if any(word in content for word in ["example", "practice", "application", "use case"]) else 0.0
            
            # Boost foundational content
            foundational_score = 0.1 if any(word in content for word in ["basic", "fundamental", "core", "essential"]) else 0.0
            
            chunk["walk_through_score"] = chunk.get("rerank_score", 0.5) + progressive_score + practical_score + foundational_score
            reranked.append(chunk)
        
        reranked.sort(key=lambda x: x["walk_through_score"], reverse=True)
        return reranked
    
    async def apply_compression_strategy(self, context: str, target_tokens: int) -> str:
        """Compress with emphasis on progressive structure"""
        # Preserve step-by-step structure
        sentences = context.split(". ")
        step_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["step", "first", "then", "next", "finally", "example"]):
                step_sentences.append(sentence)
        
        # If no step sentences found, use all sentences
        if not step_sentences:
            step_sentences = sentences
        
        # Combine step sentences
        compressed = ". ".join(step_sentences)
        words = compressed.split()
        
        if len(words) <= target_tokens:
            return compressed
        
        return " ".join(words[:target_tokens])

class NoteEditingModeStrategy(ModeStrategy):
    """Strategy for note editing mode - structured content and citations"""
    
    async def apply_retrieval_strategy(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Retrieve with emphasis on structured content"""
        # Focus on structured, citable content
        augmented_queries = [
            query,
            f"{query} structured notes",
            f"{query} key points",
            f"{query} summary",
            f"{query} bullet points"
        ]
        
        return augmented_queries
    
    async def apply_reranking_strategy(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank with emphasis on note-friendly content"""
        reranked = []
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            
            # Boost structured content
            structured_score = 0.2 if any(word in content for word in ["key", "point", "summary", "bullet", "list"]) else 0.0
            
            # Boost citable content
            citable_score = 0.15 if any(word in content for word in ["according to", "research shows", "studies indicate"]) else 0.0
            
            # Boost concise content
            concise_score = 0.1 if len(content.split()) < 30 else 0.0
            
            chunk["note_editing_score"] = chunk.get("rerank_score", 0.5) + structured_score + citable_score + concise_score
            reranked.append(chunk)
        
        reranked.sort(key=lambda x: x["note_editing_score"], reverse=True)
        return reranked
    
    async def apply_compression_strategy(self, context: str, target_tokens: int) -> str:
        """Compress with emphasis on structured notes"""
        # Create structured note format
        sentences = context.split(". ")
        key_points = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["key", "important", "main", "essential", "critical"]):
                key_points.append(f"• {sentence}")
        
        # If no key points found, create structured format from all sentences
        if not key_points:
            key_points = [f"• {sentence}" for sentence in sentences[:3]]  # Take first 3 sentences
        
        # Combine key points
        compressed = "\n".join(key_points)
        words = compressed.split()
        
        if len(words) <= target_tokens:
            return compressed
        
        return " ".join(words[:target_tokens])

class ModeAwareAssembly:
    """Mode-aware context assembly coordinator"""
    
    def __init__(self):
        self.mode_strategies = {
            'chat': ChatModeStrategy(),
            'quiz': QuizModeStrategy(),
            'deep_dive': DeepDiveModeStrategy(),
            'walk_through': WalkThroughModeStrategy(),
            'note_editing': NoteEditingModeStrategy()
        }
    
    async def get_mode_strategy(self, mode: str) -> ModeStrategy:
        """Get appropriate strategy for the specified mode"""
        return self.mode_strategies.get(mode, ChatModeStrategy())
    
    async def apply_mode_retrieval(self, mode: str, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Apply mode-specific retrieval strategy"""
        strategy = await self.get_mode_strategy(mode)
        return await strategy.apply_retrieval_strategy(query, user_context)
    
    async def apply_mode_reranking(self, mode: str, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply mode-specific reranking"""
        strategy = await self.get_mode_strategy(mode)
        return await strategy.apply_reranking_strategy(chunks, query)
    
    async def apply_mode_compression(self, mode: str, context: str, target_tokens: int) -> str:
        """Apply mode-specific compression"""
        strategy = await self.get_mode_strategy(mode)
        return await strategy.apply_compression_strategy(context, target_tokens)
