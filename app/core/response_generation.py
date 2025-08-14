"""
Response Generation for RAG Chat Core.

This module handles prompt assembly, LLM response generation, factual accuracy checking,
and response formatting for the Elevate AI chat system.
Enhanced with blueprint section hierarchy support.
"""

import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.context_assembly import AssembledContext, CognitiveProfile, SectionContext
from app.core.query_transformer import QueryTransformation, QueryIntent
from app.services.gemini_service import GeminiService


class ResponseType(Enum):
    """Types of responses the system can generate."""
    EXPLANATION = "explanation"
    CLARIFICATION = "clarification"
    QUESTION = "question"
    CORRECTION = "correction"
    ENCOURAGEMENT = "encouragement"
    SUMMARY = "summary"
    ASSESSMENT = "assessment"
    # Section-specific response types
    SECTION_OVERVIEW = "section_overview"
    SECTION_NAVIGATION = "section_navigation"
    SECTION_PROGRESS = "section_progress"
    HIERARCHY_EXPLANATION = "hierarchy_explanation"


class ToneStyle(Enum):
    """Tone styles for response generation."""
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    ENCOURAGING = "encouraging"
    SOCRATIC = "socratic"
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"
    # Section-specific tone styles
    GUIDED = "guided"  # For step-by-step section navigation
    PROGRESS_ORIENTED = "progress_oriented"  # For progress tracking


@dataclass
class ResponseGenerationRequest:
    """Request for response generation."""
    user_query: str
    query_transformation: QueryTransformation
    assembled_context: AssembledContext
    response_type: Optional[ResponseType] = None
    tone_style: Optional[ToneStyle] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    include_sources: bool = True
    metadata: Dict[str, Any] = None
    # Section-specific fields
    section_context: Optional[SectionContext] = None
    include_section_hierarchy: bool = False


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    content: str
    response_type: ResponseType
    tone_style: ToneStyle
    confidence_score: float
    sources: List[str]
    factual_accuracy_score: float
    generation_time_ms: float
    token_count: int
    metadata: Dict[str, Any]
    # Section-specific fields
    section_references: Optional[List[Dict[str, Any]]] = None
    navigation_suggestions: Optional[List[str]] = None


class ResponseGenerator:
    """
    Generates contextually appropriate responses using LLM integration.
    
    This class handles:
    - Prompt assembly with retrieved context
    - LLM response generation with user preferences
    - Factual accuracy checking
    - Response formatting and structure
    """
    
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service
        
        # Response generation templates
        self.prompt_templates = {
            ResponseType.EXPLANATION: self._get_explanation_template(),
            ResponseType.CLARIFICATION: self._get_clarification_template(),
            ResponseType.QUESTION: self._get_question_template(),
            ResponseType.CORRECTION: self._get_correction_template(),
            ResponseType.ENCOURAGEMENT: self._get_encouragement_template(),
            ResponseType.SUMMARY: self._get_summary_template(),
            ResponseType.ASSESSMENT: self._get_assessment_template(),
            # Section-specific templates
            ResponseType.SECTION_OVERVIEW: self._get_section_overview_template(),
            ResponseType.SECTION_NAVIGATION: self._get_section_navigation_template(),
            ResponseType.SECTION_PROGRESS: self._get_section_progress_template(),
            ResponseType.HIERARCHY_EXPLANATION: self._get_hierarchy_explanation_template()
        }
        
        # Tone style configurations
        self.tone_configs = {
            ToneStyle.FORMAL: {
                'instructions': "Use formal, academic language. Be precise and professional.",
                'avoid': "contractions, colloquialisms, overly casual language"
            },
            ToneStyle.CONVERSATIONAL: {
                'instructions': "Use conversational, friendly language. Be approachable and engaging.",
                'avoid': "overly formal language, jargon without explanation"
            },
            ToneStyle.ENCOURAGING: {
                'instructions': "Use encouraging, supportive language. Be positive and motivating.",
                'avoid': "negative language, discouraging phrases"
            },
            ToneStyle.SOCRATIC: {
                'instructions': "Use Socratic method. Ask guiding questions to help discovery.",
                'avoid': "direct answers, spoon-feeding information"
            },
            ToneStyle.TECHNICAL: {
                'instructions': "Use technical, precise language. Include relevant terminology.",
                'avoid': "oversimplification, vague language"
            },
            ToneStyle.SIMPLIFIED: {
                'instructions': "Use simple, clear language. Break down complex concepts.",
                'avoid': "jargon, complex sentence structures"
            },
            # Section-specific tone configurations
            ToneStyle.GUIDED: {
                'instructions': "Use guided, step-by-step language. Provide clear navigation cues.",
                'avoid': "vague directions, overwhelming information"
            },
            ToneStyle.PROGRESS_ORIENTED: {
                'instructions': "Use progress-focused language. Celebrate achievements and set clear next steps.",
                'avoid': "discouraging language, unclear progress indicators"
            }
        }
    
    async def generate_response(self, request: ResponseGenerationRequest) -> GeneratedResponse:
        """
        Generate a response using the assembled context and user preferences.
        
        Args:
            request: Response generation request
            
        Returns:
            Generated response with metadata
        """
        start_time = time.time()
        
        # Determine response type and tone if not specified
        response_type = request.response_type or self._determine_response_type(
            request.user_query, 
            request.query_transformation
        )
        
        tone_style = request.tone_style or self._determine_tone_style(
            request.assembled_context.cognitive_profile
        )
        
        # Assemble the complete prompt
        prompt = self._assemble_prompt(
            request.user_query,
            request.query_transformation,
            request.assembled_context,
            response_type,
            tone_style
        )
        
        # Generate response using LLM
        llm_response = await self._generate_llm_response(
            prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Post-process the response
        formatted_response = self._format_response(
            llm_response,
            response_type,
            tone_style,
            request.include_sources
        )
        
        # Extract sources from context
        sources = self._extract_sources(request.assembled_context)
        
        # Check factual accuracy
        factual_accuracy_score = await self._check_factual_accuracy(
            formatted_response,
            request.assembled_context.retrieved_knowledge
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            request.assembled_context,
            factual_accuracy_score
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        return GeneratedResponse(
            content=formatted_response,
            response_type=response_type,
            tone_style=tone_style,
            confidence_score=confidence_score,
            sources=sources,
            factual_accuracy_score=factual_accuracy_score,
            generation_time_ms=generation_time,
            token_count=len(formatted_response.split()),
            metadata=request.metadata or {}
        )
    
    async def generate_section_overview_response(
        self,
        request: ResponseGenerationRequest,
        section_context: SectionContext
    ) -> GeneratedResponse:
        """
        Generate a response that provides an overview of a specific section.
        
        Args:
            request: Response generation request
            section_context: Context for the specific section
            
        Returns:
            Generated response with section overview
        """
        try:
            start_time = time.time()
            
            # Prepare section-specific context
            section_info = {
                "section_title": section_context.section_title,
                "section_description": section_context.section_description,
                "section_depth": section_context.section_depth,
                "section_path": " > ".join(section_context.section_path),
                "difficulty": section_context.difficulty.value,
                "estimated_time": section_context.estimated_time_minutes,
                "key_concepts": section_context.key_concepts,
                "prerequisites": section_context.prerequisites,
                "learning_objectives": section_context.learning_objectives,
                "user_progress": f"{section_context.user_progress * 100:.1f}%"
            }
            
            # Build prompt with section context
            prompt = self._build_section_prompt(
                request, 
                ResponseType.SECTION_OVERVIEW,
                section_info
            )
            
            # Generate response using LLM
            response_content = await self.gemini_service.generate_response(prompt)
            
            # Process and format response
            formatted_response = self._format_section_response(
                response_content, 
                section_context,
                ResponseType.SECTION_OVERVIEW
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            return GeneratedResponse(
                content=formatted_response,
                response_type=ResponseType.SECTION_OVERVIEW,
                tone_style=request.tone_style or ToneStyle.CONVERSATIONAL,
                confidence_score=0.85,  # Placeholder
                sources=[f"Section: {section_context.section_title}"],
                factual_accuracy_score=0.90,  # Placeholder
                generation_time_ms=generation_time,
                token_count=len(formatted_response.split()),
                metadata={"section_id": section_context.section_id},
                section_references=[{
                    "section_id": section_context.section_id,
                    "section_title": section_context.section_title,
                    "section_path": section_context.section_path
                }],
                navigation_suggestions=self._generate_navigation_suggestions(section_context)
            )
            
        except Exception as e:
            logger.error(f"Failed to generate section overview response: {e}")
            raise
    
    async def generate_section_navigation_response(
        self,
        request: ResponseGenerationRequest,
        section_context: SectionContext
    ) -> GeneratedResponse:
        """
        Generate a response that provides navigation guidance for a section.
        
        Args:
            request: Response generation request
            section_context: Context for the specific section
            
        Returns:
            Generated response with navigation guidance
        """
        try:
            start_time = time.time()
            
            # Prepare navigation-specific context
            navigation_info = {
                "section_title": section_context.section_title,
                "section_path": " > ".join(section_context.section_path),
                "section_depth": section_context.section_depth,
                "parent_section": section_context.parent_section_id,
                "key_concepts": section_context.key_concepts,
                "prerequisites": section_context.prerequisites,
                "estimated_time": section_context.estimated_time_minutes
            }
            
            # Build prompt with navigation context
            prompt = self._build_section_prompt(
                request, 
                ResponseType.SECTION_NAVIGATION,
                navigation_info
            )
            
            # Generate response using LLM
            response_content = await self.gemini_service.generate_response(prompt)
            
            # Process and format response
            formatted_response = self._format_section_response(
                response_content, 
                section_context,
                ResponseType.SECTION_NAVIGATION
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            return GeneratedResponse(
                content=formatted_response,
                response_type=ResponseType.SECTION_NAVIGATION,
                tone_style=request.tone_style or ToneStyle.GUIDED,
                confidence_score=0.85,  # Placeholder
                sources=[f"Section: {section_context.section_title}"],
                factual_accuracy_score=0.90,  # Placeholder
                generation_time_ms=generation_time,
                token_count=len(formatted_response.split()),
                metadata={"section_id": section_context.section_id},
                section_references=[{
                    "section_id": section_context.section_id,
                    "section_title": section_context.section_title,
                    "section_path": section_context.section_path
                }],
                navigation_suggestions=self._generate_navigation_suggestions(section_context)
            )
            
        except Exception as e:
            logger.error(f"Failed to generate section navigation response: {e}")
            raise
    
    def _determine_response_type(self, user_query: str, transformation: QueryTransformation) -> ResponseType:
        """Determine the appropriate response type based on query analysis."""
        query_lower = user_query.lower()
        
        # Check for question indicators
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in query_lower for word in question_words):
            return ResponseType.EXPLANATION
        
        # Check for clarification requests
        clarification_words = ['clarify', 'explain', 'elaborate', 'detail', 'confused']
        if any(word in query_lower for word in clarification_words):
            return ResponseType.CLARIFICATION
        
        # Check for assessment requests
        assessment_words = ['evaluate', 'assess', 'check', 'correct', 'right', 'wrong']
        if any(word in query_lower for word in assessment_words):
            return ResponseType.ASSESSMENT
        
        # Check for summary requests
        summary_words = ['summarize', 'summary', 'overview', 'recap']
        if any(word in query_lower for word in summary_words):
            return ResponseType.SUMMARY
        
        # Default based on query intent
        if transformation.intent == QueryIntent.FACTUAL:
            return ResponseType.EXPLANATION
        elif transformation.intent == QueryIntent.CONCEPTUAL:
            return ResponseType.CLARIFICATION
        elif transformation.intent == QueryIntent.PROCEDURAL:
            return ResponseType.EXPLANATION
        else:
            return ResponseType.EXPLANATION
    
    def _determine_tone_style(self, cognitive_profile: CognitiveProfile) -> ToneStyle:
        """Determine appropriate tone style based on cognitive profile."""
        explanation_style = cognitive_profile.preferred_explanation_style
        
        style_mapping = {
            'detailed': ToneStyle.TECHNICAL,
            'concise': ToneStyle.SIMPLIFIED,
            'examples': ToneStyle.CONVERSATIONAL,
            'theory': ToneStyle.FORMAL,
            'socratic': ToneStyle.SOCRATIC,
            'encouraging': ToneStyle.ENCOURAGING
        }
        
        return style_mapping.get(explanation_style, ToneStyle.CONVERSATIONAL)
    
    def _assemble_prompt(
        self,
        user_query: str,
        transformation: QueryTransformation,
        context: AssembledContext,
        response_type: ResponseType,
        tone_style: ToneStyle
    ) -> str:
        """Assemble the complete prompt for LLM generation."""
        # Get base template
        base_template = self.prompt_templates[response_type]
        
        # Get tone configuration
        tone_config = self.tone_configs[tone_style]
        
        # Prepare context variables
        context_vars = {
            'user_query': user_query,
            'query_intent': transformation.intent.value,
            'expanded_query': transformation.expanded_query,
            'context_content': context.context_summary,
            'conversation_history': self._format_conversation_history(context.conversational_context),
            'retrieved_knowledge': self._format_retrieved_knowledge(context.retrieved_knowledge),
            'learning_profile': self._format_learning_profile(context.cognitive_profile),
            'tone_instructions': tone_config['instructions'],
            'tone_avoid': tone_config['avoid'],
            'current_topic': context.session_context.current_topic or "General discussion",
            'discussed_concepts': ", ".join(context.session_context.discussed_concepts[-5:]) if context.session_context.discussed_concepts else "None"
        }
        
        # Format the template
        formatted_prompt = base_template.format(**context_vars)
        
        return formatted_prompt
    
    async def _generate_llm_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using the LLM service."""
        try:
            response = await self.gemini_service.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            # Fallback response
            return f"I apologize, but I'm having trouble generating a response right now. Please try again. Error: {str(e)}"
    
    def _format_response(
        self,
        response: str,
        response_type: ResponseType,
        tone_style: ToneStyle,
        include_sources: bool
    ) -> str:
        """Format and clean the generated response."""
        # Clean up the response
        formatted = response.strip()
        
        # Remove any system instructions that might have leaked through
        system_patterns = [
            r'<\|.*?\|>',  # System tokens
            r'\[SYSTEM\].*?\[/SYSTEM\]',  # System sections
            r'AI Assistant:',  # AI prefixes
            r'Response:'  # Response prefixes
        ]
        
        for pattern in system_patterns:
            formatted = re.sub(pattern, '', formatted, flags=re.IGNORECASE | re.DOTALL)
        
        # Ensure proper formatting
        formatted = formatted.strip()
        
        # Add response type specific formatting
        if response_type == ResponseType.EXPLANATION:
            if not formatted.endswith('.') and not formatted.endswith('?') and not formatted.endswith('!'):
                formatted += '.'
        
        return formatted
    
    def _extract_sources(self, context: AssembledContext) -> List[str]:
        """Extract source references from the assembled context."""
        sources = []
        
        # Add knowledge base sources if available
        if context.retrieved_knowledge:
            for result in context.retrieved_knowledge[:5]:  # Top 5 sources
                source = f"[{result.locus_type}] {result.blueprint_id}"
                if source not in sources:
                    sources.append(source)
        
        return sources
    
    async def _check_factual_accuracy(self, response: str, retrieved_knowledge: List) -> float:
        """Check factual accuracy of the response against retrieved knowledge."""
        if not retrieved_knowledge:
            return 0.7  # Default score when no knowledge available
        
        # Simple factual accuracy check (in production, this would be more sophisticated)
        accuracy_indicators = []
        
        # Check if response contains information from retrieved knowledge
        knowledge_content = " ".join([result.content for result in retrieved_knowledge])
        
        # Simple overlap check
        response_words = set(response.lower().split())
        knowledge_words = set(knowledge_content.lower().split())
        
        overlap_ratio = len(response_words & knowledge_words) / len(response_words) if response_words else 0
        accuracy_indicators.append(overlap_ratio)
        
        # Check for contradictions (simplified)
        contradiction_words = ['not', 'never', 'incorrect', 'wrong', 'false']
        contradiction_count = sum(1 for word in contradiction_words if word in response.lower())
        contradiction_penalty = min(contradiction_count * 0.1, 0.3)
        
        accuracy_indicators.append(1.0 - contradiction_penalty)
        
        return sum(accuracy_indicators) / len(accuracy_indicators)
    
    def _calculate_confidence_score(self, context: AssembledContext, factual_accuracy: float) -> float:
        """Calculate confidence score for the response."""
        factors = []
        
        # Context quality factor
        factors.append(context.context_quality_score * 0.3)
        
        # Factual accuracy factor
        factors.append(factual_accuracy * 0.4)
        
        # Retrieved knowledge quantity factor
        knowledge_factor = min(len(context.retrieved_knowledge) / 10, 1.0)
        factors.append(knowledge_factor * 0.2)
        
        # Conversation context factor
        conv_factor = min(len(context.conversational_context) / 5, 1.0)
        factors.append(conv_factor * 0.1)
        
        return sum(factors)
    
    def _format_conversation_history(self, messages: List) -> str:
        """Format conversation history for prompt inclusion."""
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages[-3:]:  # Last 3 messages
            role = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _format_retrieved_knowledge(self, knowledge: List) -> str:
        """Format retrieved knowledge for prompt inclusion."""
        if not knowledge:
            return "No specific knowledge retrieved."
        
        formatted = []
        for i, result in enumerate(knowledge[:8], 1):  # Top 8 results
            formatted.append(f"{i}. [{result.locus_type}] {result.content}")
        
        return "\n".join(formatted)
    
    def _format_learning_profile(self, profile: CognitiveProfile) -> str:
        """Format learning profile for prompt inclusion."""
        return f"""
Learning Style: {profile.learning_style}
Preferred Difficulty: {profile.preferred_difficulty}
Explanation Style: {profile.preferred_explanation_style}
Learning Pace: {profile.learning_pace}
""".strip()
    
    # Template methods for different response types
    def _get_explanation_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Your role is to provide clear, accurate explanations tailored to the user's learning profile and context.

USER QUERY: {user_query}
QUERY INTENT: {query_intent}
EXPANDED QUERY: {expanded_query}

CONVERSATION CONTEXT:
{conversation_history}

CURRENT TOPIC: {current_topic}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide a comprehensive explanation that:
1. Directly addresses the user's question
2. Uses information from the retrieved knowledge
3. Matches the user's learning style and preferred difficulty
4. Builds on the conversation context
5. Follows the specified tone instructions

Response:
"""
    
    def _get_clarification_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. The user needs clarification on a topic.

USER QUERY: {user_query}
QUERY INTENT: {query_intent}

CONVERSATION CONTEXT:
{conversation_history}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide clarification that:
1. Identifies what might be confusing
2. Provides clearer explanations
3. Uses appropriate examples
4. Matches the user's learning preferences

Response:
"""
    
    def _get_question_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Generate thoughtful questions to guide the user's learning.

USER QUERY: {user_query}
CURRENT TOPIC: {current_topic}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}

Please generate questions that:
1. Are relevant to the current topic
2. Help the user think deeper about the concepts
3. Are appropriate for their learning level
4. Encourage active learning

Response:
"""
    
    def _get_correction_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. The user needs correction or feedback.

USER QUERY: {user_query}
CONVERSATION CONTEXT:
{conversation_history}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide correction that:
1. Gently identifies any misconceptions
2. Provides accurate information
3. Explains why something is correct/incorrect
4. Is supportive and encouraging

Response:
"""
    
    def _get_encouragement_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Provide encouragement and motivation.

USER QUERY: {user_query}
DISCUSSED CONCEPTS: {discussed_concepts}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}

Please provide encouragement that:
1. Acknowledges the user's effort
2. Highlights progress made
3. Motivates continued learning
4. Is genuine and supportive

Response:
"""
    
    def _get_summary_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Provide a comprehensive summary.

USER QUERY: {user_query}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}

Please provide a summary that:
1. Captures key points discussed
2. Organizes information logically
3. Highlights important concepts
4. Matches the user's learning style

Response:
"""
    
    def _get_assessment_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Provide assessment and evaluation.

USER QUERY: {user_query}
CONVERSATION CONTEXT:
{conversation_history}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}

Please provide assessment that:
1. Evaluates understanding accurately
2. Provides constructive feedback
3. Suggests areas for improvement
4. Is encouraging and supportive

Response:
"""

    def _get_section_overview_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Your role is to provide a clear, concise overview of a specific section or topic.

USER QUERY: {user_query}
QUERY INTENT: {query_intent}
EXPANDED QUERY: {expanded_query}

CONVERSATION CONTEXT:
{conversation_history}

CURRENT TOPIC: {current_topic}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide a concise overview of the following section:
{section_context}

Response:
"""

    def _get_section_navigation_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Your role is to provide clear, step-by-step navigation cues for a user to find specific information within a document or section.

USER QUERY: {user_query}
QUERY INTENT: {query_intent}
EXPANDED QUERY: {expanded_query}

CONVERSATION CONTEXT:
{conversation_history}

CURRENT TOPIC: {current_topic}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide a step-by-step navigation guide for the user to find the following information:
{section_context}

Response:
"""

    def _get_section_progress_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Your role is to provide progress-focused feedback and encouragement to the user.

USER QUERY: {user_query}
QUERY INTENT: {query_intent}
EXPANDED QUERY: {expanded_query}

CONVERSATION CONTEXT:
{conversation_history}

CURRENT TOPIC: {current_topic}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide progress-oriented feedback for the user's current learning journey.
{section_context}

Response:
"""

    def _get_hierarchy_explanation_template(self) -> str:
        return """
You are an expert educational AI assistant for Elevate AI. Your role is to provide a clear, hierarchical explanation of the blueprint section hierarchy.

USER QUERY: {user_query}
QUERY INTENT: {query_intent}
EXPANDED QUERY: {expanded_query}

CONVERSATION CONTEXT:
{conversation_history}

CURRENT TOPIC: {current_topic}
DISCUSSED CONCEPTS: {discussed_concepts}

RETRIEVED KNOWLEDGE:
{retrieved_knowledge}

USER LEARNING PROFILE:
{learning_profile}

TONE INSTRUCTIONS: {tone_instructions}
AVOID: {tone_avoid}

Please provide a hierarchical explanation of the blueprint section hierarchy.
{section_context}

Response:
"""

    def _build_section_prompt(
        self, 
        request: ResponseGenerationRequest, 
        response_type: ResponseType,
        section_info: Dict[str, Any]
    ) -> str:
        """
        Build a prompt for section-specific response generation.
        
        Args:
            request: Response generation request
            response_type: Type of response to generate
            section_info: Section-specific information
            
        Returns:
            Formatted prompt string
        """
        template = self.prompt_templates.get(response_type, "")
        tone_config = self.tone_configs.get(request.tone_style or ToneStyle.CONVERSATIONAL, {})
        
        # Format section context
        section_context = f"""
Section Information:
- Title: {section_info.get('section_title', 'N/A')}
- Path: {section_info.get('section_path', 'N/A')}
- Depth: {section_info.get('section_depth', 'N/A')}
- Difficulty: {section_info.get('difficulty', 'N/A')}
- Estimated Time: {section_info.get('estimated_time', 'N/A')} minutes
- Key Concepts: {', '.join(section_info.get('key_concepts', []))}
- Prerequisites: {', '.join(section_info.get('prerequisites', []))}
- Learning Objectives: {', '.join(section_info.get('learning_objectives', []))}
- User Progress: {section_info.get('user_progress', 'N/A')}
"""
        
        return template.format(
            user_query=request.user_query,
            query_intent=request.query_transformation.intent.value,
            expanded_query=request.query_transformation.expanded_query,
            conversation_history=self._format_conversation_history(request.assembled_context),
            current_topic=request.assembled_context.session_context.current_topic or "General",
            discussed_concepts=", ".join(request.assembled_context.session_context.discussed_concepts),
            retrieved_knowledge=self._format_retrieved_knowledge(request.assembled_context),
            learning_profile=self._format_learning_profile(request.assembled_context.cognitive_profile),
            tone_instructions=tone_config.get('instructions', ''),
            tone_avoid=tone_config.get('avoid', ''),
            section_context=section_context
        )
    
    def _format_section_response(
        self, 
        response_content: str, 
        section_context: SectionContext,
        response_type: ResponseType
    ) -> str:
        """
        Format a section-specific response with additional context.
        
        Args:
            response_content: Raw response content from LLM
            section_context: Section context information
            response_type: Type of response generated
            
        Returns:
            Formatted response string
        """
        # Add section context header
        header = f"📚 **{section_context.section_title}**\n"
        path_info = f"📍 **Location:** {' > '.join(section_context.section_path)}\n"
        progress_info = f"📊 **Your Progress:** {section_context.user_progress * 100:.1f}%\n\n"
        
        # Add response type indicator
        type_indicators = {
            ResponseType.SECTION_OVERVIEW: "🔍 **Section Overview**\n",
            ResponseType.SECTION_NAVIGATION: "🧭 **Navigation Guide**\n",
            ResponseType.SECTION_PROGRESS: "📈 **Progress Update**\n",
            ResponseType.HIERARCHY_EXPLANATION: "🌳 **Hierarchy Structure**\n"
        }
        
        type_indicator = type_indicators.get(response_type, "")
        
        # Combine all parts
        formatted_response = f"{header}{path_info}{progress_info}{type_indicator}\n{response_content}"
        
        return formatted_response
    
    def _generate_navigation_suggestions(self, section_context: SectionContext) -> List[str]:
        """
        Generate navigation suggestions based on section context.
        
        Args:
            section_context: Section context information
            
        Returns:
            List of navigation suggestions
        """
        suggestions = []
        
        # Add section-specific suggestions
        if section_context.user_progress < 0.3:
            suggestions.append("Start with the foundational concepts in this section")
            suggestions.append("Review any prerequisites before diving deeper")
        elif section_context.user_progress < 0.7:
            suggestions.append("Focus on the key concepts and learning objectives")
            suggestions.append("Practice with the provided examples and exercises")
        else:
            suggestions.append("Review and reinforce your understanding")
            suggestions.append("Consider exploring related sections or advanced topics")
        
        # Add navigation suggestions
        if section_context.parent_section_id:
            suggestions.append("You can return to the parent section for broader context")
        
        if section_context.section_depth < 3:
            suggestions.append("Explore child sections for more detailed information")
        
        return suggestions
# PromptTemplate class for response_generation.py
# This will be appended to the main file

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PromptTemplate:
    """Template for generating prompts for different response types."""
    template_type: str
    template_content: str
    required_variables: List[str]
    optional_variables: List[str] = None
    tone_instructions: str = ""
    examples: List[str] = None
    
    def __post_init__(self):
        if self.optional_variables is None:
            self.optional_variables = []
        if self.examples is None:
            self.examples = []
    
    def format_template(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            return self.template_content.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")
    
    def get_default_variables(self) -> Dict[str, str]:
        """Get default values for optional variables."""
        defaults = {
            'tone_instructions': self.tone_instructions,
            'conversation_history': 'No previous conversation.',
            'learning_profile': 'No specific learning profile provided.',
            'retrieved_knowledge': 'No specific knowledge retrieved.',
            'query_intent': 'General inquiry'
        }
        return {var: defaults.get(var, '') for var in self.optional_variables}
