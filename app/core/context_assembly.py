"""
Context Assembly Logic for RAG Chat Core.

This module implements the multi-tier memory system and context assembly logic:
- Tier 1: Conversational Buffer (last 5-10 messages)
- Tier 2: Session State JSON (structured scratchpad)
- Tier 3: Knowledge Base (vector database) and Cognitive Profile
Enhanced with blueprint section hierarchy support.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from app.core.rag_search import RAGSearchService, RAGSearchRequest, RAGSearchResult
from app.core.query_transformer import QueryTransformation, QueryIntent
from app.models.blueprint_centric import BlueprintSection, DifficultyLevel, UueStage

logger = logging.getLogger(__name__)


class ContextTier(Enum):
    """Different tiers of context in the memory system."""
    CONVERSATIONAL_BUFFER = "conversational_buffer"  # Tier 1
    SESSION_STATE = "session_state"                  # Tier 2
    KNOWLEDGE_BASE = "knowledge_base"                # Tier 3
    COGNITIVE_PROFILE = "cognitive_profile"          # Tier 3 - Blueprint sections
    SECTION_HIERARCHY = "section_hierarchy"          # Tier 3 - Blueprint sections


@dataclass
class ConversationMessage:
    """Structure for conversation messages in the buffer."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_id: str
    metadata: Dict[str, Any]


@dataclass
class SessionState:
    """Structure for session state information."""
    session_id: str
    user_id: str
    current_topic: Optional[str]
    learning_objectives: List[str]
    discussed_concepts: List[str]
    user_questions: List[str]
    clarifications_needed: List[str]
    progress_indicators: Dict[str, Any]
    context_summary: str
    last_updated: datetime
    metadata: Dict[str, Any]
    # Section-specific fields
    current_section: Optional[Dict[str, Any]] = None
    section_progress: Dict[int, float] = None  # section_id -> progress percentage
    section_history: List[Dict[str, Any]] = None  # sections visited in this session


@dataclass
class CognitiveProfile:
    """Structure for user cognitive profile."""
    user_id: str
    learning_style: str  # visual, auditory, kinesthetic, reading/writing
    preferred_difficulty: str  # beginner, intermediate, advanced
    knowledge_level: Dict[str, str]  # subject -> level
    learning_pace: str  # slow, medium, fast
    preferred_explanation_style: str  # detailed, concise, examples, theory
    misconceptions: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]
    last_updated: datetime


@dataclass
class SectionContext:
    """Context information for a specific blueprint section."""
    section_id: int
    section_title: str
    section_description: Optional[str]
    section_depth: int
    section_path: List[str]
    parent_section_id: Optional[int]
    difficulty: DifficultyLevel
    estimated_time_minutes: Optional[int]
    content_summary: str
    key_concepts: List[str]
    prerequisites: List[str]
    learning_objectives: List[str]
    user_progress: float  # 0.0 to 1.0
    last_accessed: datetime
    metadata: Dict[str, Any]


@dataclass
class AssembledContext:
    """Complete assembled context for RAG generation."""
    conversational_context: List[ConversationMessage]
    session_context: SessionState
    retrieved_knowledge: List[RAGSearchResult]
    cognitive_profile: CognitiveProfile
    context_summary: str
    total_tokens: int
    assembly_time_ms: float
    context_quality_score: float
    # Section-specific context
    section_context: Optional[SectionContext] = None
    section_hierarchy: Optional[List[Dict[str, Any]]] = None


class ContextAssembler:
    """
    Assembles context from the multi-tier memory system for RAG generation.
    
    This class coordinates the three tiers of memory:
    1. Conversational Buffer: Recent conversation history
    2. Session State: Structured session information
    3. Knowledge Base + Cognitive Profile: Long-term knowledge and user modeling
    """
    
    def __init__(self, search_service: RAGSearchService, max_context_tokens: int = 8000):
        self.search_service = search_service
        self.max_context_tokens = max_context_tokens
        
        # Configuration for different context tiers
        self.tier_config = {
            ContextTier.CONVERSATIONAL_BUFFER: {
                'max_messages': 10,
                'max_tokens': 2000,
                'priority_weight': 0.4
            },
            ContextTier.SESSION_STATE: {
                'max_tokens': 1000,
                'priority_weight': 0.2
            },
            ContextTier.KNOWLEDGE_BASE: {
                'max_results': 15,
                'max_tokens': 4000,
                'priority_weight': 0.3
            },
            ContextTier.COGNITIVE_PROFILE: {
                'max_tokens': 1000,
                'priority_weight': 0.1
            }
        }
        
        # In-memory storage for demo (in production, this would be a database)
        self.conversation_buffers: Dict[str, List[ConversationMessage]] = {}
        self.session_states: Dict[str, SessionState] = {}
        self.cognitive_profiles: Dict[str, CognitiveProfile] = {}
    
    async def assemble_context(
        self,
        user_id: str,
        session_id: str,
        current_query: str,
        query_transformation: QueryTransformation
    ) -> AssembledContext:
        """
        Assemble complete context from all memory tiers.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            current_query: Current user query
            query_transformation: Transformed query information
            
        Returns:
            AssembledContext with information from all tiers
        """
        start_time = time.time()
        
        # Get or create session state
        session_state = self._get_or_create_session_state(user_id, session_id)
        
        # Get or create cognitive profile
        cognitive_profile = self._get_or_create_cognitive_profile(user_id)
        
        # Get conversational buffer
        conversational_context = self._get_conversational_buffer(session_id)
        
        # Retrieve relevant knowledge
        retrieved_knowledge = await self._retrieve_relevant_knowledge(
            user_id,  # CRITICAL: Pass user_id for blueprint isolation
            current_query, 
            query_transformation, 
            session_state, 
            cognitive_profile
        )
        
        # Create context summary
        context_summary = self._create_context_summary(
            conversational_context, 
            session_state, 
            retrieved_knowledge
        )
        
        # Calculate token usage
        total_tokens = self._calculate_total_tokens(
            conversational_context, 
            session_state, 
            retrieved_knowledge, 
            cognitive_profile
        )
        
        # Apply context pruning if needed
        if total_tokens > self.max_context_tokens:
            conversational_context, retrieved_knowledge = self._prune_context(
                conversational_context, 
                retrieved_knowledge, 
                target_tokens=self.max_context_tokens
            )
            total_tokens = self._calculate_total_tokens(
                conversational_context, 
                session_state, 
                retrieved_knowledge, 
                cognitive_profile
            )
        
        # Calculate context quality score
        context_quality_score = self._calculate_context_quality(
            conversational_context, 
            session_state, 
            retrieved_knowledge, 
            query_transformation
        )
        
        assembly_time = (time.time() - start_time) * 1000
        
        return AssembledContext(
            conversational_context=conversational_context,
            session_context=session_state,
            retrieved_knowledge=retrieved_knowledge,
            cognitive_profile=cognitive_profile,
            context_summary=context_summary,
            total_tokens=total_tokens,
            assembly_time_ms=assembly_time,
            context_quality_score=context_quality_score
        )
    
    async def assemble_section_context(
        self,
        user_id: str,
        section_id: int,
        blueprint_id: int,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        include_hierarchy: bool = True
    ) -> SectionContext:
        """
        Assemble context for a specific blueprint section.
        
        Args:
            user_id: User ID for filtering
            section_id: ID of the target section
            blueprint_id: ID of the blueprint
            conversation_history: Recent conversation history
            include_hierarchy: Whether to include section hierarchy information
            
        Returns:
            SectionContext with assembled information
        """
        try:
            start_time = time.time()
            
            # Get section information from vector store
            sections = await self.search_service.vector_store.get_blueprint_sections(blueprint_id)
            target_section = None
            
            for section in sections:
                if section.get("id") == section_id:
                    target_section = section
                    break
            
            if not target_section:
                raise ValueError(f"Section {section_id} not found in blueprint {blueprint_id}")
            
            # Build section path
            section_path = self._build_section_path(sections, section_id)
            
            # Get section-specific content
            section_content = await self._get_section_content(
                user_id, section_id, blueprint_id, conversation_history
            )
            
            # Calculate user progress (placeholder - would integrate with mastery tracking)
            user_progress = await self._calculate_section_progress(user_id, section_id)
            
            # Create section context
            section_context = SectionContext(
                section_id=target_section["id"],
                section_title=target_section["title"],
                section_description=target_section.get("description"),
                section_depth=target_section["depth"],
                section_path=section_path,
                parent_section_id=target_section.get("parent_section_id"),
                difficulty=DifficultyLevel(target_section.get("difficulty", "BEGINNER")),
                estimated_time_minutes=target_section.get("estimated_time_minutes"),
                content_summary=section_content.get("summary", ""),
                key_concepts=section_content.get("key_concepts", []),
                prerequisites=section_content.get("prerequisites", []),
                learning_objectives=section_content.get("learning_objectives", []),
                user_progress=user_progress,
                last_accessed=datetime.now(),
                metadata=target_section.get("metadata", {})
            )
            
            assembly_time = (time.time() - start_time) * 1000
            logger.info(f"Section context assembled in {assembly_time:.2f}ms for section {section_id}")
            
            return section_context
            
        except Exception as e:
            logger.error(f"Failed to assemble section context: {e}")
            raise
    
    async def _get_section_content(
        self,
        user_id: str,
        section_id: int,
        blueprint_id: int,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Get content and context for a specific section.
        
        Args:
            user_id: User ID for filtering
            section_id: Section ID
            blueprint_id: Blueprint ID
            conversation_history: Recent conversation history
            
        Returns:
            Dictionary with section content information
        """
        try:
            # Create search request for section-specific content
            search_request = RAGSearchRequest(
                query="",  # Empty query to get all section content
                user_id=user_id,
                top_k=20,
                section_filter={"section_id": section_id, "blueprint_id": blueprint_id},
                include_section_hierarchy=True
            )
            
            # Perform section-specific search
            search_response = await self.search_service.search(search_request)
            
            # Extract and organize content
            content = {
                "summary": "",
                "key_concepts": [],
                "prerequisites": [],
                "learning_objectives": [],
                "related_content": []
            }
            
            if search_response.results:
                # Build summary from top results
                summaries = [result.content[:200] for result in search_response.results[:5]]
                content["summary"] = " ".join(summaries)
                
                # Extract key concepts from metadata
                for result in search_response.results:
                    if result.metadata:
                        if result.metadata.get("locus_type") == "key_term":
                            content["key_concepts"].append(result.metadata.get("locus_title", ""))
                        elif result.metadata.get("locus_type") == "foundational_concept":
                            content["prerequisites"].append(result.metadata.get("locus_title", ""))
                
                # Add related content
                content["related_content"] = [
                    {
                        "id": result.id,
                        "content": result.content[:100],
                        "type": result.metadata.get("locus_type", "unknown"),
                        "relevance": result.score
                    }
                    for result in search_response.results
                ]
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to get section content: {e}")
            return {
                "summary": "Content not available",
                "key_concepts": [],
                "prerequisites": [],
                "learning_objectives": [],
                "related_content": []
            }
    
    async def _calculate_section_progress(self, user_id: str, section_id: int) -> float:
        """
        Calculate user progress for a specific section.
        
        Args:
            user_id: User ID
            section_id: Section ID
            
        Returns:
            Progress as float between 0.0 and 1.0
        """
        try:
            # This would integrate with mastery tracking service
            # For now, return a placeholder value
            # In practice, this would query user progress from the mastery tracking system
            
            # Placeholder implementation
            import random
            return round(random.uniform(0.0, 1.0), 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate section progress: {e}")
            return 0.0
    
    def _build_section_path(self, sections: List[Dict[str, Any]], target_section_id: int) -> List[str]:
        """
        Build the path from root to a specific section.
        
        Args:
            sections: List of section dictionaries
            target_section_id: Target section ID
            
        Returns:
            List of section titles from root to target
        """
        try:
            # Find target section
            target_section = None
            for section in sections:
                if section.get("id") == target_section_id:
                    target_section = section
                    break
            
            if not target_section:
                return []
            
            # Build path recursively
            path = [target_section["title"]]
            current = target_section
            
            while current.get("parent_section_id"):
                parent_id = current["parent_section_id"]
                parent_section = None
                
                for section in sections:
                    if section.get("id") == parent_id:
                        parent_section = section
                        break
                
                if parent_section:
                    path.insert(0, parent_section["title"])
                    current = parent_section
                else:
                    break
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to build section path: {e}")
            return []
    
    async def assemble_hierarchical_context(
        self,
        user_id: str,
        blueprint_id: int,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Assemble context for the entire blueprint hierarchy.
        
        Args:
            user_id: User ID for filtering
            blueprint_id: Blueprint ID
            max_depth: Maximum depth to include
            
        Returns:
            List of section contexts organized by hierarchy
        """
        try:
            # Get all sections for the blueprint
            sections = await self.search_service.vector_store.get_blueprint_sections(blueprint_id)
            
            if not sections:
                return []
            
            # Organize sections by hierarchy
            hierarchy = self._organize_sections_by_hierarchy(sections, max_depth)
            
            # Add progress and context information
            for section_info in hierarchy:
                section_id = section_info["id"]
                section_info["user_progress"] = await self._calculate_section_progress(user_id, section_id)
                section_info["content_summary"] = await self._get_section_summary(user_id, section_id, blueprint_id)
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to assemble hierarchical context: {e}")
            return []
    
    def _organize_sections_by_hierarchy(
        self, 
        sections: List[Dict[str, Any]], 
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Organize sections into a hierarchical structure.
        
        Args:
            sections: List of section dictionaries
            max_depth: Maximum depth to include
            
        Returns:
            List of sections organized by hierarchy
        """
        try:
            # Create section map
            section_map = {section["id"]: section for section in sections}
            
            # Find root sections (no parent)
            root_sections = [s for s in sections if not s.get("parent_section_id")]
            
            # Build hierarchy recursively
            hierarchy = []
            for root_section in root_sections:
                if root_section["depth"] <= max_depth:
                    hierarchy.append(self._build_section_tree(root_section, section_map, max_depth))
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to organize sections by hierarchy: {e}")
            return []
    
    def _build_section_tree(
        self, 
        section: Dict[str, Any], 
        section_map: Dict[int, Dict[str, Any]], 
        max_depth: int
    ) -> Dict[str, Any]:
        """
        Build a tree structure for a section and its children.
        
        Args:
            section: Section dictionary
            section_map: Map of section ID to section
            max_depth: Maximum depth to include
            
        Returns:
            Section with children organized as a tree
        """
        try:
            tree_section = section.copy()
            
            # Find children
            children = [
                s for s in section_map.values() 
                if s.get("parent_section_id") == section["id"] and s["depth"] <= max_depth
            ]
            
            if children:
                tree_section["children"] = [
                    self._build_section_tree(child, section_map, max_depth)
                    for child in children
                ]
            else:
                tree_section["children"] = []
            
            return tree_section
            
        except Exception as e:
            logger.error(f"Failed to build section tree: {e}")
            return section
    
    async def _get_section_summary(self, user_id: str, section_id: int, blueprint_id: int) -> str:
        """
        Get a brief summary for a section.
        
        Args:
            user_id: User ID
            section_id: Section ID
            blueprint_id: Blueprint ID
            
        Returns:
            Section summary string
        """
        try:
            # Get section content and return a brief summary
            content = await self._get_section_content(user_id, section_id, blueprint_id)
            return content.get("summary", "No summary available")[:150] + "..."
            
        except Exception as e:
            logger.error(f"Failed to get section summary: {e}")
            return "Summary not available"
    
    def add_message_to_buffer(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a new message to the conversational buffer."""
        if session_id not in self.conversation_buffers:
            self.conversation_buffers[session_id] = []
        
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            message_id=f"{session_id}_{len(self.conversation_buffers[session_id])}",
            metadata=metadata or {}
        )
        
        self.conversation_buffers[session_id].append(message)
        
        # Maintain buffer size
        max_messages = self.tier_config[ContextTier.CONVERSATIONAL_BUFFER]['max_messages']
        if len(self.conversation_buffers[session_id]) > max_messages:
            self.conversation_buffers[session_id] = self.conversation_buffers[session_id][-max_messages:]
    
    def update_session_state(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session state with new information."""
        if session_id in self.session_states:
            session_state = self.session_states[session_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(session_state, field):
                    setattr(session_state, field, value)
            
            session_state.last_updated = datetime.utcnow()
        
    def _get_or_create_session_state(self, user_id: str, session_id: str) -> SessionState:
        """Get existing session state or create a new one."""
        if session_id not in self.session_states:
            self.session_states[session_id] = SessionState(
                session_id=session_id,
                user_id=user_id,
                current_topic=None,
                learning_objectives=[],
                discussed_concepts=[],
                user_questions=[],
                clarifications_needed=[],
                progress_indicators={},
                context_summary="",
                last_updated=datetime.utcnow(),
                metadata={}
            )
        
        return self.session_states[session_id]
    
    def _get_or_create_cognitive_profile(self, user_id: str) -> CognitiveProfile:
        """Get existing cognitive profile or create a new one."""
        if user_id not in self.cognitive_profiles:
            self.cognitive_profiles[user_id] = CognitiveProfile(
                user_id=user_id,
                learning_style="balanced",
                preferred_difficulty="intermediate",
                knowledge_level={},
                learning_pace="medium",
                preferred_explanation_style="detailed",
                misconceptions=[],
                strengths=[],
                areas_for_improvement=[],
                last_updated=datetime.utcnow()
            )
        
        return self.cognitive_profiles[user_id]
    
    def _get_conversational_buffer(self, session_id: str) -> List[ConversationMessage]:
        """Get the conversational buffer for a session."""
        return self.conversation_buffers.get(session_id, [])
    
    async def _retrieve_relevant_knowledge(
        self,
        user_id: str,  # CRITICAL: User ID for filtering blueprints by ownership
        query: str,
        transformation: QueryTransformation,
        session_state: SessionState,
        cognitive_profile: CognitiveProfile
    ) -> List[RAGSearchResult]:
        """Retrieve relevant knowledge from the knowledge base."""
        print(f"[DEBUG] === Starting _retrieve_relevant_knowledge ===")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Transformation intent: {transformation.intent}")
        print(f"[DEBUG] Search service available: {self.search_service is not None}")
        
        # Prepare user context for search
        user_context = {
            'learning_stage': session_state.metadata.get('learning_stage', 'understand'),
            'preferred_difficulty': cognitive_profile.preferred_difficulty,
            'current_topic': session_state.current_topic,
            'discussed_concepts': session_state.discussed_concepts,
            'learning_style': cognitive_profile.learning_style
        }
        print(f"[DEBUG] User context prepared: {user_context}")
        
        # Prepare conversation history
        conversation_history = []
        for msg in self.conversation_buffers.get(session_state.session_id, []):
            conversation_history.append({
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            })
        print(f"[DEBUG] Conversation history prepared: {len(conversation_history)} messages")
        
        # Create search request
        max_results = self.tier_config[ContextTier.KNOWLEDGE_BASE]['max_results']
        print(f"[DEBUG] Max results from config: {max_results}")
        
        search_request = RAGSearchRequest(
            query=query,
            user_id=user_id,  # CRITICAL: Ensure user can only access their own blueprints
            user_context=user_context,
            conversation_history=conversation_history,
            top_k=max_results,
            similarity_threshold=0.7,
            diversity_factor=0.3,
            include_relationships=True
        )
        print(f"[DEBUG] Search request created: top_k={search_request.top_k}, threshold={search_request.similarity_threshold}")
        
        # Perform search
        print(f"[DEBUG] About to call search_service.search...")
        try:
            search_response = await self.search_service.search(search_request)
            print(f"[DEBUG] Search completed successfully!")
            print(f"[DEBUG] Search response type: {type(search_response)}")
            print(f"[DEBUG] Results count: {len(search_response.results)}")
            print(f"[DEBUG] Search strategy used: {search_response.search_strategy}")
            
            for i, result in enumerate(search_response.results[:3]):
                print(f"[DEBUG] Result {i+1}: Score {result.score:.4f}, Content: {result.content[:60]}...")
            
            return search_response.results
        except Exception as e:
            print(f"[DEBUG] Search failed with error: {e}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            raise
    
    def _create_context_summary(
        self,
        conversational_context: List[ConversationMessage],
        session_state: SessionState,
        retrieved_knowledge: List[RAGSearchResult]
    ) -> str:
        """Create a concise summary of the assembled context."""
        summary_parts = []
        
        # Conversation summary
        if conversational_context:
            recent_topics = []
            for msg in conversational_context[-3:]:  # Last 3 messages
                if msg.role == 'user':
                    recent_topics.append(msg.content[:50] + "...")
            
            if recent_topics:
                summary_parts.append(f"Recent discussion: {'; '.join(recent_topics)}")
        
        # Session summary
        if session_state.current_topic:
            summary_parts.append(f"Current topic: {session_state.current_topic}")
        
        if session_state.discussed_concepts:
            concepts_str = ", ".join(session_state.discussed_concepts[-3:])  # Last 3 concepts
            summary_parts.append(f"Discussed concepts: {concepts_str}")
        
        # Knowledge summary
        if retrieved_knowledge:
            knowledge_types = set()
            for result in retrieved_knowledge[:5]:  # Top 5 results
                knowledge_types.add(result.locus_type)
            
            types_str = ", ".join(knowledge_types)
            summary_parts.append(f"Retrieved knowledge types: {types_str}")
        
        return " | ".join(summary_parts) if summary_parts else "No context available"
    
    def _calculate_total_tokens(
        self,
        conversational_context: List[ConversationMessage],
        session_state: SessionState,
        retrieved_knowledge: List[RAGSearchResult],
        cognitive_profile: CognitiveProfile
    ) -> int:
        """Calculate total token usage for the context."""
        total_tokens = 0
        
        # Conversational context tokens
        for msg in conversational_context:
            total_tokens += len(msg.content.split()) * 1.3  # Rough token estimation
        
        # Session state tokens
        session_str = json.dumps(asdict(session_state), default=str)
        total_tokens += len(session_str.split()) * 1.3
        
        # Retrieved knowledge tokens
        for result in retrieved_knowledge:
            total_tokens += len(result.content.split()) * 1.3
        
        # Cognitive profile tokens
        profile_str = json.dumps(asdict(cognitive_profile), default=str)
        total_tokens += len(profile_str.split()) * 1.3
        
        return int(total_tokens)
    
    def _prune_context(
        self,
        conversational_context: List[ConversationMessage],
        retrieved_knowledge: List[RAGSearchResult],
        target_tokens: int
    ) -> Tuple[List[ConversationMessage], List[RAGSearchResult]]:
        """Prune context to fit within token limits."""
        # Calculate current usage
        conv_tokens = sum(len(msg.content.split()) * 1.3 for msg in conversational_context)
        knowledge_tokens = sum(len(result.content.split()) * 1.3 for result in retrieved_knowledge)
        
        # Reserve tokens for session state and cognitive profile (estimated)
        reserved_tokens = 500
        available_tokens = target_tokens - reserved_tokens
        
        # Prune conversational context (keep most recent)
        max_conv_tokens = available_tokens * 0.4  # 40% for conversation
        pruned_conv = []
        current_conv_tokens = 0
        
        for msg in reversed(conversational_context):
            msg_tokens = len(msg.content.split()) * 1.3
            if current_conv_tokens + msg_tokens <= max_conv_tokens:
                pruned_conv.insert(0, msg)
                current_conv_tokens += msg_tokens
            else:
                break
        
        # Prune retrieved knowledge (keep highest scoring)
        max_knowledge_tokens = available_tokens * 0.6  # 60% for knowledge
        pruned_knowledge = []
        current_knowledge_tokens = 0
        
        for result in retrieved_knowledge:
            result_tokens = len(result.content.split()) * 1.3
            if current_knowledge_tokens + result_tokens <= max_knowledge_tokens:
                pruned_knowledge.append(result)
                current_knowledge_tokens += result_tokens
            else:
                break
        
        return pruned_conv, pruned_knowledge
    
    def _calculate_context_quality(
        self,
        conversational_context: List[ConversationMessage],
        session_state: SessionState,
        retrieved_knowledge: List[RAGSearchResult],
        query_transformation: QueryTransformation
    ) -> float:
        """Calculate a quality score for the assembled context."""
        quality_factors = []
        
        # Conversation context quality
        if conversational_context:
            # Recency factor
            latest_msg = conversational_context[-1]
            time_since_latest = (datetime.utcnow() - latest_msg.timestamp).total_seconds()
            recency_score = max(0, 1 - (time_since_latest / 3600))  # Decay over 1 hour
            quality_factors.append(recency_score * 0.2)
            
            # Conversation length factor
            length_score = min(len(conversational_context) / 10, 1.0)
            quality_factors.append(length_score * 0.1)
        
        # Session state quality
        if session_state.current_topic:
            quality_factors.append(0.15)  # Bonus for having current topic
        
        if session_state.discussed_concepts:
            concept_score = min(len(session_state.discussed_concepts) / 5, 1.0)
            quality_factors.append(concept_score * 0.1)
        
        # Retrieved knowledge quality
        if retrieved_knowledge:
            # Average relevance score
            avg_relevance = sum(result.final_score for result in retrieved_knowledge) / len(retrieved_knowledge)
            quality_factors.append(avg_relevance * 0.3)
            
            # Diversity score
            unique_types = set(result.locus_type for result in retrieved_knowledge)
            diversity_score = min(len(unique_types) / 3, 1.0)
            quality_factors.append(diversity_score * 0.1)
        
        # Intent alignment
        intent_bonus = 0.05  # Base bonus for having intent classification
        quality_factors.append(intent_bonus)
        
        return sum(quality_factors) if quality_factors else 0.0
    
    def get_context_for_prompt(self, assembled_context: AssembledContext) -> str:
        """
        Format the assembled context for LLM prompt inclusion.
        
        Returns:
            Formatted context string ready for prompt injection
        """
        context_sections = []
        
        # Add conversation history
        if assembled_context.conversational_context:
            context_sections.append("=== CONVERSATION HISTORY ===")
            for msg in assembled_context.conversational_context[-5:]:  # Last 5 messages
                role_label = "USER" if msg.role == "user" else "ASSISTANT"
                context_sections.append(f"{role_label}: {msg.content}")
        
        # Add session context
        if assembled_context.session_context.current_topic:
            context_sections.append(f"\n=== CURRENT TOPIC ===\n{assembled_context.session_context.current_topic}")
        
        if assembled_context.session_context.discussed_concepts:
            concepts = ", ".join(assembled_context.session_context.discussed_concepts)
            context_sections.append(f"\n=== DISCUSSED CONCEPTS ===\n{concepts}")
        
        # Add retrieved knowledge
        if assembled_context.retrieved_knowledge:
            context_sections.append("\n=== RELEVANT KNOWLEDGE ===")
            for i, result in enumerate(assembled_context.retrieved_knowledge[:10], 1):
                context_sections.append(f"{i}. [{result.locus_type}] {result.content}")
        
        # Add cognitive profile
        profile = assembled_context.cognitive_profile
        context_sections.append(f"\n=== USER PROFILE ===")
        context_sections.append(f"Learning Style: {profile.learning_style}")
        context_sections.append(f"Preferred Difficulty: {profile.preferred_difficulty}")
        context_sections.append(f"Explanation Style: {profile.preferred_explanation_style}")
        
        return "\n".join(context_sections)
    
    def extract_session_updates(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """
        Extract session state updates from the conversation.
        
        This is a simplified extraction - in production, this would use NLP
        to identify topics, concepts, questions, etc.
        """
        updates = {}
        
        # Extract potential topics (simplified)
        user_words = user_message.lower().split()
        assistant_words = assistant_response.lower().split()
        
        # Look for question patterns
        if any(word in user_words for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            updates['user_questions'] = [user_message]
        
        # Look for concept mentions (simplified)
        # This would be more sophisticated in production
        concept_indicators = ['concept', 'theory', 'principle', 'definition', 'explain']
        mentioned_concepts = []
        
        for word in assistant_words:
            if word in concept_indicators:
                # Find nearby words as potential concepts
                word_idx = assistant_words.index(word)
                if word_idx < len(assistant_words) - 1:
                    next_word = assistant_words[word_idx + 1]
                    mentioned_concepts.append(next_word)
        
        if mentioned_concepts:
            updates['discussed_concepts'] = mentioned_concepts
        
        return updates
