"""
Advanced memory system for premium features.
Provides hierarchical memory management with attention mechanisms.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

class EpisodicBuffer:
    """Episodic memory buffer for recent interactions"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer = []
    
    async def add_interaction(self, interaction: Dict[str, Any]):
        """Add new interaction to episodic buffer"""
        interaction["timestamp"] = datetime.utcnow().isoformat()
        self.buffer.append(interaction)
        
        # Maintain buffer size
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    async def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions from buffer"""
        return self.buffer[-limit:] if self.buffer else []
    
    async def clear_old_interactions(self, days: int = 7):
        """Clear interactions older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        self.buffer = [
            interaction for interaction in self.buffer
            if datetime.fromisoformat(interaction["timestamp"]) > cutoff_date
        ]

class SemanticStore:
    """Semantic memory store for conceptual knowledge"""
    
    def __init__(self):
        self.concepts = {}
        self.relationships = {}
    
    async def store_concept(self, concept_id: str, concept_data: Dict[str, Any]):
        """Store conceptual knowledge"""
        self.concepts[concept_id] = {
            **concept_data,
            "last_updated": datetime.utcnow().isoformat(),
            "access_count": self.concepts.get(concept_id, {}).get("access_count", 0) + 1
        }
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conceptual knowledge"""
        return self.concepts.get(concept_id)
    
    async def add_relationship(self, concept1_id: str, concept2_id: str, relationship_type: str):
        """Add relationship between concepts"""
        key = f"{concept1_id}:{concept2_id}"
        self.relationships[key] = {
            "type": relationship_type,
            "strength": 1.0,
            "last_accessed": datetime.utcnow().isoformat()
        }
    
    async def get_related_concepts(self, concept_id: str) -> List[Dict[str, Any]]:
        """Get concepts related to the specified concept"""
        related = []
        for key, relationship in self.relationships.items():
            if concept_id in key:
                other_concept = key.replace(f"{concept_id}:", "").replace(f":{concept_id}", "")
                if other_concept in self.concepts:
                    related.append({
                        "concept": self.concepts[other_concept],
                        "relationship": relationship
                    })
        return related

class ProceduralStore:
    """Procedural memory store for skills and procedures"""
    
    def __init__(self):
        self.procedures = {}
        self.skills = {}
    
    async def store_procedure(self, procedure_id: str, steps: List[str], metadata: Dict[str, Any]):
        """Store procedural knowledge"""
        self.procedures[procedure_id] = {
            "steps": steps,
            "metadata": metadata,
            "last_used": datetime.utcnow().isoformat(),
            "success_rate": metadata.get("success_rate", 0.0)
        }
    
    async def get_procedure(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve procedural knowledge"""
        return self.procedures.get(procedure_id)
    
    async def update_success_rate(self, procedure_id: str, success: bool):
        """Update procedure success rate"""
        if procedure_id in self.procedures:
            current_rate = self.procedures[procedure_id]["success_rate"]
            # Simple moving average
            new_rate = (current_rate * 0.9) + (1.0 if success else 0.0) * 0.1
            self.procedures[procedure_id]["success_rate"] = new_rate

class WorkingMemory:
    """Working memory for current session context"""
    
    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self.items = {}
        self.focus_items = []
    
    async def add_item(self, item_id: str, item_data: Dict[str, Any]):
        """Add item to working memory"""
        self.items[item_id] = {
            **item_data,
            "timestamp": datetime.utcnow().isoformat(),
            "access_count": 0
        }
        
        # Maintain size limit
        if len(self.items) > self.max_items:
            # Remove least recently accessed item
            oldest_key = min(self.items.keys(), key=lambda k: self.items[k]["timestamp"])
            del self.items[oldest_key]
    
    async def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve item from working memory"""
        if item_id in self.items:
            self.items[item_id]["access_count"] += 1
            return self.items[item_id]
        return None
    
    async def add_focus_item(self, item_id: str):
        """Add item to focus list"""
        if item_id not in self.focus_items:
            self.focus_items.append(item_id)
            if len(self.focus_items) > 5:  # Limit focus items
                self.focus_items.pop(0)
    
    async def get_focus_items(self) -> List[Dict[str, Any]]:
        """Get items currently in focus"""
        return [self.items.get(item_id) for item_id in self.focus_items if item_id in self.items]

class PremiumMemorySystem:
    """Hierarchical memory system for premium users"""
    
    def __init__(self):
        self.episodic_memory = EpisodicBuffer()
        self.semantic_memory = SemanticStore()
        self.procedural_memory = ProceduralStore()
        self.working_memory = WorkingMemory()
    
    async def retrieve_with_attention(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant memories using attention mechanisms"""
        try:
            # Get relevant concepts from semantic memory
            relevant_concepts = await self._find_relevant_concepts(query)
            
            # Get recent interactions from episodic memory
            recent_interactions = await self.episodic_memory.get_recent_interactions(5)
            
            # Get focus items from working memory
            focus_items = await self.working_memory.get_focus_items()
            
            # Get relevant procedures
            relevant_procedures = await self._find_relevant_procedures(query)
            
            return {
                "concepts": relevant_concepts,
                "recent_interactions": recent_interactions,
                "focus_items": focus_items,
                "procedures": relevant_procedures,
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return {
                "concepts": [],
                "recent_interactions": [],
                "focus_items": [],
                "procedures": [],
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def update_memory(self, interaction: Dict[str, Any]):
        """Update memory systems with new interactions"""
        try:
            # Add to episodic memory
            await self.episodic_memory.add_interaction(interaction)
            
            # Extract and store concepts
            if "concepts" in interaction:
                for concept in interaction["concepts"]:
                    await self.semantic_memory.store_concept(
                        concept["id"], concept
                    )
            
            # Extract and store procedures
            if "procedures" in interaction:
                for procedure in interaction["procedures"]:
                    await self.procedural_memory.store_procedure(
                        procedure["id"], 
                        procedure.get("steps", []),
                        procedure.get("metadata", {})
                    )
            
            # Update working memory
            if "focus_items" in interaction:
                for item_id in interaction["focus_items"]:
                    await self.working_memory.add_focus_item(item_id)
            
        except Exception as e:
            print(f"Error updating memory: {e}")
    
    async def _find_relevant_concepts(self, query: str) -> List[Dict[str, Any]]:
        """Find concepts relevant to the query"""
        # Simple keyword matching - in production, use semantic search
        relevant_concepts = []
        query_lower = query.lower()
        
        for concept_id, concept_data in self.semantic_memory.concepts.items():
            if any(keyword in query_lower for keyword in concept_data.get("keywords", [])):
                relevant_concepts.append(concept_data)
        
        return relevant_concepts[:5]  # Limit to top 5
    
    async def _find_relevant_procedures(self, query: str) -> List[Dict[str, Any]]:
        """Find procedures relevant to the query"""
        # Simple keyword matching - in production, use semantic search
        relevant_procedures = []
        query_lower = query.lower()
        
        for procedure_id, procedure_data in self.procedural_memory.procedures.items():
            if any(keyword in query_lower for keyword in procedure_data.get("metadata", {}).get("keywords", [])):
                relevant_procedures.append(procedure_data)
        
        return relevant_procedures[:3]  # Limit to top 3
    
    async def consolidate_memories(self):
        """Consolidate and optimize memory systems"""
        try:
            # Clear old episodic memories
            await self.episodic_memory.clear_old_interactions(days=7)
            
            # Update concept relationships based on co-occurrence
            recent_interactions = await self.episodic_memory.get_recent_interactions(50)
            await self._update_concept_relationships(recent_interactions)
            
        except Exception as e:
            print(f"Error consolidating memories: {e}")
    
    async def _update_concept_relationships(self, interactions: List[Dict[str, Any]]):
        """Update concept relationships based on interaction patterns"""
        concept_co_occurrences = {}
        
        for interaction in interactions:
            if "concepts" in interaction:
                concept_ids = [c["id"] for c in interaction["concepts"]]
                for i, concept1 in enumerate(concept_ids):
                    for concept2 in concept_ids[i+1:]:
                        key = f"{concept1}:{concept2}"
                        concept_co_occurrences[key] = concept_co_occurrences.get(key, 0) + 1
        
        # Update relationships based on co-occurrence
        for key, count in concept_co_occurrences.items():
            if count >= 2:  # Minimum co-occurrence threshold
                concept1, concept2 = key.split(":")
                await self.semantic_memory.add_relationship(concept1, concept2, "co_occurrence")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "episodic_buffer_size": len(self.episodic_memory.buffer),
            "semantic_concepts_count": len(self.semantic_memory.concepts),
            "procedural_count": len(self.procedural_memory.procedures),
            "working_memory_size": len(self.working_memory.items),
            "focus_items_count": len(self.working_memory.focus_items)
        }













