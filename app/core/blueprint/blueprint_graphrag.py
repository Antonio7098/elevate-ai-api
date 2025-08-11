"""
Blueprint GraphRAG module.

This module provides GraphRAG functionality for blueprints.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from app.models.blueprint import Blueprint
from app.core.blueprint.blueprint_indexer import BlueprintIndexer
from app.core.blueprint.blueprint_repository import BlueprintRepository
import json


class BlueprintGraphRAG:
    """Graph-based Retrieval-Augmented Generation system for blueprints."""
    
    def __init__(self, indexer: BlueprintIndexer, repository: BlueprintRepository):
        self.indexer = indexer
        self.repository = repository
        self.graph: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, List[Dict[str, Any]]] = {}
        self.node_embeddings: Dict[str, List[float]] = {}
    
    async def build_knowledge_graph(self) -> None:
        """Build a knowledge graph from all blueprints."""
        # Get all blueprints
        all_blueprints = await self.repository.list_all(limit=10000, offset=0)
        
        # Clear existing graph
        self.graph.clear()
        self.relationships.clear()
        
        # Create nodes for each blueprint
        for blueprint in all_blueprints:
            await self._add_blueprint_node(blueprint)
        
        # Create relationships between blueprints
        await self._create_relationships()
        
        # Calculate node embeddings (simplified)
        await self._calculate_node_embeddings()
    
    async def _add_blueprint_node(self, blueprint: Blueprint) -> None:
        """Add a blueprint as a node in the knowledge graph."""
        node_id = blueprint.id
        
        # Create node with blueprint information
        self.graph[node_id] = {
            'id': blueprint.id,
            'title': blueprint.title,
            'description': blueprint.description,
            'type': blueprint.type.value,
            'tags': blueprint.tags,
            'metadata': blueprint.metadata,
            'node_type': 'blueprint',
            'properties': {
                'status': blueprint.status.value,
                'author_id': blueprint.author_id,
                'is_public': blueprint.is_public,
                'created_at': blueprint.created_at.isoformat(),
                'updated_at': blueprint.updated_at.isoformat(),
                'version': blueprint.version
            }
        }
        
        # Initialize relationships list
        self.relationships[node_id] = []
    
    async def _create_relationships(self) -> None:
        """Create relationships between blueprint nodes."""
        blueprint_ids = list(self.graph.keys())
        
        for i, blueprint_id_1 in enumerate(blueprint_ids):
            for j, blueprint_id_2 in enumerate(blueprint_ids[i+1:], i+1):
                relationship = await self._calculate_relationship(blueprint_id_1, blueprint_id_2)
                if relationship['strength'] > 0.3:  # Only create relationships above threshold
                    await self._add_relationship(blueprint_id_1, blueprint_id_2, relationship)
    
    async def _calculate_relationship(self, node_id_1: str, node_id_2: str) -> Dict[str, Any]:
        """Calculate the relationship strength between two blueprint nodes."""
        node_1 = self.graph[node_id_1]
        node_2 = self.graph[node_id_2]
        
        # Calculate similarity based on multiple factors
        similarity_score = 0.0
        
        # Tag similarity
        tags_1 = set(node_1['tags'])
        tags_2 = set(node_2['tags'])
        if tags_1 and tags_2:
            tag_overlap = len(tags_1.intersection(tags_2))
            tag_union = len(tags_1.union(tags_2))
            tag_similarity = tag_overlap / tag_union if tag_union > 0 else 0
            similarity_score += tag_similarity * 0.4
        
        # Type similarity
        if node_1['type'] == node_2['type']:
            similarity_score += 0.3
        
        # Content similarity (simplified)
        content_similarity = await self._calculate_content_similarity(node_id_1, node_id_2)
        similarity_score += content_similarity * 0.3
        
        # Determine relationship type
        relationship_type = self._determine_relationship_type(node_1, node_2, similarity_score)
        
        return {
            'type': relationship_type,
            'strength': similarity_score,
            'properties': {
                'tag_overlap': len(tags_1.intersection(tags_2)) if tags_1 and tags_2 else 0,
                'type_match': node_1['type'] == node_2['type'],
                'content_similarity': content_similarity
            }
        }
    
    async def _calculate_content_similarity(self, node_id_1: str, node_id_2: str) -> float:
        """Calculate content similarity between two blueprints."""
        # Get content from indexer
        content_1 = self.indexer.content_index.get(node_id_1, [])
        content_2 = self.indexer.content_index.get(node_id_2, [])
        
        if not content_1 or not content_2:
            return 0.0
        
        # Simple keyword overlap calculation
        words_1 = set()
        for text in content_1:
            words_1.update(text.lower().split())
        
        words_2 = set()
        for text in content_2:
            words_2.update(text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words_1 = words_1 - stop_words
        words_2 = words_2 - stop_words
        
        if not words_1 or not words_2:
            return 0.0
        
        overlap = len(words_1.intersection(words_2))
        union = len(words_1.union(words_2))
        
        return overlap / union if union > 0 else 0.0
    
    def _determine_relationship_type(self, node_1: Dict[str, Any], node_2: Dict[str, Any], similarity: float) -> str:
        """Determine the type of relationship between two nodes."""
        if similarity > 0.8:
            return 'highly_related'
        elif similarity > 0.6:
            return 'related'
        elif similarity > 0.4:
            return 'moderately_related'
        elif similarity > 0.2:
            return 'weakly_related'
        else:
            return 'unrelated'
    
    async def _add_relationship(self, node_id_1: str, node_id_2: str, relationship: Dict[str, Any]) -> None:
        """Add a relationship between two nodes."""
        # Add bidirectional relationship
        self.relationships[node_id_1].append({
            'target_id': node_id_2,
            'relationship': relationship
        })
        
        self.relationships[node_id_2].append({
            'target_id': node_id_1,
            'relationship': relationship
        })
    
    async def _calculate_node_embeddings(self) -> None:
        """Calculate embeddings for graph nodes (simplified implementation)."""
        # In a real system, you would use a proper embedding model
        # For now, we'll create simple feature vectors
        
        for node_id, node in self.graph.items():
            # Create a simple feature vector based on node properties
            features = []
            
            # Type encoding
            type_encoding = {'learning': 0.1, 'assessment': 0.2, 'practice': 0.3, 'review': 0.4}
            features.append(type_encoding.get(node['type'], 0.0))
            
            # Tag count
            features.append(min(len(node['tags']), 10) / 10.0)
            
            # Public/private
            features.append(1.0 if node['properties']['is_public'] else 0.0)
            
            # Normalize features
            feature_sum = sum(features)
            if feature_sum > 0:
                features = [f / feature_sum for f in features]
            
            self.node_embeddings[node_id] = features
    
    async def graph_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph for relevant blueprints."""
        # First, find initial relevant nodes
        initial_nodes = await self.indexer.search(query, limit=5)
        
        if not initial_nodes:
            return []
        
        # Expand search using graph relationships
        expanded_nodes = await self._expand_search(initial_nodes, query, limit)
        
        # Rank results by relevance and graph centrality
        ranked_results = await self._rank_graph_results(expanded_nodes, query)
        
        return ranked_results[:limit]
    
    async def _expand_search(self, initial_nodes: List[str], query: str, limit: int) -> List[str]:
        """Expand search using graph relationships."""
        visited = set(initial_nodes)
        to_visit = initial_nodes.copy()
        expanded_nodes = initial_nodes.copy()
        
        # Breadth-first expansion
        while to_visit and len(expanded_nodes) < limit * 2:
            current_node = to_visit.pop(0)
            
            # Get related nodes
            related_nodes = self.relationships.get(current_node, [])
            
            for relation in related_nodes:
                target_id = relation['target_id']
                relationship = relation['relationship']
                
                # Only add if relationship is strong enough and not visited
                if (relationship['strength'] > 0.5 and 
                    target_id not in visited and 
                    len(expanded_nodes) < limit * 2):
                    
                    expanded_nodes.append(target_id)
                    visited.add(target_id)
                    to_visit.append(target_id)
        
        return expanded_nodes
    
    async def _rank_graph_results(self, node_ids: List[str], query: str) -> List[Dict[str, Any]]:
        """Rank search results using graph centrality and relevance."""
        results = []
        
        for node_id in node_ids:
            node = self.graph[node_id]
            
            # Calculate graph centrality
            centrality = self._calculate_node_centrality(node_id)
            
            # Calculate query relevance
            relevance = await self._calculate_query_relevance(node_id, query)
            
            # Combined score
            combined_score = (centrality * 0.3) + (relevance * 0.7)
            
            results.append({
                'node_id': node_id,
                'title': node['title'],
                'description': node['description'],
                'type': node['type'],
                'centrality_score': centrality,
                'relevance_score': relevance,
                'combined_score': combined_score,
                'relationships': len(self.relationships.get(node_id, [])),
                'properties': node['properties']
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results
    
    def _calculate_node_centrality(self, node_id: str) -> float:
        """Calculate centrality of a node in the graph."""
        # Simple degree centrality
        degree = len(self.relationships.get(node_id, []))
        
        # Normalize by maximum possible degree
        max_degree = max(len(rels) for rels in self.relationships.values()) if self.relationships else 1
        
        return degree / max_degree if max_degree > 0 else 0.0
    
    async def _calculate_query_relevance(self, node_id: str, query: str) -> float:
        """Calculate relevance of a node to the query."""
        # Use the indexer's search functionality
        search_results = await self.indexer.search(query, limit=100)
        
        if node_id in search_results:
            # Find position in search results (lower position = higher relevance)
            position = search_results.index(node_id)
            return 1.0 / (position + 1)  # Inverse position scoring
        
        return 0.0
    
    async def get_graph_path(self, start_node_id: str, end_node_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find a path between two nodes in the graph."""
        if start_node_id not in self.graph or end_node_id not in self.graph:
            return []
        
        # Simple breadth-first search for path finding
        queue = [(start_node_id, [start_node_id])]
        visited = {start_node_id}
        
        while queue:
            current_node, path = queue.pop(0)
            
            if current_node == end_node_id:
                return self._format_path(path)
            
            if len(path) >= max_depth:
                continue
            
            # Explore neighbors
            for relation in self.relationships.get(current_node, []):
                neighbor = relation['target_id']
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return []
    
    def _format_path(self, path: List[str]) -> List[Dict[str, Any]]:
        """Format a path into a readable structure."""
        formatted_path = []
        
        for i, node_id in enumerate(path):
            node = self.graph[node_id]
            
            # Get relationship to next node if not the last node
            relationship = None
            if i < len(path) - 1:
                next_node_id = path[i + 1]
                for rel in self.relationships.get(node_id, []):
                    if rel['target_id'] == next_node_id:
                        relationship = rel['relationship']
                        break
            
            formatted_path.append({
                'node_id': node_id,
                'title': node['title'],
                'type': node['type'],
                'relationship_to_next': relationship
            })
        
        return formatted_path
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        total_nodes = len(self.graph)
        total_relationships = sum(len(rels) for rels in self.relationships.values())
        
        # Calculate average degree
        avg_degree = total_relationships / total_nodes if total_nodes > 0 else 0
        
        # Count node types
        node_types = {}
        for node in self.graph.values():
            node_type = node['type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count relationship types
        relationship_types = {}
        for rels in self.relationships.values():
            for rel in rels:
                rel_type = rel['relationship']['type']
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            'total_nodes': total_nodes,
            'total_relationships': total_relationships,
            'average_degree': round(avg_degree, 2),
            'node_types': node_types,
            'relationship_types': relationship_types,
            'graph_density': round(total_relationships / (total_nodes * (total_nodes - 1)), 4) if total_nodes > 1 else 0
        }
    
    async def export_graph(self, format: str = 'json') -> str:
        """Export the knowledge graph in specified format."""
        if format.lower() == 'json':
            export_data = {
                'nodes': self.graph,
                'relationships': self.relationships,
                'exported_at': '2024-01-01T00:00:00Z'  # Placeholder
            }
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def clear_graph(self) -> None:
        """Clear the knowledge graph."""
        self.graph.clear()
        self.relationships.clear()
        self.node_embeddings.clear()
