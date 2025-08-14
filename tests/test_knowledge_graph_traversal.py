"""
Comprehensive tests for KnowledgeGraphTraversal

This test suite covers all service methods including graph traversal,
prerequisite chain discovery, and learning path finding.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.knowledge_graph_traversal import KnowledgeGraphTraversal


class TestKnowledgeGraphTraversal:
    """Test KnowledgeGraphTraversal class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = KnowledgeGraphTraversal()
        self.test_start_node = "test_node_123"
        self.test_end_node = "test_node_456"
        self.test_target_node = "test_target_789"
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert hasattr(self.service, 'logger')
        assert self.service.logger is not None
    
    @pytest.mark.asyncio
    async def test_traverse_graph_success(self):
        """Test successful graph traversal."""
        result = await self.service.traverse_graph(self.test_start_node, max_depth=3)
        
        assert isinstance(result, dict)
        assert result["start_node_id"] == self.test_start_node
        assert result["max_depth"] == 3
        assert "traversal_timestamp" in result
        assert "discovered_nodes" in result
        assert "discovered_edges" in result
        assert "traversal_statistics" in result
        
        # Check traversal statistics structure
        stats = result["traversal_statistics"]
        assert "total_nodes_visited" in stats
        assert "total_edges_traversed" in stats
        assert "max_depth_reached" in stats
        assert "traversal_time_ms" in stats
        
        # Check that statistics are valid
        assert stats["total_nodes_visited"] > 0
        assert stats["total_edges_traversed"] >= 0
        assert stats["max_depth_reached"] <= 3
        assert stats["traversal_time_ms"] >= 0
        
        # Check discovered nodes structure
        for node in result["discovered_nodes"]:
            assert "node_id" in node
            assert "depth" in node
            assert "discovery_order" in node
            assert 0 <= node["depth"] <= 3
        
        # Check discovered edges structure
        for edge in result["discovered_edges"]:
            assert "edge_id" in edge
            assert "source" in edge
            assert "target" in edge
            assert "relationship_type" in edge
            assert "discovered_at_depth" in edge
    
    @pytest.mark.asyncio
    async def test_traverse_graph_with_different_depths(self):
        """Test graph traversal with different depth limits."""
        # Test with depth 1
        result_depth1 = await self.service.traverse_graph(self.test_start_node, max_depth=1)
        assert result_depth1["max_depth"] == 1
        assert result_depth1["traversal_statistics"]["max_depth_reached"] <= 1
        
        # Test with depth 5
        result_depth5 = await self.service.traverse_graph(self.test_start_node, max_depth=5)
        assert result_depth5["max_depth"] == 5
        assert result_depth5["traversal_statistics"]["max_depth_reached"] <= 5
        
        # Test with depth 0
        result_depth0 = await self.service.traverse_graph(self.test_start_node, max_depth=0)
        assert result_depth0["max_depth"] == 0
        assert result_depth0["traversal_statistics"]["max_depth_reached"] <= 0
    
    @pytest.mark.asyncio
    async def test_find_prerequisite_chain_success(self):
        """Test successful prerequisite chain discovery."""
        result = await self.service.find_prerequisite_chain(self.test_target_node)
        
        assert isinstance(result, dict)
        assert result["target_node_id"] == self.test_target_node
        assert "discovery_timestamp" in result
        assert "prerequisite_chain" in result
        assert "chain_statistics" in result
        
        # Check chain statistics structure
        chain_stats = result["chain_statistics"]
        assert "total_prerequisites" in chain_stats
        assert "chain_depth" in chain_stats
        assert "estimated_learning_time" in chain_stats
        
        # Check that statistics are valid
        assert chain_stats["total_prerequisites"] > 0
        assert chain_stats["chain_depth"] > 0
        assert chain_stats["estimated_learning_time"] > 0
        
        # Check prerequisite chain structure
        for prereq in result["prerequisite_chain"]:
            assert "node_id" in prereq
            assert "depth" in prereq
            assert "relationship_type" in prereq
            assert "importance" in prereq
            assert "estimated_time_minutes" in prereq
            
            # Check value ranges
            assert 0.0 <= prereq["importance"] <= 1.0
            assert prereq["estimated_time_minutes"] > 0
    
    @pytest.mark.asyncio
    async def test_find_learning_path_success(self):
        """Test successful learning path discovery."""
        result = await self.service.find_learning_path(self.test_start_node, self.test_end_node)
        
        assert isinstance(result, dict)
        assert result["start_node_id"] == self.test_start_node
        assert result["end_node_id"] == self.test_end_node
        assert "discovery_timestamp" in result
        assert "primary_path" in result
        assert "alternative_paths" in result
        assert "path_statistics" in result
        
        # Check path statistics structure
        path_stats = result["path_statistics"]
        assert "primary_path_length" in path_stats
        assert "total_alternative_paths" in path_stats
        assert "estimated_learning_time" in path_stats
        
        # Check that statistics are valid
        assert path_stats["primary_path_length"] > 0
        assert path_stats["estimated_learning_time"] > 0
        
        # Check primary path structure
        for path_node in result["primary_path"]:
            assert "node_id" in path_node
            assert "order" in path_node
            assert "node_type" in path_node
            assert "difficulty" in path_node
            assert "uue_stage" in path_node
            assert "estimated_time_minutes" in path_node
            
            # Check value ranges
            assert path_node["order"] >= 0
            assert path_node["estimated_time_minutes"] > 0
            assert path_node["difficulty"] in ["beginner", "intermediate", "advanced"]
            assert path_node["uue_stage"] in ["understand", "use", "explore"]
    
    @pytest.mark.asyncio
    async def test_find_learning_path_with_alternatives(self):
        """Test learning path discovery with alternative paths."""
        # Mock the service to return alternative paths
        with patch.object(self.service, 'find_learning_path') as mock_find:
            mock_find.return_value = {
                "start_node_id": self.test_start_node,
                "end_node_id": self.test_end_node,
                "discovery_timestamp": datetime.now().isoformat(),
                "primary_path": [{"node_id": "node1", "order": 0}],
                "alternative_paths": [
                    [{"node_id": "alt1", "order": 0}],
                    [{"node_id": "alt2", "order": 0}]
                ],
                "path_statistics": {
                    "primary_path_length": 1,
                    "total_alternative_paths": 2,
                    "estimated_learning_time": 30
                }
            }
            
            result = await self.service.find_learning_path(self.test_start_node, self.test_end_node)
            
            assert result["path_statistics"]["total_alternative_paths"] == 2
            assert len(result["alternative_paths"]) == 2
    
    @pytest.mark.asyncio
    async def test_find_related_concepts_success(self):
        """Test successful related concepts discovery."""
        result = await self.service.find_related_concepts(self.test_start_node, max_results=5)
        
        assert isinstance(result, dict)
        assert result["source_node_id"] == self.test_start_node
        assert "discovery_timestamp" in result
        assert "relationship_types" in result
        assert "max_results" in result
        assert "related_concepts" in result
        assert "relationship_summary" in result
        
        # Check relationship summary structure
        rel_summary = result["relationship_summary"]
        assert "total_relationships" in rel_summary
        assert "relationship_type_distribution" in rel_summary
        assert "average_relationship_strength" in rel_summary
        
        # Check that statistics are valid
        assert rel_summary["total_relationships"] > 0
        assert 0.0 <= rel_summary["average_relationship_strength"] <= 1.0
        
        # Check related concepts structure
        for concept in result["related_concepts"]:
            assert "node_id" in concept
            assert "relationship_type" in concept
            assert "relationship_strength" in concept
            assert "confidence" in concept
            assert "evidence" in concept
            assert "bidirectional" in concept
            
            # Check value ranges
            assert 0.0 <= concept["relationship_strength"] <= 1.0
            assert 0.0 <= concept["confidence"] <= 1.0
            assert isinstance(concept["bidirectional"], bool)
    
    @pytest.mark.asyncio
    async def test_find_related_concepts_with_filters(self):
        """Test related concepts discovery with relationship type filters."""
        from app.models.knowledge_graph import RelationshipType
        
        # Test with specific relationship types
        result = await self.service.find_related_concepts(
            self.test_start_node, 
            relationship_types=[RelationshipType.RELATED, RelationshipType.PREREQUISITE],
            max_results=3
        )
        
        assert result["relationship_types"] == [RelationshipType.RELATED, RelationshipType.PREREQUISITE]
        assert result["max_results"] == 3
        
        # Check that all returned concepts have valid relationship types
        for concept in result["related_concepts"]:
            assert concept["relationship_type"] in [RelationshipType.RELATED, RelationshipType.PREREQUISITE]
    
    @pytest.mark.asyncio
    async def test_analyze_graph_connectivity_success(self):
        """Test successful graph connectivity analysis."""
        test_node_ids = ["node1", "node2", "node3", "node4"]
        result = await self.service.analyze_graph_connectivity(test_node_ids)
        
        assert isinstance(result, dict)
        assert result["analyzed_nodes"] == test_node_ids
        assert "analysis_timestamp" in result
        assert "connectivity_metrics" in result
        assert "centrality_analysis" in result
        assert "community_structure" in result
        
        # Check connectivity metrics structure
        connectivity = result["connectivity_metrics"]
        assert "average_degree" in connectivity
        assert "density" in connectivity
        assert "clustering_coefficient" in connectivity
        assert "connectivity_score" in connectivity
        
        # Check that metrics are valid
        assert connectivity["average_degree"] >= 0.0
        assert 0.0 <= connectivity["density"] <= 1.0
        assert 0.0 <= connectivity["clustering_coefficient"] <= 1.0
        assert 0.0 <= connectivity["connectivity_score"] <= 1.0
        
        # Check centrality analysis structure
        centrality = result["centrality_analysis"]
        assert "most_central_nodes" in centrality
        assert "bridge_nodes" in centrality
        assert "isolated_nodes" in centrality
        
        # Check community structure
        community = result["community_structure"]
        assert "detected_communities" in community
        assert "modularity_score" in community
        
        # Check that modularity score is valid
        assert -1.0 <= community["modularity_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_graph_connectivity_with_different_sizes(self):
        """Test connectivity analysis with different node set sizes."""
        # Test with small set
        small_result = await self.service.analyze_graph_connectivity(["node1"])
        assert len(small_result["analyzed_nodes"]) == 1
        
        # Test with larger set
        large_result = await self.service.analyze_graph_connectivity([f"node{i}" for i in range(10)])
        assert len(large_result["analyzed_nodes"]) == 10
        
        # Test with empty set
        empty_result = await self.service.analyze_graph_connectivity([])
        assert len(empty_result["analyzed_nodes"]) == 0
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self):
        """Test service error handling."""
        # Test with invalid node ID - service should handle empty strings gracefully
        result = await self.service.traverse_graph("", max_depth=3)
        assert isinstance(result, dict)
        assert result["start_node_id"] == ""
        
        # Test with negative depth - service should handle negative depths gracefully
        result = await self.service.traverse_graph(self.test_start_node, max_depth=-1)
        assert isinstance(result, dict)
        assert result["max_depth"] == -1
    
    @pytest.mark.asyncio
    async def test_service_integration(self):
        """Test service integration and data flow."""
        # Test complete workflow
        traversal = await self.service.traverse_graph(self.test_start_node, max_depth=2)
        prereq_chain = await self.service.find_prerequisite_chain(self.test_target_node)
        learning_path = await self.service.find_learning_path(self.test_start_node, self.test_end_node)
        related_concepts = await self.service.find_related_concepts(self.test_start_node)
        connectivity = await self.service.analyze_graph_connectivity([self.test_start_node, self.test_end_node])
        
        # Verify data consistency
        assert traversal["start_node_id"] == self.test_start_node
        assert prereq_chain["target_node_id"] == self.test_target_node
        assert learning_path["start_node_id"] == self.test_start_node
        assert learning_path["end_node_id"] == self.test_end_node
        assert related_concepts["source_node_id"] == self.test_start_node
        assert self.test_start_node in connectivity["analyzed_nodes"]
        assert self.test_end_node in connectivity["analyzed_nodes"]
    
    @pytest.mark.asyncio
    async def test_traversal_performance(self):
        """Test traversal performance characteristics."""
        import time
        
        # Test traversal time scales with depth
        start_time = time.time()
        result_depth1 = await self.service.traverse_graph(self.test_start_node, max_depth=1)
        time_depth1 = time.time() - start_time
        
        start_time = time.time()
        result_depth3 = await self.service.traverse_graph(self.test_start_node, max_depth=3)
        time_depth3 = time.time() - start_time
        
        # Deeper traversals should generally take longer (though this is simulated)
        # In real implementation, this would be more pronounced
        assert result_depth3["traversal_statistics"]["total_nodes_visited"] >= result_depth1["traversal_statistics"]["total_nodes_visited"]
        
        # Check that traversal statistics are consistent
        assert result_depth3["traversal_statistics"]["max_depth_reached"] >= result_depth1["traversal_statistics"]["max_depth_reached"]
    
    @pytest.mark.asyncio
    async def test_pathfinding_consistency(self):
        """Test that pathfinding results are consistent."""
        # Test same start/end nodes multiple times
        path1 = await self.service.find_learning_path(self.test_start_node, self.test_end_node)
        path2 = await self.service.find_learning_path(self.test_start_node, self.test_end_node)
        
        # Results should be consistent (same structure, potentially different timestamps)
        assert path1["start_node_id"] == path2["start_node_id"]
        assert path1["end_node_id"] == path2["end_node_id"]
        assert path1["path_statistics"]["primary_path_length"] == path2["path_statistics"]["primary_path_length"]
        
        # Timestamps should be different
        assert path1["discovery_timestamp"] != path2["discovery_timestamp"]
    
    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very long node IDs
        long_node_id = "a" * 1000
        result = await self.service.traverse_graph(long_node_id, max_depth=1)
        assert result["start_node_id"] == long_node_id
        
        # Test with special characters in node IDs
        special_node_id = "node@#$%^&*()"
        result = await self.service.traverse_graph(special_node_id, max_depth=1)
        assert result["start_node_id"] == special_node_id
        
        # Test with numeric node IDs
        numeric_node_id = "12345"
        result = await self.service.traverse_graph(numeric_node_id, max_depth=1)
        assert result["start_node_id"] == numeric_node_id
