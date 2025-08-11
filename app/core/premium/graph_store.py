"""
Neo4j GraphRAG implementation with Core API integration.
Provides knowledge graph creation, querying, and traversal capabilities for premium users.
"""

import os
from typing import List, Dict, Any, Optional
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
import httpx

from .core_api_client import CoreAPIClient

class Neo4jGraphStore:
    """Neo4j graph store with Core API integration for premium features"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        # Core API integration
        self.core_api_client = CoreAPIClient()
        
        if not NEO4J_AVAILABLE:
            print("Warning: Neo4j not available, using mock implementation")
            self.driver = None
            return
            
        # Get Neo4j connection details from environment
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Initialize graph schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        if not NEO4J_AVAILABLE or self.driver is None:
            print("Warning: Skipping Neo4j schema initialization")
            return
            
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT blueprint_id IF NOT EXISTS FOR (b:Blueprint) REQUIRE b.id IS UNIQUE")
            session.run("CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE")
            
            # Create indexes for better performance
            session.run("CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)")
            session.run("CREATE INDEX concept_tags IF NOT EXISTS FOR (c:Concept) ON (c.tags)")
            session.run("CREATE INDEX concept_complexity IF NOT EXISTS FOR (c:Concept) ON (c.complexityScore)")
    
    async def create_knowledge_graph(self, blueprint_id: str):
        """Create knowledge graph from blueprint using Core API data"""
        try:
            # Get blueprint and knowledge primitives from Core API
            blueprint = await self.core_api_client.get_learning_blueprint(blueprint_id)
            primitives = await self.core_api_client.get_knowledge_primitives(
                blueprint_id=blueprint_id,
                include_premium_fields=True  # complexityScore, isCoreConcept, etc.
            )
            
            with self.driver.session() as session:
                # Create blueprint node
                session.run("""
                    MERGE (b:Blueprint {id: $blueprint_id})
                    SET b.title = $title, b.sourceText = $source_text, b.createdAt = $created_at
                """, blueprint_id=blueprint_id, 
                    title=blueprint.get("title", ""),
                    source_text=blueprint.get("sourceText", ""),
                    created_at=blueprint.get("createdAt")
                )
                
                # Create concept nodes with premium fields
                for primitive in primitives:
                    session.run("""
                        MERGE (c:Concept {id: $concept_id})
                        SET c.name = $name, 
                            c.description = $description,
                            c.tags = $tags,
                            c.complexityScore = $complexity_score,
                            c.isCoreConcept = $is_core_concept,
                            c.semanticSimilarityScore = $semantic_similarity_score
                    """, 
                        concept_id=primitive["id"],
                        name=primitive["name"],
                        description=primitive["description"],
                        tags=primitive.get("conceptTags", []),
                        complexity_score=primitive.get("complexityScore", 0.5),
                        is_core_concept=primitive.get("isCoreConcept", False),
                        semantic_similarity_score=primitive.get("semanticSimilarityScore", 0.0)
                    )
                    
                    # Link concept to blueprint
                    session.run("""
                        MATCH (c:Concept {id: $concept_id})
                        MATCH (b:Blueprint {id: $blueprint_id})
                        MERGE (c)-[:PART_OF]->(b)
                    """, concept_id=primitive["id"], blueprint_id=blueprint_id)
                
                # Create prerequisite relationships
                for primitive in primitives:
                    if primitive.get("prerequisiteIds"):
                        for prereq_id in primitive["prerequisiteIds"]:
                            session.run("""
                                MATCH (c1:Concept {id: $concept_id})
                                MATCH (c2:Concept {id: $prereq_id})
                                MERGE (c2)-[:PREREQUISITE_FOR]->(c1)
                            """, concept_id=primitive["id"], prereq_id=prereq_id)
                
                # Create related concept relationships
                for primitive in primitives:
                    if primitive.get("relatedConceptIds"):
                        for related_id in primitive["relatedConceptIds"]:
                            session.run("""
                                MATCH (c1:Concept {id: $concept_id})
                                MATCH (c2:Concept {id: $related_id})
                                MERGE (c1)-[:RELATED_TO]->(c2)
                            """, concept_id=primitive["id"], related_id=related_id)
                
                # Create core concept relationships
                for primitive in primitives:
                    if primitive.get("isCoreConcept", False):
                        session.run("""
                            MATCH (c:Concept {id: $concept_id})
                            SET c:CoreConcept
                        """, concept_id=primitive["id"])
                
            return {"status": "success", "concepts_created": len(primitives)}
            
        except Exception as e:
            print(f"Error creating knowledge graph: {e}")
            return {"status": "error", "message": str(e)}
    
    async def query_graph(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Query knowledge graph with user-specific context from Core API"""
        try:
            # Get user's learning analytics and memory insights from Core API
            user_analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            memory_insights = await self.core_api_client.get_user_memory_insights(user_id)
            
            with self.driver.session() as session:
                # Perform semantic search with user context
                result = session.run("""
                    CALL db.index.vector.queryNodes('concept_embeddings', $k, $query_vector)
                    YIELD node, score
                    MATCH (c:Concept)-[:PART_OF]->(b:Blueprint)
                    RETURN c.name as concept_name, 
                           c.description as description,
                           c.complexityScore as complexity,
                           c.isCoreConcept as is_core,
                           b.title as blueprint_title,
                           score as relevance_score
                    ORDER BY score DESC
                    LIMIT $limit
                """, 
                    k=10,  # Number of nearest neighbors
                    query_vector=[0.1, 0.2, 0.3],  # TODO: Use actual query embedding
                    limit=10
                )
                
                results = []
                for record in result:
                    results.append({
                        "concept_name": record["concept_name"],
                        "description": record["description"],
                        "complexity": record["complexity"],
                        "is_core_concept": record["is_core"],
                        "blueprint_title": record["blueprint_title"],
                        "relevance_score": record["relevance_score"],
                        "user_analytics": user_analytics,
                        "memory_insights": memory_insights
                    })
                
                return results
                
        except Exception as e:
            print(f"Error querying graph: {e}")
            return []
    
    async def traverse_concepts(self, concept_id: str, user_id: str, depth: int = 3):
        """Traverse concept relationships using Core API LearningPath data"""
        try:
            # Get user's learning paths from Core API
            learning_paths = await self.core_api_client.get_user_learning_paths(user_id)
            
            with self.driver.session() as session:
                # Traverse concept relationships
                result = session.run("""
                    MATCH (start:Concept {id: $concept_id})
                    CALL apoc.path.subgraphNodes(start, {
                        maxLevel: $depth,
                        relationshipFilter: 'PREREQUISITE_FOR|RELATED_TO|PART_OF'
                    })
                    YIELD node
                    RETURN node.name as concept_name,
                           node.description as description,
                           node.complexityScore as complexity,
                           node.isCoreConcept as is_core,
                           labels(node) as labels
                    ORDER BY node.complexityScore ASC
                """, concept_id=concept_id, depth=depth)
                
                traversal_results = []
                for record in result:
                    traversal_results.append({
                        "concept_name": record["concept_name"],
                        "description": record["description"],
                        "complexity": record["complexity"],
                        "is_core_concept": record["is_core"],
                        "labels": record["labels"],
                        "learning_paths": learning_paths
                    })
                
                return traversal_results
                
        except Exception as e:
            print(f"Error traversing concepts: {e}")
            return []
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()

