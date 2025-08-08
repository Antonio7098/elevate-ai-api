"""
Indexing pipeline for orchestrating blueprint ingestion into the vector database.

This module handles the batch processing of LearningBlueprints into searchable
TextNodes with vector embeddings and metadata indexing.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.models.learning_blueprint import LearningBlueprint
from app.models.text_node import TextNode, TextNodeBatch
from app.core.blueprint_parser import BlueprintParser, BlueprintParserError
from app.core.vector_store import create_vector_store
from app.core.embeddings import get_embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class IndexingPipelineError(Exception):
    """Base exception for indexing pipeline operations."""
    pass


class IndexingProgress:
    """Tracks progress of indexing operations."""
    
    def __init__(self, total_nodes: int):
        self.total_nodes = total_nodes
        self.processed_nodes = 0
        self.failed_nodes = 0
        self.current_blueprint: Optional[str] = None
        self.current_locus: Optional[str] = None
        self.start_time = datetime.utcnow()
        self.errors = []
    
    def update_progress(self, nodes_processed: int = 1, failed: bool = False):
        """Update progress counters."""
        self.processed_nodes += nodes_processed
        if failed:
            self.failed_nodes += 1
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append({
            "timestamp": datetime.utcnow().isoformat(),
            "error": error
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        elapsed = datetime.utcnow() - self.start_time
        success_rate = ((self.processed_nodes - self.failed_nodes) / self.total_nodes * 100) if self.total_nodes > 0 else 0
        
        return {
            "total_nodes": self.total_nodes,
            "processed_nodes": self.processed_nodes,
            "failed_nodes": self.failed_nodes,
            "success_rate": round(success_rate, 2),
            "elapsed_seconds": elapsed.total_seconds(),
            "current_blueprint": self.current_blueprint,
            "current_locus": self.current_locus,
            "errors": self.errors[-10:]  # Last 10 errors
        }


class IndexingPipeline:
    """Pipeline for indexing LearningBlueprints into the vector database."""
    
    def __init__(self):
        self.settings = settings
        self.parser = BlueprintParser()
        self.vector_store = None
        self.embedding_service = None
        self.batch_size = 50  # Number of nodes to process in each batch
    
    async def _initialize_services(self):
        """Initialize vector store and embedding services."""
        if not self.vector_store:
            try:
                from app.core.services import get_vector_store as get_global_vector_store
                self.vector_store = await get_global_vector_store()
            except Exception:
                self.vector_store = create_vector_store(
                    store_type=self.settings.vector_store_type,
                    api_key=self.settings.pinecone_api_key,
                    environment=self.settings.pinecone_environment,
                    persist_directory=self.settings.chroma_persist_directory
                )
                await self.vector_store.initialize()
        
        if not self.embedding_service:
            self.embedding_service = await get_embedding_service()
        
        # Ensure index/collection exists before use
        try:
            exists = await self.vector_store.index_exists("blueprint-nodes")
            if not exists:
                # Choose dimension from embedding service if available; fallback to 768
                try:
                    dim = self.embedding_service.get_dimension()
                except Exception:
                    dim = 768
                await self.vector_store.create_index("blueprint-nodes", dimension=dim)
        except Exception:
            # Best-effort; continue even if create check fails
            pass
    
    async def index_blueprint(self, blueprint: LearningBlueprint) -> Dict[str, Any]:
        """
        Index a single LearningBlueprint into the vector database.
        
        Args:
            blueprint: The LearningBlueprint to index
            
        Returns:
            Dictionary with indexing results and statistics
        """
        try:
            logger.info(f"Starting indexing for blueprint: {blueprint.source_id}")
            
            # Parse blueprint into TextNodes
            nodes = self.parser.parse_blueprint(blueprint)
            logger.info(f"Parsed {len(nodes)} TextNodes from blueprint")
            
            # Initialize progress tracking
            progress = IndexingProgress(len(nodes))
            progress.current_blueprint = blueprint.source_id
            
            # Process nodes in batches
            results = await self._process_nodes_batch(nodes, progress)
            
            # Get final statistics
            stats = progress.get_stats()
            stats.update({
                "blueprint_id": blueprint.source_id,
                "blueprint_title": blueprint.source_title,
                "indexing_completed": True,
                "results": results
            })
            
            logger.info(f"Completed indexing for blueprint {blueprint.source_id}: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to index blueprint {blueprint.source_id}: {e}")
            raise IndexingPipelineError(f"Blueprint indexing failed: {e}")
    
    async def index_blueprints_batch(self, blueprints: List[LearningBlueprint]) -> Dict[str, Any]:
        """
        Index multiple LearningBlueprints in batch.
        
        Args:
            blueprints: List of LearningBlueprints to index
            
        Returns:
            Dictionary with batch indexing results
        """
        try:
            logger.info(f"Starting batch indexing for {len(blueprints)} blueprints")
            
            total_stats = {
                "total_blueprints": len(blueprints),
                "successful_blueprints": 0,
                "failed_blueprints": 0,
                "total_nodes_indexed": 0,
                "total_errors": 0,
                "blueprint_results": [],
                "start_time": datetime.utcnow().isoformat()
            }
            
            for blueprint in blueprints:
                try:
                    result = await self.index_blueprint(blueprint)
                    total_stats["successful_blueprints"] += 1
                    total_stats["total_nodes_indexed"] += result.get("processed_nodes", 0)
                    total_stats["blueprint_results"].append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to index blueprint {blueprint.source_id}: {e}")
                    total_stats["failed_blueprints"] += 1
                    total_stats["total_errors"] += 1
                    total_stats["blueprint_results"].append({
                        "blueprint_id": blueprint.source_id,
                        "error": str(e),
                        "indexing_completed": False
                    })
            
            total_stats["end_time"] = datetime.utcnow().isoformat()
            total_stats["success_rate"] = (
                total_stats["successful_blueprints"] / total_stats["total_blueprints"] * 100
            ) if total_stats["total_blueprints"] > 0 else 0
            
            logger.info(f"Completed batch indexing: {total_stats}")
            return total_stats
            
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
            raise IndexingPipelineError(f"Batch indexing failed: {e}")
    
    async def _process_nodes_batch(self, nodes: List[TextNode], progress: IndexingProgress) -> Dict[str, Any]:
        """Process TextNodes in batches with embeddings and vector storage."""
        results = {
            "nodes_processed": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "errors": []
        }
        
        # Process nodes in batches
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}: {len(batch)} nodes")
            
            try:
                # Generate embeddings for the batch
                batch_results = await self._process_single_batch(batch, progress)
                results["nodes_processed"] += batch_results["nodes_processed"]
                results["embeddings_generated"] += batch_results["embeddings_generated"]
                results["vectors_stored"] += batch_results["vectors_stored"]
                
            except Exception as e:
                error_msg = f"Batch processing failed: {e}"
                logger.error(error_msg)
                progress.add_error(error_msg)
                results["errors"].append(error_msg)
                
                # Mark all nodes in batch as failed
                progress.update_progress(len(batch), failed=True)
        
        return results
    
    async def _process_single_batch(self, nodes: List[TextNode], progress: IndexingProgress) -> Dict[str, Any]:
        """Process a single batch of TextNodes."""
        results = {
            "nodes_processed": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0
        }
        
        # Initialize services if needed
        await self._initialize_services()
        
        # Generate embeddings for all nodes in batch
        texts = [node.content for node in nodes]
        embeddings = await self.embedding_service.embed_batch(texts)
        
        if len(embeddings) != len(nodes):
            raise IndexingPipelineError(f"Embedding count mismatch: {len(embeddings)} vs {len(nodes)}")
        
        # Update nodes with embeddings
        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            node.embedding_dimension = len(embedding)
            node.embedding_model = self.embedding_service.model if hasattr(self.embedding_service, 'model') else 'unknown'
            progress.current_locus = node.locus_id
            results["embeddings_generated"] += 1
        
        # Store vectors in database
        vectors = []
        for node, embedding in zip(nodes, embeddings):
            # Build metadata and filter out None values (Pinecone doesn't allow null metadata)
            metadata = {
                "content": node.content,
                "blueprint_id": node.blueprint_id,
                "locus_id": node.locus_id,
            }
            
            # Only add optional fields if they have values
            if node.locus_type:
                metadata["locus_type"] = node.locus_type.value
            if node.uue_stage:
                metadata["uue_stage"] = node.uue_stage.value
                
            # Add node metadata, filtering out None values and converting lists to strings
            if node.metadata:
                for key, value in node.metadata.items():
                    if value is not None:
                        # Convert lists to comma-separated strings for ChromaDB compatibility
                        if isinstance(value, list):
                            metadata[key] = ",".join(str(item) for item in value)
                        else:
                            metadata[key] = value
            
            vectors.append({
                "id": node.id,
                "values": embedding,
                "metadata": metadata
            })
            
            # Debug logging for the first vector in each batch
            if len(vectors) == 1:
                logger.info(f" [STORAGE DEBUG] Storing vector with blueprint_id: {node.blueprint_id}")
                logger.info(f" [STORAGE DEBUG] Vector metadata: {metadata}")
        
        logger.info(f" [STORAGE DEBUG] Upserting {len(vectors)} vectors to index: blueprint-nodes")
        await self.vector_store.upsert_vectors("blueprint-nodes", vectors)
        results["vectors_stored"] = len(vectors)
        results["nodes_processed"] = len(nodes)
        
        # Update progress
        progress.update_progress(len(nodes))
        
        logger.debug(f"Processed batch: {results}")
        return results
    
    async def check_blueprint_indexed(self, blueprint_id: str) -> bool:
        """Check if a blueprint is already indexed in the vector database."""
        try:
            await self._initialize_services()
            # Search for nodes with this blueprint_id
            # For now, return False as we need to implement proper search
            return False
        except Exception as e:
            logger.warning(f"Could not check if blueprint {blueprint_id} is indexed: {e}")
            return False
    
    async def get_indexing_stats(self, blueprint_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about indexed content."""
        try:
            await self._initialize_services()

            if blueprint_id:
                # Get blueprint-specific stats by querying vector store with a filter
                filter_metadata = {"blueprint_id": blueprint_id}
                blueprint_stats = await self.vector_store.get_stats(
                    "blueprint-nodes",
                    filter_metadata=filter_metadata
                )
                node_count = blueprint_stats.get("total_vector_count", 0)
                logger.info(f"Found {node_count} indexed nodes for blueprint {blueprint_id}")

                # The get_stats call for a filtered query does not return locus_types.
                # We will return the node count and an empty dict for locus_types for now.
                # A more advanced implementation could run multiple get_stats calls for each locus type.
                return {
                    "blueprint_specific": {
                        "blueprint_id": blueprint_id,
                        "node_count": node_count,
                        "locus_types": {},  # Placeholder
                        "uue_stages": {}  # Placeholder
                    }
                }
            else:
                # Get overall stats if no blueprint_id is provided
                stats = await self.vector_store.get_stats("blueprint-nodes")
                return stats

        except Exception as e:
            logger.error(f"Failed to get indexing stats: {e}")
            raise IndexingPipelineError(f"Failed to get indexing stats: {e}")
    
    async def delete_blueprint_index(self, blueprint_id: str) -> Dict[str, Any]:
        """Delete all indexed nodes for a specific blueprint."""
        try:
            logger.info(f"🗑️ [DELETE DEBUG] Starting deletion for blueprint: {blueprint_id}")
            await self._initialize_services()
            
            if not self.vector_store:
                raise IndexingPipelineError("Vector store not initialized")

            # Define the metadata filter to target the specific blueprint
            filter_metadata = {"blueprint_id": blueprint_id}
            logger.info(f"🗑️ [DELETE DEBUG] Using filter metadata: {filter_metadata}")

            # Query the vector store to find vectors for this blueprint
            logger.info(f"🗑️ [DELETE DEBUG] Querying vector store for vectors with blueprint_id: {blueprint_id}")
            stats_before = await self.vector_store.get_stats(
                index_name="blueprint-nodes",
                filter_metadata=filter_metadata
            )
            nodes_to_delete = stats_before.get("total_vector_count", 0)
            logger.info(f"🗑️ [DELETE DEBUG] Found {nodes_to_delete} vectors to delete for blueprint {blueprint_id}")
            logger.info(f"🗑️ [DELETE DEBUG] Stats before deletion: {stats_before}")

            if nodes_to_delete > 0:
                # Delete the vectors using the new delete_by_metadata method
                logger.info(f"🗑️ [DELETE DEBUG] Calling delete_by_metadata with index: blueprint-nodes, filter: {filter_metadata}")
                await self.vector_store.delete_by_metadata(
                    index_name="blueprint-nodes",
                    filter_metadata=filter_metadata
                )
                logger.info(f"🗑️ [DELETE DEBUG] Successfully submitted deletion for {nodes_to_delete} nodes for blueprint {blueprint_id}")
                
                # Verify deletion by checking stats again with retries
                retries = 5
                backoff = 2  # seconds
                remaining_nodes = nodes_to_delete
                for attempt in range(1, retries + 1):
                    await asyncio.sleep(backoff)
                    stats_after = await self.vector_store.get_stats(
                        index_name="blueprint-nodes",
                        filter_metadata=filter_metadata
                    )
                    remaining_nodes = stats_after.get("total_vector_count", 0)
                    logger.info(
                        f"🗑️ [DELETE DEBUG] Attempt {attempt}: {remaining_nodes} vectors remain for blueprint {blueprint_id}"
                    )
                    if remaining_nodes == 0:
                        break
                    backoff *= 2  # exponential

                actual_nodes_deleted = nodes_to_delete - remaining_nodes
                logger.info(f"🗑️ [DELETE DEBUG] After deletion, {remaining_nodes} vectors remain for blueprint {blueprint_id}")
                logger.info(f"🗑️ [DELETE DEBUG] Actually deleted {actual_nodes_deleted} vectors for blueprint {blueprint_id}")
                
                # Use the actual deletion results
                result = {
                    "blueprint_id": blueprint_id,
                    "nodes_found": nodes_to_delete,
                    "nodes_deleted": actual_nodes_deleted,
                    "nodes_remaining": remaining_nodes,
                    "deletion_completed": remaining_nodes == 0
                }
            else:
                logger.info(f"🗑️ [DELETE DEBUG] No nodes found for blueprint {blueprint_id}, nothing to delete.")
                result = {
                    "blueprint_id": blueprint_id,
                    "nodes_found": 0,
                    "nodes_deleted": 0,
                    "nodes_remaining": 0,
                    "deletion_completed": True
                }
            
            logger.info(f"🗑️ [DELETE DEBUG] Deletion result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete blueprint index {blueprint_id}: {e}")
            raise IndexingPipelineError(f"Failed to delete blueprint index: {e}")