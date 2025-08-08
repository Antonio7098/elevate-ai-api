"""
Vector database client wrapper for the RAG system.

This module provides a unified interface for vector database operations,
supporting both Pinecone (production) and ChromaDB (local development).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from chromadb import Client, Collection
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector database."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class VectorStore(ABC):
    """Abstract base class for vector database operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection."""
        pass
    
    @abstractmethod
    async def create_index(self, index_name: str, dimension: int = 1536) -> None:
        """Create a new index."""
        pass
    
    @abstractmethod
    async def delete_index(self, index_name: str) -> None:
        """Delete an index."""
        pass
    
    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        pass
    
    @abstractmethod
    async def upsert_vectors(
        self,
        index_name: str,
        vectors: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors into the index."""
        pass
    
    @abstractmethod
    async def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(
        self,
        index_name: str,
        vector_ids: List[str]
    ) -> None:
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    async def get_stats(self, index_name: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get index statistics, optionally filtered by metadata."""
        pass

    @abstractmethod
    async def delete_by_metadata(
        self,
        index_name: str,
        filter_metadata: Dict[str, Any]
    ) -> None:
        """Delete vectors by metadata filter."""
        pass


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.client: Optional[Pinecone] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize Pinecone client."""
        try:
            def _init():
                return Pinecone(api_key=self.api_key)
            
            self.client = await asyncio.get_event_loop().run_in_executor(
                self._executor, _init
            )
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise VectorStoreError(f"Pinecone initialization failed: {e}")
    
    async def create_index(self, index_name: str, dimension: int = 1536) -> None:
        """Create a new Pinecone index."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            def _create():
                existing_indexes = [idx.name for idx in self.client.list_indexes()]
                if index_name not in existing_indexes:
                    self.client.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _create)
            logger.info(f"Created Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create Pinecone index {index_name}: {e}")
            raise VectorStoreError(f"Failed to create index: {e}")
    
    async def delete_index(self, index_name: str) -> None:
        """Delete a Pinecone index."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            def _delete():
                existing_indexes = [idx.name for idx in self.client.list_indexes()]
                if index_name in existing_indexes:
                    self.client.delete_index(index_name)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
            logger.info(f"Deleted Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index {index_name}: {e}")
            raise VectorStoreError(f"Failed to delete index: {e}")
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if a Pinecone index exists."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            def _check():
                existing_indexes = [idx.name for idx in self.client.list_indexes()]
                return index_name in existing_indexes
            
            return await asyncio.get_event_loop().run_in_executor(self._executor, _check)
        except Exception as e:
            logger.error(f"Failed to check Pinecone index existence: {e}")
            raise VectorStoreError(f"Failed to check index existence: {e}")
    
    async def upsert_vectors(
        self,
        index_name: str,
        vectors: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors into Pinecone index."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            # Check if index exists, create it if not
            print(f"[DEBUG] Checking if index {index_name} exists...")
            index_exists = await self.index_exists(index_name)
            print(f"[DEBUG] Index {index_name} exists: {index_exists}")
            
            if not index_exists:
                print(f"[DEBUG] Index {index_name} does not exist, creating it...")
                # Determine dimension from first vector
                dimension = len(vectors[0]["values"]) if vectors else 1536
                print(f"[DEBUG] Creating index with dimension: {dimension}")
                await self.create_index(index_name, dimension)
                
                # Wait a moment for index to be ready
                import time
                def _wait():
                    time.sleep(5)  # Increased wait time
                print(f"[DEBUG] Waiting for index {index_name} to be ready...")
                await asyncio.get_event_loop().run_in_executor(self._executor, _wait)
                
                # Verify index was created
                final_check = await self.index_exists(index_name)
                print(f"[DEBUG] Index {index_name} created successfully: {final_check}")
            else:
                print(f"[DEBUG] Index {index_name} already exists, proceeding with upsert...")
            
            index = self.client.Index(index_name)
            
            print(f"[DEBUG] About to upsert {len(vectors)} vectors to index {index_name}")
            
            def _upsert():
                print(f"[DEBUG] Executing upsert for {len(vectors)} vectors...")
                result = index.upsert(vectors=vectors)
                print(f"[DEBUG] Upsert result: {result}")
                return result
            
            result = await asyncio.get_event_loop().run_in_executor(self._executor, _upsert)
            print(f"[DEBUG] Upsert completed with result: {result}")
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {e}")
            raise VectorStoreError(f"Failed to upsert vectors: {e}")
    
    def _convert_filters_for_pinecone(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert filter format to Pinecone-compatible syntax.
        
        Pinecone requires array filters to use $in operator:
        {'field': ['val1', 'val2']} -> {'field': {'$in': ['val1', 'val2']}}
        """
        if not filters:
            return filters
            
        converted = {}
        for key, value in filters.items():
            if isinstance(value, list):
                # Convert array filters to use $in operator
                converted[key] = {"$in": value}
            else:
                # Keep scalar values as-is
                converted[key] = value
        
        return converted
    
    async def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in Pinecone."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            index = self.client.Index(index_name)
            
            # Convert filters to Pinecone-compatible format
            pinecone_filters = self._convert_filters_for_pinecone(filter_metadata)
            
            def _search():
                return index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=pinecone_filters
                )
            
            result = await asyncio.get_event_loop().run_in_executor(self._executor, _search)
            
            search_results = []
            for match in result.matches:
                search_results.append(SearchResult(
                    id=match.id,
                    content=match.metadata.get("content", ""),
                    metadata=match.metadata,
                    score=match.score
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search Pinecone index: {e}")
            raise VectorStoreError(f"Failed to search: {e}")
    
    async def delete_vectors(
        self,
        index_name: str,
        vector_ids: List[str]
    ) -> None:
        """Delete vectors by ID from Pinecone."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            index = self.client.Index(index_name)
            
            def _delete():
                index.delete(ids=vector_ids)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
            logger.info(f"Deleted {len(vector_ids)} vectors from Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")
    
    async def delete_by_metadata(
        self,
        index_name: str,
        filter_metadata: Dict[str, Any]
    ) -> None:
        """Delete vectors by metadata filter from Pinecone."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            index = self.client.Index(index_name)
            pinecone_filter = self._convert_filters_for_pinecone(filter_metadata)
            
            def _delete():
                # Pinecone's delete operation with metadata filter
                index.delete(filter=pinecone_filter)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
            logger.info(f"Submitted delete request for vectors in {index_name} with filter: {pinecone_filter}")

        except Exception as e:
            logger.error(f"Failed to delete vectors by metadata from Pinecone: {e}")
            raise VectorStoreError(f"Failed to delete by metadata: {e}")
    
    async def get_stats(self, index_name: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get Pinecone index statistics, optionally filtered by metadata.

        Serverless/Starter indexes do NOT support describe_index_stats with a filter.
        We fallback to a query-based approach: perform a blanket similarity search with
        a dummy vector and count the matches that satisfy the metadata filter. This is
        less efficient but works within plan limits.
        """
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")

        try:
            index = self.client.Index(index_name)

            # Fetch index dimension once (describe without filter is allowed)
            def _get_dim():
                return index.describe_index_stats()
            stats_overall = await asyncio.get_event_loop().run_in_executor(self._executor, _get_dim)
            dimension = stats_overall.dimension or 1536

            # Build a dummy zero vector for querying
            dummy_vector = [0.0] * dimension

            # Convert filters to Pinecone syntax
            pinecone_filter = self._convert_filters_for_pinecone(filter_metadata)

            def _query():
                return index.query(
                    vector=dummy_vector,
                    top_k=10000,  # large number to retrieve all matches (subject to service caps)
                    include_metadata=True,
                    filter=pinecone_filter,
                )

            query_result = await asyncio.get_event_loop().run_in_executor(self._executor, _query)
            matches = query_result.get("matches", [])
            node_count = len(matches)

            # Aggregate locus types if available
            locus_types: Dict[str, int] = {}
            for m in matches:
                lt = m.get("metadata", {}).get("locus_type")
                if lt:
                    locus_types[lt] = locus_types.get(lt, 0) + 1

            # For blueprint-specific stats, return the filtered count
            # For overall stats, return the total count
            if filter_metadata:
                return {
                    "total_vector_count": node_count,  # Filtered count
                    "dimension": dimension,
                    "locus_types": locus_types,
                }
            else:
                return {
                    "total_vector_count": stats_overall.total_vector_count,
                    "dimension": dimension,
                }
        except Exception as e:
            logger.error(
                "Failed to get Pinecone index stats via query fallback: %s", e
            )
            raise VectorStoreError(f"Failed to get stats: {e}")


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation for local development."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client: Optional[Client] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client."""
        try:
            def _init():
                return Client(Settings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False
                ))
            
            self.client = await asyncio.get_event_loop().run_in_executor(
                self._executor, _init
            )
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise VectorStoreError(f"ChromaDB initialization failed: {e}")
    
    async def create_index(self, index_name: str, dimension: int = 1536) -> None:
        """Create a new ChromaDB collection."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            def _create():
                if index_name not in [col.name for col in self.client.list_collections()]:
                    self.client.create_collection(
                        name=index_name,
                        metadata={"dimension": dimension}
                    )
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _create)
            logger.info(f"Created ChromaDB collection: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection {index_name}: {e}")
            raise VectorStoreError(f"Failed to create collection: {e}")
    
    async def delete_index(self, index_name: str) -> None:
        """Delete a ChromaDB collection."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            def _delete():
                self.client.delete_collection(index_name)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
            logger.info(f"Deleted ChromaDB collection: {index_name}")
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection {index_name}: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}")
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if a ChromaDB collection exists."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            def _check():
                return index_name in [col.name for col in self.client.list_collections()]
            
            return await asyncio.get_event_loop().run_in_executor(self._executor, _check)
        except Exception as e:
            logger.error(f"Failed to check ChromaDB collection existence: {e}")
            raise VectorStoreError(f"Failed to check collection existence: {e}")
    
    async def upsert_vectors(
        self,
        index_name: str,
        vectors: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors into ChromaDB collection."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_collection(index_name)
            
            # Prepare data for ChromaDB
            ids = [v["id"] for v in vectors]
            embeddings = [v["values"] for v in vectors]
            metadatas = [v.get("metadata", {}) for v in vectors]
            documents = [v.get("metadata", {}).get("content", "") for v in vectors]
            
            def _upsert():
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _upsert)
            logger.info(f"Upserted {len(vectors)} vectors to ChromaDB collection: {index_name}")
        except Exception as e:
            logger.error(f"Failed to upsert vectors to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to upsert vectors: {e}")
    
    async def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in ChromaDB."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_collection(index_name)
            
            def _search():
                # ChromaDB requires specific where clause format
                if filter_metadata:
                    # Convert metadata filters to ChromaDB where format
                    where_conditions = []
                    for key, value in filter_metadata.items():
                        if isinstance(value, list):
                            # For list values, use $in operator
                            where_conditions.append({key: {"$in": value}})
                        else:
                            # For single values, use $eq operator
                            where_conditions.append({key: {"$eq": value}})
                    
                    if len(where_conditions) == 1:
                        where_clause = where_conditions[0]
                    else:
                        # Multiple conditions need $and
                        where_clause = {"$and": where_conditions}
                else:
                    where_clause = None
                
                return collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=where_clause
                )
            
            result = await asyncio.get_event_loop().run_in_executor(self._executor, _search)
            
            search_results = []
            if result["ids"] and result["ids"][0]:
                for i, doc_id in enumerate(result["ids"][0]):
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=result["documents"][0][i] if result["documents"][0] else "",
                        metadata=result["metadatas"][0][i] if result["metadatas"][0] else {},
                        score=result["distances"][0][i] if result["distances"][0] else 0.0
                    ))
            
            return search_results
        except Exception as e:
            logger.error(f"Failed to search ChromaDB collection: {e}")
            raise VectorStoreError(f"Failed to search: {e}")
    
    async def delete_vectors(
        self,
        index_name: str,
        vector_ids: List[str]
    ) -> None:
        """Delete vectors by ID from ChromaDB."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_collection(index_name)
            
            def _delete():
                collection.delete(ids=vector_ids)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
            logger.info(f"Deleted {len(vector_ids)} vectors from ChromaDB collection: {index_name}")
        except Exception as e:
            logger.error(f"Failed to delete vectors from ChromaDB: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")

    async def delete_by_metadata(
        self,
        index_name: str,
        filter_metadata: Dict[str, Any]
    ) -> None:
        """Delete vectors by metadata filter from ChromaDB."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_collection(index_name)
            
            def _delete():
                # ChromaDB uses 'where' for filtering in delete()
                collection.delete(where=filter_metadata)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _delete)
            logger.info(f"Submitted delete request for vectors in ChromaDB {index_name} with filter: {filter_metadata}")

        except Exception as e:
            logger.error(f"Failed to delete vectors by metadata from ChromaDB: {e}")
            raise VectorStoreError(f"Failed to delete by metadata: {e}")
    
    async def get_stats(self, index_name: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get ChromaDB collection statistics, optionally filtered by metadata."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_collection(index_name)
            
            def _stats():
                # Prefer fast count(); if filtering needed, use get(where=...)
                if not filter_metadata:
                    return collection.count()
                # Use get with where filter and count returned ids
                res = collection.get(where=filter_metadata, include=[])
                ids = res.get("ids") or []
                return len(ids)
            
            count = await asyncio.get_event_loop().run_in_executor(self._executor, _stats)
            return {
                "total_vector_count": count,
                "collection_name": index_name
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB collection stats: {e}")
            raise VectorStoreError(f"Failed to get stats: {e}")


# Factory function to create the appropriate vector store
def create_vector_store(
    store_type: str = "pinecone",
    **kwargs
) -> VectorStore:
    """Create a vector store instance based on configuration."""
    if store_type.lower() == "pinecone":
        return PineconeVectorStore(
            api_key=kwargs["api_key"],
            environment=kwargs["environment"]
        )
    elif store_type.lower() == "chromadb":
        return ChromaDBVectorStore(
            persist_directory=kwargs.get("persist_directory", "./chroma_db")
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}") 