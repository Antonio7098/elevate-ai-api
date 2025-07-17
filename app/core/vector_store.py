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

import pinecone
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
    async def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics."""
        pass


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.client: Optional[pinecone.Pinecone] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize Pinecone client."""
        try:
            def _init():
                pinecone.init(api_key=self.api_key, environment=self.environment)
                return pinecone.Pinecone(api_key=self.api_key)
            
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
                if index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric="cosine"
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
                if index_name in pinecone.list_indexes():
                    pinecone.delete_index(index_name)
            
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
                return index_name in pinecone.list_indexes()
            
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
            index = self.client.Index(index_name)
            
            def _upsert():
                index.upsert(vectors=vectors)
            
            await asyncio.get_event_loop().run_in_executor(self._executor, _upsert)
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {e}")
            raise VectorStoreError(f"Failed to upsert vectors: {e}")
    
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
            
            def _search():
                return index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_metadata
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
    
    async def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self.client:
            raise VectorStoreError("Pinecone client not initialized")
        
        try:
            index = self.client.Index(index_name)
            
            def _stats():
                return index.describe_index_stats()
            
            stats = await asyncio.get_event_loop().run_in_executor(self._executor, _stats)
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone index stats: {e}")
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
                return collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=filter_metadata
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
    
    async def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        if not self.client:
            raise VectorStoreError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_collection(index_name)
            
            def _stats():
                return collection.count()
            
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