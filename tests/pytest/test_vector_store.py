"""
Unit tests for vector store operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.core.vector_store import (
    VectorStore,
    PineconeVectorStore,
    ChromaDBVectorStore,
    SearchResult,
    VectorStoreError,
    create_vector_store
)


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            id="test_id",
            content="test content",
            metadata={"key": "value"},
            score=0.95
        )
        
        assert result.id == "test_id"
        assert result.content == "test content"
        assert result.metadata == {"key": "value"}
        assert result.score == 0.95


class TestPineconeVectorStore:
    """Test PineconeVectorStore implementation."""
    
    @pytest.fixture
    def pinecone_store(self):
        """Create a PineconeVectorStore instance for testing."""
        return PineconeVectorStore(
            api_key="test_key",
            environment="test_env"
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, pinecone_store):
        """Test Pinecone client initialization."""
        with patch('pinecone.init'), patch('pinecone.Pinecone') as mock_pinecone:
            mock_client = Mock()
            mock_pinecone.return_value = mock_client
            
            await pinecone_store.initialize()
            
            assert pinecone_store.client == mock_client
    
    @pytest.mark.asyncio
    async def test_create_index(self, pinecone_store):
        """Test index creation."""
        with patch('pinecone.list_indexes', return_value=[]), \
             patch('pinecone.create_index') as mock_create:
            
            pinecone_store.client = Mock()
            
            await pinecone_store.create_index("test_index", dimension=1536)
            
            mock_create.assert_called_once_with(
                name="test_index",
                dimension=1536,
                metric="cosine"
            )
    
    @pytest.mark.asyncio
    async def test_search(self, pinecone_store):
        """Test vector search."""
        mock_index = Mock()
        mock_match = Mock()
        mock_match.id = "test_id"
        mock_match.metadata = {"content": "test content", "key": "value"}
        mock_match.score = 0.95
        
        mock_result = Mock()
        mock_result.matches = [mock_match]
        
        mock_index.query.return_value = mock_result
        pinecone_store.client = Mock()
        pinecone_store.client.Index.return_value = mock_index
        
        query_vector = [0.1, 0.2, 0.3]
        results = await pinecone_store.search("test_index", query_vector, top_k=5)
        
        assert len(results) == 1
        assert results[0].id == "test_id"
        assert results[0].content == "test content"
        assert results[0].score == 0.95


class TestChromaDBVectorStore:
    """Test ChromaDBVectorStore implementation."""
    
    @pytest.fixture
    def chroma_store(self):
        """Create a ChromaDBVectorStore instance for testing."""
        return ChromaDBVectorStore(persist_directory="./test_chroma")
    
    @pytest.mark.asyncio
    async def test_initialization(self, chroma_store):
        """Test ChromaDB client initialization."""
        with patch('chromadb.Client') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            await chroma_store.initialize()
            
            assert chroma_store.client == mock_instance
    
    @pytest.mark.asyncio
    async def test_create_collection(self, chroma_store):
        """Test collection creation."""
        mock_collections = []
        mock_client = Mock()
        mock_client.list_collections.return_value = mock_collections
        mock_client.create_collection = Mock()
        
        chroma_store.client = mock_client
        
        await chroma_store.create_index("test_collection", dimension=1536)
        
        mock_client.create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"dimension": 1536}
        )


class TestVectorStoreFactory:
    """Test vector store factory function."""
    
    def test_create_pinecone_store(self):
        """Test creating a Pinecone vector store."""
        store = create_vector_store(
            store_type="pinecone",
            api_key="test_key",
            environment="test_env"
        )
        
        assert isinstance(store, PineconeVectorStore)
        assert store.api_key == "test_key"
        assert store.environment == "test_env"
    
    def test_create_chromadb_store(self):
        """Test creating a ChromaDB vector store."""
        store = create_vector_store(
            store_type="chromadb",
            persist_directory="./test_chroma"
        )
        
        assert isinstance(store, ChromaDBVectorStore)
        assert store.persist_directory == "./test_chroma"
    
    def test_invalid_store_type(self):
        """Test creating an invalid store type."""
        with pytest.raises(ValueError, match="Unsupported vector store type"):
            create_vector_store(store_type="invalid")


class TestVectorStoreError:
    """Test VectorStoreError exception."""
    
    def test_vector_store_error_creation(self):
        """Test creating a VectorStoreError instance."""
        error = VectorStoreError("Test error message")
        assert str(error) == "Test error message" 