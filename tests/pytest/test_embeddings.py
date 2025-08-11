"""
Unit tests for embedding services.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.core.embeddings import (
    EmbeddingService,
    OpenAIEmbeddingService,
    GoogleEmbeddingService,
    LocalEmbeddingService,
    EmbeddingError,
    create_embedding_service,
    initialize_embedding_service,
    get_embedding_service
)


class TestOpenAIEmbeddingService:
    """Test OpenAIEmbeddingService implementation."""
    
    @pytest.fixture
    def openai_service(self):
        """Create an OpenAIEmbeddingService instance for testing."""
        return OpenAIEmbeddingService(
            api_key="test_key",
            model="text-embedding-3-small"
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, openai_service):
        """Test OpenAI client initialization."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            await openai_service.initialize()
            
            assert openai_service.client == mock_client
    
    @pytest.mark.asyncio
    async def test_embed_text(self, openai_service):
        """Test single text embedding."""
        mock_response = Mock()
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_data]
        
        openai_service.client = Mock()
        openai_service.client.embeddings.create.return_value = mock_response
        
        result = await openai_service.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        openai_service.client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text"
        )
    
    @pytest.mark.asyncio
    async def test_embed_batch(self, openai_service):
        """Test batch text embedding."""
        mock_response = Mock()
        mock_data1 = Mock()
        mock_data1.embedding = [0.1, 0.2, 0.3]
        mock_data2 = Mock()
        mock_data2.embedding = [0.4, 0.5, 0.6]
        mock_response.data = [mock_data1, mock_data2]
        
        openai_service.client = Mock()
        openai_service.client.embeddings.create.return_value = mock_response
        
        result = await openai_service.embed_batch(["text1", "text2"])
        
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        openai_service.client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["text1", "text2"]
        )
    
    def test_get_dimension(self, openai_service):
        """Test getting embedding dimension."""
        assert openai_service.get_dimension() == 1536


class TestGoogleEmbeddingService:
    """Test GoogleEmbeddingService implementation."""
    
    @pytest.fixture
    def google_service(self):
        """Create a GoogleEmbeddingService instance for testing."""
        return GoogleEmbeddingService(
            api_key="test_key",
            model="embedding-001"
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, google_service):
        """Test Google AI client initialization."""
        with patch('google.generativeai.configure') as mock_configure:
            await google_service.initialize()
            
            mock_configure.assert_called_once_with(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_embed_text(self, google_service):
        """Test single text embedding."""
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_model.embed_content.return_value = mock_embedding
        
        with patch('google.generativeai.GenerativeModel', return_value=mock_model):
            result = await google_service.embed_text("test text")
            
            assert result == [0.1, 0.2, 0.3]
            mock_model.embed_content.assert_called_once_with("test text")
    
    def test_get_dimension(self, google_service):
        """Test getting embedding dimension."""
        assert google_service.get_dimension() == 768


class TestLocalEmbeddingService:
    """Test LocalEmbeddingService implementation."""
    
    @pytest.fixture
    def local_service(self):
        """Create a LocalEmbeddingService instance for testing."""
        return LocalEmbeddingService(model_name="all-MiniLM-L6-v2")
    
    @pytest.mark.asyncio
    async def test_initialization(self, local_service):
        """Test sentence-transformers model initialization."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            await local_service.initialize()
            
            assert local_service.model == mock_model
            assert local_service._dimension == 384
    
    @pytest.mark.asyncio
    async def test_embed_text(self, local_service):
        """Test single text embedding."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        local_service.model = mock_model
        
        result = await local_service.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("test text", convert_to_tensor=False)
    
    def test_get_dimension(self, local_service):
        """Test getting embedding dimension."""
        assert local_service.get_dimension() == 384


class TestEmbeddingServiceFactory:
    """Test embedding service factory function."""
    
    def test_create_openai_service(self):
        """Test creating an OpenAI embedding service."""
        service = create_embedding_service(
            service_type="openai",
            api_key="test_key",
            model="text-embedding-3-small"
        )
        
        assert isinstance(service, OpenAIEmbeddingService)
        assert service.api_key == "test_key"
        assert service.model == "text-embedding-3-small"
    
    def test_create_google_service(self):
        """Test creating a Google embedding service."""
        service = create_embedding_service(
            service_type="google",
            api_key="test_key",
            model="embedding-001"
        )
        
        assert isinstance(service, GoogleEmbeddingService)
        assert service.api_key == "test_key"
        assert service.model == "embedding-001"
    
    def test_create_local_service(self):
        """Test creating a local embedding service."""
        service = create_embedding_service(
            service_type="local",
            model_name="all-MiniLM-L6-v2"
        )
        
        assert isinstance(service, LocalEmbeddingService)
        assert service.model_name == "all-MiniLM-L6-v2"
    
    def test_invalid_service_type(self):
        """Test creating an invalid service type."""
        with pytest.raises(ValueError, match="Unsupported embedding service type"):
            create_embedding_service(service_type="invalid")


class TestGlobalEmbeddingService:
    """Test global embedding service management."""
    
    @pytest.mark.asyncio
    async def test_initialize_and_get_service(self):
        """Test initializing and getting the global embedding service."""
        mock_service = Mock()
        
        with patch('app.core.embeddings.create_embedding_service', return_value=mock_service):
            await initialize_embedding_service(
                service_type="openai",
                api_key="test_key"
            )
            
            service = await get_embedding_service()
            assert service == mock_service
    
    @pytest.mark.asyncio
    async def test_get_service_without_initialization(self):
        """Test getting service without initialization."""
        with pytest.raises(EmbeddingError, match="Embedding service not initialized"):
            await get_embedding_service()


class TestEmbeddingError:
    """Test EmbeddingError exception."""
    
    def test_embedding_error_creation(self):
        """Test creating an EmbeddingError instance."""
        error = EmbeddingError("Test error message")
        assert str(error) == "Test error message" 