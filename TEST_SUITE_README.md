# 🚀 Elevate AI API - Comprehensive Test Suite

A modular test suite for testing ALL services with **REAL API and LLM calls**, including vector stores with Pinecone and LLM calls with Google Gemini.

## 🎯 Overview

This test suite is designed to test the entire Elevate AI API system with real external API calls, ensuring that all services work correctly in production-like conditions. The tests are organized into logical modules that can be run independently or together.

## 📋 Test Modules

### 1. **LLM Services** (`tests/services/test_llm_services.py`)
- **Google Gemini Service**: Tests `gemini-2.5-flash` and `gemini-1.5-pro` models
- **OpenRouter Service**: Tests various OpenRouter models
- **Mock LLM Service**: Tests fallback behavior
- **Performance Testing**: Response times and reliability

### 2. **Vector Stores** (`tests/services/test_vector_stores.py`)
- **Pinecone Vector Store**: Tests production vector database
- **ChromaDB Vector Store**: Tests local vector database
- **Embedding Operations**: Create, query, delete operations
- **Performance Testing**: Response times and throughput

### 3. **RAG & Search Services** (`tests/services/test_rag_search.py`)
- **RAG Service**: Document ingestion and retrieval
- **Search Service**: Semantic and hybrid search
- **Knowledge Retrieval**: Knowledge search and synthesis
- **Context Assembly**: Context building and enrichment

### 4. **Note Services** (`tests/services/test_note_services.py`)
- **Note Editing Service**: AI-powered note editing
- **Granular Editing Service**: Line, section, and block-level edits
- **Note Agent Orchestrator**: Multi-agent coordination
- **Content Conversion**: Markdown ↔ BlockNote conversion

### 5. **Premium Agents** (`tests/services/test_premium_agents.py`)
- **Content Curator Agent**: Content enhancement and curation
- **Explanation Agent**: Concept explanation and step breakdown
- **Context Assembly Agent**: Context building and enrichment
- **Premium Routing Agent**: Request routing and agent selection

### 6. **API Endpoints** (`tests/api/test_api_endpoints.py`)
- **Note Creation**: Create new notes via API
- **Note Editing**: Edit notes via API endpoints
- **Note Search**: Search notes via API
- **Granular Editing**: Line/section-level editing via API
- **Health Check**: API health and status

## 🚀 Running Tests

### Prerequisites

1. **Environment Variables**: Ensure your `.env` file contains:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=us-west1-gcp
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install python-dotenv httpx fastapi
   ```

### Running All Tests

```bash
# Run all test modules
python run_all_tests.py all

# Or simply run all (default)
python run_all_tests.py
```

### Running Individual Modules

```bash
# Run specific modules
python run_all_tests.py llm vector
python run_all_tests.py rag notes
python run_all_tests.py premium api

# Run single module
python run_all_tests.py llm
```

### Individual Test Runners

Each module has its own dedicated runner:

```bash
# LLM Services
python run_llm_tests.py

# Vector Stores
python run_vector_tests.py

# RAG & Search
python run_rag_tests.py

# Note Services
python run_note_tests.py

# Premium Agents
python run_premium_tests.py

# API Endpoints
python run_api_tests.py
```

### Help and Information

```bash
# Show help
python run_all_tests.py --help
python run_all_tests.py -h
```

## 🔧 Test Configuration

### Environment Setup

The test suite automatically loads environment variables from `.env` files. Make sure your API keys are properly configured:

```bash
# Check if environment variables are loaded
python -c "import os; print('GOOGLE_API_KEY:', '✅' if os.getenv('GOOGLE_API_KEY') else '❌')"
python -c "import os; print('PINECONE_API_KEY:', '✅' if os.getenv('PINECONE_API_KEY') else '❌')"
python -c "import os; print('OPENROUTER_API_KEY:', '✅' if os.getenv('OPENROUTER_API_KEY') else '❌')"
```

### Test Data

Each test module uses realistic test data:
- **Machine Learning Content**: Educational content for testing
- **Realistic Queries**: Production-like search and editing requests
- **Performance Metrics**: Response time measurements and throughput testing

## 📊 Test Results

### Output Format

Each test module provides:
- ✅ **Pass/Fail Status**: Clear success/failure indicators
- ⏱️ **Performance Metrics**: Response times and throughput
- 📝 **Detailed Results**: Specific test outcomes and data
- 🎯 **Summary**: Overall pass rate and performance

### Example Output

```
🚀 LLM Services Test Suite
============================================================

🔍 Testing Google Gemini Service
--------------------------------------------------
   🤖 Testing gemini-2.5-flash model...
   ✅ Gemini 2.5 Flash successful
   ⏱️  Response time: 1.23s
   📝 Response: 156 characters

📊 LLM Services Test Results
----------------------------------------
   ✅ PASS Google Gemini Service
   ✅ PASS OpenRouter Service
   ✅ PASS Mock LLM Service

🎯 Overall: 3/3 tests passed (100.0%)
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```bash
   # Check environment variables
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('GOOGLE_API_KEY'))"
   ```

2. **Import Errors**:
   ```bash
   # Ensure you're in the right directory
   cd elevate-ai-api
   python run_all_tests.py --help
   ```

3. **Network Issues**:
   - Check internet connectivity
   - Verify API endpoints are accessible
   - Check firewall settings

### Debug Mode

For detailed debugging, you can run individual test modules directly:

```bash
# Run with full error output
python -u tests/services/test_llm_services.py

# Check specific imports
python -c "from app.services.llm_service import create_llm_service; print('✅ Import successful')"
```

## 🎯 Test Coverage

### What's Tested

- ✅ **Real API Calls**: All external service integrations
- ✅ **Error Handling**: Graceful fallbacks and error recovery
- ✅ **Performance**: Response times and throughput
- ✅ **Data Validation**: Input/output validation
- ✅ **Integration**: Service-to-service communication
- ✅ **Edge Cases**: Boundary conditions and error scenarios

### What's NOT Tested

- ❌ **Unit Tests**: Individual function testing (use pytest for this)
- ❌ **Database Operations**: Direct database testing
- ❌ **Authentication**: User authentication flows
- ❌ **Rate Limiting**: API rate limit handling

## 🔄 Continuous Integration

### GitHub Actions

The test suite can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run Test Suite
  run: |
    cd elevate-ai-api
    python run_all_tests.py all
  env:
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
    PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

### Local Development

For development, run tests frequently:

```bash
# Quick test during development
python run_llm_tests.py

# Full test before commit
python run_all_tests.py all
```

## 📈 Performance Monitoring

### Metrics Tracked

- **Response Times**: API call latency
- **Throughput**: Requests per second
- **Success Rates**: Pass/fail ratios
- **Resource Usage**: Memory and CPU usage

### Benchmarking

Compare performance across different configurations:

```bash
# Test with different LLM providers
python run_llm_tests.py  # Google Gemini
# vs
python run_llm_tests.py  # OpenRouter
```

## 🤝 Contributing

### Adding New Tests

1. **Create Test Module**: Follow the existing pattern
2. **Add to Master Runner**: Update `run_all_tests.py`
3. **Create Individual Runner**: Add `run_<module>_tests.py`
4. **Update Documentation**: Add to this README

### Test Standards

- **Real API Calls**: Use actual external services
- **Performance Metrics**: Include timing measurements
- **Error Handling**: Test failure scenarios
- **Documentation**: Clear test descriptions

## 📚 Additional Resources

- **API Documentation**: Check individual service docs
- **Environment Setup**: See `.env.example` for required variables
- **Service Architecture**: Understand the system design
- **Troubleshooting**: Common issues and solutions

---

**Happy Testing! 🚀**

This test suite ensures your Elevate AI API is production-ready with real external service integrations.
