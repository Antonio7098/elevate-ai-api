# 🚀 E2E Testing Framework for Elevate AI API

## Overview

This comprehensive E2E testing framework tests the **real performance** of your Elevate AI API's premium chat features using **actual LLM calls** to Google Gemini and OpenRouter services. The framework measures real-world performance metrics including response times, token usage, costs, and model selection.

## 🎯 **Key Features**

- **Real LLM Calls**: Tests actual API calls to Google Gemini 1.5 Flash and OpenRouter GLM 4.5 Air
- **Model Cascading**: Tests intelligent model selection and fallback mechanisms
- **Performance Metrics**: Measures real response times, token counts, and costs
- **Concurrent Testing**: Tests system behavior under load
- **Cost Optimization**: Tests cost optimization features
- **Core API Integration**: Tests integration with the Core API

## 🏗️ **Architecture**

```
E2E Test Script → Premium Chat Endpoints → Model Cascader → LLM Services
                                                      ↓
                                              Google Gemini 1.5 Flash (Primary)
                                                      ↓
                                              OpenRouter GLM 4.5 Air (Fallback)
```

## 🚀 **Quick Start**

### 1. **Set Environment Variables**

Create a `.env` file based on `env_config_example.txt`:

```bash
# Required for real LLM calls
GOOGLE_API_KEY=your_actual_google_api_key
OPENROUTER_API_KEY=your_actual_openrouter_api_key

# Optional
CORE_API_BASE_URL=http://localhost:3000
```

### 2. **Start Services**

```bash
# Terminal 1: Start AI API
cd elevate-ai-api
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Core API
cd elevate-core-api
npm run dev  # or your start command
```

### 3. **Run Tests**

```bash
# Quick test
make test-quick

# Full test suite
make test-all

# Custom configuration
python3 run_e2e_tests.py --iterations 3 --concurrent-requests 5
```

## 🧪 **Test Types**

### 1. **Real LLM Chat Performance**
- Tests basic chat functionality with real Gemini calls
- Measures response time, token usage, and cost
- **Uses**: Google Gemini 1.5 Flash (primary)

### 2. **Model Cascading Performance**
- Tests complexity-based model selection
- Measures escalation behavior and confidence scoring
- **Uses**: Gemini 1.5 Flash → Gemini 1.5 Pro → OpenRouter GLM 4.5 Air

### 3. **Concurrent LLM Performance**
- Tests system under load with multiple simultaneous requests
- Measures throughput and resource utilization
- **Uses**: Multiple concurrent calls to primary and fallback models

### 4. **Cost Optimization Workflow**
- Tests cost optimization features
- Measures cost savings and quality preservation
- **Uses**: Early exit optimization and model selection

### 5. **Core API Integration**
- Tests integration with Core API for knowledge primitives
- Measures end-to-end workflow performance
- **Uses**: Combined AI API + Core API calls

## ⚙️ **Configuration**

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Google AI API key for Gemini models |
| `OPENROUTER_API_KEY` | ✅ | OpenRouter API key for GLM 4.5 Air |
| `CORE_API_BASE_URL` | ❌ | Core API base URL (default: localhost:3000) |

### Test Configuration

Edit `e2e_test_config.json` to customize:

```json
{
  "api_endpoints": {
    "ai_api_base": "http://localhost:8000",
    "core_api_base": "http://localhost:3000"
  },
  "test_queries": {
    "simple": ["What is photosynthesis?", "Define gravity"],
    "medium": ["Explain the water cycle", "How do plants grow?"],
    "complex": ["Analyze the impact of climate change on ecosystems"]
  },
  "performance_targets": {
    "max_response_time_ms": 10000,
    "max_cost_per_request": 0.10,
    "min_confidence_score": 0.7
  }
}
```

## 📊 **Result Interpretation**

### Performance Metrics

| Metric | Description | Good | Warning | Critical |
|--------|-------------|------|---------|----------|
| **Response Time** | Time to complete request | < 5s | 5-10s | > 10s |
| **Token Count** | Input + output tokens | < 1000 | 1000-2000 | > 2000 |
| **Cost** | Estimated cost per request | < $0.01 | $0.01-0.05 | > $0.05 |
| **Confidence** | Model confidence score | > 0.8 | 0.6-0.8 | < 0.6 |

### Model Usage Analysis

- **Primary Model**: Google Gemini 1.5 Flash (fast, cost-effective)
- **Secondary Model**: Google Gemini 1.5 Pro (balanced performance)
- **Fallback Model**: OpenRouter GLM 4.5 Air (reliability)

### Cost Analysis

- **Simple Queries**: Should use Gemini 1.5 Flash (cheapest)
- **Complex Queries**: May escalate to Gemini 1.5 Pro
- **Fallback Scenarios**: Use OpenRouter GLM 4.5 Air (free tier)

## 🔧 **Advanced Usage**

### Custom Test Scenarios

```python
# Test specific model combinations
tester = RealLLMPerformanceTester()
await tester.test_model_cascading_performance(
    queries=["complex_question_1", "complex_question_2"],
    expected_models=["gemini-1.5-flash", "gemini-1.5-pro"]
)

# Test cost optimization
await tester.test_cost_optimization_workflow(
    cost_threshold=0.05,
    quality_threshold=0.8
)
```

### Load Testing

```bash
# High concurrency test
python3 run_e2e_tests.py --concurrent-requests 20 --iterations 10

# Stress test
python3 run_e2e_tests.py --concurrent-requests 50 --iterations 5 --timeout 300
```

### Continuous Monitoring

```bash
# Run tests every 5 minutes
watch -n 300 'python3 run_e2e_tests.py --save-results --results-file "monitoring_$(date +%Y%m%d_%H%M%S).json"'
```

## 🚨 **Troubleshooting**

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **API Key Errors** | Missing or invalid API keys | Check `.env` file and API key validity |
| **Timeout Errors** | LLM services slow to respond | Increase timeout in config or check service status |
| **Model Not Found** | Incorrect model names | Verify model names in `gemini_service.py` |
| **Fallback Failures** | OpenRouter service down | Check OpenRouter status and API key |

### Debug Mode

```bash
# Enable verbose logging
python3 run_e2e_tests.py --verbose --debug

# Check service health
make check-health
```

### Service Health Checks

```bash
# AI API health
curl http://localhost:8000/api/v1/premium/health

# Core API health
curl http://localhost:3000/health
```

## 📈 **Performance Monitoring**

### Real-Time Metrics

The framework provides real-time monitoring of:

- **Response Latency**: Per-request and aggregate response times
- **Cost Tracking**: Real-time cost estimation and optimization
- **Model Performance**: Success rates and confidence scores per model
- **Resource Utilization**: Memory and CPU usage during testing

### Historical Analysis

```bash
# Compare results over time
python3 analyze_results.py results_*.json

# Generate performance report
python3 generate_report.py --time-range "24h" --format "html"
```

## 🔒 **Security Considerations**

- **API Keys**: Never commit API keys to version control
- **Rate Limiting**: Respect API rate limits to avoid service disruption
- **Cost Monitoring**: Set up alerts for unexpected cost spikes
- **Data Privacy**: Test data should not contain sensitive information

## 🤝 **Contributing**

### Adding New Tests

1. **Create Test Method**: Add new test method to `RealLLMPerformanceTester`
2. **Update Configuration**: Add test parameters to `e2e_test_config.json`
3. **Document**: Update this README with test description
4. **Validate**: Ensure tests work with real LLM services

### Test Standards

- **Real LLM Calls**: All tests must use actual API calls, no mocks
- **Error Handling**: Proper error handling and fallback mechanisms
- **Metrics**: Comprehensive performance and cost metrics
- **Documentation**: Clear test purpose and expected outcomes

## 📚 **Additional Resources**

- **API Documentation**: Check the API docs for endpoint details
- **Model Information**: [Google AI Models](https://ai.google.dev/models), [OpenRouter Models](https://openrouter.ai/models)
- **Performance Benchmarks**: Compare results with industry standards
- **Cost Optimization**: Learn about cost-effective model usage strategies

---

## 🎯 **Summary**

Your premium chat features now use **real LLM calls** with:

✅ **Primary**: Google Gemini 1.5 Flash (fast, cost-effective)  
✅ **Secondary**: Google Gemini 1.5 Pro (balanced performance)  
✅ **Fallback**: OpenRouter GLM 4.5 Air (reliability)  
✅ **No Mocks**: All responses come from real AI models  
✅ **Smart Cascading**: Intelligent model selection based on complexity  
✅ **Cost Optimization**: Automatic cost optimization and early exit  

The E2E testing framework validates this real-world performance and ensures your system works reliably with actual LLM services! 🚀
