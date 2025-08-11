# Real LLM Performance E2E Tests

This directory contains comprehensive end-to-end tests that actually call real AI LLM services to measure actual performance, cost, and reliability of your AI API system.

## 🚀 What These Tests Do

These tests are designed to:

- **Make REAL API calls** to actual LLM services (OpenAI, Google AI, OpenRouter)
- **Measure actual performance** including response times, token usage, and costs
- **Test the complete workflow** from Core API → AI API → LLM services
- **Validate cost optimization** features in real-world scenarios
- **Test concurrent performance** under realistic load
- **Verify model cascading** with different complexity levels

## ⚠️ Important Notes

**These tests will incur real costs** as they make actual API calls to LLM services. Make sure you have:
- Sufficient API credits/budget
- Understanding of the costs involved
- Proper monitoring of your API usage

## 🛠️ Prerequisites

### 1. Environment Setup
```bash
# Make sure you're in the elevate-ai-api directory
cd elevate-ai-api

# Install dependencies
pip install -r requirements.txt
# or
poetry install
```

### 2. API Keys Configuration
Set up your environment variables in a `.env` file:

```bash
# Required for real LLM testing
GOOGLE_API_KEY=your_actual_google_api_key
OPENROUTER_API_KEY=your_actual_openrouter_api_key
OPENAI_API_KEY=your_actual_openai_api_key

# Optional but recommended
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

### 3. Service Availability
Ensure these services are running:
- **AI API Server**: `http://localhost:8000`
- **Core API Server**: `http://localhost:3000`

## 🧪 Running the Tests

### Option 1: Using the Runner Script (Recommended)
```bash
# Make the script executable
chmod +x run_real_llm_tests.py

# Run the tests
python run_real_llm_tests.py
```

### Option 2: Direct Execution
```bash
# Run the main test file directly
python test_e2e_real_llm_performance.py
```

### Option 3: Individual Test Execution
```python
# You can also import and run specific tests
from test_e2e_real_llm_performance import RealLLMPerformanceTester

async def run_specific_test():
    tester = RealLLMPerformanceTester()
    result = await tester.test_real_llm_chat_performance()
    print(f"Test result: {result}")
```

## 📊 Test Coverage

### 1. **Real LLM Chat Performance**
- Tests actual chat endpoints with real queries
- Measures response times and costs
- Multiple iterations for reliable metrics
- Tests different query complexities

### 2. **Model Cascading Performance**
- Tests model selection based on complexity
- Simple → Medium → Complex queries
- Validates model selection logic
- Performance comparison across models

### 3. **Concurrent LLM Performance**
- Tests system under concurrent load
- 5 simultaneous requests
- Measures throughput and reliability
- Identifies bottlenecks

### 4. **Cost Optimization Workflow**
- Tests cost optimization features
- Validates budget management
- Tests token optimization
- Cost tracking accuracy

### 5. **Core API Integration**
- Tests Core API → AI API workflow
- Primitive creation through AI
- Data flow validation
- End-to-end integration

## 📈 Performance Metrics

The tests collect comprehensive metrics:

- **Response Times**: Individual and average response times
- **Token Usage**: Input/output token counts
- **Costs**: Estimated costs per request
- **Success Rates**: Overall test success percentages
- **Model Usage**: Which LLM models were used
- **Concurrent Performance**: Throughput under load

## 🔧 Configuration

You can customize the test behavior by modifying `REAL_LLM_CONFIG`:

```python
REAL_LLM_CONFIG = {
    "timeout": 60.0,              # Request timeout
    "retry_attempts": 3,          # Retry attempts on failure
    "retry_delay": 2.0,           # Delay between retries
    "performance_threshold": 30.0, # Performance threshold
    "cost_threshold": 1.0,        # Cost threshold
    "concurrent_requests": 5,     # Concurrent test requests
    "test_iterations": 3,         # Iterations per test
}
```

## 📋 Expected Output

The tests provide detailed output including:

```
🚀 Starting Real LLM Performance E2E Tests
📅 Test started at: 2024-01-15 10:30:00

🔧 Validating Environment for Real LLM Testing...
✅ Environment validation successful

🧪 Testing Real LLM Chat Performance...
  🔄 Testing query 1/5: Explain the concept of machine learning...
    ✅ Iteration 1: gemini-1.5-flash - 2.34s - $0.0012
    ✅ Iteration 2: gemini-1.5-flash - 2.18s - $0.0012
    ✅ Iteration 3: gemini-1.5-flash - 2.45s - $0.0012
  📊 Results: 3/3 successful, avg time: 2.32s, total cost: $0.0036

📊 REAL LLM PERFORMANCE E2E TEST DETAILED SUMMARY
🧪 TEST RESULTS:
  ✅ PASSED Real LLM Chat Performance
  ✅ PASSED Model Cascading Performance
  ✅ PASSED Concurrent LLM Performance
  ✅ PASSED Cost Optimization Workflow
  ✅ PASSED Core API Integration

🎯 Overall: 5/5 tests passed (100.0%)
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   ❌ Environment validation error: 401 Unauthorized
   ```
   - Check your API keys are correctly set
   - Verify API keys have sufficient credits
   - Ensure keys are not expired

2. **Service Unavailable**
   ```
   ❌ AI API health check failed: 503
   ```
   - Check if AI API server is running
   - Verify port 8000 is accessible
   - Check server logs for errors

3. **Timeout Errors**
   ```
   ❌ Request timed out after 60.0s
   ```
   - Increase timeout in configuration
   - Check network connectivity
   - Verify LLM service availability

4. **Cost Threshold Exceeded**
   ```
   ⚠️  Total cost exceeds threshold!
   ```
   - Review your cost threshold setting
   - Check individual query costs
   - Optimize queries if needed

### Debug Mode

Enable debug logging by setting:
```bash
export DEBUG=1
export LOG_LEVEL=DEBUG
```

## 📝 Customization

### Adding New Tests

To add new test scenarios:

1. Create a new test method in `RealLLMPerformanceTester`
2. Add it to the `tests` list in `run_all_tests()`
3. Follow the existing pattern for metrics collection

### Custom Test Data

Modify test queries and parameters in the test methods:
```python
test_queries = [
    "Your custom query here",
    "Another custom query",
    # Add more as needed
]
```

### Custom Metrics

Extend the `PerformanceMetrics` class to capture additional data:
```python
@dataclass
class PerformanceMetrics:
    response_time: float
    token_count: int
    cost: float
    model_used: str
    success: bool
    custom_metric: Optional[str] = None  # Add your custom metric
```

## 🔒 Security Considerations

- **Never commit API keys** to version control
- **Use environment variables** for sensitive configuration
- **Monitor API usage** to prevent unexpected costs
- **Set appropriate rate limits** in your LLM service accounts
- **Review test queries** to ensure they don't contain sensitive information

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review server logs for detailed error information
3. Verify all prerequisites are met
4. Test with a simple query first
5. Check your API service status pages

## 🎯 Next Steps

After running these tests:

1. **Analyze the results** to identify performance bottlenecks
2. **Optimize your queries** based on cost and performance data
3. **Tune your model selection** logic based on complexity analysis
4. **Monitor costs** and set up alerts for budget management
5. **Iterate and improve** based on test findings

---

**Happy Testing! 🚀**

Remember: These tests provide real-world performance data that can help you optimize your AI API system for production use.
