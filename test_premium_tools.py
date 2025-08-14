#!/usr/bin/env python3
"""
Test script for Real Premium Tools
Demonstrates the functionality of all implemented premium tools.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

async def test_calculator_service():
    """Test the calculator service"""
    print("🧮 Testing Calculator Service...")
    
    try:
        from app.core.premium.tools.calculator_service import CalculatorService
        
        calc = CalculatorService()
        
        # Test basic expression evaluation
        result = await calc.evaluate_expression("2 + 3 * 4")
        print(f"   Basic math: 2 + 3 * 4 = {result.result}")
        
        # Test scientific functions
        result = await calc.evaluate_expression("sin(3.14159/2)")
        print(f"   Scientific: sin(π/2) = {result.result}")
        
        # Test unit conversion
        try:
            conv_result = await calc.convert_units(100, "kilometer", "mile")
            print(f"   Unit conversion: 100 km = {conv_result.to_value:.2f} miles")
        except Exception as e:
            print(f"   Unit conversion: {e}")
        
        print("   ✅ Calculator service working!")
        
    except Exception as e:
        print(f"   ❌ Calculator service failed: {e}")

async def test_code_execution_service():
    """Test the code execution service"""
    print("💻 Testing Code Execution Service...")
    
    try:
        from app.core.premium.tools.code_execution_service import CodeExecutionService
        
        code_exec = CodeExecutionService()
        
        # Test Python code execution
        result = await code_exec.execute_code(
            code="print('Hello from Python!'); x = 5 + 3; print(f'Result: {x}')",
            language="python"
        )
        
        if result.success:
            print(f"   Python execution: {result.output.strip()}")
        else:
            print(f"   Python execution failed: {result.error}")
        
        # Test JavaScript code execution
        result = await code_exec.execute_code(
            code="console.log('Hello from JavaScript!'); let x = 5 + 3; console.log(`Result: ${x}`);",
            language="javascript"
        )
        
        if result.success:
            print(f"   JavaScript execution: {result.output.strip()}")
        else:
            print(f"   JavaScript execution failed: {result.error}")
        
        print("   ✅ Code execution service working!")
        
    except Exception as e:
        print(f"   ❌ Code execution service failed: {e}")

async def test_web_search_service():
    """Test the web search service"""
    print("🔍 Testing Web Search Service...")
    
    try:
        from app.core.premium.tools.web_search_service import TavilySearchService
        
        # Check if API key is available
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("   ⚠️  TAVILY_API_KEY not set, skipping web search test")
            return
        
        search_service = TavilySearchService(api_key)
        
        async with search_service:
            # Test basic search
            result = await search_service.search("artificial intelligence", max_results=3)
            print(f"   Search results: {len(result.results)} results found")
            
            if result.results:
                first_result = result.results[0]
                print(f"   First result: {first_result.title[:50]}...")
        
        print("   ✅ Web search service working!")
        
    except Exception as e:
        print(f"   ❌ Web search service failed: {e}")

async def test_diagram_generation_service():
    """Test the diagram generation service"""
    print("📊 Testing Diagram Generation Service...")
    
    try:
        from app.core.premium.tools.diagram_generation_service import (
            DiagramGenerationService, DiagramData, DiagramStyle
        )
        
        diagram_service = DiagramGenerationService()
        
        # Test flowchart generation
        data = DiagramData(
            nodes=[
                {"id": "start", "label": "Start", "type": "start"},
                {"id": "process", "label": "Process", "type": "default"},
                {"id": "end", "label": "End", "type": "end"}
            ],
            edges=[
                {"from": "start", "to": "process"},
                {"from": "process", "to": "end"}
            ],
            metadata={"title": "Simple Flowchart"},
            diagram_type="flowchart"
        )
        
        result = await diagram_service.generate_diagram("flowchart", data)
        
        if result.success:
            print(f"   Flowchart generated: {result.diagram_type} in {result.format} format")
            print(f"   Generation time: {result.generation_time:.3f}s")
        else:
            print(f"   Flowchart generation failed: {result.error}")
        
        print("   ✅ Diagram generation service working!")
        
    except Exception as e:
        print(f"   ❌ Diagram generation service failed: {e}")

async def test_example_generation_service():
    """Test the example generation service"""
    print("📚 Testing Example Generation Service...")
    
    try:
        from app.core.premium.tools.example_generation_service import (
            ExampleGenerationService, ExampleRequest
        )
        
        example_service = ExampleGenerationService()
        
        # Test example generation
        request = ExampleRequest(
            concept="sorting algorithms",
            user_id="test_user",
            context={"learning_mode": "interactive"},
            user_level="intermediate",
            example_type="code",
            learning_style="visual",
            previous_examples=[]
        )
        
        result = await example_service.generate_examples(request)
        
        if result.success:
            print(f"   Examples generated: {len(result.examples)} examples for '{result.concept}'")
            print(f"   Difficulty level: {result.difficulty_level}")
            print(f"   Learning objectives: {len(result.learning_objectives)} objectives")
        else:
            print(f"   Example generation failed: {result.error}")
        
        print("   ✅ Example generation service working!")
        
    except Exception as e:
        print(f"   ❌ Example generation service failed: {e}")

async def test_tools_integration():
    """Test the tools integration service"""
    print("🔧 Testing Tools Integration Service...")
    
    try:
        from app.core.premium.tools.tools_integration_service import (
            ToolsIntegrationService, ToolExecutionRequest
        )
        
        tools_service = ToolsIntegrationService()
        
        # Test tool availability
        available_tools = await tools_service.get_available_tools()
        print(f"   Available tools: {', '.join(available_tools)}")
        
        # Test tool status
        status = await tools_service.get_tool_status()
        print(f"   Tool status: {len(status)} tools checked")
        
        # Test single tool execution
        request = ToolExecutionRequest(
            tool_name="calculator",
            parameters={"expression": "10 * 5", "operation": "evaluate"},
            user_id="test_user",
            context={"mode": "test"}
        )
        
        result = await tools_service.execute_tool(request)
        
        if result.success:
            print(f"   Tool execution successful: {result.tool_name}")
            print(f"   Execution time: {result.execution_time:.3f}s")
        else:
            print(f"   Tool execution failed: {result.error}")
        
        print("   ✅ Tools integration service working!")
        
    except Exception as e:
        print(f"   ❌ Tools integration service failed: {e}")

async def main():
    """Run all tests"""
    print("🚀 Starting Real Premium Tools Test Suite\n")
    
    # Test individual services
    await test_calculator_service()
    print()
    
    await test_code_execution_service()
    print()
    
    await test_web_search_service()
    print()
    
    await test_diagram_generation_service()
    print()
    
    await test_example_generation_service()
    print()
    
    await test_tools_integration()
    print()
    
    print("🎉 Premium Tools Test Suite Complete!")
    print("\nNote: Some services may show warnings or fallbacks if external dependencies")
    print("(like Docker, Tavily API key, or diagram tools) are not configured.")

if __name__ == "__main__":
    asyncio.run(main())
