"""
Tools Integration Service for Premium Tools
Coordinates all premium tools and provides unified interface for the context assembly agent.
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from .calculator_service import CalculatorService, CalculationResult
from .code_execution_service import CodeExecutionService, CodeExecutionResult
from .web_search_service import TavilySearchService, SearchResponse, RealtimeData
from .diagram_generation_service import DiagramGenerationService, DiagramResult, DiagramData
from .example_generation_service import ExampleGenerationService, ExampleResult, ExampleRequest

@dataclass
class ToolExecutionRequest:
    """Request for tool execution"""
    tool_name: str
    parameters: Dict[str, Any]
    user_id: str
    context: Dict[str, Any]

@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class ToolsIntegrationService:
    """Service that integrates all premium tools"""
    
    def __init__(self):
        # Initialize all tool services
        self.calculator_service = CalculatorService()
        self.code_execution_service = CodeExecutionService()
        self.web_search_service = TavilySearchService()
        self.diagram_generation_service = DiagramGenerationService()
        self.example_generation_service = ExampleGenerationService()
        
        # Tool registry
        self.tools = {
            "calculator": self.calculator_service,
            "code_executor": self.code_execution_service,
            "web_search": self.web_search_service,
            "diagram_generator": self.diagram_generation_service,
            "example_generator": self.example_generation_service
        }
    
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """
        Execute a specific tool based on the request
        
        Args:
            request: Tool execution request
            
        Returns:
            ToolExecutionResult with execution results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if request.tool_name not in self.tools:
                return ToolExecutionResult(
                    tool_name=request.tool_name,
                    success=False,
                    result=None,
                    execution_time=0.0,
                    error=f"Unknown tool: {request.tool_name}"
                )
            
            # Execute the appropriate tool
            if request.tool_name == "calculator":
                result = await self._execute_calculator(request.parameters)
            elif request.tool_name == "code_executor":
                result = await self._execute_code_executor(request.parameters)
            elif request.tool_name == "web_search":
                result = await self._execute_web_search(request.parameters)
            elif request.tool_name == "diagram_generator":
                result = await self._execute_diagram_generator(request.parameters)
            elif request.tool_name == "example_generator":
                result = await self._execute_example_generator(request.parameters, request.user_id, request.context)
            else:
                result = None
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ToolExecutionResult(
                tool_name=request.tool_name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"tool_version": "real", "execution_mode": "production"}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolExecutionResult(
                tool_name=request.tool_name,
                success=False,
                result=None,
                execution_time=execution_time,
                error=f"Tool execution failed: {str(e)}"
            )
    
    async def execute_multiple_tools(self, requests: List[ToolExecutionRequest]) -> List[ToolExecutionResult]:
        """
        Execute multiple tools concurrently
        
        Args:
            requests: List of tool execution requests
            
        Returns:
            List of ToolExecutionResult
        """
        tasks = [self.execute_tool(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolExecutionResult(
                    tool_name=requests[i].tool_name,
                    success=False,
                    result=None,
                    execution_time=0.0,
                    error=f"Tool execution failed: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_calculator(self, parameters: Dict[str, Any]) -> CalculationResult:
        """Execute calculator tool"""
        operation = parameters.get("operation", "evaluate")
        expression = parameters.get("expression", "")
        variables = parameters.get("variables", {})
        
        if operation == "evaluate":
            return await self.calculator_service.evaluate_expression(expression, variables)
        elif operation == "solve":
            variable = parameters.get("variable", "x")
            return await self.calculator_service.solve_equation(expression, variable)
        elif operation == "derivative":
            variable = parameters.get("variable", "x")
            order = parameters.get("order", 1)
            return await self.calculator_service.calculate_derivative(expression, variable, order)
        elif operation == "integral":
            variable = parameters.get("variable", "x")
            limits = parameters.get("limits")
            return await self.calculator_service.calculate_integral(expression, variable, limits)
        elif operation == "convert_units":
            value = parameters.get("value", 0.0)
            from_unit = parameters.get("from_unit", "")
            to_unit = parameters.get("to_unit", "")
            return await self.calculator_service.convert_units(value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown calculator operation: {operation}")
    
    async def _execute_code_executor(self, parameters: Dict[str, Any]) -> CodeExecutionResult:
        """Execute code execution tool"""
        code = parameters.get("code", "")
        language = parameters.get("language", "python")
        inputs = parameters.get("inputs", {})
        limits = parameters.get("limits")
        
        return await self.code_execution_service.execute_code(code, language, inputs, limits)
    
    async def _execute_web_search(self, parameters: Dict[str, Any]) -> SearchResponse:
        """Execute web search tool"""
        query = parameters.get("query", "")
        search_type = parameters.get("search_type", "basic")
        max_results = parameters.get("max_results", 10)
        include_answer = parameters.get("include_answer", True)
        
        return await self.web_search_service.search(query, search_type, max_results, include_answer)
    
    async def _execute_diagram_generator(self, parameters: Dict[str, Any]) -> DiagramResult:
        """Execute diagram generation tool"""
        diagram_type = parameters.get("diagram_type", "flowchart")
        data = parameters.get("data", {})
        style = parameters.get("style")
        output_format = parameters.get("output_format", "svg")
        
        # Convert data to DiagramData structure
        diagram_data = DiagramData(
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            metadata=data.get("metadata", {}),
            diagram_type=diagram_type
        )
        
        return await self.diagram_generation_service.generate_diagram(
            diagram_type, diagram_data, style, output_format
        )
    
    async def _execute_example_generator(self, parameters: Dict[str, Any], 
                                        user_id: str, context: Dict[str, Any]) -> ExampleResult:
        """Execute example generation tool"""
        concept = parameters.get("concept", "")
        user_level = parameters.get("user_level", "intermediate")
        example_type = parameters.get("example_type", "code")
        learning_style = parameters.get("learning_style", "visual")
        previous_examples = parameters.get("previous_examples", [])
        
        request = ExampleRequest(
            concept=concept,
            user_id=user_id,
            context=context,
            user_level=user_level,
            example_type=example_type,
            learning_style=learning_style,
            previous_examples=previous_examples
        )
        
        return await self.example_generation_service.generate_examples(request)
    
    async def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        if tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        
        if tool_name == "calculator":
            return {
                "name": "Calculator",
                "description": "Mathematical expression evaluation and unit conversion",
                "supported_operations": ["evaluate", "solve", "derivative", "integral", "convert_units"],
                "capabilities": ["Basic math", "Scientific functions", "Unit conversion", "Equation solving"]
            }
        elif tool_name == "code_executor":
            return {
                "name": "Code Executor",
                "description": "Secure code execution in isolated containers",
                "supported_languages": await tool.get_supported_languages(),
                "capabilities": ["Safe execution", "Multiple languages", "Resource limits", "Input/output handling"]
            }
        elif tool_name == "web_search":
            return {
                "name": "Web Search",
                "description": "Real-time information retrieval using Tavily API",
                "supported_queries": ["General search", "News", "Weather", "Stocks", "Sports"],
                "capabilities": ["Live data", "Structured results", "Rate limiting", "Caching"]
            }
        elif tool_name == "diagram_generator":
            return {
                "name": "Diagram Generator",
                "description": "Visual diagram creation for various types",
                "supported_types": await tool.get_supported_diagram_types(),
                "capabilities": ["Multiple formats", "Custom styling", "Real-time updates", "Export options"]
            }
        elif tool_name == "example_generator":
            return {
                "name": "Example Generator",
                "description": "AI-powered, context-aware example generation",
                "supported_types": await tool.get_supported_example_types(),
                "capabilities": ["Personalization", "Learning style adaptation", "Difficulty scaling", "Feedback integration"]
            }
        
        return None
    
    async def get_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tools"""
        status = {}
        
        for tool_name, tool in self.tools.items():
            try:
                if tool_name == "calculator":
                    status[tool_name] = {
                        "status": "available",
                        "supported_functions": await tool.get_supported_functions(),
                        "supported_units": await tool.get_supported_units()
                    }
                elif tool_name == "code_executor":
                    status[tool_name] = {
                        "status": "available" if tool.docker_available else "limited",
                        "supported_languages": await tool.get_supported_languages(),
                        "docker_available": tool.docker_available
                    }
                elif tool_name == "web_search":
                    status[tool_name] = {
                        "status": "available",
                        "analytics": await tool.get_search_analytics()
                    }
                elif tool_name == "diagram_generator":
                    status[tool_name] = {
                        "status": "available",
                        "available_tools": await tool.get_available_tools()
                    }
                elif tool_name == "example_generation_service":
                    status[tool_name] = {
                        "status": "available",
                        "supported_types": await tool.get_supported_example_types(),
                        "learning_styles": await tool.get_learning_style_adaptations()
                    }
            except Exception as e:
                status[tool_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return status
    
    async def validate_tool_request(self, request: ToolExecutionRequest) -> Dict[str, Any]:
        """Validate a tool execution request"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if tool exists
        if request.tool_name not in self.tools:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unknown tool: {request.tool_name}")
            return validation_result
        
        # Tool-specific validation
        if request.tool_name == "calculator":
            if "expression" not in request.parameters:
                validation_result["valid"] = False
                validation_result["errors"].append("Calculator requires 'expression' parameter")
        
        elif request.tool_name == "code_executor":
            if "code" not in request.parameters:
                validation_result["valid"] = False
                validation_result["errors"].append("Code executor requires 'code' parameter")
            if "language" not in request.parameters:
                validation_result["warnings"].append("Language not specified, defaulting to Python")
        
        elif request.tool_name == "web_search":
            if "query" not in request.parameters:
                validation_result["valid"] = False
                validation_result["errors"].append("Web search requires 'query' parameter")
        
        elif request.tool_name == "diagram_generator":
            if "diagram_type" not in request.parameters:
                validation_result["valid"] = False
                validation_result["errors"].append("Diagram generator requires 'diagram_type' parameter")
            if "data" not in request.parameters:
                validation_result["valid"] = False
                validation_result["errors"].append("Diagram generator requires 'data' parameter")
        
        elif request.tool_name == "example_generator":
            if "concept" not in request.parameters:
                validation_result["valid"] = False
                validation_result["errors"].append("Example generator requires 'concept' parameter")
        
        return validation_result
