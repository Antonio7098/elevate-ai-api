"""
Real Code Execution Service for Premium Tools
Provides secure, containerized code execution for multiple programming languages.
"""

import asyncio
import json
import tempfile
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import docker
from docker.errors import DockerException
import re

@dataclass
class CodeExecutionResult:
    """Result of code execution"""
    code: str
    language: str
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: Optional[str] = None
    exit_code: int = 0
    success: bool = True

@dataclass
class ExecutionLimits:
    """Resource limits for code execution"""
    cpu_time: int = 30  # seconds
    memory: str = "512m"  # memory limit
    network_access: bool = False
    file_access: bool = False

class CodeExecutionService:
    """Secure code execution service using Docker containers"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except DockerException:
            self.docker_available = False
            print("Warning: Docker not available, falling back to local execution")
        
        # Language configurations
        self.language_configs = {
            "python": {
                "image": "python:3.11-slim",
                "command": ["python", "-c"],
                "file_extension": ".py"
            },
            "javascript": {
                "image": "node:18-slim",
                "command": ["node", "-e"],
                "file_extension": ".js"
            },
            "typescript": {
                "image": "node:18-slim",
                "command": ["npx", "ts-node", "-e"],
                "file_extension": ".ts"
            },
            "sql": {
                "image": "postgres:15",
                "command": ["psql", "-c"],
                "file_extension": ".sql"
            }
        }
    
    async def execute_code(self, code: str, language: str, inputs: Optional[Dict[str, Any]] = None, 
                          limits: Optional[ExecutionLimits] = None) -> CodeExecutionResult:
        """
        Execute code in a secure container
        
        Args:
            code: Source code to execute
            language: Programming language
            inputs: Optional input data
            limits: Resource limits
            
        Returns:
            CodeExecutionResult with execution results
        """
        if not limits:
            limits = ExecutionLimits()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate language support
            if language not in self.language_configs:
                return CodeExecutionResult(
                    code=code,
                    language=language,
                    output="",
                    error=f"Unsupported language: {language}",
                    success=False
                )
            
            # Validate code safety
            if not self._validate_code_safety(code, language):
                return CodeExecutionResult(
                    code=code,
                    language=language,
                    output="",
                    error="Code contains potentially dangerous operations",
                    success=False
                )
            
            # Execute code
            if self.docker_available:
                result = await self._execute_in_container(code, language, inputs, limits)
            else:
                result = await self._execute_locally(code, language, inputs, limits)
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return CodeExecutionResult(
                code=code,
                language=language,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=execution_time,
                success=False
            )
    
    async def _execute_in_container(self, code: str, language: str, inputs: Optional[Dict[str, Any]], 
                                   limits: ExecutionLimits) -> CodeExecutionResult:
        """Execute code in Docker container"""
        config = self.language_configs[language]
        
        # Prepare code with inputs
        prepared_code = self._prepare_code_with_inputs(code, language, inputs)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=config["file_extension"], delete=False) as f:
            f.write(prepared_code)
            temp_file = f.name
        
        try:
            # Run container
            container = self.docker_client.containers.run(
                config["image"],
                command=config["command"] + [prepared_code],
                detach=True,
                mem_limit=limits.memory,
                network_disabled=not limits.network_access,
                volumes={temp_file: {'bind': f'/tmp/code{config["file_extension"]}', 'mode': 'ro'}} if limits.file_access else {},
                remove=True
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=limits.cpu_time)
                logs = container.logs().decode('utf-8')
                
                return CodeExecutionResult(
                    code=code,
                    language=language,
                    output=logs,
                    exit_code=result['StatusCode'],
                    success=result['StatusCode'] == 0
                )
                
            except Exception as e:
                container.kill()
                return CodeExecutionResult(
                    code=code,
                    language=language,
                    output="",
                    error=f"Execution timeout or error: {str(e)}",
                    success=False
                )
                
        finally:
            # Cleanup
            Path(temp_file).unlink(missing_ok=True)
    
    async def _execute_locally(self, code: str, language: str, inputs: Optional[Dict[str, Any]], 
                              limits: ExecutionLimits) -> CodeExecutionResult:
        """Execute code locally (fallback when Docker unavailable)"""
        config = self.language_configs[language]
        
        # Prepare code with inputs
        prepared_code = self._prepare_code_with_inputs(code, language, inputs)
        
        try:
            # Execute with subprocess
            process = await asyncio.create_subprocess_exec(
                *config["command"],
                prepared_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=limits.cpu_time
            )
            
            output = stdout.decode('utf-8')
            error = stderr.decode('utf-8') if stderr else None
            
            return CodeExecutionResult(
                code=code,
                language=language,
                output=output,
                error=error,
                exit_code=process.returncode,
                success=process.returncode == 0
            )
            
        except asyncio.TimeoutError:
            return CodeExecutionResult(
                code=code,
                language=language,
                output="",
                error="Execution timeout",
                success=False
            )
    
    def _prepare_code_with_inputs(self, code: str, language: str, inputs: Optional[Dict[str, Any]]) -> str:
        """Prepare code with input data"""
        if not inputs:
            return code
        
        if language == "python":
            return self._prepare_python_inputs(code, inputs)
        elif language == "javascript":
            return self._prepare_javascript_inputs(code, inputs)
        elif language == "typescript":
            return self._prepare_typescript_inputs(code, inputs)
        elif language == "sql":
            return self._prepare_sql_inputs(code, inputs)
        
        return code
    
    def _prepare_python_inputs(self, code: str, inputs: Dict[str, Any]) -> str:
        """Prepare Python code with inputs"""
        input_code = "\n".join([f"{k} = {repr(v)}" for k, v in inputs.items()])
        return f"{input_code}\n\n{code}"
    
    def _prepare_javascript_inputs(self, code: str, inputs: Dict[str, Any]) -> str:
        """Prepare JavaScript code with inputs"""
        input_code = "\n".join([f"const {k} = {json.dumps(v)};" for k, v in inputs.items()])
        return f"{input_code}\n\n{code}"
    
    def _prepare_typescript_inputs(self, code: str, inputs: Dict[str, Any]) -> str:
        """Prepare TypeScript code with inputs"""
        input_code = "\n".join([f"const {k}: any = {json.dumps(v)};" for k, v in inputs.items()])
        return f"{input_code}\n\n{code}"
    
    def _prepare_sql_inputs(self, code: str, inputs: Dict[str, Any]) -> str:
        """Prepare SQL code with inputs"""
        # For SQL, we'll use parameterized queries
        return code
    
    def _validate_code_safety(self, code: str, language: str) -> bool:
        """Validate code for safety"""
        dangerous_patterns = [
            r'import\s+os', r'import\s+subprocess', r'import\s+sys',
            r'__import__\s*\(', r'eval\s*\(', r'exec\s*\(',
            r'open\s*\(', r'file\s*\(', r'input\s*\(',
            r'raw_input\s*\(', r'compile\s*\(', r'execfile\s*\(',
            r'require\s*\(', r'process\.', r'child_process',
            r'fs\.', r'path\.', r'url\.', r'http\.', r'https\.'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        return True
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return list(self.language_configs.keys())
    
    async def get_language_info(self, language: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific language"""
        if language in self.language_configs:
            return self.language_configs[language]
        return None
