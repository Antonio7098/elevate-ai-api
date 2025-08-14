"""
Real Calculator Service for Premium Tools
Provides mathematical expression evaluation, scientific functions, and unit conversion.
"""

import re
import math
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import sympy as sp
from sympy import symbols, solve, diff, integrate, limit, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import pint
from pint import UnitRegistry

@dataclass
class CalculationResult:
    """Result of a mathematical calculation"""
    expression: str
    result: Union[str, float, complex]
    result_type: str
    units: Optional[str] = None
    steps: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 1.0

@dataclass
class UnitConversionResult:
    """Result of a unit conversion"""
    from_value: float
    from_unit: str
    to_value: float
    to_unit: str
    conversion_factor: float

class CalculatorService:
    """Real calculator service with mathematical expression evaluation and unit conversion"""
    
    def __init__(self):
        # Initialize sympy for mathematical operations
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
        # Initialize pint for unit conversion
        self.ureg = UnitRegistry()
        
        # Common mathematical constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'inf': float('inf'),
            'infinity': float('inf'),
            'nan': float('nan')
        }
        
        # Supported mathematical functions
        self.functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'log': math.log, 'log10': math.log10, 'ln': math.log,
            'sqrt': math.sqrt, 'exp': math.exp, 'abs': abs,
            'floor': math.floor, 'ceil': math.ceil, 'round': round,
            'factorial': math.factorial, 'gcd': math.gcd, 'lcm': lambda a, b: abs(a * b) // math.gcd(a, b)
        }
    
    async def evaluate_expression(self, expression: str, variables: Optional[Dict[str, float]] = None) -> CalculationResult:
        """
        Evaluate a mathematical expression safely
        
        Args:
            expression: Mathematical expression as string
            variables: Optional dictionary of variable values
            
        Returns:
            CalculationResult with the evaluation result
        """
        try:
            # Clean and validate expression
            cleaned_expression = self._clean_expression(expression)
            
            if not self._validate_expression(cleaned_expression):
                return CalculationResult(
                    expression=expression,
                    result="",
                    result_type="error",
                    error="Invalid mathematical expression"
                )
            
            # Parse expression with sympy
            parsed_expr = parse_expr(cleaned_expression, transformations=self.transformations)
            
            # Substitute variables if provided
            if variables:
                for var_name, var_value in variables.items():
                    parsed_expr = parsed_expr.subs(symbols(var_name), var_value)
            
            # Evaluate the expression
            result = parsed_expr.evalf()
            
            # Determine result type
            result_type = self._determine_result_type(result)
            
            # Generate solution steps for complex expressions
            steps = self._generate_solution_steps(cleaned_expression, result)
            
            return CalculationResult(
                expression=expression,
                result=str(result),
                result_type=result_type,
                steps=steps,
                confidence=1.0
            )
            
        except Exception as e:
            return CalculationResult(
                expression=expression,
                result="",
                result_type="error",
                error=f"Calculation error: {str(e)}"
            )
    
    async def solve_equation(self, equation: str, variable: str = "x") -> CalculationResult:
        """
        Solve an equation for a given variable
        
        Args:
            equation: Equation as string (e.g., "2*x + 3 = 7")
            variable: Variable to solve for
            
        Returns:
            CalculationResult with the solution
        """
        try:
            # Parse equation
            if "=" in equation:
                left_side, right_side = equation.split("=", 1)
                parsed_equation = parse_expr(f"({left_side}) - ({right_side})", transformations=self.transformations)
            else:
                parsed_equation = parse_expr(equation, transformations=self.transformations)
            
            # Solve the equation
            solutions = solve(parsed_equation, symbols(variable))
            
            if not solutions:
                return CalculationResult(
                    expression=equation,
                    result="",
                    result_type="error",
                    error="No solution found"
                )
            
            # Format solutions
            if len(solutions) == 1:
                result = str(solutions[0])
            else:
                result = f"[{', '.join(str(sol) for sol in solutions)}]"
            
            return CalculationResult(
                expression=equation,
                result=result,
                result_type="solution",
                steps=f"Solved {equation} for {variable}",
                confidence=1.0
            )
            
        except Exception as e:
            return CalculationResult(
                expression=equation,
                result="",
                result_type="error",
                error=f"Equation solving error: {str(e)}"
            )
    
    async def calculate_derivative(self, expression: str, variable: str = "x", order: int = 1) -> CalculationResult:
        """
        Calculate derivative of an expression
        
        Args:
            expression: Mathematical expression
            variable: Variable to differentiate with respect to
            order: Order of derivative (1 for first derivative, 2 for second, etc.)
            
        Returns:
            CalculationResult with the derivative
        """
        try:
            parsed_expr = parse_expr(expression, transformations=self.transformations)
            var_symbol = symbols(variable)
            
            # Calculate derivative
            derivative = parsed_expr
            for _ in range(order):
                derivative = diff(derivative, var_symbol)
            
            # Simplify the result
            simplified_derivative = simplify(derivative)
            
            return CalculationResult(
                expression=f"d^{order}/{variable}^{order}({expression})",
                result=str(simplified_derivative),
                result_type="derivative",
                steps=f"Calculated {order}{'st' if order == 1 else 'nd' if order == 2 else 'th'} derivative",
                confidence=1.0
            )
            
        except Exception as e:
            return CalculationResult(
                expression=expression,
                result="",
                result_type="error",
                error=f"Derivative calculation error: {str(e)}"
            )
    
    async def calculate_integral(self, expression: str, variable: str = "x", limits: Optional[tuple] = None) -> CalculationResult:
        """
        Calculate integral of an expression
        
        Args:
            expression: Mathematical expression
            variable: Variable to integrate with respect to
            limits: Optional tuple (lower_limit, upper_limit) for definite integral
            
        Returns:
            CalculationResult with the integral
        """
        try:
            parsed_expr = parse_expr(expression, transformations=self.transformations)
            var_symbol = symbols(variable)
            
            if limits:
                lower, upper = limits
                integral = integrate(parsed_expr, (var_symbol, lower, upper))
                result_type = "definite_integral"
                steps = f"Calculated definite integral from {lower} to {upper}"
            else:
                integral = integrate(parsed_expr, var_symbol)
                result_type = "indefinite_integral"
                steps = "Calculated indefinite integral"
            
            # Simplify the result
            simplified_integral = simplify(integral)
            
            return CalculationResult(
                expression=f"∫{expression}d{variable}" + (f" from {limits[0]} to {limits[1]}" if limits else ""),
                result=str(simplified_integral),
                result_type=result_type,
                steps=steps,
                confidence=1.0
            )
            
        except Exception as e:
            return CalculationResult(
                expression=expression,
                result="",
                result_type="error",
                error=f"Integral calculation error: {str(e)}"
            )
    
    async def convert_units(self, value: float, from_unit: str, to_unit: str) -> UnitConversionResult:
        """
        Convert between different units
        
        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            UnitConversionResult with the conversion
        """
        try:
            # Create quantity with source unit
            quantity = value * self.ureg(from_unit)
            
            # Convert to target unit
            converted_quantity = quantity.to(to_unit)
            
            # Calculate conversion factor
            conversion_factor = float(converted_quantity / quantity)
            
            return UnitConversionResult(
                from_value=value,
                from_unit=from_unit,
                to_value=float(converted_quantity.magnitude),
                to_unit=to_unit,
                conversion_factor=conversion_factor
            )
            
        except Exception as e:
            raise ValueError(f"Unit conversion error: {str(e)}")
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize mathematical expression"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', '', expression)
        
        # Replace common mathematical symbols
        replacements = {
            '×': '*',
            '÷': '/',
            '²': '**2',
            '³': '**3',
            'π': 'pi',
            '∞': 'inf',
            '√': 'sqrt',
            '∫': 'integrate',
            '∂': 'diff'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def _validate_expression(self, expression: str) -> bool:
        """Validate mathematical expression for safety"""
        # Check for potentially dangerous operations
        dangerous_patterns = [
            r'__.*__',  # Python magic methods
            r'import\s+',  # Import statements
            r'exec\s*\(',  # Exec function
            r'eval\s*\(',  # Eval function
            r'open\s*\(',  # File operations
            r'file\s*\(',  # File operations
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False
        
        return True
    
    def _determine_result_type(self, result) -> str:
        """Determine the type of mathematical result"""
        if result.is_integer:
            return "integer"
        elif result.is_rational and not result.is_integer:
            return "rational"
        elif result.is_real and not result.is_rational:
            return "real"
        elif result.is_complex and not result.is_real:
            return "complex"
        elif result == sp.oo or result == -sp.oo:
            return "infinity"
        elif result == sp.nan:
            return "undefined"
        else:
            return "expression"
    
    def _generate_solution_steps(self, expression: str, result) -> str:
        """Generate human-readable solution steps"""
        try:
            # For simple expressions, just show the evaluation
            if len(expression) < 20:
                return f"Evaluated {expression} = {result}"
            
            # For more complex expressions, show intermediate steps
            parsed = parse_expr(expression, transformations=self.transformations)
            simplified = simplify(parsed)
            
            if simplified != parsed:
                return f"1. Parsed: {expression}\n2. Simplified: {simplified}\n3. Result: {result}"
            else:
                return f"1. Parsed: {expression}\n2. Result: {result}"
                
        except:
            return f"Result: {result}"
    
    async def get_supported_functions(self) -> Dict[str, str]:
        """Get list of supported mathematical functions"""
        return {
            "Basic": ["+", "-", "*", "/", "**", "sqrt", "abs"],
            "Trigonometric": ["sin", "cos", "tan", "asin", "acos", "atan"],
            "Hyperbolic": ["sinh", "cosh", "tanh"],
            "Logarithmic": ["log", "log10", "ln", "exp"],
            "Other": ["factorial", "gcd", "lcm", "floor", "ceil", "round"]
        }
    
    async def get_supported_units(self) -> Dict[str, list]:
        """Get list of supported unit categories"""
        return {
            "Length": ["meter", "kilometer", "mile", "foot", "inch", "centimeter"],
            "Mass": ["kilogram", "gram", "pound", "ounce"],
            "Volume": ["liter", "gallon", "cubic_meter", "cubic_foot"],
            "Temperature": ["celsius", "fahrenheit", "kelvin"],
            "Area": ["square_meter", "square_foot", "acre", "hectare"],
            "Time": ["second", "minute", "hour", "day", "year"],
            "Speed": ["meter_per_second", "kilometer_per_hour", "mile_per_hour"],
            "Currency": ["USD", "EUR", "GBP", "JPY", "CAD"]
        }
