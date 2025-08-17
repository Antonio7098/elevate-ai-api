# PromptTemplate class for response_generation.py

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Base prompt template class."""
    
    template: str
    required_variables: list
    optional_variables: list
    
    def __post_init__(self):
        """Validate template variables."""
        if not self.template:
            raise ValueError("Template cannot be empty")
    
    def format_template(self, **kwargs) -> str:
        """Format template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def get_default_variables(self) -> Dict[str, str]:
        """Get default values for optional variables."""
        defaults = {
            "style": "neutral",
            "tone": "professional",
            "length": "medium"
        }
        return {var: defaults.get(var, '') for var in self.optional_variables}


class ResponseGenerationPromptTemplate:
    """Template service for response generation prompts."""
    
    def __init__(self):
        self.base_templates = {
            "explanation": """
            Based on the following context, answer the question clearly and accurately.
            
            Context: {context}
            Question: {query}
            
            Please provide a {response_type} response.
            """,
            "detailed_explanation": """
            Based on the following context, provide a comprehensive and detailed answer.
            
            Context: {context}
            Question: {query}
            
            Please provide a {response_type} response with examples and explanations.
            """,
            "simple_explanation": """
            Based on the following context, provide a simple and easy-to-understand answer.
            
            Context: {context}
            Question: {query}
            
            Please provide a {response_type} response in simple terms.
            """
        }
    
    async def generate_prompt(
        self, 
        query: str, 
        context: str = "", 
        response_type: str = "explanation"
    ) -> str:
        """Generate a prompt based on query, context, and response type."""
        template = self.base_templates.get(response_type, self.base_templates["explanation"])
        
        return template.format(
            context=context,
            query=query,
            response_type=response_type
        )
    
    async def customize_template(
        self, 
        base_template: str, 
        customizations: Dict[str, str]
    ) -> str:
        """Customize a base template with specific parameters."""
        # Simple customization - append customization info
        custom_text = "\n".join([f"{k}: {v}" for k, v in customizations.items()])
        
        return f"{base_template}\n\nCustomizations:\n{custom_text}"
    
    async def apply_template(
        self, 
        template: str, 
        variables: Dict[str, str]
    ) -> str:
        """Apply variables to a template."""
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
