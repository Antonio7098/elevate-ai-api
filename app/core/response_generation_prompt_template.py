# PromptTemplate class for response_generation.py
# This will be appended to the main file

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PromptTemplate:
    """Template for generating prompts for different response types."""
    template_type: str
    template_content: str
    required_variables: List[str]
    optional_variables: List[str] = None
    tone_instructions: str = ""
    examples: List[str] = None
    
    def __post_init__(self):
        if self.optional_variables is None:
            self.optional_variables = []
        if self.examples is None:
            self.examples = []
    
    def format_template(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            return self.template_content.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")
    
    def get_default_variables(self) -> Dict[str, str]:
        """Get default values for optional variables."""
        defaults = {
            'tone_instructions': self.tone_instructions,
            'conversation_history': 'No previous conversation.',
            'learning_profile': 'No specific learning profile provided.',
            'retrieved_knowledge': 'No specific knowledge retrieved.',
            'query_intent': 'General inquiry'
        }
        return {var: defaults.get(var, '') for var in self.optional_variables}
