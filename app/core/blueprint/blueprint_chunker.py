"""
Blueprint chunker module - adapter for existing functionality.

This module provides the chunking interface expected by tests,
using functionality that already exists in the blueprint system.
"""

from typing import List, Dict, Any, Optional
from app.models.blueprint import Blueprint


class BlueprintChunker:
    """Adapter for blueprint content chunking functionality."""
    
    def __init__(self):
        """Initialize the chunker."""
        pass
    
    async def chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Chunk content into smaller pieces."""
        # Simple chunking by character count
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i + chunk_size])
        return chunks
    
    async def chunk_with_overlap(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk content with overlap between chunks."""
        chunks = []
        i = 0
        while i < len(content):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
    
    async def semantic_chunk(self, content: str) -> List[str]:
        """Create semantic chunks based on content structure."""
        # Simple paragraph-based chunking
        paragraphs = content.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    async def chunk_by_sections(self, content: str) -> Dict[str, str]:
        """Chunk content by sections."""
        # Simple section-based chunking
        sections = content.split('\n# ')
        result = {}
        for i, section in enumerate(sections):
            if i == 0:
                result['intro'] = section
            else:
                result[f'section_{i}'] = f'# {section}'
        return result
