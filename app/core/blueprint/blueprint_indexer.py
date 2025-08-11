"""
Blueprint indexer for indexing operations.

This module provides indexing functionality for blueprints.
"""

from typing import List, Dict, Any, Optional
from app.models.blueprint import Blueprint
import json
import hashlib


class BlueprintIndexingError(Exception):
    """Exception raised when blueprint indexing fails."""
    pass


class BlueprintIndexer:
    """Indexer class for blueprint content and metadata."""
    
    def __init__(self):
        self.index: Dict[str, Dict[str, Any]] = {}
        self.content_index: Dict[str, List[str]] = {}
        self.metadata_index: Dict[str, Dict[str, List[str]]] = {}
    
    async def index_blueprint(self, blueprint: Blueprint) -> None:
        """Index a blueprint for search and retrieval."""
        # Create document index
        doc_id = blueprint.id
        self.index[doc_id] = {
            'id': blueprint.id,
            'title': blueprint.title,
            'description': blueprint.description,
            'type': blueprint.type.value,
            'status': blueprint.status.value,
            'author_id': blueprint.author_id,
            'is_public': blueprint.is_public,
            'tags': blueprint.tags,
            'created_at': blueprint.created_at.isoformat(),
            'updated_at': blueprint.updated_at.isoformat(),
            'version': blueprint.version
        }
        
        # Index content
        await self._index_content(doc_id, blueprint.content)
        
        # Index metadata
        await self._index_metadata(doc_id, blueprint.metadata)
        
        # Index tags
        await self._index_tags(doc_id, blueprint.tags)
    
    async def _index_content(self, doc_id: str, content: Dict[str, Any]) -> None:
        """Index blueprint content."""
        if not content:
            return
        
        # Extract text content from different sections
        text_content = []
        
        # Extract from sections if they exist
        if 'sections' in content and isinstance(content['sections'], list):
            for section in content['sections']:
                if isinstance(section, dict):
                    # Add section title
                    if 'title' in section and section['title']:
                        text_content.append(section['title'])
                    
                    # Add section content
                    if 'content' in section:
                        section_content = section['content']
                        if isinstance(section_content, str):
                            text_content.append(section_content)
                        elif isinstance(section_content, dict):
                            # Recursively extract text from nested content
                            text_content.extend(self._extract_text_from_dict(section_content))
        
        # Extract from learning objectives if they exist
        if 'learning_objectives' in content and isinstance(content['learning_objectives'], list):
            for objective in content['learning_objectives']:
                if isinstance(objective, str) and objective.strip():
                    text_content.append(objective)
        
        # Extract from other content fields
        for key, value in content.items():
            if key not in ['sections', 'learning_objectives']:
                if isinstance(value, str):
                    text_content.append(value)
                elif isinstance(value, dict):
                    text_content.extend(self._extract_text_from_dict(value))
        
        # Store indexed content
        self.content_index[doc_id] = text_content
    
    async def _extract_text_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """Recursively extract text content from nested dictionaries."""
        text_content = []
        
        for key, value in data.items():
            if isinstance(value, str):
                text_content.append(value)
            elif isinstance(value, dict):
                text_content.extend(self._extract_text_from_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        text_content.append(item)
                    elif isinstance(item, dict):
                        text_content.extend(self._extract_text_from_dict(item))
        
        return text_content
    
    async def _index_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Index blueprint metadata."""
        if not metadata:
            return
        
        self.metadata_index[doc_id] = {}
        
        for key, value in metadata.items():
            if isinstance(value, str):
                self.metadata_index[doc_id][key] = [value]
            elif isinstance(value, list):
                self.metadata_index[doc_id][key] = [str(item) for item in value if item]
            elif value is not None:
                self.metadata_index[doc_id][key] = [str(value)]
    
    async def _index_tags(self, doc_id: str, tags: List[str]) -> None:
        """Index blueprint tags."""
        if not tags:
            return
        
        # Tags are already stored in the main index, but we can create a reverse index
        for tag in tags:
            if tag not in self.metadata_index:
                self.metadata_index[tag] = {}
            if 'tagged_documents' not in self.metadata_index[tag]:
                self.metadata_index[tag]['tagged_documents'] = []
            self.metadata_index[tag]['tagged_documents'].append(doc_id)
    
    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[str]:
        """Search for blueprints by text query."""
        query_lower = query.lower()
        results = []
        
        for doc_id, content in self.content_index.items():
            # Check if query matches any content
            for text in content:
                if query_lower in text.lower():
                    results.append(doc_id)
                    break
            
            # Check title and description in main index
            doc_info = self.index.get(doc_id, {})
            if (query_lower in doc_info.get('title', '').lower() or
                (doc_info.get('description') and query_lower in doc_info.get('description', '').lower())):
                if doc_id not in results:
                    results.append(doc_id)
        
        # Remove duplicates and apply pagination
        unique_results = list(dict.fromkeys(results))
        return unique_results[offset:offset + limit]
    
    async def search_by_tags(self, tags: List[str], limit: int = 100, offset: int = 0) -> List[str]:
        """Search for blueprints by tags."""
        if not tags:
            return []
        
        matching_docs = set()
        
        for tag in tags:
            if tag in self.metadata_index and 'tagged_documents' in self.metadata_index[tag]:
                matching_docs.update(self.metadata_index[tag]['tagged_documents'])
        
        results = list(matching_docs)
        return results[offset:offset + limit]
    
    async def search_by_type(self, blueprint_type: str, limit: int = 100, offset: int = 0) -> List[str]:
        """Search for blueprints by type."""
        results = []
        
        for doc_id, doc_info in self.index.items():
            if doc_info.get('type') == blueprint_type:
                results.append(doc_id)
        
        return results[offset:offset + limit]
    
    async def search_by_status(self, status: str, limit: int = 100, offset: int = 0) -> List[str]:
        """Search for blueprints by status."""
        results = []
        
        for doc_id, doc_info in self.index.items():
            if doc_info.get('status') == status:
                results.append(doc_id)
        
        return results[offset:offset + limit]
    
    async def search_by_author(self, author_id: str, limit: int = 100, offset: int = 0) -> List[str]:
        """Search for blueprints by author."""
        results = []
        
        for doc_id, doc_info in self.index.items():
            if doc_info.get('author_id') == author_id:
                results.append(doc_id)
        
        return results[offset:offset + limit]
    
    async def remove_blueprint(self, blueprint_id: str) -> None:
        """Remove a blueprint from the index."""
        if blueprint_id in self.index:
            del self.index[blueprint_id]
        
        if blueprint_id in self.content_index:
            del self.content_index[blueprint_id]
        
        if blueprint_id in self.metadata_index:
            del self.metadata_index[blueprint_id]
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            'total_documents': len(self.index),
            'total_content_entries': len(self.content_index),
            'total_metadata_entries': len(self.metadata_index),
            'index_size_bytes': len(json.dumps(self.index, default=str)),
            'content_index_size_bytes': len(json.dumps(self.content_index, default=str)),
            'metadata_index_size_bytes': len(json.dumps(self.metadata_index, default=str))
        }
    
    async def clear_index(self) -> None:
        """Clear all indexes."""
        self.index.clear()
        self.content_index.clear()
        self.metadata_index.clear()
