#!/usr/bin/env python3
"""
Debug script to test the extraction functions.
"""

import sys
import os
import asyncio

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.deconstruction import (
    extract_foundational_concepts,
    extract_key_terms,
    extract_processes,
    identify_relationships,
    Proposition,
    Entity,
    Process
)

async def main():
    """Main function to test the extraction functions."""
    print("🧪 DEBUGGING EXTRACTION FUNCTIONS")
    print("=" * 50)
    
    # Test data
    test_text = """
    Photosynthesis is a process used by plants and other organisms to convert light energy 
    into chemical energy that can later be released to fuel the organisms' activities. 
    This chemical energy is stored in carbohydrate molecules, such as sugars, which are 
    synthesized from carbon dioxide and water.
    
    Key terms include:
    - Photosynthesis: The process of converting light energy to chemical energy
    - Chlorophyll: The green pigment in plants that absorbs light
    - Stomata: Small openings in leaves that allow gas exchange
    
    The process involves several steps:
    1. Light absorption by chlorophyll
    2. Water absorption through roots
    3. Carbon dioxide intake through stomata
    4. Conversion of light energy to chemical energy
    5. Production of glucose and oxygen
    """
    
    section_id = "test_section_1"
    
    print("📝 TESTING FOUNDATIONAL CONCEPTS EXTRACTION")
    try:
        propositions = await extract_foundational_concepts(test_text, section_id)
        print(f"✅ Extracted {len(propositions)} propositions")
        for i, prop in enumerate(propositions, 1):
            print(f"  {i}. {prop.statement}")
    except Exception as e:
        print(f"❌ Error extracting propositions: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("📝 TESTING KEY TERMS EXTRACTION")
    try:
        entities = await extract_key_terms(test_text, section_id)
        print(f"✅ Extracted {len(entities)} entities")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity.entity}: {entity.definition}")
    except Exception as e:
        print(f"❌ Error extracting entities: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("📝 TESTING PROCESSES EXTRACTION")
    try:
        processes = await extract_processes(test_text, section_id)
        print(f"✅ Extracted {len(processes)} processes")
        for i, process in enumerate(processes, 1):
            print(f"  {i}. {process.process_name}: {len(process.steps)} steps")
    except Exception as e:
        print(f"❌ Error extracting processes: {e}")
        import traceback
        traceback.print_exc()
    
    # Test relationships with the extracted data
    print("\n" + "=" * 50)
    print("📝 TESTING RELATIONSHIPS IDENTIFICATION")
    try:
        # Create some dummy data for testing
        dummy_propositions = [
            Proposition(
                id="prop_1",
                statement="Photosynthesis converts light energy to chemical energy",
                supporting_evidence=[],
                sections=[section_id]
            )
        ]
        
        dummy_entities = [
            Entity(
                id="entity_1",
                entity="Photosynthesis",
                definition="The process of converting light energy to chemical energy",
                category="Concept",
                sections=[section_id]
            )
        ]
        
        dummy_processes = [
            Process(
                id="process_1",
                process_name="Photosynthesis",
                steps=[
                    "Light absorption by chlorophyll",
                    "Water absorption through roots",
                    "Carbon dioxide intake through stomata",
                    "Conversion of light energy to chemical energy",
                    "Production of glucose and oxygen"
                ],
                sections=[section_id]
            )
        ]
        
        relationships = await identify_relationships(dummy_propositions, dummy_entities, dummy_processes)
        print(f"✅ Identified {len(relationships)} relationships")
        for i, rel in enumerate(relationships, 1):
            print(f"  {i}. {rel.relationship_type}: {rel.source_primitive_id} -> {rel.target_primitive_id}")
    except Exception as e:
        print(f"❌ Error identifying relationships: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
