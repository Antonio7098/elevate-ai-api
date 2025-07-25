import asyncio
import json
import sys
sys.path.append('/home/antonio/programming/elevate/core_and_ai/elevate-ai-api')

from app.core.blueprint_parser import BlueprintParser
from app.utils.blueprint_translator import translate_blueprint
from app.models.learning_blueprint import LearningBlueprint

async def debug_blueprint_parsing():
    # Get blueprint 78 data from Core API
    import requests
    
    # First get auth token
    auth_response = requests.post('http://localhost:3000/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    token = auth_response.json()['token']
    
    # Get blueprint 78 data
    blueprint_response = requests.get('http://localhost:3000/api/ai-rag/learning-blueprints/78', 
                                      headers={'Authorization': f'Bearer {token}'})
    blueprint_response_data = blueprint_response.json()
    
    # Extract the blueprintJson field which contains the actual blueprint structure
    blueprint_data = blueprint_response_data.get('blueprintJson', {})
    
    print("=== Original Blueprint Data ===")
    print(f"Response has blueprintJson: {'blueprintJson' in blueprint_response_data}")
    print(f"Has sections: {'sections' in blueprint_data}")
    print(f"Has knowledge_primitives: {'knowledge_primitives' in blueprint_data}")
    if 'sections' in blueprint_data:
        print(f"Sections count: {len(blueprint_data['sections'])}")
        print(f"First section: {blueprint_data['sections'][0] if 'sections' in blueprint_data and blueprint_data['sections'] else 'None'}")
    if 'knowledge_primitives' in blueprint_data:
        print(f"Propositions count: {len(blueprint_data['knowledge_primitives'].get('key_propositions_and_facts', []))}")
        print(f"Entities count: {len(blueprint_data['knowledge_primitives'].get('key_entities_and_definitions', []))}")  
        print(f"Processes count: {len(blueprint_data['knowledge_primitives'].get('described_processes_and_steps', []))}")  
        print(f"Relationships count: {len(blueprint_data['knowledge_primitives'].get('identified_relationships', []))}")
    
    print("\n=== Blueprint Translation ===")
    try:
        # Translate blueprint data to LearningBlueprint object
        learning_blueprint = translate_blueprint(blueprint_data)
        print(f"Translation successful: {learning_blueprint.source_id}")
        print(f"Translated sections count: {len(learning_blueprint.sections)}")
        print(f"Translated propositions count: {len(learning_blueprint.knowledge_primitives.key_propositions_and_facts)}")
        print(f"Translated entities count: {len(learning_blueprint.knowledge_primitives.key_entities_and_definitions)}")
        
        print("\n=== Blueprint Parsing ===")
        # Parse blueprint into TextNodes
        parser = BlueprintParser()
        nodes = parser.parse_blueprint(learning_blueprint)
        print(f"Total nodes parsed: {len(nodes)}")
        
        if nodes:
            print(f"First node: {nodes[0].id} - {nodes[0].locus_type} - {nodes[0].content[:100]}...")
        else:
            print("No nodes were parsed!")
            
            # Debug each parsing method individually
            source_text_hash = parser._generate_source_hash(learning_blueprint)
            
            section_nodes = parser._parse_sections(learning_blueprint, source_text_hash)
            print(f"Section nodes: {len(section_nodes)}")
            
            primitive_nodes = parser._parse_knowledge_primitives(learning_blueprint, source_text_hash)
            print(f"Knowledge primitive nodes: {len(primitive_nodes)}")
            
            if len(section_nodes) == 0:
                print("Issue: No section nodes were parsed")
                print(f"Blueprint sections type: {type(learning_blueprint.sections)}")
                print(f"Blueprint sections: {learning_blueprint.sections}")
            
            if len(primitive_nodes) == 0:
                print("Issue: No knowledge primitive nodes were parsed")
                print(f"Knowledge primitives type: {type(learning_blueprint.knowledge_primitives)}")
        
    except Exception as e:
        print(f"Error during translation or parsing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_blueprint_parsing())
