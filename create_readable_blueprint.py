#!/usr/bin/env python3
"""
Convert JSON test results into a readable text file.
"""

import json
import os
from datetime import datetime

def create_readable_blueprint():
    """Create a readable text file from the test results."""
    
    # Find the most recent content test results file
    results_files = [f for f in os.listdir('.') if f.startswith('content_test_results_') and f.endswith('.json')]
    
    if not results_files:
        print("❌ No content test results files found. Please run the test first.")
        return
    
    # Get the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    print(f"📁 Using results file: {latest_file}")
    
    # Load the results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"readable_blueprint_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SEQUENTIAL GENERATION WORKFLOW - READABLE BLUEPRINT\n")
        f.write("=" * 80 + "\n\n")
        
        # Test Information
        f.write("TEST INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Session ID: {results.get('session_id', 'N/A')}\n")
        f.write(f"Start Time: {results.get('start_time', 'N/A')}\n")
        f.write(f"End Time: {results.get('end_time', 'N/A')}\n")
        f.write(f"Total Execution Time: {results.get('total_time', 'N/A')} seconds\n")
        f.write(f"Steps Completed: {', '.join(results.get('steps_completed', []))}\n")
        f.write(f"Errors: {len(results.get('errors', []))}\n\n")
        
        # Source Content
        f.write("SOURCE CONTENT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Type: {results.get('source_type', 'N/A')}\n")
        f.write(f"Length: {len(results.get('source_content', ''))} characters\n\n")
        f.write(results.get('source_content', 'No source content') + "\n\n")
        
        # Generated Content
        f.write("GENERATED CONTENT\n")
        f.write("-" * 40 + "\n")
        
        generated_content = results.get('generated_content', {})
        
        # Session Start Content
        if 'session_start' in generated_content:
            f.write("Session Start Content:\n")
            session_content = generated_content['session_start']
            f.write(f"  Source Content: {session_content.get('source_content', 'N/A')[:100]}...\n")
            f.write(f"  Source Type: {session_content.get('source_type', 'N/A')}\n")
            f.write(f"  User Preferences: {session_content.get('user_preferences', 'N/A')}\n")
            f.write(f"  Analysis Timestamp: {session_content.get('analysis_timestamp', 'N/A')}\n\n")
        
        # Blueprint Content
        if 'blueprint' in generated_content:
            f.write("Generated Blueprint:\n")
            blueprint = generated_content['blueprint']
            
            if isinstance(blueprint, dict):
                f.write(f"  Title: {blueprint.get('title', 'N/A')}\n")
                f.write(f"  Description: {blueprint.get('description', 'N/A')}\n")
                f.write(f"  Type: {blueprint.get('type', 'N/A')}\n")
                f.write(f"  Status: {blueprint.get('status', 'N/A')}\n")
                f.write(f"  ID: {blueprint.get('id', 'N/A')}\n")
                f.write(f"  Author ID: {blueprint.get('author_id', 'N/A')}\n")
                f.write(f"  Created At: {blueprint.get('created_at', 'N/A')}\n")
                f.write(f"  Updated At: {blueprint.get('updated_at', 'N/A')}\n")
                f.write(f"  Version: {blueprint.get('version', 'N/A')}\n")
                f.write(f"  Is Public: {blueprint.get('is_public', 'N/A')}\n")
                f.write(f"  Tags: {blueprint.get('tags', 'N/A')}\n")
                f.write(f"  Metadata: {blueprint.get('metadata', 'N/A')}\n\n")
                
                # Blueprint Sections
                blueprint_sections = blueprint.get("sections", [])
                if blueprint_sections:
                    f.write(f"  Blueprint Sections ({len(blueprint_sections)}):\n")
                    for i, section in enumerate(blueprint_sections):
                        f.write(f"    Section {i+1}:\n")
                        f.write(f"      ID: {section.get('id', 'N/A')}\n")
                        f.write(f"      Title: {section.get('title', 'N/A')}\n")
                        f.write(f"      Description: {section.get('description', 'N/A')}\n")
                        f.write(f"      Blueprint ID: {section.get('blueprint_id', 'N/A')}\n")
                        f.write(f"      Parent Section ID: {section.get('parent_section_id', 'N/A')}\n")
                        f.write(f"      Depth: {section.get('depth', 'N/A')}\n")
                        f.write(f"      Order Index: {section.get('order_index', 'N/A')}\n")
                        f.write(f"      Difficulty: {section.get('difficulty', 'N/A')}\n")
                        f.write(f"      Estimated Time: {section.get('estimated_time_minutes', 'N/A')} minutes\n")
                        f.write(f"      User ID: {section.get('user_id', 'N/A')}\n")
                        f.write(f"      Created At: {section.get('created_at', 'N/A')}\n")
                        f.write(f"      Updated At: {section.get('updated_at', 'N/A')}\n")
                        
                        # Show children if they exist
                        children = section.get('children', [])
                        if children:
                            f.write(f"      Children ({len(children)}):\n")
                            for j, child in enumerate(children):
                                f.write(f"        {j+1}. {child.get('title', 'N/A')}\n")
                        
                        f.write("\n")
                else:
                    f.write("  No blueprint sections found.\n\n")
            else:
                f.write(f"  Blueprint (raw): {blueprint}\n\n")
        
        # Generated Sections
        if 'sections' in generated_content:
            f.write("Generated Sections:\n")
            sections = generated_content['sections']
            
            if sections:
                f.write(f"  Number of sections: {len(sections)}\n\n")
                for i, section in enumerate(sections):
                    f.write(f"  Section {i+1}:\n")
                    f.write(f"    ID: {section.get('id', 'N/A')}\n")
                    f.write(f"    Title: {section.get('title', 'N/A')}\n")
                    f.write(f"    Description: {section.get('description', 'N/A')}\n")
                    f.write(f"    Blueprint ID: {section.get('blueprint_id', 'N/A')}\n")
                    f.write(f"    Parent Section ID: {section.get('parent_section_id', 'N/A')}\n")
                    f.write(f"    Depth: {section.get('depth', 'N/A')}\n")
                    f.write(f"    Order Index: {section.get('order_index', 'N/A')}\n")
                    f.write(f"    Difficulty: {section.get('difficulty', 'N/A')}\n")
                    f.write(f"    Estimated Time: {section.get('estimated_time_minutes', 'N/A')} minutes\n")
                    f.write(f"    User ID: {section.get('user_id', 'N/A')}\n")
                    f.write(f"    Created At: {section.get('created_at', 'N/A')}\n")
                    f.write(f"    Updated At: {section.get('updated_at', 'N/A')}\n")
                    
                    # Show children if they exist
                    children = section.get('children', [])
                    if children:
                        f.write(f"    Children ({len(children)}):\n")
                        for j, child in enumerate(children):
                            f.write(f"      {j+1}. {child.get('title', 'N/A')}\n")
                    
                    f.write("\n")
            else:
                f.write("  No sections generated yet.\n\n")
        
        # Current Content (if available)
        if 'current_content' in generated_content:
            f.write("Current Content State:\n")
            current_content = generated_content['current_content']
            f.write(f"  Keys available: {list(current_content.keys())}\n")
            
            # Show what's in the current content
            for key, value in current_content.items():
                if key == 'sections' and isinstance(value, list):
                    f.write(f"  {key}: {len(value)} items\n")
                elif key == 'blueprint' and isinstance(value, dict):
                    f.write(f"  {key}: {value.get('title', 'N/A')} - {value.get('description', 'N/A')[:50]}...\n")
                else:
                    f.write(f"  {key}: {str(value)[:100]}...\n")
            f.write("\n")
        
        # Error Content (if available)
        if 'error_content' in generated_content:
            f.write("Error State Content:\n")
            error_content = generated_content['error_content']
            f.write(f"  Keys available: {list(error_content.keys())}\n")
            f.write(f"  Content: {str(error_content)[:200]}...\n\n")
        
        # Errors
        if results.get('errors'):
            f.write("ERRORS ENCOUNTERED\n")
            f.write("-" * 40 + "\n")
            for error in results['errors']:
                f.write(f"  - {error}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF READABLE BLUEPRINT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Readable blueprint created: {output_file}")
    
    # Show a quick summary
    print(f"\n📋 SUMMARY:")
    print(f"   Source length: {len(results.get('source_content', ''))} characters")
    
    generated_content = results.get('generated_content', {})
    if 'blueprint' in generated_content:
        blueprint = generated_content['blueprint']
        if isinstance(blueprint, dict):
            print(f"   Blueprint sections: {len(blueprint.get('sections', []))}")
    
    if 'sections' in generated_content:
        print(f"   Generated sections: {len(generated_content['sections'])}")
    
    print(f"   Steps completed: {len(results.get('steps_completed', []))}")
    print(f"   Errors: {len(results.get('errors', []))}")
    
    return output_file

if __name__ == "__main__":
    output_file = create_readable_blueprint()
    if output_file:
        print(f"\n📄 Full readable content saved to: {output_file}")
        print("💡 You can open this file to see the complete generated content in a readable format.")




