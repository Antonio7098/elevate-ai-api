#!/usr/bin/env python3
"""
Extract and display the source text and generated blueprint content.
"""

import json
import os
import sys
from datetime import datetime

# Add the app directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.generation_orchestrator import GenerationOrchestrator, GenerationStep, GenerationStatus

async def extract_blueprint_content():
    """Extract the source text and generated blueprint content."""
    
    # Find the most recent test results file
    results_files = [f for f in os.listdir('.') if f.startswith('simple_test_results_') and f.endswith('.json')]
    
    if not results_files:
        print("❌ No test results files found. Please run the test first.")
        return
    
    # Get the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    print(f"📁 Using results file: {latest_file}")
    
    # Load the results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Session ID: {results.get('session_id', 'N/A')}")
    print(f"📅 Generated at: {results.get('start_time', 'N/A')}")
    print(f"⏱️  Execution time: {results.get('total_time', 'N/A')} seconds")
    
    # Get the orchestrator instance to access the session data
    orchestrator = GenerationOrchestrator()
    
    try:
        # Get the generation progress
        progress = await orchestrator.get_generation_progress(results['session_id'])
        
        # Extract source content
        source_content = progress.current_content.get("source_content", "No source content found")
        source_type = progress.current_content.get("source_type", "Unknown")
        
        # Extract blueprint content
        blueprint = progress.current_content.get("blueprint", {})
        sections = progress.current_content.get("sections", [])
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"blueprint_content_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SEQUENTIAL GENERATION WORKFLOW - BLUEPRINT CONTENT\n")
            f.write("=" * 80 + "\n\n")
            
            # Source Content Section
            f.write("SOURCE CONTENT\n")
            f.write("-" * 40 + "\n")
            f.write(f"Type: {source_type}\n")
            f.write(f"Length: {len(source_content)} characters\n\n")
            f.write(source_content + "\n\n")
            
            # Blueprint Section
            f.write("GENERATED BLUEPRINT\n")
            f.write("-" * 40 + "\n")
            
            if isinstance(blueprint, dict):
                f.write(f"Title: {blueprint.get('title', 'N/A')}\n")
                f.write(f"Description: {blueprint.get('description', 'N/A')}\n")
                f.write(f"Type: {blueprint.get('type', 'N/A')}\n")
                f.write(f"Status: {blueprint.get('status', 'N/A')}\n\n")
                
                # Show sections if they exist in the blueprint
                blueprint_sections = blueprint.get("sections", [])
                if blueprint_sections:
                    f.write(f"Blueprint Sections ({len(blueprint_sections)}):\n")
                    for i, section in enumerate(blueprint_sections):
                        f.write(f"  {i+1}. {section.get('title', 'N/A')}\n")
                        if section.get('description'):
                            f.write(f"     Description: {section.get('description')}\n")
                        f.write(f"     Depth: {section.get('depth', 'N/A')}\n")
                        f.write(f"     Order: {section.get('order_index', 'N/A')}\n\n")
            else:
                f.write(f"Blueprint (raw): {blueprint}\n\n")
            
            # Generated Sections Section
            f.write("GENERATED SECTIONS\n")
            f.write("-" * 40 + "\n")
            
            if sections:
                f.write(f"Number of sections: {len(sections)}\n\n")
                for i, section in enumerate(sections):
                    f.write(f"Section {i+1}:\n")
                    f.write(f"  ID: {section.get('id', 'N/A')}\n")
                    f.write(f"  Title: {section.get('title', 'N/A')}\n")
                    f.write(f"  Description: {section.get('description', 'N/A')}\n")
                    f.write(f"  Blueprint ID: {section.get('blueprint_id', 'N/A')}\n")
                    f.write(f"  Parent Section ID: {section.get('parent_section_id', 'N/A')}\n")
                    f.write(f"  Depth: {section.get('depth', 'N/A')}\n")
                    f.write(f"  Order Index: {section.get('order_index', 'N/A')}\n")
                    f.write(f"  Difficulty: {section.get('difficulty', 'N/A')}\n")
                    f.write(f"  Estimated Time: {section.get('estimated_time_minutes', 'N/A')} minutes\n")
                    f.write(f"  User ID: {section.get('user_id', 'N/A')}\n")
                    f.write(f"  Created At: {section.get('created_at', 'N/A')}\n")
                    f.write(f"  Updated At: {section.get('updated_at', 'N/A')}\n")
                    
                    # Show children if they exist
                    children = section.get('children', [])
                    if children:
                        f.write(f"  Children ({len(children)}):\n")
                        for j, child in enumerate(children):
                            f.write(f"    {j+1}. {child.get('title', 'N/A')}\n")
                    
                    f.write("\n")
            else:
                f.write("No sections generated yet.\n\n")
            
            # Current Status Section
            f.write("CURRENT STATUS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Current Step: {progress.current_step}\n")
            f.write(f"Status: {progress.status}\n")
            f.write(f"Completed Steps: {progress.completed_steps}\n")
            f.write(f"Started At: {progress.started_at}\n")
            f.write(f"Last Updated: {progress.last_updated}\n")
            
            # User Edits Section
            if progress.user_edits:
                f.write(f"\nUser Edits ({len(progress.user_edits)}):\n")
                for i, edit in enumerate(progress.user_edits):
                    f.write(f"  Edit {i+1}:\n")
                    f.write(f"    Step: {edit.get('step', 'N/A')}\n")
                    f.write(f"    Content ID: {edit.get('content_id', 'N/A')}\n")
                    f.write(f"    User Notes: {edit.get('user_notes', 'N/A')}\n")
                    f.write(f"    Edited Content: {edit.get('edited_content', 'N/A')}\n\n")
            
            # Errors Section
            if progress.errors:
                f.write("ERRORS\n")
                f.write("-" * 40 + "\n")
                for error in progress.errors:
                    f.write(f"  - {error}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF BLUEPRINT CONTENT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Content extracted and saved to: {output_file}")
        
        # Also display a summary in the terminal
        print(f"\n📋 SUMMARY:")
        print(f"   Source length: {len(source_content)} characters")
        print(f"   Blueprint sections: {len(blueprint.get('sections', []))}")
        print(f"   Generated sections: {len(sections)}")
        print(f"   Current step: {progress.current_step}")
        print(f"   Status: {progress.status}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error extracting content: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main function."""
    print("🔍 Extracting Blueprint Content from Test Results")
    print("=" * 60)
    
    output_file = await extract_blueprint_content()
    
    if output_file:
        print(f"\n📄 Full content saved to: {output_file}")
        print("💡 You can open this file to see the complete generated content.")
    else:
        print("\n❌ Failed to extract content.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())




