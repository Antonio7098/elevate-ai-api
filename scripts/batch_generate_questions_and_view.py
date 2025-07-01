#!/usr/bin/env python3
"""
Batch generate question sets for all deconstructions in deconstructions/ folder, save results, and display them nicely.

Usage:
  python batch_generate_questions_and_view.py [--count N] [--auto-count]
  python batch_generate_questions_and_view.py --count 10
  python batch_generate_questions_and_view.py --auto-count
"""
import os
import json
import requests
import argparse
from pathlib import Path
from rich import print
from rich.table import Table

API_URL = "http://127.0.0.1:8000/api/v1/ai-rag/learning-blueprints/{blueprint_id}/question-sets"
API_KEY = os.environ.get("ELEVATE_API_KEY", "test_api_key_123")
DECONSTRUCTIONS_DIR = Path("deconstructions")
QUESTION_SETS_DIR = Path("question_sets")
SOURCES_DIR = Path("sources")


def ensure_dirs():
    DECONSTRUCTIONS_DIR.mkdir(exist_ok=True)
    QUESTION_SETS_DIR.mkdir(exist_ok=True)

def get_deconstruction_files():
    return sorted([f for f in DECONSTRUCTIONS_DIR.glob("*.json")])

def calculate_auto_question_count(blueprint_data):
    """Calculate appropriate number of questions to cover as many concepts as possible efficiently."""
    kp = blueprint_data.get("blueprint_json", {}).get("knowledge_primitives", {})
    
    # Count different types of knowledge primitives
    propositions = len(kp.get('key_propositions_and_facts', []))
    entities = len(kp.get('key_entities_and_definitions', []))
    processes = len(kp.get('described_processes_and_steps', []))
    relationships = len(kp.get('identified_relationships', []))
    
    total_primitives = propositions + entities + processes + relationships
    
    if total_primitives == 0:
        return 3  # Minimum for basic content
    
    # Strategy: Cover multiple concepts per question when possible
    # Start with a base that aims to cover most concepts efficiently
    base_count = max(3, min(12, total_primitives * 0.8))
    
    # Adjust based on content type:
    # - More questions for processes (they're complex)
    # - Fewer questions for simple entity lists
    # - Relationships can often be tested in combination with other concepts
    
    if processes > 0:
        # Processes need dedicated questions, but try to combine with related concepts
        base_count += min(processes, 2)  # Cap process questions
    if relationships > 0:
        # Relationships can often be tested alongside other concepts
        base_count += min(relationships // 2, 2)  # Every 2 relationships = 1 question
    
    # Ensure we don't exceed reasonable limits
    return int(min(15, base_count))  # Cap at 15 questions

def display_source_text(source_text):
    """Display source text."""
    if not source_text:
        print("[yellow]No source text available[/yellow]")
        return
    
    print(f"\n[bold blue]Source Text:[/bold blue]")
    # Truncate if too long for display
    display_text = source_text[:1000] + "..." if len(source_text) > 1000 else source_text
    print(f"[dim]{display_text}[/dim]")

def display_knowledge_primitives(kp):
    """Display all knowledge primitives in detail."""
    if not kp:
        print("[yellow]No knowledge primitives found[/yellow]")
        return
    
    print(f"\n[bold yellow]Knowledge Primitives:[/bold yellow]")
    
    # Propositions
    propositions = kp.get('key_propositions_and_facts', [])
    if propositions:
        print(f"\n[bold cyan]Propositions ({len(propositions)}):[/bold cyan]")
        for i, prop in enumerate(propositions, 1):
            print(f"  {i}. [bold]{prop.get('statement', 'N/A')}[/bold]")
            if prop.get('supporting_evidence'):
                print(f"     Evidence: {', '.join(prop.get('supporting_evidence', []))}")
            print()
    
    # Entities
    entities = kp.get('key_entities_and_definitions', [])
    if entities:
        print(f"\n[bold green]Entities ({len(entities)}):[/bold green]")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. [bold]{entity.get('entity', 'N/A')}[/bold] - {entity.get('definition', 'N/A')}")
            print(f"     Category: {entity.get('category', 'N/A')}")
            print()
    
    # Processes
    processes = kp.get('described_processes_and_steps', [])
    if processes:
        print(f"\n[bold magenta]Processes ({len(processes)}):[/bold magenta]")
        for i, process in enumerate(processes, 1):
            print(f"  {i}. [bold]{process.get('process_name', 'N/A')}[/bold]")
            steps = process.get('steps', [])
            if steps:
                for j, step in enumerate(steps, 1):
                    print(f"     {j}. {step}")
            print()
    
    # Relationships
    relationships = kp.get('identified_relationships', [])
    if relationships:
        print(f"\n[bold orange]Relationships ({len(relationships)}):[/bold orange]")
        for i, rel in enumerate(relationships, 1):
            print(f"  {i}. [bold]{rel.get('relationship_type', 'N/A')}[/bold]")
            print(f"     {rel.get('description', 'N/A')}")
            print(f"     Source: {rel.get('source_primitive_id', 'N/A')} → Target: {rel.get('target_primitive_id', 'N/A')}")
            print()

def display_blueprint(blueprint_data):
    """Display blueprint information."""
    print(f"\n[bold green]Blueprint Information:[/bold green]")
    print(f"  [cyan]Blueprint ID:[/cyan] {blueprint_data.get('blueprint_id', 'N/A')}")
    
    # Display source text
    source_text = blueprint_data.get('source_text', '')
    display_source_text(source_text)
    
    # Display sections
    sections = blueprint_data.get("blueprint_json", {}).get("sections", [])
    if sections:
        print(f"\n[bold magenta]Sections ({len(sections)}):[/bold magenta]")
        for section in sections:
            print(f"  • {section.get('section_name', 'N/A')} - {section.get('description', '')[:50]}...")
    
    # Display all knowledge primitives in detail
    kp = blueprint_data.get("blueprint_json", {}).get("knowledge_primitives", {})
    display_knowledge_primitives(kp)

def generate_questions(deconstruction_path, question_count=None, auto_count=False):
    with open(deconstruction_path, "r") as f:
        data = json.load(f)
    blueprint_id = data.get("blueprint_id")
    if not blueprint_id:
        print(f"[red]No blueprint_id found in {deconstruction_path.name}[/red]")
        return None
    
    # Determine question count
    if auto_count:
        question_count = calculate_auto_question_count(data)
        print(f"[bold yellow]Auto-calculated question count: {question_count}[/bold yellow]")
    elif question_count is None:
        question_count = 5  # Default
    
    name = f"Questions for {deconstruction_path.stem}"
    payload = {
        "name": name,
        "question_options": {
            "count": question_count
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    url = API_URL.format(blueprint_id=blueprint_id)
    print(f"\n[bold cyan]Generating {question_count} questions for:[/bold cyan] {deconstruction_path.name}")
    
    # Display the blueprint first
    display_blueprint(data)
    
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        print(f"[red]Failed to generate questions for {deconstruction_path.name}: {resp.status_code} {resp.text}[/red]")
        return None
    result = resp.json()
    out_path = QUESTION_SETS_DIR / f"questions_{deconstruction_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path, result

def view_questions(json_path, result):
    print(f"\n[bold green]Generated Questions:[/bold green] {json_path.name}\n")
    questions = result.get("questions", [])
    if not questions:
        print("[yellow]No questions generated.[/yellow]")
        return
    print(f"[bold]Total questions generated: {len(questions)}[/bold]\n")
    for i, q in enumerate(questions, 1):
        print(f"[bold]{i}. {q.get('text')}[/bold]")
        print(f"   [blue]Answer:[/blue] {q.get('answer')}")
        print(f"   [magenta]Type:[/magenta] {q.get('question_type')} | [cyan]Marks:[/cyan] {q.get('total_marks_available')}")
        if q.get('marking_criteria'):
            print(f"   [dim]Marking:[/dim] {q.get('marking_criteria')}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Batch generate question sets with configurable counts")
    parser.add_argument("--count", type=int, help="Number of questions to generate per blueprint")
    parser.add_argument("--auto-count", action="store_true", help="Let the system determine appropriate question count based on content complexity")
    
    args = parser.parse_args()
    
    if args.count and args.auto_count:
        print("[red]Cannot specify both --count and --auto-count. Choose one.[/red]")
        return
    
    ensure_dirs()
    files = get_deconstruction_files()
    if not files:
        print(f"[yellow]No deconstruction files found in {DECONSTRUCTIONS_DIR}/. Run batch_deconstruct_and_view.py first.[/yellow]")
        return
    
    for f in files:
        out_path, result = generate_questions(f, args.count, args.auto_count) or (None, None)
        if out_path:
            view_questions(out_path, result)
        print("-" * 80)  # Separator between files
    print("\n[bold magenta]Batch question generation complete.[/bold magenta]")

if __name__ == "__main__":
    main() 