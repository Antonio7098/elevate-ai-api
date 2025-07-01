#!/usr/bin/env python3
"""
View blueprint, source text, and generated questions together in a comprehensive format.

Usage:
  python scripts/view_blueprint_with_questions.py [blueprint_id]
  python scripts/view_blueprint_with_questions.py --all
"""
import os
import json
import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint
from typing import Optional, Dict, Any

console = Console()

API_URL = "http://127.0.0.1:8000/api/v1/ai-rag/learning-blueprints/{blueprint_id}/question-sets"
API_KEY = os.environ.get("ELEVATE_API_KEY", "test_api_key_123")
DECONSTRUCTIONS_DIR = Path("deconstructions")
QUESTION_SETS_DIR = Path("question_sets")
SOURCES_DIR = Path("sources")


def load_deconstruction(blueprint_id: str) -> Optional[Dict[str, Any]]:
    """Load deconstruction data by blueprint_id."""
    for file_path in DECONSTRUCTIONS_DIR.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            if data.get("blueprint_id") == blueprint_id:
                return data
    return None


def load_source_text(source_id: str) -> Optional[str]:
    """Load source text by source_id."""
    for file_path in SOURCES_DIR.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            if data.get("source_id") == source_id:
                return data.get("source_text", "")
    return None


def load_questions(blueprint_id: str) -> Optional[Dict[str, Any]]:
    """Load generated questions by blueprint_id."""
    question_file = QUESTION_SETS_DIR / f"questions_{blueprint_id}.json"
    if question_file.exists():
        with open(question_file, "r") as f:
            return json.load(f)
    return None


def display_source_text(source_text: str, title: str = "Source Text"):
    """Display source text in a panel."""
    if not source_text:
        console.print(Panel("No source text available", title=title, border_style="red"))
        return
    
    # Truncate if too long
    display_text = source_text[:2000] + "..." if len(source_text) > 2000 else source_text
    console.print(Panel(display_text, title=title, border_style="blue"))


def display_blueprint(blueprint_data: Dict[str, Any]):
    """Display blueprint information."""
    console.print(Panel(f"[bold]Blueprint ID:[/bold] {blueprint_data.get('blueprint_id', 'N/A')}", 
                       title="Blueprint Information", border_style="green"))
    
    # Display sections
    sections = blueprint_data.get("blueprint_json", {}).get("sections", [])
    if sections:
        table = Table(title="Sections", show_header=True, header_style="bold magenta")
        table.add_column("Section ID", style="cyan")
        table.add_column("Section Name", style="green")
        table.add_column("Description", style="white")
        
        for section in sections:
            table.add_row(
                section.get("section_id", ""),
                section.get("section_name", ""),
                section.get("description", "")[:50] + "..." if len(section.get("description", "")) > 50 else section.get("description", "")
            )
        console.print(table)
    
    # Display knowledge primitives summary
    kp = blueprint_data.get("blueprint_json", {}).get("knowledge_primitives", {})
    if kp:
        console.print("\n[bold]Knowledge Primitives Summary:[/bold]")
        console.print(f"  • Propositions: {len(kp.get('key_propositions_and_facts', []))}")
        console.print(f"  • Entities: {len(kp.get('key_entities_and_definitions', []))}")
        console.print(f"  • Processes: {len(kp.get('described_processes_and_steps', []))}")
        console.print(f"  • Relationships: {len(kp.get('identified_relationships', []))}")


def display_questions(questions_data: Dict[str, Any]):
    """Display generated questions."""
    questions = questions_data.get("questions", [])
    if not questions:
        console.print(Panel("No questions generated yet", title="Questions", border_style="yellow"))
        return
    
    console.print(f"\n[bold green]Generated Questions ({len(questions)} total)[/bold green]")
    
    for i, question in enumerate(questions, 1):
        # Create a panel for each question
        question_text = f"[bold]Q{i}:[/bold] {question.get('text', '')}"
        answer_text = f"[blue]Answer:[/blue] {question.get('answer', '')}"
        details_text = f"[magenta]Type:[/magenta] {question.get('question_type', '')} | [cyan]Marks:[/cyan] {question.get('total_marks_available', '')}"
        
        content = f"{question_text}\n\n{answer_text}\n\n{details_text}"
        
        if question.get('marking_criteria'):
            content += f"\n\n[dim]Marking Criteria:[/dim] {question.get('marking_criteria')}"
        
        console.print(Panel(content, title=f"Question {i}", border_style="bright_blue"))
        console.print()  # Add spacing between questions


def view_single_blueprint(blueprint_id: str):
    """View a single blueprint with its source and questions."""
    console.print(f"\n[bold cyan]Viewing Blueprint: {blueprint_id}[/bold cyan]")
    console.print("=" * 80)
    
    # Load deconstruction data
    deconstruction_data = load_deconstruction(blueprint_id)
    if not deconstruction_data:
        console.print(f"[red]No deconstruction found for blueprint_id: {blueprint_id}[/red]")
        return
    
    # Load source text
    source_id = deconstruction_data.get("source_id")
    source_text = load_source_text(source_id) if source_id else None
    
    # Load questions
    questions_data = load_questions(blueprint_id)
    
    # Display everything
    if source_text:
        display_source_text(source_text, f"Source Text (ID: {source_id})")
        console.print()
    display_blueprint(deconstruction_data)
    console.print()
    if questions_data:
        display_questions(questions_data)


def view_all_blueprints():
    """View all available blueprints."""
    deconstruction_files = list(DECONSTRUCTIONS_DIR.glob("*.json"))
    if not deconstruction_files:
        console.print("[yellow]No deconstruction files found. Run batch_deconstruct_and_view.py first.[/yellow]")
        return
    
    console.print(f"\n[bold cyan]Available Blueprints ({len(deconstruction_files)} total)[/bold cyan]")
    console.print("=" * 80)
    
    for file_path in sorted(deconstruction_files):
        with open(file_path, "r") as f:
            data = json.load(f)
            blueprint_id = data.get("blueprint_id", "N/A")
            source_text = data.get("source_text", "")[:100] + "..." if len(data.get("source_text", "")) > 100 else data.get("source_text", "")
            
            # Check if questions exist
            questions_file = QUESTION_SETS_DIR / f"questions_{blueprint_id}.json"
            has_questions = questions_file.exists()
            
            status = "[green]✓ Questions available[/green]" if has_questions else "[yellow]No questions yet[/yellow]"
            
            console.print(f"\n[bold]Blueprint ID:[/bold] {blueprint_id}")
            console.print(f"[bold]Source Preview:[/bold] {source_text}")
            console.print(f"[bold]Status:[/bold] {status}")
            console.print("-" * 60)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            view_all_blueprints()
        else:
            blueprint_id = sys.argv[1]
            view_single_blueprint(blueprint_id)
    else:
        # Show available blueprints
        view_all_blueprints()
        console.print("\n[bold]Usage:[/bold]")
        console.print("  python scripts/view_blueprint_with_questions.py [blueprint_id]")
        console.print("  python scripts/view_blueprint_with_questions.py --all")


if __name__ == "__main__":
    main() 