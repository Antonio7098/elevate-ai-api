#!/usr/bin/env python3
"""
View a LearningBlueprint deconstruction result in a human-friendly way.

Usage:
  python view_deconstruction.py --file result.json
  cat result.json | python view_deconstruction.py
"""
import sys
import json
import argparse
from typing import Any, Dict
from pathlib import Path

try:
    from rich import print
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
except ImportError:
    print = __builtins__.print
    Console = None
    Table = None
    Panel = None
    Syntax = None


def load_json(path: str = None) -> Dict[str, Any]:
    if path:
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def print_source_text(source_text: str):
    print("[bold cyan]Source Text:[/bold cyan]")
    if Syntax:
        print(Syntax(source_text, "markdown", theme="ansi_dark", line_numbers=True))
    else:
        print(source_text)
    print()


def print_sections(sections):
    print("[bold green]Sections:[/bold green]")
    if Table:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Section ID", style="dim")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Parent")
        for s in sections:
            table.add_row(
                s.get("section_id", ""),
                s.get("section_name", ""),
                s.get("description", ""),
                str(s.get("parent_section_id", ""))
            )
        print(table)
    else:
        for s in sections:
            print(f"- [{s.get('section_id')}] {s.get('section_name')}: {s.get('description')} (parent: {s.get('parent_section_id')})")
    print()


def print_knowledge_primitives(kp):
    print("[bold yellow]Knowledge Primitives:[/bold yellow]")
    for key, label in [
        ("key_propositions_and_facts", "Propositions & Facts"),
        ("key_entities_and_definitions", "Entities & Definitions"),
        ("described_processes_and_steps", "Processes & Steps"),
        ("identified_relationships", "Relationships"),
        ("implicit_and_open_questions", "Questions")
    ]:
        items = kp.get(key, [])
        if not items:
            continue
        print(f"[bold]{label}[/bold] ({len(items)}):")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {json.dumps(item, indent=2)}")
        print()


def print_summary(blueprint):
    print("[bold magenta]LearningBlueprint Summary:[/bold magenta]")
    print(f"Source ID: [cyan]{blueprint.get('source_id')}[/cyan]")
    print(f"Title: [cyan]{blueprint.get('source_title')}[/cyan]")
    print(f"Type: [cyan]{blueprint.get('source_type')}[/cyan]")
    print(f"Summary: {json.dumps(blueprint.get('source_summary', {}), indent=2)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="View a LearningBlueprint deconstruction result.")
    parser.add_argument('--file', '-f', help='Path to JSON file (default: stdin)', default=None)
    args = parser.parse_args()

    data = load_json(args.file)
    # Accept both direct blueprint or API response
    blueprint = data.get('blueprint_json', data)
    source_text = data.get('source_text', '')

    print_source_text(source_text)
    print_summary(blueprint)
    print_sections(blueprint.get('sections', []))
    print_knowledge_primitives(blueprint.get('knowledge_primitives', {}))

if __name__ == "__main__":
    main() 