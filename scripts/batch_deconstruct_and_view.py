#!/usr/bin/env python3
"""
Batch deconstruct all sources in sources/ folder, save results, and display them nicely.

Usage:
  python batch_deconstruct_and_view.py
"""
import os
import json
import requests
from pathlib import Path
import subprocess

API_URL = "http://127.0.0.1:8000/api/v1/deconstruct"
API_KEY = os.environ.get("ELEVATE_API_KEY", "test_api_key_123")
SOURCES_DIR = Path("sources")
DECONSTRUCTIONS_DIR = Path("deconstructions")
VIEW_SCRIPT = "view_deconstruction.py"


def ensure_dirs():
    SOURCES_DIR.mkdir(exist_ok=True)
    DECONSTRUCTIONS_DIR.mkdir(exist_ok=True)

def get_source_files():
    return sorted([f for f in SOURCES_DIR.glob("*.json")])

def deconstruct_source(source_path):
    with open(source_path, "r") as f:
        data = json.load(f)
    description = data.get("title", source_path.stem)
    source_text = data["text"]
    payload = {
        "source_text": source_text,
        "source_type_hint": data.get("type", "article")
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    print(f"\n[bold cyan]Deconstructing:[/bold cyan] {description}")
    resp = requests.post(API_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        print(f"[red]Failed to deconstruct {source_path.name}: {resp.status_code} {resp.text}[/red]")
        return None
    result = resp.json()
    out_path = DECONSTRUCTIONS_DIR / f"deconstruction_{source_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path

def view_deconstruction(json_path):
    print(f"\n[bold green]Viewing deconstruction:[/bold green] {json_path.name}\n")
    subprocess.run(["python", VIEW_SCRIPT, "--file", str(json_path)])

def main():
    ensure_dirs()
    files = get_source_files()
    if not files:
        print(f"[yellow]No source files found in {SOURCES_DIR}/. Add .json files with 'title' and 'text' fields.[/yellow]")
        return
    for f in files:
        out_path = deconstruct_source(f)
        if out_path:
            view_deconstruction(out_path)
    print("\n[bold magenta]Batch deconstruction complete.[/bold magenta]")

if __name__ == "__main__":
    main() 