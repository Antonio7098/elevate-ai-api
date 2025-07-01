#!/usr/bin/env python3
"""
CLI tool for viewing LLM API usage statistics.
"""

import argparse
import json
from datetime import date, datetime
from app.core.usage_tracker import usage_tracker


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.6f}"
    else:
        return f"${cost:.4f}"


def display_summary(summary: dict):
    """Display usage summary in a formatted way."""
    print("\n" + "="*60)
    print("LLM API USAGE SUMMARY")
    print("="*60)
    
    print(f"Total Calls: {summary['total_calls']}")
    print(f"Total Tokens: {summary['total_tokens']:,}")
    print(f"Total Cost: {format_cost(summary['total_cost_usd'])}")
    print(f"Success Rate: {summary['success_rate']}%")
    
    if summary['by_provider']:
        print("\nBy Provider:")
        for provider, stats in summary['by_provider'].items():
            print(f"  {provider.upper()}: {stats['calls']} calls, {stats['tokens']:,} tokens, {format_cost(stats['cost_usd'])}")
    
    if summary['by_model']:
        print("\nBy Model:")
        for model, stats in summary['by_model'].items():
            print(f"  {model}: {stats['calls']} calls, {stats['tokens']:,} tokens, {format_cost(stats['cost_usd'])}")
    
    if summary['by_operation']:
        print("\nBy Operation:")
        for operation, stats in summary['by_operation'].items():
            print(f"  {operation}: {stats['calls']} calls, {stats['tokens']:,} tokens, {format_cost(stats['cost_usd'])}")


def display_recent_usage(usage_records: list, limit: int = 10):
    """Display recent usage records."""
    print(f"\n{'-'*60}")
    print(f"RECENT USAGE (Last {min(limit, len(usage_records))} records)")
    print("-"*60)
    
    for record in usage_records[-limit:]:
        timestamp = datetime.fromisoformat(record['timestamp'])
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"{record['provider'].upper()} | "
              f"{record['model']} | "
              f"{record['operation']} | "
              f"{record['total_tokens']} tokens | "
              f"{format_cost(record['estimated_cost_usd'])} | "
              f"{'✓' if record['success'] else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="View LLM API usage statistics")
    parser.add_argument("--summary", action="store_true", help="Show usage summary")
    parser.add_argument("--recent", action="store_true", help="Show recent usage records")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent records to show")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    # Default to summary if no specific command given
    if not args.summary and not args.recent:
        args.summary = True
    
    try:
        if args.summary:
            # Parse dates
            start = None
            end = None
            if args.start_date:
                start = date.fromisoformat(args.start_date)
            if args.end_date:
                end = date.fromisoformat(args.end_date)
            
            summary = usage_tracker.get_usage_summary(start, end)
            
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                display_summary(summary)
        
        if args.recent:
            recent_usage = usage_tracker.get_recent_usage(args.limit)
            
            if args.json:
                print(json.dumps(recent_usage, indent=2))
            else:
                display_recent_usage(recent_usage, args.limit)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 