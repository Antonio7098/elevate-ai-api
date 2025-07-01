"""
Usage tracking for LLM API calls.

This module tracks API usage, token consumption, and estimated costs
for transparency and cost control.
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from app.core.config import settings


@dataclass
class UsageRecord:
    """Record of a single LLM API call."""
    timestamp: str
    model: str
    provider: str  # "google" or "openai"
    operation: str  # e.g., "extract_sections", "extract_entities"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    success: bool
    error_message: Optional[str] = None
    request_id: Optional[str] = None


class UsageTracker:
    """Tracks and reports LLM API usage."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the usage tracker.
        
        Args:
            log_file: Path to the JSON log file. If None, uses default location.
        """
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "llm_usage.json"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            self._write_log([])
    
    def log_usage(self, record: UsageRecord) -> None:
        """Log a usage record to the JSON file.
        
        Args:
            record: The usage record to log
        """
        try:
            # Read existing logs
            logs = self._read_log()
            
            # Add new record
            logs.append(asdict(record))
            
            # Write back to file
            self._write_log(logs)
            
        except Exception as e:
            print(f"Warning: Failed to log usage: {e}")
    
    def get_usage_summary(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, Any]:
        """Get a summary of usage statistics.
        
        Args:
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            
        Returns:
            Dictionary with usage summary statistics
        """
        logs = self._read_log()
        
        # Filter by date if specified
        if start_date or end_date:
            filtered_logs = []
            for log in logs:
                log_date = datetime.fromisoformat(log["timestamp"]).date()
                if start_date and log_date < start_date:
                    continue
                if end_date and log_date > end_date:
                    continue
                filtered_logs.append(log)
            logs = filtered_logs
        
        if not logs:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "success_rate": 0.0,
                "by_model": {},
                "by_operation": {},
                "by_provider": {}
            }
        
        # Calculate summary statistics
        total_calls = len(logs)
        total_tokens = sum(log["total_tokens"] for log in logs)
        total_cost = sum(log["estimated_cost_usd"] for log in logs)
        successful_calls = sum(1 for log in logs if log["success"])
        success_rate = (successful_calls / total_calls) * 100 if total_calls > 0 else 0
        
        # Group by model
        by_model = {}
        for log in logs:
            model = log["model"]
            if model not in by_model:
                by_model[model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0
                }
            by_model[model]["calls"] += 1
            by_model[model]["tokens"] += log["total_tokens"]
            by_model[model]["cost_usd"] += log["estimated_cost_usd"]
        
        # Group by operation
        by_operation = {}
        for log in logs:
            operation = log["operation"]
            if operation not in by_operation:
                by_operation[operation] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0
                }
            by_operation[operation]["calls"] += 1
            by_operation[operation]["tokens"] += log["total_tokens"]
            by_operation[operation]["cost_usd"] += log["estimated_cost_usd"]
        
        # Group by provider
        by_provider = {}
        for log in logs:
            provider = log["provider"]
            if provider not in by_provider:
                by_provider[provider] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0
                }
            by_provider[provider]["calls"] += 1
            by_provider[provider]["tokens"] += log["total_tokens"]
            by_provider[provider]["cost_usd"] += log["estimated_cost_usd"]
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "success_rate": round(success_rate, 2),
            "by_model": by_model,
            "by_operation": by_operation,
            "by_provider": by_provider
        }
    
    def get_recent_usage(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent usage records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent usage records
        """
        logs = self._read_log()
        return logs[-limit:] if logs else []
    
    def _read_log(self) -> List[Dict[str, Any]]:
        """Read the usage log file."""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _write_log(self, logs: List[Dict[str, Any]]) -> None:
        """Write logs to the usage log file."""
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)


# Global usage tracker instance
usage_tracker = UsageTracker()


def estimate_google_ai_cost(input_tokens: int, output_tokens: int, model: str = "gemini-1.5-flash") -> float:
    """Estimate cost for Google AI API call.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for pricing
        
    Returns:
        Estimated cost in USD
    """
    # Google AI pricing (as of 2024, may need updates)
    pricing = {
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},  # per 1K tokens
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},      # per 1K tokens
        "gemini-pro": {"input": 0.0005, "output": 0.0015}           # per 1K tokens
    }
    
    model_pricing = pricing.get(model, pricing["gemini-1.5-flash"])
    
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return input_cost + output_cost


def log_llm_call(
    model: str,
    provider: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    success: bool = True,
    error_message: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """Log an LLM API call.
    
    Args:
        model: Model name
        provider: Provider name ("google" or "openai")
        operation: Operation name (e.g., "extract_sections")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        success: Whether the call was successful
        error_message: Error message if call failed
        request_id: Optional request ID for tracing
    """
    total_tokens = input_tokens + output_tokens
    
    if provider == "google":
        estimated_cost = estimate_google_ai_cost(input_tokens, output_tokens, model)
    else:
        # Default to Google pricing for unknown providers
        estimated_cost = estimate_google_ai_cost(input_tokens, output_tokens, model)
    
    record = UsageRecord(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        provider=provider,
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
        success=success,
        error_message=error_message,
        request_id=request_id
    )
    
    usage_tracker.log_usage(record) 