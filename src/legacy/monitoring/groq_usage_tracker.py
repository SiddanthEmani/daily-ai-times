#!/usr/bin/env python3
"""
Groq API Usage Tracker

Tracks and reports Groq API usage for Daily AI Times pipeline testing.
Compatible with both direct aiohttp calls and Groq Python SDK.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class APICall:
    """Represents a single API call to Groq."""
    timestamp: str
    model: str
    endpoint: str
    request_tokens: int
    response_tokens: int
    total_tokens: int
    processing_time: float
    status_code: int
    success: bool
    agent: Optional[str] = None  # Which agent made the call
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[str] = None

@dataclass
class UsageSummary:
    """Summary of API usage during a test session."""
    session_start: str
    session_end: str
    total_duration: float
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_tokens: int
    total_request_tokens: int
    total_response_tokens: int
    models_used: Dict[str, int]
    endpoints_used: Dict[str, int]
    agents_used: Dict[str, int]  # New field for agent usage
    average_processing_time: float
    rate_limit_hits: int
    estimated_cost: float
    calls: List[APICall]

class GroqUsageTracker:
    """Tracks Groq API usage and generates detailed reports."""
    
    def __init__(self):
        """Initialize the usage tracker."""
        self.calls: List[APICall] = []
        self.session_start = datetime.now(timezone.utc)
        self.rate_limit_hits = 0
        
        # Groq pricing (as of 2025) - approximate values
        self.pricing = {
            'gemma2-9b-it': {'input': 0.00000027, 'output': 0.00000027},  # $0.27 per 1M tokens
            'llama3-8b-8192': {'input': 0.00000010, 'output': 0.00000010},  # $0.10 per 1M tokens
            'llama-3.1-8b-instant': {'input': 0.00000010, 'output': 0.00000010},
            'llama3-70b-8192': {'input': 0.00000059, 'output': 0.00000079},  # $0.59/$0.79 per 1M tokens
            'deepseek-r1-distill-llama-70b': {'input': 0.00000059, 'output': 0.00000079},  # Similar to llama3-70b-8192
            'default': {'input': 0.00000050, 'output': 0.00000050}  # Default fallback
        }
    
    def record_call(self, 
                   model: str,
                   endpoint: str = "chat/completions",
                   request_tokens: int = 0,
                   response_tokens: int = 0,
                   processing_time: float = 0.0,
                   status_code: int = 200,
                   success: bool = True,
                   agent: Optional[str] = None,
                   error_message: Optional[str] = None,
                   request_id: Optional[str] = None,
                   rate_limit_remaining: Optional[int] = None,
                   rate_limit_reset: Optional[str] = None) -> None:
        """Record an API call."""
        
        call = APICall(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            endpoint=endpoint,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=request_tokens + response_tokens,
            processing_time=processing_time,
            status_code=status_code,
            success=success,
            agent=agent,
            error_message=error_message,
            request_id=request_id,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset=rate_limit_reset
        )
        
        self.calls.append(call)
        
        if status_code == 429:
            self.rate_limit_hits += 1
            
        logger.debug(f"Recorded API call: {model} ({agent}) - {request_tokens + response_tokens} tokens - {processing_time:.2f}s")
    
    def estimate_cost(self, model: str, request_tokens: int, response_tokens: int) -> float:
        """Estimate the cost of an API call."""
        pricing = self.pricing.get(model, self.pricing['default'])
        input_cost = request_tokens * pricing['input']
        output_cost = response_tokens * pricing['output']
        return input_cost + output_cost
    
    def get_usage_summary(self) -> UsageSummary:
        """Generate a comprehensive usage summary."""
        session_end = datetime.now(timezone.utc)
        total_duration = (session_end - self.session_start).total_seconds()
        
        if not self.calls:
            return UsageSummary(
                session_start=self.session_start.isoformat(),
                session_end=session_end.isoformat(),
                total_duration=total_duration,
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                total_tokens=0,
                total_request_tokens=0,
                total_response_tokens=0,
                models_used={},
                endpoints_used={},
                agents_used={},
                average_processing_time=0.0,
                rate_limit_hits=0,
                estimated_cost=0.0,
                calls=[]
            )
        
        successful_calls = sum(1 for call in self.calls if call.success)
        failed_calls = len(self.calls) - successful_calls
        total_tokens = sum(call.total_tokens for call in self.calls)
        total_request_tokens = sum(call.request_tokens for call in self.calls)
        total_response_tokens = sum(call.response_tokens for call in self.calls)
        
        # Count models, endpoints, and agents
        models_used = {}
        endpoints_used = {}
        agents_used = {}
        for call in self.calls:
            models_used[call.model] = models_used.get(call.model, 0) + 1
            endpoints_used[call.endpoint] = endpoints_used.get(call.endpoint, 0) + 1
            if call.agent:
                agents_used[call.agent] = agents_used.get(call.agent, 0) + 1
        
        # Calculate average processing time for successful calls
        successful_times = [call.processing_time for call in self.calls if call.success]
        average_processing_time = sum(successful_times) / len(successful_times) if successful_times else 0.0
        
        # Estimate total cost
        total_cost = 0.0
        for call in self.calls:
            if call.success:
                total_cost += self.estimate_cost(call.model, call.request_tokens, call.response_tokens)
        
        return UsageSummary(
            session_start=self.session_start.isoformat(),
            session_end=session_end.isoformat(),
            total_duration=total_duration,
            total_calls=len(self.calls),
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_tokens=total_tokens,
            total_request_tokens=total_request_tokens,
            total_response_tokens=total_response_tokens,
            models_used=models_used,
            endpoints_used=endpoints_used,
            agents_used=agents_used,
            average_processing_time=average_processing_time,
            rate_limit_hits=self.rate_limit_hits,
            estimated_cost=total_cost,
            calls=self.calls
        )
    
    def save_usage_report(self, output_path: Path) -> None:
        """Save detailed usage report to JSON file."""
        summary = self.get_usage_summary()
        
        # Convert to JSON-serializable format
        report_data = asdict(summary)
        
        # Add rate limit analysis
        report_data['rate_limit_analysis'] = self._analyze_rate_limits()
        
        # Add efficiency metrics
        report_data['efficiency_metrics'] = self._calculate_efficiency_metrics(summary)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Usage report saved to {output_path}")
    
    def _analyze_rate_limits(self) -> Dict[str, Any]:
        """Analyze rate limit usage based on Groq limits."""
        if not self.calls:
            return {}
        
        # Get timing data
        call_times = [datetime.fromisoformat(call.timestamp.replace('Z', '+00:00')) for call in self.calls]
        if len(call_times) < 2:
            return {}
        
        duration_minutes = (call_times[-1] - call_times[0]).total_seconds() / 60
        calls_per_minute = len(self.calls) / duration_minutes if duration_minutes > 0 else 0
        
        # Calculate token usage rate
        total_tokens = sum(call.total_tokens for call in self.calls)
        tokens_per_minute = total_tokens / duration_minutes if duration_minutes > 0 else 0
        
        # Determine model limits (using most common model)
        models_used = {}
        for call in self.calls:
            models_used[call.model] = models_used.get(call.model, 0) + 1
        
        primary_model = max(models_used.items(), key=lambda x: x[1])[0] if models_used else 'unknown'
        
        # Model-specific limits (from groq-limits.md - updated July 2025)
        model_limits = {
            'gemma2-9b-it': {'rpm': 30, 'tpm': 15000, 'daily_tokens': 500000},
            'llama3-8b-8192': {'rpm': 30, 'tpm': 6000, 'daily_tokens': 500000},
            'llama-3.1-8b-instant': {'rpm': 30, 'tpm': 6000, 'daily_tokens': 500000},
            'llama3-70b-8192': {'rpm': 30, 'tpm': 6000, 'daily_tokens': 500000},
            'llama-3.3-70b-versatile': {'rpm': 30, 'tpm': 12000, 'daily_tokens': 100000},
            'deepseek-r1-distill-llama-70b': {'rpm': 30, 'tpm': 6000, 'daily_tokens': -1},
            'allam-2-7b': {'rpm': 30, 'tpm': 6000, 'daily_tokens': -1},
            'mistral-saba-24b': {'rpm': 30, 'tpm': 6000, 'daily_tokens': 500000},
        }
        
        limits = model_limits.get(primary_model, {'rpm': 30, 'tpm': 6000, 'daily_tokens': 500000})
        
        return {
            'primary_model': primary_model,
            'actual_rpm': round(calls_per_minute, 2),
            'actual_tpm': round(tokens_per_minute, 2),
            'limit_rpm': limits['rpm'],
            'limit_tpm': limits['tpm'],
            'rpm_utilization': round((calls_per_minute / limits['rpm']) * 100, 1),
            'tpm_utilization': round((tokens_per_minute / limits['tpm']) * 100, 1),
            'rate_limit_hits': self.rate_limit_hits,
            'daily_token_limit': limits['daily_tokens'],
            'daily_utilization': round((total_tokens / limits['daily_tokens']) * 100, 3) if limits['daily_tokens'] > 0 else 0
        }
    
    def _calculate_efficiency_metrics(self, summary: UsageSummary) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        if summary.total_calls == 0:
            return {
                'success_rate': 0,
                'average_tokens_per_call': 0,
                'tokens_per_second': 0,
                'calls_per_minute': 0,
                'input_output_ratio': 0,
                'cost_per_call': 0,
                'cost_per_1k_tokens': 0
            }
        
        return {
            'success_rate': round((summary.successful_calls / summary.total_calls) * 100, 1),
            'average_tokens_per_call': round(summary.total_tokens / summary.total_calls, 1),
            'tokens_per_second': round(summary.total_tokens / summary.total_duration, 1) if summary.total_duration > 0 else 0,
            'calls_per_minute': round(summary.total_calls / (summary.total_duration / 60), 1) if summary.total_duration > 0 else 0,
            'input_output_ratio': round(summary.total_request_tokens / summary.total_response_tokens, 2) if summary.total_response_tokens > 0 else 0,
            'cost_per_call': round(summary.estimated_cost / summary.successful_calls, 6) if summary.successful_calls > 0 else 0,
            'cost_per_1k_tokens': round((summary.estimated_cost / summary.total_tokens) * 1000, 4) if summary.total_tokens > 0 else 0
        }

# Global instance for easy access
usage_tracker = GroqUsageTracker()
