#!/usr/bin/env python3
"""
Intelligent Article Distribution System for Daily AI Times
Advanced workload distribution with comprehensive token analysis and optimization.

COMPREHENSIVE DISTRIBUTION STRATEGY:
- Analyzes system prompts, article content, and expected output tokens
- Considers TPM, RPM, and daily token budgets with 95% utilization
- Implements smart load balancing based on real agent capabilities
- Optimizes for concurrent processing and pipeline efficiency
- Accounts for model-specific characteristics and batch sizes

Key Features:
1. Multi-factor token estimation (input + system + output)
2. Agent capacity modeling with real-time constraints
3. Intelligent workload balancing and fallback handling
4. Performance monitoring and optimization feedback
5. Pipeline runs optimization (6 runs per day)
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentCapacity:
    """Agent capacity and characteristics (per-day focus, no per-run budgets)."""
    agent_name: str
    model_name: str
    tpm_limit: int
    rpm_limit: int
    daily_token_limit: int
    tokens_used_today: int
    batch_size: int
    processing_speed: float  # articles per minute (derived from TPM/RPM & batch size)
    current_load: float  # 0.0 to 1.0
    safety_margin: float

    @property
    def effective_tpm(self) -> int:
        return int(self.tpm_limit * self.safety_margin)

    @property
    def remaining_daily_budget(self) -> float:
        if self.daily_token_limit == -1:
            return float('inf')
        return max(0, self.daily_token_limit - self.tokens_used_today)


@dataclass
class TokenEstimate:
    """Comprehensive token estimation for article processing."""
    system_prompt_tokens: int
    article_content_tokens: int
    expected_output_tokens: int
    total_tokens: int
    
    @classmethod
    def estimate_for_batch(cls, articles: List[Dict[str, Any]], 
                          system_prompt: str, agent_name: str) -> 'TokenEstimate':
        """Estimate tokens for a batch of articles."""
        # System prompt tokens (shared across batch)
        system_tokens = cls._estimate_text_tokens(system_prompt)
        
        # Article content tokens
        article_tokens = 0
        for article in articles:
            title = article.get('title', '')[:70]  # Truncated for processing
            description = article.get('description', '')[:120]
            content = f"{title} {description}"
            article_tokens += cls._estimate_text_tokens(content)
        
        # Expected output tokens (model-specific)
        output_tokens = cls._estimate_output_tokens(len(articles), agent_name)
        
        total = system_tokens + article_tokens + output_tokens
        
        return cls(
            system_prompt_tokens=system_tokens,
            article_content_tokens=article_tokens,
            expected_output_tokens=output_tokens,
            total_tokens=total
        )
    
    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Estimate tokens from text (approximate: 1 token ≈ 0.75 words)."""
        if not text:
            return 0
        word_count = len(text.split())
        return int(word_count * 1.33)  # Conservative estimate
    
    @staticmethod
    def _estimate_output_tokens(article_count: int, agent_name: str) -> int:
        """Estimate output tokens based on model and article count."""
        # Base tokens per article for JSON response
        base_tokens_per_article = 35
        
        # Model-specific adjustments
        if 'gemma2' in agent_name.lower():
            multiplier = 1.2  # Gemma2 tends to be more verbose
        elif 'llama-4' in agent_name.lower():
            multiplier = 1.1  # Scout models are detailed but concise
        else:
            multiplier = 1.0
        
        return int(article_count * base_tokens_per_article * multiplier)


class ArticleDistributor:
    """Article distributor (Phase 1 bulk) using daily + instantaneous limits.

    Heuristic:
        Allocate proportionally to each agent's safe capacity where safe capacity is the
        minimum articles the agent can process now without breaching ANY of:
            - TPM (tokens per minute) @ utilization target over a 10‑minute planning window
            - RPM (requests per minute) converted to article capacity via batch size
            - Remaining DAILY token budget (no per‑run concept)

    Steps:
        1. Build AgentCapacity objects
        2. For each agent compute safe_capacity = min(by_tpm, by_rpm, by_remaining_daily)
        3. Allocate proportionally; redistribute leftovers respecting headroom
        4. Fallback to round‑robin if all capacities are zero
    """

    def __init__(self):
        """Initialize distributor defaults."""
        self.utilization_target = 0.95
        self.target_window_minutes = 10  # planning horizon window for TPM -> articles
        self.conservative_tokens_per_article = 280  # conservative per article token estimate
        self.last_unassigned = []

    def distribute_articles(self, articles: List[Dict[str, Any]], agents: Dict[str, Any], system_prompt: str = "") -> Dict[str, List[Dict[str, Any]]]:
        if not articles or not agents:
            return {}

        logger.info(f"Distributing {len(articles)} articles across {len(agents)} agents (single strategy)")

        capacities = self._analyze_agent_capacities(agents)
        est_tokens_per_article = self.conservative_tokens_per_article

        # Compute safe capacity (max articles) per agent
        per_agent_caps = {}
        total_weight = 0
        for cap in capacities:
            # Effective limits
            effective_tpm = cap.effective_tpm
            remaining_daily = cap.remaining_daily_budget
            rpm_limit = cap.rpm_limit
            batch_size = cap.batch_size or 1

            # Constraint translations
            by_tpm = (effective_tpm * self.target_window_minutes) // (est_tokens_per_article) if effective_tpm > 0 else 0
            # Each batch consumes batch_size * est_tokens_per_article tokens
            max_batches_rpm_window = rpm_limit * self.target_window_minutes  # over 10 minutes
            by_rpm = max_batches_rpm_window * batch_size  # articles
            by_remaining_daily = (remaining_daily // est_tokens_per_article) if remaining_daily != float('inf') else float('inf')

            numeric_limits = [l for l in [by_tpm, by_rpm, by_remaining_daily] if l != float('inf')]
            if not numeric_limits:
                safe_capacity = len(articles)  # unlimited agent
            else:
                safe_capacity = int(max(0, min(numeric_limits)))

            per_agent_caps[cap.agent_name] = {
                'safe_capacity': safe_capacity,
                'capacity_obj': cap
            }
            total_weight += safe_capacity
            logger.info(f"Agent {cap.agent_name} safe capacity: {safe_capacity} articles (TPM:{by_tpm}, RPM:{by_rpm}, DailyRemain:{by_remaining_daily})")

        if total_weight == 0:
            logger.warning("All agents report zero capacity; falling back to round-robin")
            return self._emergency_distribution(articles, agents)

        # Proportional allocation with hard capacity cap
        total_requested = len(articles)
        total_safe_capacity = sum(info['safe_capacity'] for info in per_agent_caps.values())
        distribution: Dict[str, List[Dict[str, Any]]] = {cap.agent_name: [] for cap in capacities}

        logger.info(f"Capacity summary: requested={total_requested}, aggregate_safe_capacity={total_safe_capacity}")

        remaining = total_requested
        provisional_alloc: Dict[str, int] = {}
        for name, info in per_agent_caps.items():
            if remaining <= 0:
                provisional_alloc[name] = 0
                continue
            if total_safe_capacity == 0:
                provisional_alloc[name] = 0
                continue
            weight = info['safe_capacity'] / total_safe_capacity if total_safe_capacity > 0 else 0
            alloc = int(weight * total_requested)
            # Clamp to safe capacity
            alloc = min(alloc, info['safe_capacity'])
            # Don't allocate more than remaining
            alloc = min(alloc, remaining)
            provisional_alloc[name] = alloc
            remaining -= alloc

        # Leftover distribution only to agents with headroom
        if remaining > 0:
            for name, info in sorted(per_agent_caps.items(), key=lambda x: x[1]['safe_capacity'] - provisional_alloc.get(x[0], 0), reverse=True):
                if remaining <= 0:
                    break
                headroom = info['safe_capacity'] - provisional_alloc.get(name, 0)
                if headroom <= 0:
                    continue
                take = min(headroom, remaining)
                provisional_alloc[name] += take
                remaining -= take

        # Any residual remaining means we exceeded aggregate capacity; mark unassigned
        unassigned_articles: List[Dict[str, Any]] = []
        if remaining > 0:
            logger.warning(f"Insufficient capacity: {remaining} articles unassigned (no agent headroom left)")
            # We'll place the tail of the list into _unassigned bucket
            # Determine how many articles assigned so far to slice correctly later

        # Slice article list deterministically following allocation order
        cursor = 0
        for name in provisional_alloc.keys():
            count = provisional_alloc[name]
            assigned_slice = articles[cursor:cursor+count]
            distribution[name] = assigned_slice
            cursor += count
            logger.info(f"Assign {count} articles to {name} (capacity {per_agent_caps[name]['safe_capacity']})")

        if cursor < len(articles):
            unassigned_articles = articles[cursor:]
            self.last_unassigned = unassigned_articles
            logger.info(f"Unassigned (capacity exceeded): {len(unassigned_articles)} articles retained for later processing")
        else:
            self.last_unassigned = []

        # Validation (only agent keys)
        total_assigned = sum(len(v) for v in distribution.values())
        if total_assigned > total_requested:
            logger.error(f"Distribution over-allocation detected: {total_assigned} > {total_requested}")
        elif total_assigned < total_requested:
            logger.info(f"Distributed {total_assigned}/{total_requested} articles; {total_requested - total_assigned} unassigned")

        return distribution
    
    def _analyze_agent_capacities(self, agents: Dict[str, Any]) -> List[AgentCapacity]:
        """Analyze comprehensive agent capacities and constraints."""
        capacities = []
        
        for agent_name, agent in agents.items():
            # Extract agent characteristics
            model_limits = agent.model_limits
            tpm_limit = model_limits.get('tpm', 6000)
            rpm_limit = model_limits.get('rpm', 30)
            daily_limit = model_limits.get('daily_tokens', 500000)
            
            # Get current usage
            tokens_used = getattr(agent, 'daily_tokens_used', 0)
            batch_size = getattr(agent, 'adaptive_batch_size', 10)
            safety_margin = getattr(agent, 'safety_margin', 0.95)
            
            # Calculate processing speed (articles per minute)
            estimated_tokens_per_article = 280  # More accurate estimate including all components
            tokens_per_batch = batch_size * estimated_tokens_per_article
            batches_per_minute = min(
                (tpm_limit * safety_margin) / tokens_per_batch,
                (rpm_limit * safety_margin) / 1  # 1 request per batch
            )
            processing_speed = batches_per_minute * batch_size
            
            capacity = AgentCapacity(
                agent_name=agent_name,
                model_name=agent.model_name,
                tpm_limit=tpm_limit,
                rpm_limit=rpm_limit,
                daily_token_limit=daily_limit,
                tokens_used_today=tokens_used,
                batch_size=batch_size,
                processing_speed=processing_speed,
                current_load=0.0,
                safety_margin=safety_margin
            )
            
            capacities.append(capacity)
            
            logger.debug(f"Agent {agent_name} capacity: {processing_speed:.1f} articles/min, daily_limit={daily_limit:,}")
        # Sort by processing speed (highest first) for stable deterministic ordering
        capacities.sort(key=lambda x: x.processing_speed, reverse=True)
        return capacities
    
    def _emergency_distribution(self, articles: List[Dict[str, Any]], 
                              agents: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Emergency fallback distribution when system capacity is insufficient."""
        logger.warning("Using emergency distribution - system capacity insufficient")
        
        # Simple round-robin distribution
        distribution = {name: [] for name in agents.keys()}
        agent_names = list(agents.keys())
        
        for i, article in enumerate(articles):
            agent_name = agent_names[i % len(agent_names)]
            distribution[agent_name].append(article)
        
        return distribution
    
    def _validate_distribution(self, distribution: Dict[str, List[Dict[str, Any]]], 
                             original_articles: List[Dict[str, Any]],
                             capacities: List[AgentCapacity]) -> None:
        """Validate the distribution and log comprehensive metrics."""
        total_assigned = sum(len(articles) for articles in distribution.values())
        
        if total_assigned != len(original_articles):
            logger.error(f"Distribution validation failed: {total_assigned} assigned vs "
                        f"{len(original_articles)} original articles")
        
        # Log distribution metrics
        total_estimated_tokens = 0
        max_processing_time = 0
        
        for agent_name, articles in distribution.items():
            if not articles:
                continue
                
            capacity = next((c for c in capacities if c.agent_name == agent_name), None)
            if not capacity:
                continue
            
            estimated_tokens = len(articles) * 280  # Conservative estimate
            processing_time = len(articles) / capacity.processing_speed if capacity.processing_speed > 0 else 0
            
            total_estimated_tokens += estimated_tokens
            max_processing_time = max(max_processing_time, processing_time)
            
            logger.info(f"Final: {agent_name} -> {len(articles)} articles, "
                       f"{estimated_tokens:,} tokens, {processing_time:.1f}min")
        
        # System-wide metrics
        total_system_capacity = sum(cap.effective_tpm for cap in capacities)
        system_utilization = (total_estimated_tokens / (total_system_capacity * 10)) * 100  # 10-minute window
        
        logger.info(f"Distribution complete: {total_assigned} articles, "
                   f"{total_estimated_tokens:,} total tokens, "
                   f"{system_utilization:.1f}% system utilization, "
                   f"~{max_processing_time:.1f}min max processing time")
    
    def get_distribution_metrics(self, distribution: Dict[str, List[Dict[str, Any]]],
                               capacities: List[AgentCapacity]) -> Dict[str, Any]:
        """Get comprehensive distribution metrics for monitoring."""
        metrics = {
            'total_agents': len(distribution),
            'total_articles': sum(len(articles) for articles in distribution.values()),
            'agent_utilization': {},
            'load_balance_score': 0.0,
            'estimated_completion_time': 0.0
        }
        
        processing_times = []
        for agent_name, articles in distribution.items():
            capacity = next((c for c in capacities if c.agent_name == agent_name), None)
            if capacity and articles:
                processing_time = len(articles) / capacity.processing_speed
                processing_times.append(processing_time)
                
                metrics['agent_utilization'][agent_name] = {
                    'articles': len(articles),
                    'estimated_tokens': len(articles) * 280,
                    'processing_time': processing_time,
                    'load_percentage': capacity.current_load * 100
                }
        
        # Calculate load balance score (lower variance = better balance)
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            variance = sum((t - avg_time) ** 2 for t in processing_times) / len(processing_times)
            metrics['load_balance_score'] = 1.0 / (1.0 + variance)  # Score between 0 and 1
            metrics['estimated_completion_time'] = max(processing_times)
        
        return metrics
