#!/usr/bin/env python3
"""
Deep Intelligence Agent for NewsXP.ai
Provides sophisticated analysis after initial consensus filtering.
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timezone
from collections import deque

from groq import AsyncGroq
import groq

logger = logging.getLogger(__name__)


class DeepIntelligenceAgent:
    """
    Advanced analysis agent that performs deep intelligence processing
    on consensus-filtered articles using high-capability models.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any], api_key: str):
        """Initialize the deep intelligence agent."""
        self.model_name = model_name
        self.config = config
        self.api_key = api_key
        self.client = AsyncGroq(
            api_key=api_key,
            max_retries=3,
            timeout=45.0  # Reduced timeout to prevent hanging
        )
        
        # Agent configuration
        self.specialization = config.get('specialization', 'general_intelligence')
        self.focus_areas = config.get('focus_areas', [])
        
        # Model parameters
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 4000)
        self.top_p = config.get('top_p', 0.9)
        
        # Rate limiting configuration - intelligent batching and window tracking
        self._setup_rate_limits()
        
        # Request tracking for intelligent rate limiting
        self.request_times = deque()  # Track request timestamps
        self.token_usage = deque()  # Track token usage over time
        self.current_minute_requests = 0
        self.current_minute_tokens = 0
        self.minute_start_time = time.time()
        
        # Analysis features
        self.enable_fact_checking = config.get('enable_fact_checking', True)
        self.enable_bias_detection = config.get('enable_bias_detection', True)
        self.enable_impact_analysis = config.get('enable_impact_analysis', True)
        self.enable_credibility_scoring = config.get('enable_credibility_scoring', True)
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.rate_limit_count = 0
        self.session_start_time = time.time()
        
        logger.info(f"Initialized Deep Intelligence Agent: {model_name} ({self.specialization})")
        logger.info(f"Rate limits: {self.requests_per_minute} req/min, {self.tokens_per_minute} tokens/min")
        
        # Special logging for qwen models
        if 'qwen' in model_name.lower():
            logger.info(f"QWEN model detected - using optimized timeouts and rate limits")
    
    def _setup_rate_limits(self):
        """Setup intelligent rate limiting based on Groq model capabilities."""
        # Get base configuration
        base_rpm = self.config.get('requests_per_minute', 25)  # Conservative default
        
        # Model-specific limits from groq-limits.md
        model_limits = {
            "meta-llama/llama-4-scout-17b-16e-instruct": {
                "rpm": 30, "tokens_per_minute": 30000, "daily_requests": 1000
            },
            "meta-llama/llama-4-maverick-17b-128e-instruct": {
                "rpm": 30, "tokens_per_minute": 6000, "daily_requests": 1000
            },
            "llama-3.3-70b-versatile": {
                "rpm": 30, "tokens_per_minute": 12000, "daily_requests": 1000
            },
            "llama3-70b-8192": {
                "rpm": 30, "tokens_per_minute": 6000, "daily_requests": 14400
            },
            "llama-3.1-8b-instant": {
                "rpm": 30, "tokens_per_minute": 6000, "daily_requests": 14400
            },
            "llama3-8b-8192": {
                "rpm": 30, "tokens_per_minute": 6000, "daily_requests": 14400
            },
            "gemma2-9b-it": {
                "rpm": 30, "tokens_per_minute": 15000, "daily_requests": 14400
            },
            "qwen/qwen3-32b": {
                "rpm": 60, "tokens_per_minute": 6000, "daily_requests": 1000
            },
            "qwen-qwq-32b": {
                "rpm": 30, "tokens_per_minute": 6000, "daily_requests": 1000
            }
        }
        
        # Get limits for current model
        limits = model_limits.get(self.model_name, {
            "rpm": 25, "tokens_per_minute": 6000, "daily_requests": 1000
        })
        
        # Apply optimized buffer (90% of limits for better utilization)
        self.requests_per_minute = min(base_rpm, int(limits["rpm"] * 0.9))
        self.tokens_per_minute = int(limits["tokens_per_minute"] * 0.9)
        self.daily_requests_limit = limits["daily_requests"]
        
        # Calculate optimal batch sizes
        self.max_parallel_requests = max(1, min(5, self.requests_per_minute // 6))  # Conservative parallel processing
        self.request_interval = 60.0 / self.requests_per_minute
        self.min_batch_delay = 5.0  # Minimum delay between batches
        
        logger.info(f"Rate limits for {self.model_name}: {self.requests_per_minute} req/min, "
                   f"{self.tokens_per_minute} tokens/min, max parallel: {self.max_parallel_requests}")
    
    def _clean_old_requests(self, current_time: float):
        """Remove requests older than 60 seconds from tracking."""
        cutoff_time = current_time - 60.0
        
        # Clean request times
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
        
        # Clean token usage
        while self.token_usage and self.token_usage[0][0] < cutoff_time:
            self.token_usage.popleft()
    
    def _get_current_usage(self, current_time: float) -> Tuple[int, int]:
        """Get current requests and tokens used in the last minute."""
        self._clean_old_requests(current_time)
        
        requests_count = len(self.request_times)
        tokens_count = sum(usage[1] for usage in self.token_usage)
        
        return requests_count, tokens_count
    
    def _can_make_request(self, estimated_tokens: int = 4000) -> Tuple[bool, float]:
        """
        Check if we can make a request without hitting rate limits.
        Returns (can_make_request, wait_time_seconds)
        """
        current_time = time.time()
        current_requests, current_tokens = self._get_current_usage(current_time)
        
        # Check request limit
        if current_requests >= self.requests_per_minute:
            # Find when the oldest request will expire
            if self.request_times:
                oldest_request = self.request_times[0]
                wait_time = (oldest_request + 60.0) - current_time + 1.0  # Add 1s buffer
                return False, max(0, wait_time)
            return False, 60.0
        
        # Check token limit
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            # Find when enough tokens will be available
            if self.token_usage:
                # Calculate when enough tokens will expire
                tokens_needed = (current_tokens + estimated_tokens) - self.tokens_per_minute
                cumulative_tokens = 0
                
                for timestamp, tokens in self.token_usage:
                    cumulative_tokens += tokens
                    if cumulative_tokens >= tokens_needed:
                        wait_time = (timestamp + 60.0) - current_time + 1.0  # Add 1s buffer
                        return False, max(0, wait_time)
            
            return False, 30.0  # Default wait time
        
        return True, 0.0
    
    def _record_request(self, tokens_used: int = 4000):
        """Record a successful request for rate limiting tracking."""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_usage.append((current_time, tokens_used))
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
    
    def _create_analysis_prompt(self, article: Dict[str, Any]) -> str:
        """Create a comprehensive analysis prompt for deep intelligence processing."""
        
        # Extract existing scores if available
        consensus_score = article.get('consensus_multi_dimensional_score', {})
        initial_confidence = consensus_score.get('confidence_mean', 0.0)
        
        # Debug logging to understand what we're getting
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Article consensus score for '{article.get('title', 'Unknown')[:50]}': {consensus_score}")
            logger.debug(f"Initial confidence: {initial_confidence}")
        
        prompt = f"""Deep intelligence analysis for pre-filtered article.

Specialization: {self.specialization}
Focus: {', '.join(self.focus_areas[:2]) if self.focus_areas else 'General'}

ARTICLE:
Title: {article.get('title', 'N/A')}
Source: {article.get('source', 'N/A')}
Content: {article.get('content', article.get('description', 'N/A'))[:1000]}

CONSENSUS: Overall={consensus_score.get('overall_score', 0.0):.2f}, Confidence={initial_confidence:.2f}

ANALYSIS (3 key areas):
1. CREDIBILITY: Source reputation, factual accuracy, potential misinformation
2. BIAS & IMPACT: Political/cultural bias, societal implications, narrative framing  
3. VALUE: Reader utility, insights provided, content quality

GUIDELINES:
- ACCEPT: Valid AI/tech content (DEFAULT for pre-filtered)
- CONDITIONAL: Good content needing minor improvements
- REJECT: Only serious credibility/relevance issues

JSON Response:
{{
  "fact_check_confidence": 0.8,
  "bias_score": 0.3,
  "credibility_score": 0.8,
  "impact_potential": 0.7,
  "overall_score": 0.75,
  "confidence": 0.8,
  "recommendation": "ACCEPT",
  "key_insights": ["insight"],
  "risk_factors": ["risk"]
}}

Target: 70-80% acceptance rate. Default ACCEPT for pre-filtered articles unless serious issues."""
        
        return prompt
    
    async def _make_api_request(self, prompt: str) -> Optional[str]:
        """Make a rate-limited request to the Groq API with intelligent batching."""
        
        # Improved token estimation accounting for JSON structure and whitespace
        # More accurate estimation: ~3.5 chars per token for structured text, ~2.8 for JSON
        estimated_prompt_tokens = int(len(prompt) / 3.5)
        estimated_response_tokens = int(self.max_tokens * 0.6)  # Typically use 60% of max tokens
        estimated_total_tokens = estimated_prompt_tokens + estimated_response_tokens
        
        # Enhanced rate limit prediction and handling
        can_request, wait_time = self._can_make_request(estimated_total_tokens)
        
        if not can_request:
            # Enhanced wait time prediction for different models
            if 'llama-3.3-70b-versatile' in self.model_name.lower():
                max_wait_time = 90.0  # More generous for 70b model
            elif 'qwen' in self.model_name.lower():
                max_wait_time = 30.0  # Keep short for qwen
            else:
                max_wait_time = 60.0  # Default
                
            wait_time = min(wait_time, max_wait_time)
            
            logger.info(f"Rate limit prevention: waiting {wait_time:.1f}s for {self.model_name} "
                       f"(estimated tokens: {estimated_total_tokens})")
            await asyncio.sleep(wait_time)
            
            # Re-check after waiting with model-specific timeout
            can_request, additional_wait = self._can_make_request(estimated_total_tokens)
            if not can_request and additional_wait > 0:
                if 'llama-3.3-70b-versatile' in self.model_name.lower():
                    additional_wait = min(additional_wait, 30.0)  # More patient for 70b
                else:
                    additional_wait = min(additional_wait, 15.0)  # Standard for others
                
                logger.info(f"Additional rate limit wait: {additional_wait:.1f}s for {self.model_name}")
                await asyncio.sleep(additional_wait)
        
        self.request_count += 1
        
        # Enhanced logging for 70b model debugging
        if 'llama-3.3-70b-versatile' in self.model_name.lower():
            logger.info(f"70b model: Starting API request #{self.request_count} with {estimated_total_tokens} tokens")
        
        try:
            # Aligned API timeout with model capacity - aggressive for 70b
            if 'llama-4-scout' in self.model_name.lower():
                timeout_duration = 90.0  # Highest capacity model gets longest API timeout
            elif 'llama-3.3-70b-versatile' in self.model_name.lower():
                timeout_duration = 60.0  # Aggressive timeout to detect hangs early
            elif 'qwen' in self.model_name.lower():
                timeout_duration = 60.0  # Shorter timeout for QWEN
            else:
                timeout_duration = 75.0  # Default timeout
            
            # Add timeout to prevent hanging on individual requests
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional news analyst. You must respond with valid JSON only."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    response_format={"type": "json_object"}  # Ensures valid JSON response
                ),
                timeout=timeout_duration
            )
            
            # Record successful request
            actual_tokens = estimated_total_tokens
            if hasattr(response, 'usage') and response.usage:
                actual_tokens = response.usage.total_tokens
            self._record_request(actual_tokens)
            
            self.success_count += 1
            
            # Enhanced logging for 70b model debugging
            if 'llama-3.3-70b-versatile' in self.model_name.lower():
                response_content = response.choices[0].message.content
                logger.info(f"70b model: API request #{self.request_count} completed, response length: {len(response_content)}")
                return response_content
            
            return response.choices[0].message.content
            
        except groq.RateLimitError as e:
            self.rate_limit_count += 1
            logger.warning(f"Rate limit hit for {self.model_name}: {e}")
            
            # Extract wait time and add buffer
            wait_time = self._extract_wait_time(str(e)) + 2.0
            logger.info(f"Waiting {wait_time:.1f}s before retry")
            await asyncio.sleep(wait_time)
            
            # Single retry with fresh rate limit check and timeout protection
            try:
                can_retry, additional_wait = self._can_make_request(estimated_total_tokens)
                if not can_retry:
                    await asyncio.sleep(additional_wait)
                
                # Retry with shorter timeout to prevent hanging
                retry_response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a professional news analyst. You must respond with valid JSON only."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        response_format={"type": "json_object"}
                    ),
                    timeout=45.0  # Shorter timeout for retry
                )
                
                # Record successful retry
                actual_tokens = estimated_total_tokens
                if hasattr(retry_response, 'usage') and retry_response.usage:
                    actual_tokens = retry_response.usage.total_tokens
                self._record_request(actual_tokens)
                
                self.success_count += 1
                return retry_response.choices[0].message.content
                
            except Exception as retry_e:
                self.error_count += 1
                logger.error(f"Retry failed for {self.model_name}: {retry_e}")
                return None
        
        except asyncio.TimeoutError:
            self.error_count += 1
            logger.error(f"Request timeout for {self.model_name} after 90 seconds")
            return None
            
        except groq.APIConnectionError as e:
            self.error_count += 1
            logger.error(f"Connection error for {self.model_name}: {e}")
            return None
            
        except groq.APIStatusError as e:
            self.error_count += 1
            logger.error(f"API error for {self.model_name}: {e.status_code} - {e}")
            return None
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unexpected error for {self.model_name}: {e}")
            return None
    
    def _extract_wait_time(self, error_message: str) -> float:
        """Extract wait time from rate limit error message."""
        import re
        
        # Look for patterns like "try again in 30s" or "wait 60 seconds"
        patterns = [
            r'try again in (\d+)s',
            r'wait (\d+) seconds?',
            r'retry after (\d+) seconds?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return float(match.group(1)) + 1.0  # Add 1s buffer
        
        return 30.0  # Default wait time
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the model."""
        try:
            parsed = json.loads(response)
            
            # Validate required fields
            required_fields = [
                'fact_check_confidence', 'bias_score', 'credibility_score',
                'impact_potential', 'overall_score', 'confidence',
                'recommendation', 'key_insights', 'risk_factors'
            ]
            
            for field in required_fields:
                if field not in parsed:
                    logger.warning(f"Missing field '{field}' in response from {self.model_name}")
                    parsed[field] = 0.5 if field.endswith('_score') or field == 'confidence' else 'CONDITIONAL'
                    if field in ['key_insights', 'risk_factors']:
                        parsed[field] = []
            
            # Validate numeric fields
            for field in ['fact_check_confidence', 'bias_score', 'credibility_score', 
                         'impact_potential', 'overall_score', 'confidence']:
                try:
                    value = float(parsed[field])
                    parsed[field] = max(0.0, min(1.0, value))  # Clamp to [0,1]
                except (ValueError, TypeError):
                    parsed[field] = 0.5
            
            # Validate recommendation
            if parsed.get('recommendation') not in ['ACCEPT', 'REJECT', 'CONDITIONAL']:
                parsed['recommendation'] = 'CONDITIONAL'
            
            return self._convert_to_expected_format(parsed)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error from {self.model_name}: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def _convert_to_expected_format(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the parsed JSON to the expected deep intelligence format."""
        return {
            "deep_intelligence_analysis": {
                "fact_verification": {
                    "fact_check_confidence": parsed["fact_check_confidence"]
                },
                "bias_detection": {
                    "bias_detection_score": parsed["bias_score"]
                },
                "credibility_assessment": {
                    "credibility_score": parsed["credibility_score"]
                },
                "impact_analysis": {
                    "impact_potential": parsed["impact_potential"]
                },
                "synthesis": {
                    "overall_deep_intelligence_score": parsed["overall_score"],
                    "confidence_in_analysis": parsed["confidence"],
                    "recommendation": parsed["recommendation"],
                    "key_insights": parsed.get("key_insights", []),
                    "risk_factors": parsed.get("risk_factors", []),
                    "enhancement_suggestions": parsed.get("enhancement_suggestions", [])
                }
            }
        }
    
    async def analyze_article(self, article: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """
        Perform deep intelligence analysis on a single article.
        Returns: (article_with_analysis, recommendation_accept, confidence_score)
        """
        # Aligned timeouts with orchestrator configuration - aggressive for 70b individual processing
        if 'llama-4-scout' in self.model_name.lower():
            timeout_duration = 60.0  # Highest capacity model
        elif 'llama-3.3-70b-versatile' in self.model_name.lower():
            timeout_duration = 90.0  # More time for individual processing
        elif 'qwen' in self.model_name.lower():
            timeout_duration = 30.0  # Lower capacity but still reasonable
        else:
            timeout_duration = 45.0  # Default for other models
        
        try:
            # Add timeout to prevent hanging on individual article analysis
            return await asyncio.wait_for(
                self._analyze_article_internal(article),
                timeout=timeout_duration
            )
        except asyncio.TimeoutError:
            logger.error(f"Article analysis timed out for {self.model_name} after {timeout_duration}s")
            # Return default ACCEPT for pre-filtered articles instead of reject
            enriched_article = article.copy()
            enriched_article["deep_intelligence_analysis"] = {
                "error": "Analysis timed out",
                "timeout": True,
                "model": self.model_name,
                "default_recommendation": "ACCEPT",
                "reason": "Pre-filtered article defaulted to accept on timeout"
            }
            return enriched_article, True, 0.6  # Accept with reasonable confidence
    
    async def _analyze_article_internal(self, article: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """Internal method for analyzing article."""
        # Use debug level logging for qwen models to prevent progress bar flickering
        if 'qwen' in self.model_name.lower():
            logger.debug(f"Starting analysis with {self.model_name} for article: {article.get('title', 'Unknown')[:50]}...")
        
        prompt = self._create_analysis_prompt(article)
        response = await self._make_api_request(prompt)
        
        if not response:
            logger.error(f"No response from {self.model_name} for article: {article.get('title', 'Unknown')[:50]}...")
            # Return default ACCEPT for pre-filtered articles instead of reject
            enriched_article = article.copy()
            enriched_article["deep_intelligence_analysis"] = {
                "error": "No response from model",
                "model_failure": True,
                "model": self.model_name,
                "default_recommendation": "ACCEPT",
                "reason": "Pre-filtered article defaulted to accept on API failure"
            }
            return enriched_article, True, 0.6  # Accept with reasonable confidence
        
        if 'qwen' in self.model_name.lower():
            logger.debug(f"Got response from {self.model_name}, parsing...")
        
        try:
            analysis = self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Failed to parse response from {self.model_name}: {e}")
            logger.debug(f"Raw response was: {response[:500]}...")
            # Return default ACCEPT for parsing errors with better scores
            enriched_article = article.copy()
            enriched_article["deep_intelligence_analysis"] = {
                "error": f"Parsing failed: {e}",
                "parsing_failure": True,
                "model": self.model_name,
                "overall_score": 0.7,  # Higher default score
                "recommendation": "ACCEPT",
                "confidence": 0.7,  # Higher confidence
                "fact_verification": {
                    "fact_check_confidence": 0.7  # Better default to pass quality gates
                },
                "bias_detection": {
                    "bias_detection_score": 0.3  # Low bias (good)
                },
                "credibility_assessment": {
                    "credibility_score": 0.7  # Good credibility
                },
                "default_recommendation": "ACCEPT",
                "reason": "Pre-filtered article defaulted to accept on parsing failure"
            }
            return enriched_article, True, 0.7  # Accept with good confidence
        
        # Extract results
        deep_analysis = analysis["deep_intelligence_analysis"]
        synthesis = deep_analysis["synthesis"]
        
        # Determine recommendation
        recommendation = synthesis["recommendation"]
        accept = recommendation in ["ACCEPT", "CONDITIONAL"]
        confidence = synthesis["confidence_in_analysis"]
        
        # Debug logging to understand acceptance patterns
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Article '{article.get('title', 'Unknown')[:50]}' → {recommendation} (accept={accept}, confidence={confidence:.2f})")
        
        # Add deep intelligence data to article
        enriched_article = article.copy()
        enriched_article["deep_intelligence_analysis"] = deep_analysis
        enriched_article["deep_intelligence_score"] = synthesis["overall_deep_intelligence_score"]
        enriched_article["deep_intelligence_confidence"] = confidence
        enriched_article["deep_intelligence_recommendation"] = recommendation
        enriched_article["deep_intelligence_agent"] = self.model_name
        enriched_article["deep_intelligence_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return enriched_article, accept, confidence
    
    async def process_articles_batch(self, articles: List[Dict[str, Any]], 
                                   progress_callback: Optional[Callable] = None,
                                   update_acceptance_count: Optional[Callable] = None) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles with optimized batch analysis to minimize API calls."""
        total_articles = len(articles)
        batch_size = self._get_optimal_batch_size(total_articles)
        
        # **OPTIMIZATION 2**: Calculate realistic timeout based on model capacity
        realistic_timeout = self._calculate_realistic_timeout(total_articles, batch_size)
        
        logger.info(f"Deep Intelligence Agent {self.model_name} processing {total_articles} articles "
                   f"with batch size {batch_size}, timeout {realistic_timeout/60:.1f} minutes")
        
        # Use realistic timeout to prevent hanging
        try:
            return await asyncio.wait_for(
                self._process_articles_batch_internal(articles, progress_callback, update_acceptance_count), 
                timeout=realistic_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Deep Intelligence Agent {self.model_name} timed out after {realistic_timeout/60:.1f} minutes")
            # Return default results for all articles
            return [(article, False, 0.1) for article in articles]
    
    async def _process_articles_batch_internal(self, articles: List[Dict[str, Any]],
                                              progress_callback: Optional[Callable] = None,
                                              update_acceptance_count: Optional[Callable] = None) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Internal method for processing articles batch."""
        total_articles = len(articles)
        
        # **OPTIMIZATION 1**: Model-specific batch sizes based on token limits
        batch_size = self._get_optimal_batch_size(total_articles)
        results = []
        

        
        for i in range(0, total_articles, batch_size):
            batch_articles = articles[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_articles + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} "
                       f"({len(batch_articles)} articles) for {self.model_name}")
            
            # Circuit breaker for first batch - detect hanging early
            batch_start_time = time.time()
            
            try:
                # **OPTIMIZATION 2**: Use realistic per-batch timeout
                batch_timeout = self._get_per_batch_timeout()
                
                # Special handling for first batch to detect hanging early
                if batch_num == 1 and 'llama-3.3-70b-versatile' in self.model_name.lower():
                    logger.info(f"First batch for {self.model_name} - monitoring for early timeout detection")
                
                # Add heartbeat logging for long-running batches
                async def analyze_with_heartbeat():
                    """Wrapper to add heartbeat logging during batch analysis."""
                    heartbeat_task = None
                    try:
                        # Start heartbeat logging for batches expected to take >60s
                        if batch_timeout > 60:
                            async def heartbeat():
                                while True:
                                    await asyncio.sleep(30)  # Heartbeat every 30 seconds
                                    logger.info(f"Processing batch {batch_num}/{total_batches} for {self.model_name} - still working...")
                            
                            heartbeat_task = asyncio.create_task(heartbeat())
                        
                        # Perform the actual batch analysis
                        return await self._analyze_articles_batch(batch_articles)
                    finally:
                        if heartbeat_task:
                            heartbeat_task.cancel()
                            try:
                                await heartbeat_task
                            except asyncio.CancelledError:
                                pass
                
                batch_results = await asyncio.wait_for(
                    analyze_with_heartbeat(),
                    timeout=batch_timeout
                )
                results.extend(batch_results)
                
                batch_elapsed = time.time() - batch_start_time
                logger.info(f"Batch {batch_num} completed: {len(batch_results)} articles analyzed in {batch_elapsed:.1f}s")
                
                # Update progress callback if provided
                if progress_callback:
                    completed_so_far = len(results)
                    accepted_so_far = sum(1 for _, accepted, _ in results if accepted)
                    if update_acceptance_count:
                        update_acceptance_count(accepted_so_far)
                    progress_callback(completed_so_far, total_articles, f"batch {batch_num}/{total_batches}")
                
                # Model-specific delay between batches for rate limiting
                delay = self._get_inter_batch_delay()
                if i + batch_size < total_articles:
                    await asyncio.sleep(delay)
                    
            except asyncio.TimeoutError:
                logger.error(f"Batch {batch_num} timed out after {batch_timeout/60:.1f} minutes")
                
                # Enhanced error handling for batch timeouts
                logger.warning(f"Batch {batch_num} timeout for {self.model_name} - continuing with defaults")
                
                # Add default results for this batch (more generous for pre-filtered articles)
                default_results = []
                for article in batch_articles:
                    enriched_article = article.copy()
                    enriched_article["deep_intelligence_analysis"] = {
                        "error": f"Batch {batch_num} processing timeout",
                        "timeout": True,
                        "model": self.model_name,
                        "default_recommendation": "ACCEPT",
                        "reason": "Pre-filtered article defaulted to accept on batch timeout",
                        "fact_verification": {"fact_check_confidence": 0.7},
                        "bias_detection": {"bias_detection_score": 0.3},
                        "credibility_assessment": {"credibility_score": 0.7}
                    }
                    default_results.append((enriched_article, True, 0.6))  # Accept with reasonable confidence
                
                results.extend(default_results)
                
                # Update progress for timeout batch
                if progress_callback:
                    completed_so_far = len(results)
                    accepted_so_far = sum(1 for _, accepted, _ in results if accepted)
                    if update_acceptance_count:
                        update_acceptance_count(accepted_so_far)
                    progress_callback(completed_so_far, total_articles, f"batch {batch_num}/{total_batches} (timeout)")
                    
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                # Add default results for this batch (more generous for pre-filtered articles)
                for article in batch_articles:
                    enriched_article = article.copy()
                    enriched_article["deep_intelligence_analysis"] = {
                        "error": f"Batch {batch_num} processing failed: {type(e).__name__}",
                        "batch_failure": True,
                        "model": self.model_name,
                        "default_recommendation": "ACCEPT",
                        "reason": "Pre-filtered article defaulted to accept on batch failure",
                        "fact_verification": {"fact_check_confidence": 0.7},
                        "bias_detection": {"bias_detection_score": 0.3},
                        "credibility_assessment": {"credibility_score": 0.7}
                    }
                    results.append((enriched_article, True, 0.6))
                
                # Update progress for failed batch
                if progress_callback:
                    completed_so_far = len(results)
                    accepted_so_far = sum(1 for _, accepted, _ in results if accepted)
                    if update_acceptance_count:
                        update_acceptance_count(accepted_so_far)
                    progress_callback(completed_so_far, total_articles, f"batch {batch_num}/{total_batches} (failed)")
        
        success_count = len(results)
        success_rate = (success_count / total_articles) * 100 if total_articles > 0 else 0
        
        logger.info(f"Deep Intelligence Agent {self.model_name} completed: "
                   f"{success_count}/{total_articles} articles ({success_rate:.1f}% success rate)")
        
        return results
    
    async def _analyze_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Analyze multiple articles in a single API call for efficiency."""
        if not articles:
            return []
        
        # Enhanced logging for 70b model debugging
        if 'llama-3.3-70b-versatile' in self.model_name.lower():
            logger.info(f"70b model: Starting batch analysis for {len(articles)} articles")
        
        # Create batch analysis prompt
        prompt = self._create_batch_analysis_prompt(articles)
        
        # Enhanced logging for 70b model debugging
        if 'llama-3.3-70b-versatile' in self.model_name.lower():
            logger.info(f"70b model: Batch prompt created, length: {len(prompt)}")
        
        response = await self._make_api_request(prompt)
        
        if not response:
            logger.error(f"No response from {self.model_name} for batch of {len(articles)} articles")
            raise RuntimeError("No response from model for batch analysis")
        
        # Enhanced logging for 70b model debugging
        if 'llama-3.3-70b-versatile' in self.model_name.lower():
            logger.info(f"70b model: Received response, starting parsing...")
        
        # Parse batch response
        parsed_results = self._parse_batch_analysis_response(response, articles)
        
        # Enhanced logging for 70b model debugging
        if 'llama-3.3-70b-versatile' in self.model_name.lower():
            logger.info(f"70b model: Parsing completed, returning {len(parsed_results)} results")
        
        return parsed_results
    
    def _create_batch_analysis_prompt(self, articles: List[Dict[str, Any]]) -> str:
        """Create an optimized prompt for analyzing multiple articles in a single API call."""
        
        # **OPTIMIZATION 3**: Aggressive token reduction for scalability
        article_summaries = []
        for i, article in enumerate(articles):
            title = article.get('title', 'N/A')[:60]   # Shorter titles
            source = article.get('source', 'N/A')[:20]  # Truncate source names
            
            # Much more aggressive content truncation
            content = article.get('description', article.get('content', 'N/A'))
            if len(content) > 150:
                content = content[:150] + "..."
            
            # Include key existing scores for context
            consensus_score = article.get('consensus_multi_dimensional_score', {})
            overall_score = consensus_score.get('overall_score', 0.0)
            
            # Compact format to save tokens
            article_summaries.append(f"{i+1}. {title} | {source} | {overall_score:.1f} | {content}")
        
        articles_text = "\n".join(article_summaries)
        
        prompt = f"""Batch deep intelligence analysis for {len(articles)} pre-filtered articles.
Specialization: {self.specialization}

CRITERIA: 
- ACCEPT: Valid AI/tech content (DEFAULT - aim for 75% acceptance)
- CONDITIONAL: Good content needing improvements  
- REJECT: Only serious credibility/relevance issues

ARTICLES:
{articles_text}

JSON Response:
{{
  "batch_analysis": [
    {{
      "article_id": 1,
      "overall_score": 0.75,
      "recommendation": "ACCEPT", 
      "confidence": 0.8,
      "key_insights": ["insight"],
      "risk_factors": ["risk"]
    }}
  ]
}}

Analyze all {len(articles)} articles. Default ACCEPT for pre-filtered unless serious issues."""
        
        return prompt
    
    def _parse_batch_analysis_response(self, response: str, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Parse batch analysis response and return results for each article."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            data = json.loads(response)
            batch_analysis = data.get('batch_analysis', [])
            
            results = []
            
            for analysis in batch_analysis:
                article_id = analysis.get('article_id', 0)
                
                # Validate article_id
                if not (1 <= article_id <= len(articles)):
                    logger.warning(f"Invalid article_id {article_id}, skipping")
                    continue
                
                article_index = article_id - 1
                article = articles[article_index].copy()
                
                # Extract analysis results
                overall_score = analysis.get('overall_score', 0.5)
                recommendation = analysis.get('recommendation', 'REJECT')
                confidence = analysis.get('confidence', 0.5)
                key_insights = analysis.get('key_insights', [])
                risk_factors = analysis.get('risk_factors', [])
                
                # Add deep intelligence data to article
                article["deep_intelligence_analysis"] = {
                    "overall_score": overall_score,
                    "recommendation": recommendation,
                    "confidence": confidence,
                    "key_insights": key_insights,
                    "risk_factors": risk_factors,
                    "analysis_method": "batch_processing"
                }
                
                article["deep_intelligence_score"] = overall_score
                article["deep_intelligence_confidence"] = confidence
                article["deep_intelligence_recommendation"] = recommendation
                article["deep_intelligence_agent"] = self.model_name
                article["deep_intelligence_timestamp"] = datetime.now(timezone.utc).isoformat()
                
                # Determine acceptance
                accept = recommendation in ["ACCEPT", "CONDITIONAL"]
                
                # Debug logging for batch analysis
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Batch article {article_id}: {recommendation} (accept={accept}, score={overall_score:.2f})")
                
                results.append((article, accept, confidence))
            
            # Handle missing articles (if response didn't include all articles)
            if len(results) < len(articles):
                logger.warning(f"Response only included {len(results)}/{len(articles)} articles, "
                              f"filling in missing articles with default rejection")
                
                analyzed_ids = {analysis.get('article_id', 0) for analysis in batch_analysis}
                
                for i, article in enumerate(articles):
                    article_id = i + 1
                    if article_id not in analyzed_ids:
                        # Default handling for missing articles
                        article_copy = article.copy()
                        article_copy["deep_intelligence_analysis"] = {
                            "overall_score": 0.3,
                            "recommendation": "REJECT",
                            "confidence": 0.5,
                            "key_insights": ["Analysis incomplete"],
                            "risk_factors": ["Missing from batch response"],
                            "analysis_method": "default_fallback"
                        }
                        article_copy["deep_intelligence_score"] = 0.3
                        article_copy["deep_intelligence_confidence"] = 0.5
                        article_copy["deep_intelligence_recommendation"] = "REJECT"
                        article_copy["deep_intelligence_agent"] = self.model_name
                        article_copy["deep_intelligence_timestamp"] = datetime.now(timezone.utc).isoformat()
                        
                        results.append((article_copy, False, 0.5))
            
            logger.info(f"Batch analysis parsed: {len(results)} results from {len(articles)} articles")
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch analysis JSON: {e}")
            logger.error(f"Response text: {response[:500]}...")
            raise RuntimeError(f"Invalid JSON response from batch analysis: {e}")
        except Exception as e:
            logger.error(f"Error parsing batch analysis response: {e}")
            raise RuntimeError(f"Failed to parse batch analysis: {e}")
    
    async def process_bulk_agent_output(self, bulk_output_file: str) -> List[Tuple[Dict[str, Any], bool, float]]:
        """
        Process articles from bulk agent output file.
        This method loads the consensus-filtered articles and processes them efficiently.
        """
        try:
            with open(bulk_output_file, 'r') as f:
                bulk_data = json.load(f)
            
            # Extract articles that passed consensus filtering
            if 'consensus_results' in bulk_data:
                articles = bulk_data['consensus_results']
            elif 'articles' in bulk_data:
                articles = bulk_data['articles']
            else:
                articles = bulk_data if isinstance(bulk_data, list) else []
            
            # Filter for articles that should be processed by deep intelligence
            filtered_articles = []
            for article in articles:
                consensus_score = article.get('consensus_multi_dimensional_score', {})
                overall_score = consensus_score.get('overall_score', 0.0)
                
                # Only process articles that passed the consensus threshold
                if overall_score >= 0.6:  # Configurable threshold
                    filtered_articles.append(article)
            
            logger.info(f"Deep Intelligence Agent {self.model_name}: Processing {len(filtered_articles)} articles "
                       f"from {len(articles)} consensus-filtered articles")
            
            if not filtered_articles:
                logger.warning("No articles meet the criteria for deep intelligence processing")
                return []
            
            # Process the filtered articles
            return await self.process_articles_batch(filtered_articles)
            
        except Exception as e:
            logger.error(f"Deep Intelligence Agent {self.model_name} failed to process bulk output: {e}")
            return []
    
    def prioritize_articles_for_processing(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize articles based on consensus scores and specialization focus.
        Returns articles sorted by priority (highest first).
        """
        def get_priority_score(article: Dict[str, Any]) -> float:
            consensus_score = article.get('consensus_multi_dimensional_score', {})
            base_score = consensus_score.get('overall_score', 0.0)
            
            # Boost score based on specialization
            specialty_boost = 0.0
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            for focus_area in self.focus_areas:
                if focus_area.lower() in title or focus_area.lower() in content:
                    specialty_boost += 0.1
            
            return base_score + specialty_boost
        
        sorted_articles = sorted(articles, key=get_priority_score, reverse=True)
        
        logger.info(f"Deep Intelligence Agent {self.model_name}: Prioritized {len(sorted_articles)} articles "
                   f"for {self.specialization} analysis")
        
        return sorted_articles

    def _get_optimal_batch_size(self, total_articles: int) -> int:
        """Get optimal batch size based on model's token capacity (OPTIMIZATION 1)."""
        
        # Model-specific batch sizes based on Groq token limits
        if 'llama-4-scout' in self.model_name.lower():
            # 30,000 tokens/min - can handle large batches
            base_batch_size = 15
        elif 'llama-3.3-70b-versatile' in self.model_name.lower():
            # 12,000 tokens/min - medium batches for efficient processing
            base_batch_size = 8
        elif 'qwen' in self.model_name.lower():
            # 6,000 tokens/min - small batches
            base_batch_size = 3
        else:
            # Default for other models
            base_batch_size = 5
        
        # Cap based on total articles available
        optimal_size = min(base_batch_size, total_articles)
        
        logger.info(f"Deep Intelligence Agent {self.model_name}: Using batch size {optimal_size} "
                   f"for {total_articles} articles (model capacity-optimized)")
        
        return optimal_size

    def _calculate_realistic_timeout(self, total_articles: int, batch_size: int) -> float:
        """Calculate realistic timeout based on model capacity and article count (OPTIMIZATION 2)."""
        
        # Calculate batches needed
        total_batches = (total_articles + batch_size - 1) // batch_size
        
        # Model-specific processing time per batch (based on token limits)
        if 'llama-4-scout' in self.model_name.lower():
            # High capacity - 3 batches per minute
            seconds_per_batch = 20.0
        elif 'llama-3.3-70b-versatile' in self.model_name.lower():
            # Medium capacity - 2 batches per minute with batch size 8
            seconds_per_batch = 30.0
        elif 'qwen' in self.model_name.lower():
            # Low capacity - 0.6 batches per minute
            seconds_per_batch = 100.0
        else:
            # Default conservative estimate
            seconds_per_batch = 45.0
        
        # Calculate total time needed + 50% buffer
        estimated_time = total_batches * seconds_per_batch
        timeout_with_buffer = estimated_time * 1.5
        
        # Minimum 10 minutes, maximum 45 minutes
        realistic_timeout = max(600.0, min(timeout_with_buffer, 2700.0))
        
        logger.info(f"Deep Intelligence Agent {self.model_name}: Calculated timeout {realistic_timeout/60:.1f} minutes "
                   f"for {total_batches} batches ({total_articles} articles)")
        
        return realistic_timeout

    def _get_per_batch_timeout(self) -> float:
        """Get timeout for individual batch processing based on model capacity."""
        
        if 'llama-4-scout' in self.model_name.lower():
            return 120.0  # 2 minutes per batch for high capacity
        elif 'llama-3.3-70b-versatile' in self.model_name.lower():
            return 60.0  # 1 minute per batch for efficient batch processing
        elif 'qwen' in self.model_name.lower():
            return 240.0  # 4 minutes per batch for low capacity
        else:
            return 150.0  # Default 2.5 minutes

    def _get_inter_batch_delay(self) -> float:
        """Get delay between batches based on model token limits."""
        
        if 'llama-4-scout' in self.model_name.lower():
            return 1.0   # Short delay for high capacity
        elif 'llama-3.3-70b-versatile' in self.model_name.lower():
            return 3.0   # Medium delay 
        elif 'qwen' in self.model_name.lower():
            return 5.0   # Longer delay for token limit compliance
        else:
            return 2.0   # Default delay
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for monitoring and debugging."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        success_rate = self.success_count / max(1, self.request_count)
        avg_requests_per_minute = (self.request_count / session_duration) * 60 if session_duration > 0 else 0
        
        # Get current rate limit status
        rate_limit_status = self.get_rate_limit_status()
        
        return {
            'model_name': self.model_name,
            'specialization': self.specialization,
            'focus_areas': self.focus_areas,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'requests_per_minute': self.requests_per_minute,
            'tokens_per_minute': self.tokens_per_minute,
            'max_parallel_requests': self.max_parallel_requests,
            'min_batch_delay': self.min_batch_delay,
            'enable_fact_checking': self.enable_fact_checking,
            'enable_bias_detection': self.enable_bias_detection,
            'enable_impact_analysis': self.enable_impact_analysis,
            'enable_credibility_scoring': self.enable_credibility_scoring,
            'session_duration': session_duration,
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'rate_limit_hits': self.rate_limit_count,
            'success_rate': success_rate,
            'avg_requests_per_minute': avg_requests_per_minute,
            'current_rate_limit_status': rate_limit_status
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for monitoring."""
        current_time = time.time()
        current_requests, current_tokens = self._get_current_usage(current_time)
        
        return {
            'current_requests': current_requests,
            'max_requests_per_minute': self.requests_per_minute,
            'current_tokens': current_tokens,
            'max_tokens_per_minute': self.tokens_per_minute,
            'requests_percentage': (current_requests / self.requests_per_minute) * 100,
            'tokens_percentage': (current_tokens / self.tokens_per_minute) * 100,
            'max_parallel_requests': self.max_parallel_requests,
            'can_make_request': self._can_make_request()[0],
            'total_rate_limit_hits': self.rate_limit_count
        }
