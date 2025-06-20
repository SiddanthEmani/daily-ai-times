#!/usr/bin/env python3
"""
Stage 2: Compound Intelligence Agent

Second stage of the optimized 4-stage pipeline using deepseek-r1-distill-llama-70b reasoning model.
Processes articles that passed Stage 1 bulk filtering with multi-dimensional compound reasoning.
Target: Filter down to 30% of input articles (deep analysis and reasoning).
"""

import os
import json
import logging
import asyncio
import aiohttp
import re
import sys
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
import time

# Import usage tracker
try:
    from .groq_usage_tracker import usage_tracker
except ImportError:
    try:
        from groq_usage_tracker import usage_tracker
    except ImportError:
        class DummyTracker:
            def record_call(self, **kwargs):
                pass
        usage_tracker = DummyTracker()

logger = logging.getLogger(__name__)

class Stage2CompoundAgent:
    """Stage 2 Compound Intelligence - Deep analysis with reasoning model."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the Stage 2 compound agent."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Load configuration if provided
        if config:
            self.model = config.get('model', 'gemma2-9b-it')
            self.batch_size = config.get('batch_size', 10)
            self.max_requests_per_cycle = config.get('max_requests_per_cycle', 25)
            self.target_pass_rate = config.get('target_pass_rate', 0.30)
            self.time_budget_minutes = config.get('time_budget_minutes', 8)
        else:
            self.model = "gemma2-9b-it"  # Best free model: 30 req/min, 15,000 tokens/min, 500k tokens/day
            self.batch_size = 10  # Larger batches due to higher token limit
            self.max_requests_per_cycle = 25  # Process more articles per cycle
            self.target_pass_rate = 0.30  # Fixed to 30% as specified
            self.time_budget_minutes = 8
            
        self.session = None
        
        # Stage 2 optimized settings - Updated for gemma2-9b-it (FREE MODEL)
        # Model limits: 30 req/min, 14,400 req/day, 15,000 tokens/min, 500,000 tokens/day (FREE)
        self.requests_per_minute_limit = 30
        self.tokens_per_minute_limit = 15000  # Much higher token limit!
        self.requests_per_day_limit = 14400
        
        # More aggressive rate limiting for speed (use 95% of limits)
        self.safe_requests_per_minute = int(self.requests_per_minute_limit * 0.95)  # 28 req/min
        self.safe_tokens_per_minute = int(self.tokens_per_minute_limit * 0.95)     # 14,250 tokens/min
        
        self.tokens_per_article = 150  # Higher token budget per article
        
        # Enhanced rate limiting with proper reset tracking
        self.last_request_time = 0
        self.consecutive_rate_limits = 0
        self.max_retries = 5
        self.base_retry_delay = 2.0
        
        # Rate limit tracking windows (rolling 1-minute windows)
        self.request_timestamps = []  # Track request times for rate limiting
        self.token_usage_window = []  # Track token usage over time
        self.rate_limit_reset_time = None  # Track when rate limits reset
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),  # Longer timeout for reasoning models
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self, task_id: Optional[str] = None):
        """Optimized rate limiting for free models - minimal delays."""
        if task_id is None:
            task_id = "S2-XXXX"
            
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff_time]
        self.token_usage_window = [(t, tokens) for t, tokens in self.token_usage_window if t > cutoff_time]
        
        # Only enforce rate limits if we're actually close to hitting them
        requests_in_window = len(self.request_timestamps)
        tokens_in_window = sum(tokens for _, tokens in self.token_usage_window)
        
        # Log current rate limit status
        logger.debug(f"[{task_id}] Rate limit status: {requests_in_window}/{self.safe_requests_per_minute} requests, "
                    f"{tokens_in_window}/{self.safe_tokens_per_minute} tokens in current window")
        
        # Only wait if we're at 100% of safe limits (not before)
        if requests_in_window >= self.safe_requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_request) + 0.1  # Minimal safety margin
            if wait_time > 0:
                logger.info(f"[{task_id}] Request rate limit reached ({requests_in_window}/{self.safe_requests_per_minute}). "
                          f"Waiting {wait_time:.1f}s for window reset")
                await asyncio.sleep(wait_time)
                current_time = time.time()
        
        # Token limit check (with high limit, this should rarely trigger)
        if tokens_in_window >= self.safe_tokens_per_minute:
            oldest_token_time = min(t for t, _ in self.token_usage_window) if self.token_usage_window else current_time
            wait_time = 60 - (current_time - oldest_token_time) + 0.1  # Minimal safety margin
            if wait_time > 0:
                logger.info(f"[{task_id}] Token rate limit reached ({tokens_in_window}/{self.safe_tokens_per_minute}). "
                          f"Waiting {wait_time:.1f}s for window reset")
                await asyncio.sleep(wait_time)
                current_time = time.time()
        
        # REMOVE: No longer wait for arbitrary rate limit reset times from headers
        # This was causing unnecessary 65s waits
        
        # Minimal delay between requests (much reduced)
        time_since_last = current_time - self.last_request_time
        minimal_delay = 2.2  # Just over 2s for 28 req/min limit
        if time_since_last < minimal_delay:
            wait_time = minimal_delay - time_since_last
            logger.debug(f"[{task_id}] Minimal delay: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # Record this request time
        self.last_request_time = time.time()
        self.request_timestamps.append(self.last_request_time)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for rate limit hits."""
        # For rate limit errors (429), wait for the rate limit window to reset
        if self.consecutive_rate_limits > 0:
            # Wait for at least 60 seconds to ensure rate limit window resets
            base_wait = 60.0 + (attempt * 10.0)  # 60s + 10s per attempt
            return min(base_wait, 120.0)  # Cap at 2 minutes
        
        # For other errors, use exponential backoff
        base_delay = self.base_retry_delay
        exponential_delay = base_delay * (2 ** attempt)
        return min(exponential_delay, 30.0)  # Cap at 30 seconds for non-rate-limit errors
    
    def _record_token_usage(self, tokens: int):
        """Record token usage for rate limiting."""
        current_time = time.time()
        self.token_usage_window.append((current_time, tokens))
    
    def _update_rate_limit_info(self, headers: dict):
        """Update rate limit information from response headers (simplified)."""
        # Extract rate limit information from headers if available
        remaining_requests = headers.get('x-ratelimit-remaining-requests')
        remaining_tokens = headers.get('x-ratelimit-remaining-tokens')
        
        # Log rate limit status for monitoring (but don't wait based on headers)
        if remaining_requests and remaining_tokens:
            logger.debug(f"Rate limits from headers - Requests: {remaining_requests}/{self.requests_per_minute_limit}, "
                        f"Tokens: {remaining_tokens}/{self.tokens_per_minute_limit}")
            
            # Warning if we're getting very close to limits
            try:
                req_remaining = int(remaining_requests)
                tok_remaining = int(remaining_tokens)
                
                if req_remaining < 3:
                    logger.warning(f"Very low request limit remaining: {req_remaining}")
                if tok_remaining < 1000:
                    logger.warning(f"Low token limit remaining: {tok_remaining}")
                    
            except (ValueError, TypeError):
                pass  # Ignore conversion errors
    
    def _create_compound_analysis_prompt(self, articles: List[Dict]) -> str:
        """Create a concise compound analysis prompt for a batch of articles."""
        
        # Create brief article summaries to reduce token usage
        article_details = []
        for i, article in enumerate(articles):
            title = article.get('title', 'No title')[:100]  # Truncate title
            description = article.get('description', 'No description')[:150]  # Shorter description
            source = article.get('source', 'Unknown')
            
            article_details.append(f"{i+1}. {title} ({source})\n{description}")
        
        articles_text = "\n\n".join(article_details)
        
        # Calculate expected advance count (25-35%)
        expected_advances = max(1, round(len(articles) * 0.30))  # Target 30%
        
        prompt = f"""CRITICAL: You MUST filter most articles. Advance ONLY the best {expected_advances} articles.

TARGET: Advance exactly {expected_advances} articles out of {len(articles)} (REJECT the other {len(articles) - expected_advances}).

{articles_text}

SELECTION CRITERIA: Only advance articles with significant AI/tech breakthroughs, major industry impacts, or critical research developments.

REQUIRED OUTPUT FORMAT (NO OTHER TEXT):
1: [ADVANCE] Score: 85 | [FILTER] Score: 45
2: [FILTER] Score: 30 | [ADVANCE] Score: 70
3: [ADVANCE] Score: 92 | [FILTER] Score: 25
{len(articles)}: [FILTER] Score: 40 | [ADVANCE] Score: 60

RESPOND NOW WITH DECISIONS 1-{len(articles)}:"""

        return prompt
    
    async def _call_groq_api(self, prompt: str, max_tokens: int = 400, task_id: Optional[str] = None) -> Optional[str]:
        """Make a call to the Groq API for Stage 2 compound analysis with enhanced rate limiting."""
        if not self.session:
            logger.error("Session not initialized")
            return None
        
        if task_id is None:
            task_id = "S2-XXXX"
        
        estimated_input_tokens = len(prompt) // 4
        estimated_total_tokens = estimated_input_tokens + max_tokens
        
        for attempt in range(self.max_retries):
            await self._rate_limit(task_id)
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Stage 2 AI/tech news filter. RESPOND EXACTLY AS SHOWN:\n\nFORMAT REQUIREMENTS:\n- Start immediately with '1:'\n- No explanations, no reasoning, no extra text\n- Use exactly: '1: [ADVANCE] Score: 85 | [FILTER] Score: 45'\n- Advance only ~30% of articles\n- Never use <think> tags or XML\n- No markdown formatting"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Very low temperature for format consistency
                "top_p": 0.8,
                "stream": False
            }
            
            start_time = time.time()
            
            try:
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                    # Extract rate limit information from headers
                    self._update_rate_limit_info(dict(response.headers))
                    
                    if response.status == 429:
                        self.consecutive_rate_limits += 1
                        retry_delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Rate limit hit (429), waiting {retry_delay:.1f}s for reset "
                                     f"(attempt {attempt + 1}, consecutive: {self.consecutive_rate_limits})")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    # Reset consecutive rate limits on successful response
                    if response.status == 200:
                        self.consecutive_rate_limits = 0
                    
                    if response.status != 200:
                        logger.error(f"API request failed with status {response.status}")
                        response_text = await response.text()
                        logger.error(f"Error response: {response_text}")
                        if attempt == self.max_retries - 1:
                            return None
                        retry_delay = self._calculate_retry_delay(attempt)
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    result = await response.json()
                    response_time = time.time() - start_time
                    
                    # Extract response content
                    choice = result['choices'][0]['message']
                    content = choice['content']
                    
                    # Record token usage for rate limiting
                    usage = result.get('usage', {})
                    actual_total_tokens = usage.get('total_tokens', estimated_total_tokens)
                    prompt_tokens = usage.get('prompt_tokens', estimated_input_tokens)
                    completion_tokens = usage.get('completion_tokens', len(content) // 4)
                    
                    self._record_token_usage(actual_total_tokens)
                    
                    # Detailed token logging for Stage 2
                    current_tokens_in_window = sum(tokens for _, tokens in self.token_usage_window)
                    logger.info(f"[{task_id}] 🔤 Stage 2 Token Usage:")
                    logger.info(f"[{task_id}]   📊 This Request: {prompt_tokens} prompt + {completion_tokens} completion = {actual_total_tokens} total tokens")
                    logger.info(f"[{task_id}]   📈 Window Usage: {current_tokens_in_window}/{self.safe_tokens_per_minute} tokens ({current_tokens_in_window/self.safe_tokens_per_minute*100:.1f}%)")
                    logger.info(f"[{task_id}]   ⚡ Model: {self.model} (15,000 tokens/min limit)")
                    logger.info(f"[{task_id}]   💰 Cost: $0.00 (FREE MODEL)")
                    
                    # Record usage
                    usage_tracker.record_call(
                        model=self.model,
                        agent="stage2_compound",
                        request_tokens=prompt_tokens,
                        response_tokens=completion_tokens,
                        processing_time=response_time,
                        success=True
                    )
                    
                    return content
                    
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    usage_tracker.record_call(
                        model=self.model,
                        agent="stage2_compound",
                        request_tokens=estimated_input_tokens,
                        response_tokens=0,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="timeout"
                    )
                    return None
                retry_delay = self._calculate_retry_delay(attempt)
                await asyncio.sleep(retry_delay)
                continue
            except Exception as e:
                logger.error(f"API request error: {e} (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    usage_tracker.record_call(
                        model=self.model,
                        agent="stage2_compound",
                        request_tokens=estimated_input_tokens,
                        response_tokens=0,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message=str(e)
                    )
                    return None
                retry_delay = self._calculate_retry_delay(attempt)
                await asyncio.sleep(retry_delay)
                continue
        
        return None
    
    def _parse_compound_response(self, response: str, articles: List[Dict]) -> List[Dict]:
        """Parse the simplified compound analysis response and score articles."""
        advanced_articles = []
        
        if not response:
            logger.warning("Empty response from compound analysis")
            return advanced_articles
        
        # Aggressive cleaning of response - handle various verbose formats
        original_response = response
        
        # Remove <think> blocks completely - improved handling
        while '<think>' in response and '</think>' in response:
            start = response.find('<think>')
            end = response.find('</think>') + 8
            response = response[:start] + response[end:]
            logger.info("Removed <think> block from response")
        
        # If the response still contains <think> without closing tag, extract content before it
        if '<think>' in response:
            before_think = response.split('<think>', 1)[0].strip()
            if before_think and len(before_think) > 20:
                response = before_think
                logger.info("Extracted content before unclosed <think> tag")
            else:
                # Try to extract decision lines from anywhere in the response
                import re
                decision_lines = re.findall(r'\d+:\s*\[(?:ADVANCE|FILTER)\].*?(?:\n|$)', original_response, re.IGNORECASE | re.MULTILINE)
                if decision_lines:
                    response = '\n'.join(decision_lines)
                    logger.info(f"Extracted {len(decision_lines)} decision lines from response with unclosed <think>")
                else:
                    # Fallback: look for any numbered lines with advance/filter keywords
                    decision_lines = re.findall(r'\d+.*?(?:ADVANCE|FILTER).*?(?:\n|$)', original_response, re.IGNORECASE)
                    response = '\n'.join(decision_lines)
                    logger.info(f"Fallback: extracted {len(decision_lines)} decision lines from original")
        
        # Look for actual decision lines by finding numbered patterns
        response = response.strip()
        
        # If response is still empty after cleaning, try aggressive extraction
        if not response or len(response) < 10:
            logger.warning("Response too short after cleaning, trying aggressive extraction from original")
            import re
            # Try multiple patterns to find decision lines
            patterns = [
                r'\d+:\s*\[(?:ADVANCE|FILTER)\].*?(?:\n|$)',  # Format: "1: [ADVANCE] Score: 85"
                r'\d+:.*?(?:ADVANCE|FILTER).*?(?:\n|$)',      # Format: "1: ADVANCE Score: 85"
                r'\d+\s*-?\s*(?:ADVANCE|FILTER).*?(?:\n|$)', # Format: "1 - ADVANCE"
            ]
            
            for pattern in patterns:
                decision_lines = re.findall(pattern, original_response, re.IGNORECASE)
                if decision_lines:
                    response = '\n'.join(decision_lines)
                    logger.info(f"Extracted {len(decision_lines)} decision lines using pattern: {pattern}")
                    break
            
            if not response:
                logger.error("Could not extract any decision lines from response")
                return self._emergency_fallback_parsing(articles)
        
        # Debug logging - show actual response
        logger.info(f"Parsing cleaned response length: {len(response)} characters")
        logger.info(f"Cleaned response preview: {response[:200]}...")
        
        lines = response.strip().split('\n')
        
        # Track which articles we've seen decisions for
        decisions_found = set()
        import re
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for pattern: "1: [ADVANCE/FILTER] Score: XX" or "1: [ADVANCE] Score: 85 | [FILTER] Score: 45"
            if ':' in line and any(keyword in line.upper() for keyword in ['ADVANCE', 'FILTER']):
                try:
                    # Parse article number
                    parts = line.split(':', 1)
                    article_num_str = parts[0].strip()
                    rest = parts[1].strip()
                    
                    # Extract article number (handle various formats like "1", "Article 1", etc.)
                    number_match = re.search(r'\d+', article_num_str)
                    if not number_match:
                        continue
                    
                    article_num = int(number_match.group())
                    article_index = article_num - 1
                    
                    if article_index < 0 or article_index >= len(articles):
                        logger.warning(f"Article index {article_index} out of range")
                        continue
                    
                    if article_num in decisions_found:
                        continue  # Skip duplicates
                    decisions_found.add(article_num)
                    
                    # Determine decision - be more strict about ADVANCE detection
                    decision = 'FILTER'  # Default to FILTER
                    
                    # Look for ADVANCE decision with higher score
                    if 'ADVANCE' in rest.upper():
                        # Extract both scores if available
                        advance_score = 0
                        filter_score = 0
                        
                        advance_match = re.search(r'\[ADVANCE\].*?Score:\s*(\d+)', rest, re.IGNORECASE)
                        filter_match = re.search(r'\[FILTER\].*?Score:\s*(\d+)', rest, re.IGNORECASE)
                        
                        if advance_match:
                            advance_score = int(advance_match.group(1))
                        if filter_match:
                            filter_score = int(filter_match.group(1))
                        
                        # Only advance if ADVANCE score is significantly higher than FILTER score
                        if advance_score > filter_score + 20:  # At least 20 point difference
                            decision = 'ADVANCE'
                            score = advance_score
                        else:
                            decision = 'FILTER'
                            score = filter_score if filter_score > 0 else advance_score
                    else:
                        # Extract filter score
                        score_match = re.search(r'Score:\s*(\d+)', rest, re.IGNORECASE)
                        if score_match:
                            score = int(score_match.group(1))
                        else:
                            score = 50  # Default score
                    
                    # Extract reason
                    reason = "Processed by Stage 2"
                    if '-' in rest:
                        reason = rest.split('-', 1)[1].strip()
                    elif '|' in rest and 'FILTER' in rest:
                        # Handle format like "| [FILTER] Score: 45"
                        reason = "Filtered by compound analysis"
                    
                    logger.info(f"Found article decision line: {line}")
                    logger.info(f"Parsed article {article_num}: {decision} with score {score}")
                    
                    # Add to advanced articles if decision is ADVANCE
                    if decision == 'ADVANCE':
                        article = articles[article_index].copy()
                        article['stage2_score'] = score
                        article['stage2_reasoning'] = {'overall': reason}
                        article['stage2_decision'] = 'ADVANCE'
                        article['compound_analysis'] = True
                        advanced_articles.append(article)
                        logger.info(f"Advanced article {article_num} with score {score}")
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing article line: {line} - {e}")
                    continue
        
        # Calculate expected advances (30% target)
        expected_advances = max(1, round(len(articles) * 0.30))
        actual_advances = len(advanced_articles)
        
        logger.info(f"Final parsing result: {actual_advances} articles advanced from {len(articles)} input articles")
        logger.info(f"Expected ~{expected_advances} advances (30% target), got {actual_advances}")
        
        # If we got significantly more advances than expected, be more selective
        if actual_advances > expected_advances * 1.5:  # If more than 150% of target
            logger.warning(f"Too many advances ({actual_advances} vs {expected_advances}), applying stricter filtering")
            
            # Sort by score and take only the top articles
            scored_articles = [(article.get('stage2_score', 0), article) for article in advanced_articles]
            scored_articles.sort(reverse=True, key=lambda x: x[0])
            
            # Take only the expected number of top-scoring articles
            advanced_articles = [article for score, article in scored_articles[:expected_advances]]
            actual_advances = len(advanced_articles)
            
            logger.info(f"Filtered down to {actual_advances} articles using score-based selection")
        
        # If no articles were advanced, log more details for debugging
        if actual_advances == 0:
            logger.warning("No articles were advanced by Stage 2! Checking response format...")
            logger.warning(f"Original response length: {len(original_response)}")
            logger.warning(f"Cleaned response length: {len(response)}")
            logger.warning(f"First 500 chars of original response:\n{original_response[:500]}")
        
        return advanced_articles
    
    def _emergency_fallback_parsing(self, articles: List[Dict]) -> List[Dict]:
        """Emergency fallback when all parsing methods fail - select articles by simple criteria."""
        logger.warning("Using emergency fallback parsing - selecting articles by title keywords")
        
        # Select approximately 30% of articles based on title keywords
        target_count = max(1, round(len(articles) * 0.30))
        advanced_articles = []
        
        # Keywords that indicate high-quality AI/tech content
        high_value_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'llm', 'gpt', 'transformer', 'model', 'algorithm',
            'breakthrough', 'research', 'innovation', 'development', 'technology',
            'open source', 'api', 'framework', 'platform', 'tool', 'release'
        ]
        
        # Score articles by keyword presence
        scored_articles = []
        for i, article in enumerate(articles):
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            combined_text = f"{title} {content}"
            
            score = sum(1 for keyword in high_value_keywords if keyword in combined_text)
            scored_articles.append((score, i, article))
        
        # Sort by score and take top articles
        scored_articles.sort(reverse=True, key=lambda x: x[0])
        
        for score, idx, article in scored_articles[:target_count]:
            article_copy = article.copy()
            article_copy['stage2_score'] = min(85, 60 + score * 5)  # Score between 60-85
            article_copy['stage2_reasoning'] = {'overall': f'Emergency fallback selection (keyword score: {score})'}
            article_copy['stage2_decision'] = 'ADVANCE'
            article_copy['compound_analysis'] = True
            article_copy['emergency_fallback'] = True
            advanced_articles.append(article_copy)
        
        logger.info(f"Emergency fallback selected {len(advanced_articles)}/{len(articles)} articles")
        return advanced_articles

    async def compound_analyze_articles(self, articles: List[Dict], max_articles: Optional[int] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform compound analysis on articles using Stage 2 compound reasoning.
        
        Args:
            articles: List of articles that passed Stage 1
            max_articles: Maximum number of articles to process (for testing)
            task_id: Unique identifier for this task (for parallel processing logs)
        
        Returns:
            Dict containing advanced articles and analysis metrics
        """
        start_time = time.time()
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"S2-{int(time.time() * 1000) % 10000:04d}"
        
        if not articles:
            logger.warning(f"[{task_id}] No articles provided for compound analysis")
            return {
                'model': self.model,
                'stage': 'stage2_compound',
                'input_count': 0,
                'output_count': 0,
                'articles': [],
                'pass_rate': 0.0,
                'target_pass_rate': self.target_pass_rate,
                'requests_made': 0,
                'processing_time': 0,
                'articles_per_second': 0
            }
        
        # Limit articles for testing if specified
        if max_articles:
            articles = articles[:max_articles]
        
        logger.info(f"[{task_id}] Starting Stage 2 compound analysis for {len(articles)} articles")
        logger.info(f"[{task_id}] Rate limiting: ~2.2s between requests, {self.batch_size} articles per batch")
        logger.info(f"[{task_id}] Model limits: {self.requests_per_minute_limit} req/min, {self.tokens_per_minute_limit} tokens/min")
        logger.info(f"[{task_id}] Safe limits: {self.safe_requests_per_minute} req/min, {self.safe_tokens_per_minute} tokens/min")
        logger.info(f"[{task_id}] Estimated time: {(len(articles) / self.batch_size) * (2.2 / 60):.1f} minutes")
        
        all_advanced_articles = []
        total_requests = 0
        
        # Process articles in batches
        for i in range(0, len(articles), self.batch_size):
            if total_requests >= self.max_requests_per_cycle:
                logger.warning(f"[{task_id}] Reached max requests limit ({self.max_requests_per_cycle}), stopping early")
                break
            
            batch = articles[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = min((len(articles) + self.batch_size - 1) // self.batch_size, self.max_requests_per_cycle)
            
            # Progress reporting
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                rate = (i / elapsed_time) * 60  # articles per minute
                remaining_articles = len(articles) - i
                estimated_time_remaining = remaining_articles / rate if rate > 0 else 0
                logger.info(f"[{task_id}] Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")
                logger.info(f"[{task_id}] Progress: {i}/{len(articles)} articles ({i/len(articles)*100:.1f}%), "
                          f"Rate: {rate:.1f} articles/min, ETA: {estimated_time_remaining:.1f}min")
            else:
                logger.info(f"[{task_id}] Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")
            
            # Create compound analysis prompt
            prompt = self._create_compound_analysis_prompt(batch)
            
            # Get compound analysis
            response = await self._call_groq_api(prompt, max_tokens=400, task_id=task_id)
            total_requests += 1
            
            if response:
                # Parse response and get advanced articles
                batch_advanced = self._parse_compound_response(response, batch)
                
                # Check if we got a reasonable number of advances - use fallback if not
                expected_advances = max(1, round(len(batch) * 0.30))
                if len(batch_advanced) == 0 and len(batch) > 1:
                    logger.warning(f"[{task_id}] Batch {batch_num}: No articles advanced, trying fallback method")
                    try:
                        batch_advanced = await self._fallback_simple_analysis(batch, task_id)
                        total_requests += 1  # Count fallback request
                    except Exception as e:
                        logger.error(f"[{task_id}] Fallback method failed: {e}")
                
                all_advanced_articles.extend(batch_advanced)
                
                logger.info(f"[{task_id}] Batch {batch_num}: {len(batch_advanced)}/{len(batch)} articles advanced")
                    
            else:
                logger.error(f"[{task_id}] Failed to process batch {batch_num}")
                # Don't break on single failure, continue with remaining batches
        
        processing_time = time.time() - start_time
        articles_processed = min(len(articles), (total_requests * self.batch_size))
        pass_rate = len(all_advanced_articles) / articles_processed if articles_processed > 0 else 0
        articles_per_second = articles_processed / processing_time if processing_time > 0 else 0
        total_tokens_used = sum(tokens for _, tokens in self.token_usage_window)
        
        result = {
            'model': self.model,
            'stage': 'stage2_compound',
            'input_count': articles_processed,
            'output_count': len(all_advanced_articles),
            'articles': all_advanced_articles,
            'pass_rate': pass_rate,
            'target_pass_rate': self.target_pass_rate,
            'requests_made': total_requests,
            'processing_time': processing_time,
            'articles_per_second': articles_per_second,
            'batch_size': self.batch_size,
            'batches_processed': total_requests,
            'compound_analysis_completed': True,
            'rate_limit_hits': self.consecutive_rate_limits,
            'token_usage_per_minute': total_tokens_used,
            'total_tokens_used': total_tokens_used
        }
        
        # 📊 STAGE 2 SUMMARY LOGGING 📊
        logger.info(f"[{task_id}] 🏁 Stage 2 Complete: {len(all_advanced_articles)}/{articles_processed} articles "
                   f"({pass_rate:.1%}) in {processing_time:.2f}s")
        logger.info(f"[{task_id}] 🔤 Token Summary:")
        logger.info(f"[{task_id}]   📊 Total Tokens Used: {total_tokens_used}")
        logger.info(f"[{task_id}]   📈 Avg Tokens/Request: {total_tokens_used/total_requests:.0f}" if total_requests > 0 else "No requests")
        logger.info(f"[{task_id}]   📈 Avg Tokens/Article: {total_tokens_used/articles_processed:.0f}" if articles_processed > 0 else "No articles")
        logger.info(f"[{task_id}]   ⚡ Model: {self.model} (15,000 tokens/min FREE)")
        logger.info(f"[{task_id}]   💰 Total Cost: $0.00 (FREE MODEL)")
        logger.info(f"[{task_id}]   ⚡ Performance: {articles_per_second:.1f} articles/sec")
        logger.info(f"[{task_id}] 🚫 Rate limit hits: {self.consecutive_rate_limits}")
        
        return result
    
    async def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process a batch of articles for streaming pipeline."""
        # Generate unique task ID for this batch
        task_id = f"S2-{int(time.time() * 1000) % 10000:04d}"
        result = await self.compound_analyze_articles(articles, task_id=task_id)
        return result.get('articles', [])

    async def _fallback_simple_analysis(self, articles: List[Dict], task_id: str) -> List[Dict]:
        """Fallback method with ultra-simple prompt when main prompt fails."""
        logger.warning(f"[{task_id}] Using fallback simple analysis due to format issues")
        
        # Create an extremely simple prompt
        titles = [f"{i+1}. {article.get('title', 'No title')[:80]}" for i, article in enumerate(articles)]
        titles_text = '\n'.join(titles)
        
        simple_prompt = f"""NO <think> tags! Output ONLY the format below:

{titles_text}

Format (advance ~{max(1, round(len(articles) * 0.30))} articles):
1: ADVANCE
2: FILTER
3: ADVANCE
{len(articles)}: FILTER"""

        try:
            await self._rate_limit(task_id)
            response = await self._call_groq_api(simple_prompt, max_tokens=200, task_id=task_id)
            
            if not response:
                logger.error(f"[{task_id}] Fallback analysis failed - no response, using emergency fallback")
                return self._emergency_fallback_parsing(articles)
            
            # Parse simple response
            advanced_articles = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line and 'ADVANCE' in line.upper():
                    try:
                        article_num = int(line.split(':')[0].strip())
                        if 1 <= article_num <= len(articles):
                            article = articles[article_num - 1].copy()
                            article['stage2_score'] = 70  # Default score for fallback
                            article['stage2_reasoning'] = {'overall': 'Fallback analysis - advanced due to format issues'}
                            article['stage2_decision'] = 'ADVANCE'
                            article['compound_analysis'] = True
                            article['fallback_method'] = 'simple'
                            advanced_articles.append(article)
                            logger.info(f"[{task_id}] Fallback advanced article {article_num}")
                    except (ValueError, IndexError):
                        continue
            
            # If still no articles, use emergency fallback
            if not advanced_articles:
                logger.warning(f"[{task_id}] Simple fallback failed, using emergency fallback")
                return self._emergency_fallback_parsing(articles)
            
            logger.info(f"[{task_id}] Fallback analysis: {len(advanced_articles)}/{len(articles)} articles advanced")
            return advanced_articles
            
        except Exception as e:
            logger.error(f"[{task_id}] Fallback analysis failed: {e}, using emergency fallback")
            return self._emergency_fallback_parsing(articles)

async def main():
    """Test the Stage 2 compound agent independently."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stage2_compound_agent.py <articles_json_file>")
        return
    
    # Load test articles
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])[:10]  # Test with first 10 articles
    
    print(f"Testing Stage 2 Compound Agent with {len(articles)} articles")
    
    async with Stage2CompoundAgent() as agent:
        result = await agent.compound_analyze_articles(articles)
    
    print("\nStage 2 Results:")
    print(f"Model: {result['model']}")
    print(f"Input: {result['input_count']} articles")
    print(f"Output: {result['output_count']} articles")
    print(f"Pass rate: {result['pass_rate']:.1%} (target: {result['target_pass_rate']:.1%})")
    print(f"Requests made: {result['requests_made']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Articles/second: {result['articles_per_second']:.1f}")
    
    # Show sample advanced articles
    if result['articles']:
        print(f"\nSample Advanced Articles:")
        for i, article in enumerate(result['articles'][:3]):
            print(f"\n{i+1}. {article.get('title', 'No title')}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Stage 2 Score: {article.get('stage2_score', 'N/A')}")
            if article.get('stage2_reasoning'):
                reasoning = article['stage2_reasoning']
                print(f"   Technical: {reasoning.get('technical', 'N/A')}")
                print(f"   Impact: {reasoning.get('impact', 'N/A')}")
                print(f"   Overall: {reasoning.get('overall', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
