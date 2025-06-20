#!/usr/bin/env python3
"""
Stage 1: Bulk Processing Filter

First stage of the optimized 4-stage pipeline using meta-llama/llama-4-scout-17b-16e-instruct.
Processes up to 2,988 articles in batches of 18 articles per request.
Target: Filter down to 60% of input articles (high-quality but broad filtering).
"""

import os
import json
import logging
import asyncio
import aiohttp
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

class Stage1BulkFilter:
    """Stage 1 Bulk Processing Filter - High-throughput initial filtering."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the Stage 1 bulk filter."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Optimized for bulk processing
        self.session = None
        
        # Stage 1 optimized settings - Updated for meta-llama/llama-4-scout-17b-16e-instruct
        # Model limits: 30 req/min, 30,000 tokens/min, 1,000 req/day
        self.requests_per_minute_limit = 30
        self.tokens_per_minute_limit = 30000
        
        # Conservative rate limiting (90% of limits since Stage 1 has better token limits)
        self.safe_requests_per_minute = int(self.requests_per_minute_limit * 0.9)  # 27 req/min
        self.safe_tokens_per_minute = int(self.tokens_per_minute_limit * 0.9)     # 27,000 tokens/min
        
        self.rate_limit_delay = 60.0 / self.safe_requests_per_minute  # ~2.2s between requests
        
        # Load configuration if provided
        if config:
            self.batch_size = config.get('batch_size', 5)
            self.max_requests_per_cycle = config.get('max_requests_per_cycle', 15)
            self.target_pass_rate = config.get('target_pass_rate', 0.60)
        else:
            self.batch_size = 5  # Smaller batches for testing
            self.max_requests_per_cycle = 15  # Increased to handle 50 articles
            self.target_pass_rate = 0.60  # 60% pass rate as specified
            
        self.tokens_per_article = 100  # Average tokens per article analysis
        
        # Rate limit tracking
        self.last_request_time = 0
        self.max_retries = 5
        self.base_retry_delay = 2.0  # Base delay for exponential backoff
        self.request_timestamps = []  # Track request times
        self.token_usage_window = []  # Track token usage
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=45),  # Longer timeout for batch processing
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
    
    async def _rate_limit(self):
        """Enforce rate limiting for Stage 1."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _create_batch_filtering_prompt(self, articles: List[Dict]) -> str:
        """Create the bulk filtering prompt for a batch of articles."""
        
        # Create article summaries for efficient processing
        article_summaries = []
        for i, article in enumerate(articles):
            title = article.get('title', 'No title')[:100]
            description = article.get('description', 'No description')[:200]
            source = article.get('source', 'Unknown')
            
            article_summaries.append(f"{i+1}. [{source}] {title}\n   {description}")
        
        articles_text = "\n\n".join(article_summaries)
        
        prompt = f"""You are a Stage 1 bulk filter for AI/tech news. Your job is to identify high-potential articles with genuine AI/technology value.

FILTERING CRITERIA (Stage 1 - Selective but Efficient):
✅ INCLUDE articles about:
- Significant AI/ML research, breakthroughs, or novel applications
- Major technology developments or innovations
- Advanced software engineering or architectural advances
- Tech industry news with substantial impact
- Important open source projects or major releases
- Research papers with clear technical contributions
- Robotics, automation, or cutting-edge hardware
- Emerging technologies with real potential

❌ EXCLUDE articles that are:
- Basic marketing/promotional content
- Simple tutorials, how-to guides, or listicles
- Minor product updates or routine announcements
- Celebrity tech news or industry gossip
- Basic company news without technical substance
- General business news without innovation focus
- Rehashed or derivative content
- Opinion pieces without technical depth

TARGET: Accept ~70% of articles (be moderately selective - quality over quantity).

ARTICLES TO EVALUATE:
{articles_text}

RESPONSE FORMAT:
For each article, provide: [INCLUDE/EXCLUDE] Brief reason (max 10 words)

Example:
1. [INCLUDE] Novel AI research breakthrough
2. [EXCLUDE] Basic promotional content
3. [INCLUDE] Significant technical innovation

Respond for articles 1-{len(articles)}:"""

        return prompt
    
    async def _call_groq_api(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Make a call to the Groq API for Stage 1 bulk processing."""
        if not self.session:
            logger.error("Session not initialized")
            return None
        
        estimated_input_tokens = len(prompt) // 4
        
        for attempt in range(self.max_retries):
            await self._rate_limit()
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Stage 1 bulk filter for AI/tech news. Focus on speed and broad filtering - accept quality articles while rejecting obvious low-value content. Be decisive and consistent."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,  # Low temperature for consistent bulk filtering
                "top_p": 0.9
            }
            
            call_start_time = time.time()
            
            try:
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                    processing_time = time.time() - call_start_time
                    
                    rate_limit_remaining = response.headers.get('x-ratelimit-remaining-requests')
                    rate_limit_reset = response.headers.get('x-ratelimit-reset-requests')
                    request_id = response.headers.get('x-request-id')
                    
                    if response.status == 200:
                        data = await response.json()
                        response_content = data['choices'][0]['message']['content'].strip()
                        
                        usage = data.get('usage', {})
                        actual_input_tokens = usage.get('prompt_tokens', estimated_input_tokens)
                        actual_output_tokens = usage.get('completion_tokens', len(response_content) // 4)
                        
                        # Record successful API call
                        usage_tracker.record_call(
                            model=self.model,
                            endpoint="chat/completions",
                            request_tokens=actual_input_tokens,
                            response_tokens=actual_output_tokens,
                            processing_time=processing_time,
                            status_code=response.status,
                            success=True,
                            agent="stage1_bulk_filter",
                            request_id=request_id,
                            rate_limit_remaining=int(rate_limit_remaining) if rate_limit_remaining else None,
                            rate_limit_reset=rate_limit_reset
                        )
                        
                        return response_content
                        
                    elif response.status == 401:
                        error_text = await response.text()
                        logger.error(f"Groq API error {response.status}: {error_text}")
                        
                        usage_tracker.record_call(
                            model=self.model,
                            endpoint="chat/completions",
                            request_tokens=estimated_input_tokens,
                            response_tokens=0,
                            processing_time=processing_time,
                            status_code=response.status,
                            success=False,
                            agent="stage1_bulk_filter",
                            error_message=f"Authentication failed: {error_text}",
                            request_id=request_id
                        )
                        
                        raise ValueError("Invalid API key - authentication failed")
                        
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Stage 1 rate limit hit on attempt {attempt + 1}: {error_text}")
                        
                        usage_tracker.record_call(
                            model=self.model,
                            endpoint="chat/completions",
                            request_tokens=estimated_input_tokens,
                            response_tokens=0,
                            processing_time=processing_time,
                            status_code=response.status,
                            success=False,
                            agent="stage1_bulk_filter",
                            error_message=f"Rate limit hit: {error_text}",
                            request_id=request_id,
                            rate_limit_remaining=int(rate_limit_remaining) if rate_limit_remaining else None,
                            rate_limit_reset=rate_limit_reset
                        )
                        
                        if attempt < self.max_retries - 1:
                            wait_time = self.base_retry_delay * (2 ** attempt)
                            logger.info(f"Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error("Max retries reached for rate limit")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Groq API error {response.status}: {error_text}")
                        
                        usage_tracker.record_call(
                            model=self.model,
                            endpoint="chat/completions",
                            request_tokens=estimated_input_tokens,
                            response_tokens=0,
                            processing_time=processing_time,
                            status_code=response.status,
                            success=False,
                            agent="stage1_bulk_filter",
                            error_message=f"API error: {error_text}",
                            request_id=request_id
                        )
                        
                        if attempt < self.max_retries - 1:
                            wait_time = self.base_retry_delay * (2 ** attempt)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return None
                            
            except aiohttp.ClientError as e:
                logger.error(f"Stage 1 network error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.base_retry_delay * (2 ** attempt))
                    continue
                else:
                    return None
        
        return None
    
    def _parse_batch_response(self, response: str, articles: List[Dict]) -> Tuple[List[Dict], int]:
        """Parse the batch filtering response and return filtered articles."""
        if not response:
            logger.warning("Empty response from Stage 1 API")
            return articles, 0  # Return all articles if parsing fails
        
        lines = response.strip().split('\n')
        included_articles = []
        include_count = 0
        exclude_count = 0
        
        # Parse each line for INCLUDE/EXCLUDE decisions
        for i, line in enumerate(lines):
            if i >= len(articles):
                break
                
            line = line.strip()
            if '[INCLUDE]' in line.upper():
                if i < len(articles):
                    article = articles[i].copy()
                    # Add Stage 1 filtering metadata
                    article['stage1_filter'] = {
                        'decision': 'INCLUDE',
                        'reason': line.split(']', 1)[1].strip() if ']' in line else 'Stage 1 approved',
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'model': self.model
                    }
                    included_articles.append(article)
                    include_count += 1
            elif '[EXCLUDE]' in line.upper():
                exclude_count += 1
                # Don't include excluded articles
            else:
                # If parsing is unclear, default to include (Stage 1 is broad)
                if i < len(articles):
                    article = articles[i].copy()
                    article['stage1_filter'] = {
                        'decision': 'INCLUDE',
                        'reason': 'Default include (unclear response)',
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'model': self.model
                    }
                    included_articles.append(article)
                    include_count += 1
        
        logger.info(f"Stage 1 batch result: {include_count} included, {exclude_count} excluded")
        return included_articles, include_count
    
    async def bulk_filter_articles(self, articles: List[Dict], max_articles: int = 25) -> Dict[str, Any]:
        """
        Perform Stage 1 bulk filtering on articles.
        
        Args:
            articles: List of articles to filter
            max_articles: Maximum articles to process (default: 2988 per cycle)
            
        Returns:
            Dictionary with filtering results and metadata
        """
        start_time = time.time()
        
        # Limit input to maximum capacity
        input_articles = articles[:max_articles]
        
        logger.info(f"Stage 1: Starting bulk filtering of {len(input_articles)} articles")
        logger.info(f"Target: {self.batch_size} articles/request, {self.max_requests_per_cycle} max requests")
        
        filtered_articles = []
        total_requests = 0
        total_includes = 0
        total_excludes = 0
        
        # Process articles in batches
        for i in range(0, len(input_articles), self.batch_size):
            if total_requests >= self.max_requests_per_cycle:
                logger.warning(f"Stage 1: Reached max requests limit ({self.max_requests_per_cycle})")
                break
                
            batch = input_articles[i:i + self.batch_size]
            logger.info(f"Processing batch {total_requests + 1}/{min(len(input_articles) // self.batch_size + 1, self.max_requests_per_cycle)}: {len(batch)} articles")
            
            # Create prompt for this batch
            prompt = self._create_batch_filtering_prompt(batch)
            
            # Call API
            response = await self._call_groq_api(prompt, max_tokens=len(batch) * 50)  # Scale tokens with batch size
            
            if response:
                # Parse response and get filtered articles
                batch_filtered, batch_includes = self._parse_batch_response(response, batch)
                filtered_articles.extend(batch_filtered)
                total_includes += batch_includes
                total_excludes += len(batch) - batch_includes
                total_requests += 1
            else:
                logger.warning(f"Stage 1: Failed to process batch {total_requests + 1}, including all articles")
                # On failure, include all articles from batch with failure metadata
                for article in batch:
                    article_copy = article.copy()
                    article_copy['stage1_filter'] = {
                        'decision': 'INCLUDE',
                        'reason': 'API failure - default include',
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'model': self.model,
                        'api_failed': True
                    }
                    filtered_articles.append(article_copy)
                total_includes += len(batch)
                total_requests += 1
            
            # Brief pause between batches
            await asyncio.sleep(0.5)
        
        processing_time = time.time() - start_time
        pass_rate = total_includes / len(input_articles) if input_articles else 0
        
        result = {
            'stage': 'stage1_bulk_filter',
            'model': self.model,
            'input_count': len(input_articles),
            'output_count': len(filtered_articles),
            'includes': total_includes,
            'excludes': total_excludes,
            'pass_rate': pass_rate,
            'target_pass_rate': self.target_pass_rate,
            'requests_made': total_requests,
            'max_requests': self.max_requests_per_cycle,
            'batch_size': self.batch_size,
            'processing_time': processing_time,
            'articles_per_second': len(input_articles) / processing_time if processing_time > 0 else 0,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'articles': filtered_articles
        }
        
        logger.info(f"Stage 1 complete: {len(filtered_articles)}/{len(input_articles)} articles passed ({pass_rate:.1%})")
        logger.info(f"Processing time: {processing_time:.2f}s, Rate: {result['articles_per_second']:.1f} articles/sec")
        
        return result
    
    async def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process a batch of articles for streaming pipeline."""
        result = await self.bulk_filter_articles(articles)
        return result.get('articles', [])


async def main():
    """Test the Stage 1 bulk filter."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stage1_bulk_filter.py <articles_json_file>")
        return
    
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    
    async with Stage1BulkFilter() as filter_agent:
        result = await filter_agent.bulk_filter_articles(articles)
    
    print(f"Stage 1 Results:")
    print(f"Input: {result['input_count']} articles")
    print(f"Output: {result['output_count']} articles")
    print(f"Pass Rate: {result['pass_rate']:.1%}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
    print(f"Requests Made: {result['requests_made']}")


if __name__ == "__main__":
    asyncio.run(main())
