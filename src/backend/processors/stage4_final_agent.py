#!/usr/bin/env python3
"""
Stage 4: Final Ranking Agent

Fourth and final stage of the optimized 4-stage pipeline using qwen/qwen3-32b.
Processes articles that passed Stage 3 expert analysis with comprehensive evaluation and final ranking.
Target: Select the best 25 articles from ~108 candidates through individual assessment.
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
            def get_usage_summary(self):
                return type('UsageSummary', (), {
                    'total_request_tokens': 0,
                    'total_response_tokens': 0,
                    'total_tokens': 0,
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0
                })()
        usage_tracker = DummyTracker()

logger = logging.getLogger(__name__)

class Stage4FinalAgent:
    """Stage 4 Final Ranking - Comprehensive evaluation and final selection."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the Stage 4 final ranking agent."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Load configuration if provided
        if config:
            self.model = config.get('model', 'llama-3.1-8b-instant')
            self.final_count = config.get('final_count', 25)
            self.time_budget_minutes = config.get('time_budget_minutes', 20)
            self.max_requests_per_cycle = config.get('max_requests_per_cycle', 30)
        else:
            self.model = "llama-3.1-8b-instant"  # Same reliable free model as Stage 3
            self.final_count = 25  # Target final article count
            self.time_budget_minutes = 20
            self.max_requests_per_cycle = 30
            
        self.session = None
        
        # Stage 4 optimized settings - Using llama-3.1-8b-instant (FREE MODEL)
        # Model limits: 30 req/min, 6,000 tokens/min, 500k tokens/day (FREE)
        # IMPORTANT: After Stages 1-3, we need very conservative rate limiting
        self.requests_per_minute_limit = 30
        self.tokens_per_minute_limit = 6000
        self.requests_per_day_limit = 14400
        
        # Very conservative rate limiting for Stage 4 (use 60% of limits after previous stages)
        self.safe_requests_per_minute = int(self.requests_per_minute_limit * 0.6)  # 18 req/min
        self.safe_tokens_per_minute = int(self.tokens_per_minute_limit * 0.6)     # 3,600 tokens/min
        
        self.rate_limit_delay = 60.0 / self.safe_requests_per_minute  # ~3.3s between requests
        self.tokens_per_article = 400  # Conservative tokens per article analysis
        
        # Individual processing (1 article per request for maximum quality)
        self.batch_size = 1
        self.max_articles_to_process = 108  # Based on pipeline capacity
        
        # Rate limit tracking
        self.last_request_time = 0
        self.max_retries = 3
        self.base_retry_delay = 5.0  # Much longer delay for Stage 4 rate limit retries
        self.request_timestamps = []
        self.token_usage_window = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),  # Standard timeout for individual processing
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
        """Enforce rate limiting for Stage 4 with improved safety."""
        current_time = time.time()
        
        # Always wait minimum delay between requests (extra conservative for Stage 4)
        time_since_last = current_time - self.last_request_time
        min_delay = self.rate_limit_delay
        
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        else:
            # Even if enough time has passed, add a small buffer
            buffer_delay = 0.5  # Extra 0.5s buffer
            logger.debug(f"Adding rate limit buffer: sleeping {buffer_delay:.2f}s")
            await asyncio.sleep(buffer_delay)
        
        self.last_request_time = time.time()
    
    def _create_final_ranking_prompt(self, article: Dict) -> str:
        """Create the comprehensive evaluation prompt for final ranking."""
        
        title = article.get('title', 'No title')
        description = article.get('description', article.get('summary', 'No description'))
        source = article.get('source', 'Unknown')
        url = article.get('url', 'No URL')
        
        # Get enrichment data from previous stages
        stage1_reason = article.get('stage1_reason', 'No Stage 1 analysis')
        stage2_analysis = article.get('stage2_analysis', 'No Stage 2 analysis')
        stage3_category = article.get('category', 'uncategorized')
        stage3_quality = article.get('quality_score', 0)
        stage3_reasoning = article.get('stage3_reasoning', 'No Stage 3 analysis')
        
        # Published date for recency evaluation
        published_date = article.get('published_date', article.get('pubDate', 'Unknown'))
        
        return f"""You are a news curator performing final evaluation. Rate this article across 6 criteria and provide a comprehensive score.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Description: {description}

PREVIOUS ANALYSIS:
Stage 1: {stage1_reason}
Stage 2: {stage2_analysis}
Stage 3: Category={stage3_category}, Quality={stage3_quality}/10, Reasoning={stage3_reasoning}

EVALUATION CRITERIA (Score each 0-100):
1. SIGNIFICANCE & IMPACT: Global importance, affects many people, long-term consequences
2. TIMELINESS & RELEVANCE: Breaking news, recent developments, time-sensitive
3. CREDIBILITY & SOURCE: Reputable organization, authoritative sources, factual accuracy
4. UNIQUENESS & NOVELTY: Exclusive reporting, unique angle, not widely covered
5. AUDIENCE VALUE: Actionable information, educational value, decision-making utility
6. PRESENTATION QUALITY: Clear headline, well-written, comprehensive coverage

SCORING GUIDELINES:
- 85-100: Excellent (top-tier news)
- 70-84: Good (solid journalism)
- 55-69: Fair (average content)
- 0-54: Poor (reject)

FINAL RECOMMENDATION:
- ACCEPT if total score ≥ 70
- REJECT if total score < 70

Return ONLY valid JSON:
{{
    "significance_impact": {{"score": [0-100], "reasoning": "brief explanation"}},
    "timeliness_relevance": {{"score": [0-100], "reasoning": "brief explanation"}},
    "credibility_source": {{"score": [0-100], "reasoning": "brief explanation"}},
    "uniqueness_novelty": {{"score": [0-100], "reasoning": "brief explanation"}},
    "audience_value": {{"score": [0-100], "reasoning": "brief explanation"}},
    "presentation_quality": {{"score": [0-100], "reasoning": "brief explanation"}},
    "total_score": [average of 6 scores],
    "recommendation": "ACCEPT or REJECT",
    "final_reasoning": "comprehensive justification",
    "ranking_priority": "HIGH or MEDIUM or LOW"
}}"""
    
    async def _make_groq_request(self, prompt: str, article: Dict) -> Optional[Dict]:
        """Make a request to Groq API for final ranking evaluation."""
        
        if not self.session:
            logger.error("Session not initialized. Use async context manager.")
            return None
            
        await self._rate_limit()
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Low temperature for consistent evaluation
            "max_tokens": 1200,  # Comprehensive analysis
            "top_p": 0.9
        }
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status == 429:
                        retry_delay = self.base_retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit on attempt {attempt + 1}, waiting {retry_delay}s")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    response_data = await response.json()
                    
                    if response.status != 200:
                        error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"Groq API error (attempt {attempt + 1}): {error_msg}")
                        
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.base_retry_delay * (2 ** attempt))
                            continue
                        else:
                            # Record failed call
                            usage_tracker.record_call(
                                agent="stage4_final",
                                model=self.model,
                                success=False,
                                processing_time=processing_time,
                                error_message=error_msg,
                                request_tokens=0,
                                response_tokens=0,
                                status_code=response.status
                            )
                            return None
                    
                    # Parse usage information
                    usage = response_data.get('usage', {})
                    request_tokens = usage.get('prompt_tokens', 0)
                    response_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', request_tokens + response_tokens)
                    
                    # Record successful call
                    usage_tracker.record_call(
                        agent="stage4_final",
                        model=self.model,
                        success=True,
                        processing_time=processing_time,
                        request_tokens=request_tokens,
                        response_tokens=response_tokens,
                        status_code=response.status
                    )
                    
                    # Log token usage for this request
                    logger.info(f"[S4] 🔤 Stage 4 Token Usage:")
                    logger.info(f"[S4]   📊 This Request: {request_tokens} prompt + {response_tokens} completion = {total_tokens} total tokens")
                    logger.info(f"[S4]   ⚡ Model: {self.model} (6,000 tokens/min, 500k daily limit)")
                    logger.info(f"[S4]   💰 Cost: $0.00 (FREE MODEL)")
                    
                    content = response_data['choices'][0]['message']['content'].strip()
                    
                    # Parse JSON response with improved error handling
                    try:
                        # First, try direct JSON parsing
                        evaluation = json.loads(content)
                        return evaluation
                    except json.JSONDecodeError:
                        logger.debug(f"Direct JSON parsing failed, trying cleanup methods")
                        
                        # Try to clean common formatting issues
                        cleaned_content = content.strip()
                        
                        # Remove markdown code blocks
                        if cleaned_content.startswith('```json'):
                            cleaned_content = cleaned_content[7:]
                        elif cleaned_content.startswith('```'):
                            cleaned_content = cleaned_content[3:]
                        if cleaned_content.endswith('```'):
                            cleaned_content = cleaned_content[:-3]
                        
                        cleaned_content = cleaned_content.strip()
                        
                        # Try parsing cleaned content
                        try:
                            evaluation = json.loads(cleaned_content)
                            return evaluation
                        except json.JSONDecodeError:
                            pass
                        
                        # Try to extract JSON object using regex
                        import re
                        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                        json_matches = re.findall(json_pattern, content, re.DOTALL)
                        
                        for match in json_matches:
                            try:
                                evaluation = json.loads(match)
                                # Validate it has required fields
                                if 'total_score' in evaluation and 'recommendation' in evaluation:
                                    return evaluation
                            except json.JSONDecodeError:
                                continue
                        
                        # Final fallback: Create structured evaluation with neutral score
                        logger.warning(f"All JSON parsing failed, creating fallback evaluation")
                        return {
                            "significance_impact": {"score": 60, "reasoning": "Parsing failed - using default"},
                            "timeliness_relevance": {"score": 60, "reasoning": "Parsing failed - using default"},
                            "credibility_source": {"score": 60, "reasoning": "Parsing failed - using default"},
                            "uniqueness_novelty": {"score": 60, "reasoning": "Parsing failed - using default"},
                            "audience_value": {"score": 60, "reasoning": "Parsing failed - using default"},
                            "presentation_quality": {"score": 60, "reasoning": "Parsing failed - using default"},
                            "total_score": 60,
                            "recommendation": "REJECT",
                            "final_reasoning": f"JSON parsing failed. Raw response: {content[:300]}...",
                            "ranking_priority": "LOW"
                        }
                        
            except Exception as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.base_retry_delay * (2 ** attempt))
                    continue
                else:
                    # Record failed call
                    usage_tracker.record_call(
                        agent="stage4_final",
                        model=self.model,
                        success=False,
                        processing_time=time.time() - start_time,
                        error_message=str(e),
                        request_tokens=0,
                        response_tokens=0,
                        status_code=0
                    )
                    return None
        
        return None
    
    async def final_rank_articles(self, articles: List[Dict], target_count: int = 25) -> Dict[str, Any]:
        """
        Perform final ranking and selection of the best articles.
        
        Args:
            articles: List of articles from Stage 3
            target_count: Number of final articles to select (default 25)
        
        Returns:
            Dict containing the final ranked articles and processing metrics
        """
        start_time = time.time()
        logger.info(f"Stage 4: Starting final ranking of {len(articles)} articles -> {target_count}")
        
        if not articles:
            logger.warning("No articles provided for final ranking")
            return {
                'model': self.model,
                'input_count': 0,
                'output_count': 0,
                'articles': [],
                'target_count': target_count,
                'processing_time': 0,
                'requests_made': 0,
                'articles_per_second': 0,
                'error': 'no_articles_provided'
            }
        
        # Limit articles to maximum capacity
        articles_to_process = articles[:self.max_articles_to_process]
        if len(articles) > self.max_articles_to_process:
            logger.warning(f"Limiting input from {len(articles)} to {self.max_articles_to_process} articles")
        
        # Process each article individually for maximum quality
        evaluated_articles = []
        successful_evaluations = 0
        failed_evaluations = 0
        total_request_tokens = 0
        total_response_tokens = 0
        
        logger.info(f"Processing {len(articles_to_process)} articles individually...")
        
        # Add initial delay to let rate limits reset after previous stages
        initial_delay = 5.0  # 5 second delay to let previous API calls clear
        logger.info(f"Waiting {initial_delay}s for rate limits to reset after previous stages...")
        await asyncio.sleep(initial_delay)
        
        for i, article in enumerate(articles_to_process):
            try:
                logger.debug(f"Evaluating article {i+1}/{len(articles_to_process)}: {article.get('title', 'No title')[:50]}")
                
                # Add article-level rate limiting info
                logger.debug(f"Stage 4 rate limiting: {self.rate_limit_delay:.1f}s delay, model: {self.model}")
                
                # Create comprehensive evaluation prompt
                prompt = self._create_final_ranking_prompt(article)
                
                # Make API request
                evaluation = await self._make_groq_request(prompt, article)
                
                if evaluation:
                    # Add evaluation data to article
                    enriched_article = article.copy()
                    enriched_article.update({
                        'stage4_evaluation': evaluation,
                        'final_score': evaluation.get('total_score', 0),
                        'recommendation': evaluation.get('recommendation', 'REJECT'),
                        'ranking_priority': evaluation.get('ranking_priority', 'LOW'),
                        'final_reasoning': evaluation.get('final_reasoning', ''),
                        'processed_at': datetime.now(timezone.utc).isoformat()
                    })
                    
                    evaluated_articles.append(enriched_article)
                    successful_evaluations += 1
                    
                    logger.debug(f"Article evaluated: score={evaluation.get('total_score', 0)}, recommendation={evaluation.get('recommendation', 'REJECT')}")
                else:
                    # Keep article but mark as failed evaluation
                    fallback_article = article.copy()
                    fallback_article.update({
                        'stage4_evaluation': None,
                        'final_score': 50,  # Neutral score for failed evaluations
                        'recommendation': 'REJECT',
                        'ranking_priority': 'LOW',
                        'final_reasoning': 'Evaluation failed - API error or parsing issue',
                        'evaluation_failed': True,
                        'processed_at': datetime.now(timezone.utc).isoformat()
                    })
                    
                    evaluated_articles.append(fallback_article)
                    failed_evaluations += 1
                    
                    logger.warning(f"Failed to evaluate article {i+1}: {article.get('title', 'No title')[:50]}")
                
            except Exception as e:
                logger.error(f"Error processing article {i+1}: {e}")
                failed_evaluations += 1
                continue
        
        # Sort articles by final score (highest first)
        evaluated_articles.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Filter to only accepted articles
        accepted_articles = [a for a in evaluated_articles if a.get('recommendation') == 'ACCEPT']
        
        # If we don't have enough accepted articles, take the highest scored ones
        if len(accepted_articles) < target_count:
            logger.warning(f"Only {len(accepted_articles)} articles accepted, taking top {target_count} by score")
            final_articles = evaluated_articles[:target_count]
            
            # Mark additional articles as accepted
            for i in range(len(accepted_articles), min(target_count, len(final_articles))):
                final_articles[i]['recommendation'] = 'ACCEPT'
                final_articles[i]['final_reasoning'] += ' (Promoted to meet target count)'
        else:
            # Take the top accepted articles
            final_articles = accepted_articles[:target_count]
        
        processing_time = time.time() - start_time
        articles_per_second = len(articles_to_process) / processing_time if processing_time > 0 else 0
        
        # Get final token usage statistics from usage tracker
        try:
            stage4_usage = usage_tracker.get_usage_summary()
            total_request_tokens = getattr(stage4_usage, 'total_request_tokens', 0)
            total_response_tokens = getattr(stage4_usage, 'total_response_tokens', 0)
            total_tokens = getattr(stage4_usage, 'total_tokens', 0)
        except (AttributeError, Exception):
            # Fallback if usage tracker doesn't have the method or fails
            total_request_tokens = 0
            total_response_tokens = 0
            total_tokens = 0
        
        # Calculate quality distribution
        score_distribution = {
            'excellent': len([a for a in evaluated_articles if a.get('final_score', 0) >= 85]),
            'good': len([a for a in evaluated_articles if 75 <= a.get('final_score', 0) < 85]),
            'fair': len([a for a in evaluated_articles if 60 <= a.get('final_score', 0) < 75]),
            'poor': len([a for a in evaluated_articles if a.get('final_score', 0) < 60])
        }
        
        priority_distribution = {
            'HIGH': len([a for a in final_articles if a.get('ranking_priority') == 'HIGH']),
            'MEDIUM': len([a for a in final_articles if a.get('ranking_priority') == 'MEDIUM']),
            'LOW': len([a for a in final_articles if a.get('ranking_priority') == 'LOW'])
        }
        
        result = {
            'model': self.model,
            'input_count': len(articles_to_process),
            'output_count': len(final_articles),
            'target_count': target_count,
            'articles': final_articles,
            'processing_time': processing_time,
            'requests_made': successful_evaluations + failed_evaluations,
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'articles_per_second': articles_per_second,
            'acceptance_rate': len([a for a in evaluated_articles if a.get('recommendation') == 'ACCEPT']) / len(evaluated_articles) if evaluated_articles else 0,
            'average_score': sum(a.get('final_score', 0) for a in evaluated_articles) / len(evaluated_articles) if evaluated_articles else 0,
            'score_distribution': score_distribution,
            'priority_distribution': priority_distribution,
            'token_usage': {
                'total_request_tokens': total_request_tokens,
                'total_response_tokens': total_response_tokens,
                'total_tokens': total_tokens,
                'average_tokens_per_request': total_tokens / successful_evaluations if successful_evaluations > 0 else 0
            },
            'quality_metrics': {
                'top_score': max(a.get('final_score', 0) for a in evaluated_articles) if evaluated_articles else 0,
                'bottom_score': min(a.get('final_score', 0) for a in evaluated_articles) if evaluated_articles else 0,
                'median_score': sorted([a.get('final_score', 0) for a in evaluated_articles])[len(evaluated_articles)//2] if evaluated_articles else 0
            }
        }
        
        logger.info(f"Stage 4 complete: {len(final_articles)} final articles selected")
        logger.info(f"  Average score: {result['average_score']:.1f}, Acceptance rate: {result['acceptance_rate']:.1%}")
        logger.info(f"  Processing: {successful_evaluations} successful, {failed_evaluations} failed in {processing_time:.2f}s")
        logger.info(f"  Token usage: {total_tokens} total ({total_request_tokens} req + {total_response_tokens} resp)")
        logger.info(f"  Priority distribution: HIGH={priority_distribution['HIGH']}, MED={priority_distribution['MEDIUM']}, LOW={priority_distribution['LOW']}")
        
        return result
    
    async def rank_articles(self, articles: List[Dict], target_count: int) -> List[Dict]:
        """Rank articles for streaming pipeline."""
        result = await self.final_rank_articles(articles, target_count)
        return result.get('articles', [])
    
    async def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process a batch of articles for streaming pipeline (alias for rank_articles)."""
        # Stage 4 typically handles all articles at once for final ranking
        return articles  # Pass through for now, actual ranking happens in rank_articles
