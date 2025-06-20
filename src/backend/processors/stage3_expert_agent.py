#!/usr/bin/env python3
"""
Stage 3: Expert Analysis Agent

Third stage of the optimized 4-stage pipeline using llama-3.3-70b-versatile.
Processes articles that passed Stage 2 compound analysis with expert-level categorization.
Target: Filter down to 20% of input articles (expert categorization and quality assessment).
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

class Stage3ExpertAgent:
    """Stage 3 Expert Analysis - Professional categorization and quality assessment."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the Stage 3 expert agent."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Load configuration if provided
        if config:
            self.model = config.get('model', 'llama-3.1-8b-instant')
            self.target_pass_rate = config.get('target_pass_rate', 0.35)
            self.time_budget_minutes = config.get('time_budget_minutes', 10)
            self.articles_per_request = config.get('batch_size', 4)
            self.max_requests_per_cycle = config.get('max_requests_per_cycle', 15)
            self.daily_token_limit = config.get('daily_token_limit', 10000)
        else:
            self.model = "llama-3.1-8b-instant"  # Reliable free model with good JSON output
            self.target_pass_rate = 0.35  # Updated pass rate
            self.time_budget_minutes = 10
            self.articles_per_request = 4  # Batch size for expert analysis
            self.max_requests_per_cycle = 15  # Reduced to match config and stay under rate limits
            self.daily_token_limit = 10000  # Not enforced due to unlimited daily tokens
            
        self.session = None
        
        # Stage 3 optimized settings - Updated for llama-3.1-8b-instant
        # Model limits: 30 req/min, 14,400 req/day, 6,000 tokens/min, 500,000 tokens/day (FREE)
        self.requests_per_minute_limit = 30
        self.tokens_per_minute_limit = 6000
        self.tokens_per_day_limit = 500000  # High daily limit = FREE
        self.requests_per_day_limit = 14400
        
        # Safe limits (95% of max for optimal performance without rate limiting)
        self.safe_requests_per_minute = int(self.requests_per_minute_limit * 0.95)  # 28 req/min
        self.safe_tokens_per_minute = int(self.tokens_per_minute_limit * 0.95)     # 5,700 tokens/min
        
        # Pipeline configuration - optimized for llama-3.1-8b-instant (FREE model)
        self.tokens_per_article = 150  # Increased for expert analysis
        
        # Calculate processing capacity
        self.max_requests_per_cycle = min(
            self.max_requests_per_cycle,  # From config or default
            int(self.time_budget_minutes * self.safe_requests_per_minute)  # Time-based limit
        )
        
        # Token constraint - high daily limit (500k tokens)
        self.tokens_per_cycle = 100000  # Higher limit since model is free with high daily limit
        self.max_articles_per_cycle = self.tokens_per_cycle // self.tokens_per_article
        
        # Rate limiting - optimized for llama-3.1-8b-instant
        self.rate_limit_delay = 60.0 / self.safe_requests_per_minute  # ~2.1 seconds between requests
        
        # Categories for expert analysis
        self.expert_categories = {
            'breakthrough': 'Major breakthrough or paradigm shift',
            'research': 'Significant research advancement',
            'industry': 'Important industry development',
            'safety': 'AI safety or ethics concern',
            'policy': 'Government or regulatory development',
            'open_source': 'Open source release or community development',
            'startup': 'Startup funding or product launch',
            'acquisition': 'Company acquisition or merger',
            'partnership': 'Strategic partnership or collaboration',
            'controversy': 'Significant controversy or debate'
        }
        
        logger.info(f"Stage 3 Expert Agent initialized")
        logger.info(f"Model: {self.model}")
        logger.info(f"Capacity: {self.max_articles_per_cycle} articles/cycle, {self.max_requests_per_cycle} requests/cycle")
        logger.info(f"Rate: {self.articles_per_request} articles/request, {self.rate_limit_delay:.1f}s delay")
        logger.info(f"Target pass rate: {self.target_pass_rate:.1%}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
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
        """Apply rate limiting between requests."""
        if self.rate_limit_delay > 0:
            await asyncio.sleep(self.rate_limit_delay)
    
    def _create_expert_analysis_prompt(self, articles: List[Dict]) -> str:
        """Create expert analysis prompt for a batch of articles."""
        
        # Create article summaries for analysis
        article_summaries = []
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            description = article.get('description', 'No description')
            source = article.get('source', 'Unknown source')
            
            # Include any existing analysis from previous stages
            stage1_analysis = article.get('stage1_analysis', {})
            stage2_analysis = article.get('stage2_analysis', {})
            
            summary = f"""Article {i}:
Title: {title}
Source: {source}
Description: {description[:300]}{'...' if len(description) > 300 else ''}"""
            
            if stage1_analysis:
                summary += f"\nStage 1 Relevance: {stage1_analysis.get('relevance_score', 'N/A')}"
            
            if stage2_analysis:
                summary += f"\nStage 2 Quality: {stage2_analysis.get('quality_score', 'N/A')}"
                if stage2_analysis.get('significance'):
                    summary += f"\nSignificance: {stage2_analysis['significance']}"
            
            article_summaries.append(summary)
        
        categories_list = '\n'.join([f"- {cat}: {desc}" for cat, desc in self.expert_categories.items()])
        
        prompt = f"""You are an expert AI news analyst. Analyze these {len(articles)} articles and respond with VALID JSON ONLY.

ARTICLES:
{chr(10).join(article_summaries)}

CATEGORIES: breakthrough, research, industry, safety, policy, open_source, startup, acquisition, partnership, controversy

INSTRUCTIONS:
- Analyze each article for AI/tech significance
- Rate quality 1-10 (8+ = high quality, include these)
- Select approximately {int(len(articles) * self.target_pass_rate)} articles ({self.target_pass_rate:.0%})
- RESPOND WITH VALID JSON ONLY

{{
  "analysis": [
    {{
      "article_id": 1,
      "category": "research",
      "quality_score": 8.5,
      "impact_level": "significant",
      "novelty": "novel",
      "credibility": "high",
      "audience_relevance": "important",
      "expert_reasoning": "Important AI research development",
      "recommendation": "INCLUDE"
    }}
  ]
}}"""

        return prompt
    
    async def _call_groq_api(self, prompt: str, max_tokens: int = 1000, task_id: Optional[str] = None) -> Optional[str]:
        """Make API call to Groq with expert analysis prompt."""
        if not self.session:
            logger.error("Session not initialized. Use async context manager.")
            return None
        
        if task_id is None:
            task_id = "S3-XXXX"
        
        call_start_time = time.time()
        estimated_input_tokens = len(prompt) // 4
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI news analyst. ALWAYS respond with valid JSON only. No explanations, no markdown, no extra text."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Very low temperature for consistent JSON output
            "top_p": 0.8,
            "stop": None
        }
        
        try:
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract usage information
                    usage = data.get('usage', {})
                    request_tokens = usage.get('prompt_tokens', estimated_input_tokens)
                    response_tokens = usage.get('completion_tokens', len(str(data)) // 4)
                    total_tokens = usage.get('total_tokens', request_tokens + response_tokens)
                    
                    # Detailed token logging for Stage 3
                    logger.info(f"[{task_id}] 🔤 Stage 3 Token Usage:")
                    logger.info(f"[{task_id}]   📊 This Request: {request_tokens} prompt + {response_tokens} completion = {total_tokens} total tokens")
                    logger.info(f"[{task_id}]   ⚡ Model: {self.model} (6,000 tokens/min, 500k daily limit)")
                    logger.info(f"[{task_id}]   💰 Cost: $0.00 (FREE MODEL)")
                    
                    # Record usage
                    usage_tracker.record_call(
                        model=self.model,
                        agent="stage3_expert",
                        success=True,
                        request_tokens=request_tokens,
                        response_tokens=response_tokens,
                        processing_time=time.time() - call_start_time
                    )
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['message']['content']
                        logger.debug(f"[{task_id}] API call successful: {total_tokens} tokens, {time.time() - call_start_time:.2f}s")
                        return content
                    else:
                        logger.error(f"[{task_id}] No choices in API response")
                        return None
                        
                elif response.status == 429:
                    logger.warning(f"[{task_id}] Rate limit hit, waiting...")
                    # Record rate limit hit
                    usage_tracker.record_call(
                        model=self.model,
                        agent="stage3_expert",
                        success=False,
                        processing_time=time.time() - call_start_time,
                        status_code=429,
                        error_message="rate_limit"
                    )
                    await asyncio.sleep(60)  # Wait 1 minute
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"[{task_id}] API call failed: {response.status} - {error_text}")
                    usage_tracker.record_call(
                        model=self.model,
                        agent="stage3_expert",
                        success=False,
                        processing_time=time.time() - call_start_time,
                        status_code=response.status,
                        error_message=f"http_{response.status}"
                    )
                    return None
                    
        except Exception as e:
            logger.error(f"[{task_id}] API call exception: {e}")
            usage_tracker.record_call(
                model=self.model,
                agent="stage3_expert",
                success=False,
                processing_time=time.time() - call_start_time,
                error_message=str(e)
            )
            return None
    
    def _parse_expert_analysis(self, response_text: str, batch_articles: List[Dict]) -> List[Dict]:
        """Parse expert analysis response and apply to articles."""
        try:
            # Clean up response text aggressively
            response_text = response_text.strip()
            
            # Remove common markdown or formatting issues
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Remove any leading/trailing whitespace again
            response_text = response_text.strip()
            
            # If response is empty or very short, log and return empty
            if len(response_text) < 10:
                logger.error(f"Response too short ({len(response_text)} chars): '{response_text}'")
                return []
            
            # Log the cleaned response for debugging
            logger.debug(f"Parsing JSON response (length: {len(response_text)}): {response_text[:200]}...")
            
            # Parse JSON response
            analysis_data = json.loads(response_text)
            
            # Validate expected structure
            if not isinstance(analysis_data, dict) or 'analysis' not in analysis_data:
                logger.error(f"Invalid JSON structure: missing 'analysis' key")
                return []
            
            analysis_list = analysis_data.get('analysis', [])
            if not isinstance(analysis_list, list):
                logger.error(f"Invalid JSON structure: 'analysis' is not a list")
                return []
            
            logger.info(f"Successfully parsed JSON with {len(analysis_list)} analysis items")
            
            analyzed_articles = []
            
            # Process each analysis item
            for analysis_item in analysis_list:
                if not isinstance(analysis_item, dict):
                    logger.warning(f"Skipping invalid analysis item: {analysis_item}")
                    continue
                
                article_id = analysis_item.get('article_id')
                recommendation = analysis_item.get('recommendation', '').upper()
                
                # Validate article_id
                if isinstance(article_id, int) and 1 <= article_id <= len(batch_articles):
                    article = batch_articles[article_id - 1].copy()  # article_id is 1-based
                    
                    # Add Stage 3 expert analysis
                    article['stage3_analysis'] = {
                        'category': analysis_item.get('category', 'unknown'),
                        'quality_score': analysis_item.get('quality_score', 0),
                        'impact_level': analysis_item.get('impact_level', 'unknown'),
                        'novelty': analysis_item.get('novelty', 'unknown'),
                        'credibility': analysis_item.get('credibility', 'unknown'),
                        'audience_relevance': analysis_item.get('audience_relevance', 'unknown'),
                        'expert_reasoning': analysis_item.get('expert_reasoning', 'No reasoning provided'),
                        'recommendation': recommendation,
                        'analyzed_at': datetime.now(timezone.utc).isoformat(),
                        'model': self.model,
                        'stage': 'stage3_expert'
                    }
                    
                    article['stage3_decision'] = recommendation
                    article['expert_analysis'] = True
                        
                    # Only include articles recommended by expert analysis
                    if recommendation == 'INCLUDE':
                        analyzed_articles.append(article)
                        logger.debug(f"Article {article_id}: INCLUDED - {analysis_item.get('category', 'unknown')} - Quality: {analysis_item.get('quality_score', 0)}")
                    else:
                        logger.debug(f"Article {article_id}: EXCLUDED - {analysis_item.get('category', 'unknown')} - Quality: {analysis_item.get('quality_score', 0)}")
                else:
                    logger.warning(f"Invalid article_id: {article_id} (expected 1-{len(batch_articles)})")
            
            logger.info(f"Expert analysis result: {len(analyzed_articles)}/{len(batch_articles)} articles included")
            return analyzed_articles
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text (first 500 chars): {response_text[:500]}")
            logger.error(f"Response text (last 200 chars): {response_text[-200:]}")
            return []
        except Exception as e:
            logger.error(f"Error processing expert analysis: {e}")
            logger.error(f"Response text: {response_text[:300]}...")
            return []
    
    async def expert_analyze_articles(self, articles: List[Dict], max_articles: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform expert analysis on articles using Stage 3 professional categorization.
        
        Args:
            articles: List of articles that passed Stage 2 compound analysis
            max_articles: Maximum number of articles to process (for testing)
        
        Returns:
            Dict containing expert-analyzed articles and analysis metrics
        """
        start_time = time.time()
        
        if not articles:
            logger.warning("No articles provided for expert analysis")
            return {
                'model': self.model,
                'stage': 'stage3_expert',
                'input_count': 0,
                'output_count': 0,
                'articles': [],
                'pass_rate': 0.0,
                'target_pass_rate': self.target_pass_rate,
                'requests_made': 0,
                'processing_time': 0,
                'articles_per_second': 0,
                'categories_found': {},
                'quality_distribution': {}
            }
        
        # Limit articles for testing if specified
        if max_articles:
            articles = articles[:max_articles]
        
        # Apply token constraint
        if len(articles) > self.max_articles_per_cycle:
            logger.warning(f"Limiting to {self.max_articles_per_cycle} articles due to token constraints")
            articles = articles[:self.max_articles_per_cycle]
        
        # Generate task ID for logging
        task_id = f"S3-{int(time.time() * 1000) % 10000:04d}"
        
        logger.info(f"[{task_id}] Starting Stage 3 expert analysis for {len(articles)} articles")
        logger.info(f"[{task_id}] Model: {self.model}")
        logger.info(f"[{task_id}] Rate limiting: {self.rate_limit_delay:.1f}s between requests, {self.articles_per_request} articles per batch")
        logger.info(f"[{task_id}] Model limits: {self.requests_per_minute_limit} req/min, {self.tokens_per_minute_limit} tokens/min")
        logger.info(f"[{task_id}] Target pass rate: {self.target_pass_rate:.1%}")
        logger.info(f"[{task_id}] Estimated time: {(len(articles) / self.articles_per_request) * (self.rate_limit_delay / 60):.1f} minutes")
        
        all_expert_articles = []
        total_requests = 0
        categories_found = {}
        quality_scores = []
        
        # Process articles in batches for expert analysis
        for i in range(0, len(articles), self.articles_per_request):
            if total_requests >= self.max_requests_per_cycle:
                logger.warning(f"Reached max requests limit ({self.max_requests_per_cycle})")
                break
            
            batch = articles[i:i + self.articles_per_request]
            logger.info(f"[{task_id}] Processing batch {total_requests + 1}/{min(len(articles) // self.articles_per_request + 1, self.max_requests_per_cycle)}: {len(batch)} articles")
            
            # Create expert analysis prompt
            prompt = self._create_expert_analysis_prompt(batch)
            
            # Apply rate limiting
            if total_requests > 0:
                await self._rate_limit()
            
            # Make API call
            response = await self._call_groq_api(prompt, max_tokens=1200, task_id=task_id)
            total_requests += 1
            
            if response:
                # Parse expert analysis
                expert_articles = self._parse_expert_analysis(response, batch)
                all_expert_articles.extend(expert_articles)
                
                # Track categories and quality scores
                for article in expert_articles:
                    stage3_analysis = article.get('stage3_analysis', {})
                    category = stage3_analysis.get('category', 'unknown')
                    quality_score = stage3_analysis.get('quality_score', 0)
                    
                    categories_found[category] = categories_found.get(category, 0) + 1
                    quality_scores.append(quality_score)
                
                logger.info(f"[{task_id}] Batch {total_requests} complete: {len(expert_articles)}/{len(batch)} articles passed expert analysis")
            else:
                logger.warning(f"[{task_id}] Batch {total_requests} failed - no response from API")
        
        # Calculate quality distribution
        quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
        for score in quality_scores:
            if score >= 8.0:
                quality_distribution['high'] += 1
            elif score >= 6.0:
                quality_distribution['medium'] += 1
            else:
                quality_distribution['low'] += 1
        
        # Calculate metrics
        processing_time = time.time() - start_time
        pass_rate = len(all_expert_articles) / len(articles) if articles else 0
        articles_per_second = len(all_expert_articles) / processing_time if processing_time > 0 else 0
        
        # 📊 STAGE 3 SUMMARY LOGGING 📊
        logger.info(f"[{task_id}] 🏁 Stage 3 Complete: {len(all_expert_articles)}/{len(articles)} articles ({pass_rate:.1%}) in {processing_time:.2f}s")
        logger.info(f"[{task_id}] 🔤 Token Summary:")
        logger.info(f"[{task_id}]   📊 Total Requests: {total_requests}")
        logger.info(f"[{task_id}]   ⚡ Model: {self.model} (6,000 tokens/min, 500k daily limit FREE)")
        logger.info(f"[{task_id}]   💰 Total Cost: $0.00 (FREE MODEL)")
        logger.info(f"[{task_id}]   ⚡ Performance: {articles_per_second:.2f} articles/sec")
        logger.info(f"[{task_id}] 📂 Top categories: {dict(list(sorted(categories_found.items(), key=lambda x: x[1], reverse=True))[:3])}")
        logger.info(f"[{task_id}] 📈 Quality distribution: {quality_distribution}")
        logger.info(f"  Top categories: {dict(sorted(categories_found.items(), key=lambda x: x[1], reverse=True)[:3])}")
        logger.info(f"  Quality distribution: {quality_distribution}")
        
        return {
            'model': self.model,
            'stage': 'stage3_expert',
            'input_count': len(articles),
            'output_count': len(all_expert_articles),
            'articles': all_expert_articles,
            'pass_rate': pass_rate,
            'target_pass_rate': self.target_pass_rate,
            'requests_made': total_requests,
            'processing_time': processing_time,
            'articles_per_second': articles_per_second,
            'batch_size': self.articles_per_request,
            'batches_processed': total_requests,
            'categories_found': categories_found,
            'quality_distribution': quality_distribution,
            'tokens_per_article': self.tokens_per_article,
            'max_articles_per_cycle': self.max_articles_per_cycle,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
    
    async def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process a batch of articles for streaming pipeline."""
        result = await self.expert_analyze_articles(articles)
        return result.get('articles', [])


async def main():
    """Main function for testing Stage 3 Expert Agent."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stage3_expert_agent.py <articles_json_file>")
        return
    
    # Load test articles
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    if not articles:
        print("No articles found in input file")
        return
    
    print(f"Testing Stage 3 Expert Agent with {len(articles)} articles")
    
    # Test the expert agent
    async with Stage3ExpertAgent() as expert_agent:
        result = await expert_agent.expert_analyze_articles(articles, max_articles=20)
    
    print(f"\nStage 3 Expert Analysis Results:")
    print(f"Input articles: {result['input_count']}")
    print(f"Output articles: {result['output_count']}")
    print(f"Pass rate: {result['pass_rate']:.1%}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Requests made: {result['requests_made']}")
    print(f"Categories found: {result['categories_found']}")
    print(f"Quality distribution: {result['quality_distribution']}")
    
    # Save results
    output_file = "stage3_expert_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
