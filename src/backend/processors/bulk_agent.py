#!/usr/bin/env python3
"""
Bulk Filtering Agent for Daily AI Times Swarm Intelligence System
Enhanced with Multi-Dimensional Scoring System (Phase 3.0 - EFFICIENCY OPTIMIZED)

MAJOR IMPROVEMENTS in this version:
- Migrated from manual aiohttp to official Groq Python library for cleaner, more reliable API interactions
- Implemented JSON Mode for guaranteed structured output, eliminating complex text parsing
- Added Pydantic models (ArticleScores, BatchScoringResponse) for type-safe JSON validation
- Leveraged Groq's built-in retry logic and error handling, reducing custom rate limiting complexity
- Improved error handling with specific Groq exception types
- Enhanced logging and monitoring integration
- Maintained backward compatibility with existing scoring systems

UNIFIED APPROACH (Phase 2.0):
- ELIMINATED artificial specialization bias: All agents now evaluate ALL dimensions equally
- ENHANCED consensus quality: True model diversity instead of artificial prompt constraints  
- IMPROVED evaluation balance: Comprehensive assessment across relevance, quality, novelty, impact
- PRESERVED model diversity: Natural LLM differences provide variation, not artificial bias
- MAINTAINED compatibility: All configurations, interfaces, and data flows unchanged

EFFICIENCY OPTIMIZATIONS (Phase 3.0):
- CACHED expensive calculations: Agent variance, default scores, jitter ranges computed once
- CONSOLIDATED rate limiting state: Removed redundant counters and tracking arrays
- STREAMLINED token estimation: Direct calculations without string joining operations  
- OPTIMIZED delay logic: Pre-calculated inter-batch delays with cached jitter ranges
- SIMPLIFIED API key loading: Efficient .env parsing while preserving local testing support
- REDUCED time.time() calls: Single time calculations for multiple operations

Key Benefits:
1. Much more reliable JSON parsing with Pydantic validation
2. Cleaner code with less manual HTTP handling
3. Better error recovery with Groq's built-in retry mechanisms
4. Type safety and validation for all API responses
5. Easier maintenance and debugging
6. UNIFIED: Better consensus through balanced evaluation across all agents
7. UNIFIED: Reduced bias and improved objectivity in article scoring
8. OPTIMIZED: Reduced computational overhead and memory usage
9. OPTIMIZED: Faster initialization and processing with cached calculations
"""

import os, logging, asyncio, time, random, json, re
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from groq import AsyncGroq
import groq

# Consolidated imports with fallbacks
try:
    from ...shared.types.scoring import MultiDimensionalScore, ArticleScore, create_multi_dimensional_score
except ImportError:
    try:
        from shared.types.scoring import MultiDimensionalScore, ArticleScore, create_multi_dimensional_score
    except ImportError:
        class DummyScore:
            def __init__(self, **kwargs):
                self.binary_decision = kwargs.get('binary_decision', False)
                self.confidence_mean = kwargs.get('confidence_mean', 0.0)
                self.overall_score = kwargs.get('overall_score', 0.0)
            def get_legacy_format(self): return self.binary_decision, self.confidence_mean
            def to_dict(self): return {'binary_decision': self.binary_decision, 'confidence_mean': self.confidence_mean, 'overall_score': self.overall_score}
        MultiDimensionalScore = ArticleScore = DummyScore
        create_multi_dimensional_score = lambda **kwargs: DummyScore(**kwargs)

try:
    from ..monitoring.groq_usage_tracker import usage_tracker
except ImportError:
    try:
        from monitoring.groq_usage_tracker import usage_tracker
    except ImportError:
        try:
            from backend.monitoring.groq_usage_tracker import usage_tracker
        except ImportError:
            class DummyTracker:
                def record_call(self, **kwargs):
                    pass
            usage_tracker = DummyTracker()

logger = logging.getLogger(__name__)

# Pydantic models for structured JSON output
class ArticleScores(BaseModel):
    """Individual article scores in structured JSON format."""
    relevance: float = Field(ge=0.0, le=1.0, description="Tech professional relevance")
    quality: float = Field(ge=0.0, le=1.0, description="Content depth and technical accuracy")
    novelty: float = Field(ge=0.0, le=1.0, description="How new/unique the content is")
    impact: float = Field(ge=0.0, le=1.0, description="Importance to tech community")
    confidence: float = Field(ge=0.0, le=1.0, description="Assessment confidence")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Assessment uncertainty")
    ai: float = Field(ge=0.0, le=1.0, description="AI/ML category probability")
    entertainment: float = Field(ge=0.0, le=1.0, description="Entertainment category probability") 
    sports: float = Field(ge=0.0, le=1.0, description="Sports tech category probability")
    health: float = Field(ge=0.0, le=1.0, description="Health tech category probability")

class BatchScoringResponse(BaseModel):
    """Complete batch response with all article scores."""
    articles: List[ArticleScores] = Field(description="Scores for each article in batch")

class BulkFilteringAgent:
    """Individual agent in the bulk intelligence swarm with unified evaluation approach.
    
    UNIFIED APPROACH: This agent evaluates articles across ALL dimensions (relevance, quality, 
    novelty, impact) equally, without artificial specialization bias. Model diversity comes 
    from natural LLM differences rather than prompt constraints, providing better consensus 
    quality and more objective scoring.
    """
    
    def __init__(self, model_name: str, agent_config: Dict[str, Any], api_key: Optional[str] = None, agent_id: Optional[str] = None):
        """Initialize a bulk filtering agent with configuration.
        
        OPTIMIZED: Cached expensive calculations, consolidated rate limiting state, 
        preserved .env.local loading for local testing.
        """
        self.model_name = model_name
        self.agent_id = agent_id or "Unknown"
        
        # API key loading - preserved for local testing but optimized
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            # OPTIMIZED: Streamlined .env loading (preserved for local testing)
            try:
                from dotenv import load_dotenv
                load_dotenv('.env.local')
                load_dotenv()
                self.api_key = os.getenv('GROQ_API_KEY')
            except ImportError:
                # Efficient manual loading fallback
                for env_file in ['.env.local', '.env']:
                    if os.path.exists(env_file):
                        with open(env_file, 'r') as f:
                            for line in f:
                                if line.startswith('GROQ_API_KEY='):
                                    self.api_key = line.split('=', 1)[1].strip()
                                    os.environ['GROQ_API_KEY'] = self.api_key
                                    break
                        if self.api_key:
                            break
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Configuration
        self.consensus_weight = agent_config.get('consensus_weight', 0.35)
        
        # OPTIMIZED: Cache expensive calculations done once
        self.model_limits = self._get_model_limits(model_name)
        self.tokens_per_minute = self.model_limits['tpm']
        self.requests_per_minute = self.model_limits['rpm']
        self.daily_tokens_limit = self.model_limits['daily_tokens']
        
        # OPTIMIZED: Pre-calculate agent variance (used in _get_default_scores)
        self.cached_agent_variance = hash(self.model_name) % 100 / 1000.0
        
        # Model-specific optimizations with cached delay values
        if model_name == 'gemma2-9b-it':
            self.safety_margin = 0.90
            self.inter_batch_delay = 5.0
            self.jitter_range = (0, 0.5)  # Cached jitter calculation
        elif model_name in ['llama3-8b-8192', 'llama-3.1-8b-instant']:
            self.safety_margin = 0.85
            self.inter_batch_delay = 4.0
            self.jitter_range = (0, 0.2)
        else:
            self.safety_margin = 0.80
            self.inter_batch_delay = 5.0
            self.jitter_range = (0, 0.25)
            
        # OPTIMIZED: Consolidated rate limiting calculations
        self.requests_per_minute_limit = int(self.requests_per_minute * self.safety_margin)
        self.tokens_per_minute_limit = int(self.tokens_per_minute * self.safety_margin)
        self.estimated_tokens_per_article = 250
        
        # OPTIMIZED: Unified timing state (removed redundant counters)
        current_time = time.time()
        self.current_minute_start = current_time
        self.current_minute_tokens = 0
        self.requests_this_minute = 0
        
        # Daily token tracking
        self.daily_tokens_used = 0
        self.daily_start_time = current_time
        self.tokens_per_day_limit = self.daily_tokens_limit if self.daily_tokens_limit != -1 else float('inf')
        
        # Batch sizing optimization - use calculated optimal batch size
        self.adaptive_batch_size = self._calculate_optimal_batch_size()
        
        # Processing constants
        self.temperature = 0.1
        self.top_p = 0.9
        self.max_tokens_per_article = 35
        self.title_max_length = 70
        self.description_max_length = 120
        self.max_retries = 3
        
        # OPTIMIZED: Cache default scores template (computed once)
        self._cached_default_scores = None
        
        self.groq_client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        try:
            if not self.api_key:
                raise ValueError(f"API key is missing for agent {self.agent_id}")
            
            # Create AsyncGroq client with basic parameters - some versions don't support timeout/max_retries
            self.groq_client = AsyncGroq(api_key=self.api_key)
            logger.debug(f"BulkFilteringAgent {self.agent_id} AsyncGroq client created successfully")
            return self
        except Exception as e:
            logger.error(f"Failed to create AsyncGroq client for agent {self.agent_id}: {e}")
            logger.error(f"API key present: {'Yes' if self.api_key else 'No'}")
            logger.error(f"API key length: {len(self.api_key) if self.api_key else 0}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.groq_client:
            # AsyncGroq doesn't have aclose method in older versions
            pass
    
    def _get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific rate limits from GROQ API documentation."""
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
        
        # Default limits for unknown models
        default_limits = {'rpm': 30, 'tpm': 6000, 'daily_tokens': 500000}
        
        limits = model_limits.get(model_name, default_limits)
        return limits

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on model-specific token limits - SPEED OPTIMIZED."""
        # Model-specific optimizations - MAXIMIZED for GitHub Actions speed
        if self.model_name == 'gemma2-9b-it':
            # Gemma2 has issues with large batches in JSON mode - limit to smaller, reliable batches
            # The model tends to only return 10 scores regardless of batch size, so keep it manageable
            return 10  # Fixed batch size that the model can reliably handle
        
        elif self.tokens_per_minute_limit <= 6000:
            # 6k token/min models - larger batches for speed
            max_tokens_per_batch = int(self.tokens_per_minute_limit * 0.90 / 10)  # 10 batches per minute
            optimal_size = max_tokens_per_batch // self.estimated_tokens_per_article
            return max(8, min(optimal_size, 25))  # Larger batches for speed
        
        elif self.tokens_per_minute_limit <= 12000:
            # 12k token/min models - aggressive batching
            max_tokens_per_batch = int(self.tokens_per_minute_limit * 0.90 / 5)  # 5 batches per minute
            optimal_size = max_tokens_per_batch // self.estimated_tokens_per_article
            return max(12, min(optimal_size, 40))
        
        # Default fallback - speed optimized
        max_tokens_per_batch = int(self.tokens_per_minute_limit * 0.85 / 8)
        optimal_size = max_tokens_per_batch // self.estimated_tokens_per_article
        return max(8, min(optimal_size, 25))
    
    def _reset_token_counter_if_needed(self):
        """OPTIMIZED: Simplified token counter reset with single time check."""
        current_time = time.time()
        
        # OPTIMIZED: Single time calculation for both checks
        if current_time - self.current_minute_start >= 60:
            self.current_minute_tokens = 0
            self.current_minute_start = current_time
            self.requests_this_minute = 0
            
        # Reset daily counter if needed
        if current_time - self.daily_start_time >= 86400:  # 24 hours
            self.daily_tokens_used = 0
            self.daily_start_time = current_time
    
    def _check_daily_token_limit(self, estimated_tokens: int) -> bool:
        """Check if we have enough daily tokens remaining."""
        remaining_tokens = self.tokens_per_day_limit - self.daily_tokens_used
        return remaining_tokens >= estimated_tokens
    
    def _estimate_batch_tokens(self, messages_or_articles) -> int:
        """OPTIMIZED: Streamlined token estimation with reduced string operations."""
        if isinstance(messages_or_articles, list) and len(messages_or_articles) > 0:
            if isinstance(messages_or_articles[0], dict) and 'role' in messages_or_articles[0]:
                # OPTIMIZED: Direct calculation without string joining
                total_words = sum(len(msg.get('content', '').split()) for msg in messages_or_articles)
                return int(total_words * 1.3)
            return len(messages_or_articles) * self.estimated_tokens_per_article
        return self.estimated_tokens_per_article
    
    async def _intelligent_rate_limit(self, estimated_tokens: int):
        """OPTIMIZED: Simplified rate limiting with consolidated state and fewer time calls."""
        current_time = time.time()
        self._reset_token_counter_if_needed()
        
        # Check daily token limit first
        if not self._check_daily_token_limit(estimated_tokens):
            remaining_tokens = self.tokens_per_day_limit - self.daily_tokens_used
            logger.warning(f"Daily token limit approaching for {self.model_name}: {remaining_tokens} tokens remaining")
            if remaining_tokens <= 0:
                logger.error(f"Daily token limit exceeded for {self.model_name}")
                raise ValueError(f"Daily token limit exceeded for {self.model_name}")
        
        # OPTIMIZED: Simplified request rate limit check
        if self.requests_this_minute >= self.requests_per_minute_limit:
            wait_time = 60 - (current_time - self.current_minute_start) + 0.2
            if wait_time > 0.2:
                # OPTIMIZED: Use cached jitter range
                jitter = random.uniform(*self.jitter_range)
                logger.info(f"Rate limiting: waiting {wait_time + jitter:.1f}s for {self.model_name}")
                await asyncio.sleep(wait_time + jitter)
                self._reset_token_counter_if_needed()
        
        # OPTIMIZED: Simplified token limit check
        if self.current_minute_tokens + estimated_tokens > self.tokens_per_minute_limit:
            wait_time = 60 - (current_time - self.current_minute_start) + 0.2
            if wait_time > 0.2:
                # OPTIMIZED: Use cached jitter range
                jitter = random.uniform(*self.jitter_range)
                logger.info(f"Token limit: waiting {wait_time + jitter:.1f}s for {self.model_name}")
                await asyncio.sleep(wait_time + jitter)
                self._reset_token_counter_if_needed()
        
        # OPTIMIZED: Simplified counter updates (removed redundant tracking)
        self.current_minute_tokens += estimated_tokens
        self.daily_tokens_used += estimated_tokens
        self.requests_this_minute += 1
    
    def _get_filtering_prompt(self, article_count: int) -> str:
        """Get the UNIFIED multi-dimensional scoring prompt for JSON mode.
        
        UNIFIED APPROACH: All agents now evaluate ALL dimensions (relevance, quality, novelty, impact) 
        equally, leveraging true model diversity instead of artificial prompt specialization.
        This provides better consensus quality and removes artificial bias.
        
        OPTIMIZED: Single prompt template eliminates code duplication and maintenance overhead.
        """
        
        # UNIFIED PROMPT: Single template for all models (eliminates 40+ lines of duplication)
        json_emphasis = ""
        if self.model_name == 'gemma2-9b-it':
            json_emphasis = "\n\nIMPORTANT: You MUST respond with valid JSON only. Do not include any explanatory text, markdown formatting, or code blocks. Start directly with the opening curly brace."
        
        base_prompt = f"""You are an expert news evaluator for a technology-focused publication. You will analyze EXACTLY {article_count} articles and provide balanced, comprehensive scores from 0.0 to 1.0 for each article.

CRITICAL: Your response must contain EXACTLY {article_count} articles in the "articles" array. No more, no less.{json_emphasis}

EVALUATION GUIDELINES - Be reasonably generous with tech-related content:
- RELEVANCE: How relevant to technology professionals (0.6+ for any tech topic, 0.8+ for major tech news)
- QUALITY: Content depth and journalistic quality (0.6+ for standard news, 0.8+ for in-depth analysis)
- NOVELTY: How new or unique the content is (0.5+ for recent developments, 0.7+ for breaking news)
- IMPACT: Importance to the tech community (0.5+ for industry updates, 0.8+ for major announcements)
- CONFIDENCE: Your confidence in this assessment (0.7+ for clear assessments)
- UNCERTAINTY: Your uncertainty about this assessment (0.3 or lower for most assessments)

CATEGORY CLASSIFICATION: Assign probability scores for each category (values between 0.0 and 1.0):
- AI: Artificial intelligence, machine learning, neural networks, AI companies, LLMs, AI research
- ENTERTAINMENT: Movies, gaming, streaming, social media, content creation, digital entertainment  
- SPORTS: Sports technology, esports, fitness tech, sports analytics, athlete performance tech
- HEALTH: Medical technology, healthtech, biotech, mental health apps, medical AI, health devices

Respond with a JSON object containing an "articles" array with EXACTLY {article_count} entries. Each article should have scores as decimal numbers between 0.0 and 1.0.

Required JSON format:
{{
  "articles": [
    {{
      "relevance": 0.8,
      "quality": 0.7,
      "novelty": 0.6,
      "impact": 0.7,
      "confidence": 0.8,
      "uncertainty": 0.2,
      "ai": 0.8,
      "entertainment": 0.1,
      "sports": 0.05,
      "health": 0.05
    }}"""
        
        # Dynamic examples based on article count (works for all models)
        if article_count > 1:
            base_prompt += """,
    {
      "relevance": 0.6,
      "quality": 0.6,
      "novelty": 0.5,
      "impact": 0.5,
      "confidence": 0.8,
      "uncertainty": 0.2,
      "ai": 0.3,
      "entertainment": 0.6,
      "sports": 0.05,
      "health": 0.05
    }"""
            
        if article_count > 2:
            # Model-specific instructions for handling larger batches
            if self.model_name == 'gemma2-9b-it':
                base_prompt += f"""
    ... (continue this pattern for ALL {article_count} articles - do not stop until you have {article_count} complete entries)"""
            else:
                base_prompt += f"""
    ... (continue for all {article_count} articles)"""
        
        base_prompt += """
  ]
}

IMPORTANT: The "articles" array must contain EXACTLY """ + str(article_count) + """ entries, one for each article provided."""
        
        # Model-specific additional instructions (only where genuinely needed)
        if self.model_name == 'gemma2-9b-it':
            base_prompt += f"""

MANDATORY REQUIREMENT: Return exactly {article_count} score objects in the "articles" array. Count them: 1, 2, 3... up to {article_count}."""
        
        return base_prompt
    
    async def process_batch(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process a batch of articles and return decisions with confidence scores.
        
        OPTIMIZED: Separates scoring data from metadata - only sends title/description to prompt.
        """
        if not articles:
            return []
        
        # OPTIMIZED: Extract only scoring-relevant data for prompt, keep metadata separate
        scoring_data = []
        article_metadata = []
        
        for i, article in enumerate(articles):
            # Extract only data needed for scoring (sent to prompt)
            title = article.get('title', 'No title')[:self.title_max_length]
            description = article.get('description', '')
            
            scoring_item = {
                'index': i + 1,
                'title': title,
                'description': description[:self.description_max_length] if description.strip() else ''
            }
            scoring_data.append(scoring_item)
            
            # Keep full article metadata (not sent to prompt)
            article_metadata.append(article)
        
        # Build prompt content from scoring data only
        user_content_parts = []
        for item in scoring_data:
            if item['description']:
                user_content_parts.append(f"{item['index']}. {item['title']} - {item['description']}")
            else:
                user_content_parts.append(f"{item['index']}. {item['title']}")
        
        user_content = '\n'.join(user_content_parts)
        
        messages = [
            {"role": "system", "content": self._get_filtering_prompt(len(articles))},
            {"role": "user", "content": user_content}
        ]
        
        batch_response = await self._make_api_call(messages)
        if not batch_response:
            logger.error(f"API call failed for {self.model_name} - NO FALLBACK")
            raise RuntimeError(f"Agent {self.model_name} failed to generate response")
        
        # Enhanced logging for debugging
        logger.info(f"{self.model_name} processing {len(articles)} articles")
        logger.info(f"{self.model_name} received {len(batch_response.articles)} scores")
        
        # Debug: Check first response structure
        if batch_response.articles:
            first_score = batch_response.articles[0]
            logger.info(f"{self.model_name} sample scores: R={first_score.relevance:.2f}, Q={first_score.quality:.2f}, I={first_score.impact:.2f}")
            logger.info(f"{self.model_name} sample categories: AI={first_score.ai:.2f}, Ent={first_score.entertainment:.2f}, Sports={first_score.sports:.2f}, Health={first_score.health:.2f}")
        
        # Process structured JSON response with original article metadata
        if batch_response.articles and len(batch_response.articles) >= len(articles) * 0.7:
            results = []
            accept_count = 0
            
            # Ensure we have exactly the right number of scores
            scores_to_use = batch_response.articles[:len(articles)]
            
            # Pad with default scores if needed
            defaults_added = 0
            while len(scores_to_use) < len(articles):
                defaults = self._get_default_scores()
                default_score = ArticleScores(**defaults)
                scores_to_use.append(default_score)
                defaults_added += 1
            
            if defaults_added > 0:
                logger.warning(f"{self.model_name} added {defaults_added} default scores (API returned {len(batch_response.articles)}, needed {len(articles)})")
            
            for i, (original_article, article_scores) in enumerate(zip(article_metadata, scores_to_use)):
                # Create multi-dimensional score with flat category structure
                category_probs = {
                    'ai': article_scores.ai,
                    'entertainment': article_scores.entertainment,
                    'sports': article_scores.sports,
                    'health': article_scores.health
                }
                
                md_score = create_multi_dimensional_score(
                    relevance=article_scores.relevance,
                    quality=article_scores.quality,
                    novelty=article_scores.novelty,
                    impact=article_scores.impact,
                    category_probabilities=category_probs,
                    confidence_mean=article_scores.confidence,
                    confidence_std=article_scores.uncertainty,
                    agent_name=self.model_name,
                    model_name=self.model_name
                )
                
                decision, confidence = md_score.get_legacy_format()
                results.append((original_article, decision, confidence))
                original_article['multi_dimensional_score'] = md_score.to_dict()
                
                if decision:
                    accept_count += 1
                
                # Log a sample of scores for debugging
                if i < 3:  # Log first 3 articles
                    logger.info(f"{self.model_name} Article {i+1}: "
                              f"R={article_scores.relevance:.2f}, "
                              f"Q={article_scores.quality:.2f}, "
                              f"N={article_scores.novelty:.2f}, "
                              f"I={article_scores.impact:.2f}, "
                              f"C={article_scores.confidence:.2f}, "
                              f"Decision={decision}")
            
            logger.info(f"{self.model_name} batch results: {accept_count}/{len(articles)} accepted")
            return results
        else:
            received_count = len(batch_response.articles) if batch_response.articles else 0
            logger.warning(f"Agent {self.model_name}: JSON parsing failed or insufficient scores, got {received_count} scores for {len(articles)} articles - using fallback scores")
            
            # Provide fallback scores for all articles instead of failing
            results = []
            defaults = self._get_default_scores()
            accept_count = 0
            
            for i, article in enumerate(article_metadata):
                # Create fallback multi-dimensional score
                # Make gemma2-9b-it fallbacks more generous to compensate for JSON issues
                if self.model_name == 'gemma2-9b-it':
                    fallback_relevance = min(1.0, defaults['relevance'] + 0.1)
                    fallback_quality = min(1.0, defaults['quality'] + 0.1) 
                    fallback_novelty = min(1.0, defaults['novelty'] + 0.1)
                    fallback_impact = min(1.0, defaults['impact'] + 0.1)
                    logger.debug(f"Using enhanced fallback scores for gemma2-9b-it: R={fallback_relevance:.2f}, Q={fallback_quality:.2f}")
                else:
                    fallback_relevance = defaults['relevance']
                    fallback_quality = defaults['quality']
                    fallback_novelty = defaults['novelty']
                    fallback_impact = defaults['impact']
                
                md_score = create_multi_dimensional_score(
                    relevance=fallback_relevance,
                    quality=fallback_quality,
                    novelty=fallback_novelty,
                    impact=fallback_impact,
                    category_probabilities={
                        'ai': defaults['ai'],
                        'entertainment': defaults['entertainment'],
                        'sports': defaults['sports'],
                        'health': defaults['health']
                    },
                    confidence_mean=defaults['confidence'],
                    confidence_std=defaults['uncertainty'],
                    agent_name=self.model_name,
                    model_name=self.model_name
                )
                
                decision, confidence = md_score.get_legacy_format()
                results.append((article, decision, confidence))
                article['multi_dimensional_score'] = md_score.to_dict()
                
                if decision:
                    accept_count += 1
            
            logger.warning(f"{self.model_name} fallback results: {accept_count}/{len(articles)} accepted")
            return results
    
    async def process_articles_with_rate_limiting(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles in batches with intelligent rate limiting.
        
        OPTIMIZED: Uses cached delay values and simplified batch processing logic.
        """
        if not articles:
            return []
        
        effective_batch_size = self.adaptive_batch_size
        all_results = []
        
        for i in range(0, len(articles), effective_batch_size):
            batch = articles[i:i + effective_batch_size]
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
            
            # OPTIMIZED: Use cached inter-batch delay with simplified jitter
            if i + effective_batch_size < len(articles):
                jitter = random.uniform(*self.jitter_range)
                total_delay = self.inter_batch_delay + jitter
                await asyncio.sleep(total_delay)
        
        return all_results
    
    async def process_articles(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles in batches (legacy method)."""
        return await self.process_articles_with_rate_limiting(articles)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for monitoring."""
        return {
            'model_name': self.model_name,
            'evaluation_approach': 'unified_comprehensive',  # NEW: Indicates balanced evaluation
            'bias_elimination': True,  # NEW: Indicates removal of artificial specialization bias
            'consensus_weight': self.consensus_weight,
            'requests_per_minute': self.requests_per_minute,
            'tokens_per_minute': self.tokens_per_minute,
            'tokens_per_day_limit': self.tokens_per_day_limit,
            'daily_tokens_used': self.daily_tokens_used,
            'daily_tokens_remaining': self.tokens_per_day_limit - self.daily_tokens_used if self.tokens_per_day_limit != float('inf') else 'unlimited',
            'adaptive_batch_size': self.adaptive_batch_size,
            'safety_margin': self.safety_margin,
            'api_client': 'groq-python',  # Using official Groq library
            'json_mode_enabled': True,    # Structured JSON output
            'max_retries': self.max_retries,
            'prompt_version': 'unified_v2.0',  # NEW: Indicates unified prompt approach
            'diversity_source': 'natural_model_differences',  # NEW: Indicates source of agent diversity
            # OPTIMIZED: New optimization metadata
            'optimization_level': 'phase_3_efficiency',  # NEW: Indicates optimization phase
            'cached_calculations': True,  # NEW: Uses cached agent variance and default scores
            'consolidated_state': True,   # NEW: Simplified rate limiting state
            'inter_batch_delay': self.inter_batch_delay,  # NEW: Cached delay value
            'jitter_range': self.jitter_range,  # NEW: Cached jitter configuration
            'env_loading_optimized': True  # NEW: Streamlined .env loading
        }

    async def _make_api_call(self, messages: List[Dict[str, Any]]) -> Optional[BatchScoringResponse]:
        """Make API call using Groq client with JSON mode for structured output."""
        if not self.groq_client:
            raise RuntimeError("Groq client not initialized. Use async context manager.")

        await self._intelligent_rate_limit(self._estimate_batch_tokens(messages))

        start_time = time.time()

        try:
            # Model-specific token limits - increased for better JSON completion
            if self.model_name == 'gemma2-9b-it':
                # Give gemma2 significantly more tokens for reliable JSON completion
                estimated_output = len(messages[1]['content'].split()) * 0.8  # Estimate output based on input
                max_tokens = max(2000, min(int(estimated_output), 4000))  # At least 2000, up to 4000
            else:
                max_tokens = min(self.max_tokens_per_article * 30, 1500)  # Increased for better completion
                
            # Use Groq's JSON mode for guaranteed structured output
            # Special handling for gemma2-9b-it
            if self.model_name == 'gemma2-9b-it':
                completion = await self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    temperature=0.05,  # Lower temperature for more consistent JSON
                    top_p=0.95,  # Slightly higher top_p for gemma2
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},  # Enable JSON mode
                    stream=False
                )
            else:
                completion = await self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},  # Enable JSON mode
                    stream=False
                )
            
            processing_time = time.time() - start_time
            
            # Parse the JSON response into our Pydantic model
            response_content = completion.choices[0].message.content
            
            try:
                # Parse with Pydantic validation
                response_data = json.loads(response_content)
                batch_response = BatchScoringResponse(**response_data)
                
                # Record successful call
                if hasattr(usage_tracker, 'record_call'):
                    try:
                        usage_info = completion.usage
                        usage_tracker.record_call(
                            model=self.model_name,
                            endpoint="chat/completions",
                            request_tokens=usage_info.prompt_tokens if usage_info else 0,
                            response_tokens=usage_info.completion_tokens if usage_info else 0,
                            processing_time=processing_time,
                            success=True,
                            agent=self.agent_id or "bulk_agent",
                            status_code=200
                        )
                    except Exception as e:
                        logger.debug(f"Usage tracking failed: {e}")
                
                return batch_response
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON response from {self.model_name}: {e}")
                logger.debug(f"Raw response: {response_content[:500]}")
                
                # Special handling for gemma2-9b-it - try to repair common JSON issues
                if self.model_name == 'gemma2-9b-it':
                    logger.info(f"Attempting JSON repair for {self.model_name}")
                    try:
                        # Try to fix common gemma2 JSON issues
                        repaired_json = self._repair_gemma_json(response_content)
                        if repaired_json:
                            batch_response = BatchScoringResponse(**repaired_json)
                            logger.info(f"Successfully repaired JSON for {self.model_name}")
                            return batch_response
                    except Exception as repair_error:
                        logger.debug(f"JSON repair failed: {repair_error}")
                
                return None
                
        except groq.RateLimitError as e:
            logger.warning(f"Rate limit error for {self.model_name}: {e}")
            # Wait longer before retrying when hitting rate limits
            await asyncio.sleep(60)  # Wait 1 minute for rate limit reset
            return None
            
        except groq.APITimeoutError as e:
            logger.error(f"Timeout for {self.model_name}: {e}")
            return None
            
        except groq.APIConnectionError as e:
            logger.error(f"Connection error for {self.model_name}: {e}")
            return None
            
        except groq.APIStatusError as e:
            # Special handling for gemma2-9b-it JSON validation errors
            if self.model_name == 'gemma2-9b-it' and e.status_code == 400:
                error_details = str(e)
                if 'json_validate_failed' in error_details and 'failed_generation' in error_details:
                    logger.warning(f"gemma2-9b-it JSON validation failed, attempting to extract scores from failed generation")
                    
                    # Try to extract the failed generation JSON
                    try:
                        import re
                        failed_gen_match = re.search(r"'failed_generation': '({.*})'", error_details)
                        if failed_gen_match:
                            failed_json = failed_gen_match.group(1)
                            # Unescape the JSON
                            failed_json = failed_json.replace('\\n', '\n').replace('\\"', '"')
                            
                            response_data = json.loads(failed_json)
                            batch_response = BatchScoringResponse(**response_data)
                            
                            logger.info(f"Successfully recovered {len(batch_response.articles)} scores from failed generation")
                            return batch_response
                    except Exception as recovery_error:
                        logger.debug(f"Failed to recover from failed generation: {recovery_error}")
            
            logger.error(f"API status error {e.status_code} for {self.model_name}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error for {self.model_name}: {e}")
            return None
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Get default scores with natural model-based variance.
        
        UNIFIED APPROACH: Removed specialization-based score adjustments to eliminate 
        artificial bias. Natural model diversity is preserved through model-based variance
        while ensuring balanced evaluation across all dimensions.
        
        OPTIMIZED: Uses cached agent variance and score caching for performance.
        """
        # OPTIMIZED: Return cached scores if already computed
        if self._cached_default_scores is not None:
            return self._cached_default_scores
        
        # OPTIMIZED: Use pre-computed agent variance instead of recalculating
        agent_variance = self.cached_agent_variance
        
        # Improved defaults to be more accepting for news articles
        defaults = {
            'relevance': 0.6 + agent_variance,  # Higher relevance baseline
            'quality': 0.6 + agent_variance,    # Higher quality baseline
            'novelty': 0.5 + agent_variance,    # Slightly higher novelty
            'impact': 0.5 + agent_variance,     # Slightly higher impact
            'confidence': 0.7 + agent_variance, # Higher confidence in assessment
            'uncertainty': 0.2 + agent_variance, # Lower uncertainty
            'ai': 0.7,  # Default to AI-focused for tech pipeline
            'entertainment': 0.1,
            'sports': 0.1,
            'health': 0.1
        }
        
        # Clamp all values to valid range [0.0, 1.0] and cache result
        for key in defaults:
            defaults[key] = max(0.0, min(1.0, defaults[key]))
        
        # OPTIMIZED: Cache the computed defaults
        self._cached_default_scores = defaults
        return defaults

    def _repair_gemma_json(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Repair common JSON issues specific to gemma2-9b-it model."""
        if not response_content or not response_content.strip():
            return None
            
        try:
            # Clean up the response
            content = response_content.strip()
            
            # Remove common prefixes that gemma2 might add
            prefixes_to_remove = [
                "Here is the JSON response:",
                "```json",
                "```",
                "JSON:",
                "Response:",
            ]
            
            for prefix in prefixes_to_remove:
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = ["```", "```json"]
            for suffix in suffixes_to_remove:
                if content.endswith(suffix):
                    content = content[:-len(suffix)].strip()
            
            # Find JSON boundaries
            start_idx = content.find('{')
            if start_idx == -1:
                return None
                
            # Find the last closing brace
            end_idx = content.rfind('}')
            if end_idx == -1 or end_idx <= start_idx:
                # Try to complete the JSON
                content = content[start_idx:] + '}'
            else:
                content = content[start_idx:end_idx + 1]
            
            # Try to parse the extracted JSON
            try:
                data = json.loads(content)
                
                # Validate that we have the expected structure
                if 'articles' in data and isinstance(data['articles'], list):
                    return data
                else:
                    logger.debug(f"Repaired JSON missing 'articles' key or wrong structure")
                    return None
                    
            except json.JSONDecodeError:
                # Try more aggressive repair
                logger.debug("Attempting aggressive JSON repair for gemma2-9b-it")
                
                # Try to fix common issues like missing quotes, trailing commas, etc.
                import re
                
                # Fix missing quotes around keys
                content = re.sub(r'(\w+):', r'"\1":', content)
                
                # Remove trailing commas before closing braces/brackets
                content = re.sub(r',(\s*[}\]])', r'\1', content)
                
                # Try parsing again
                try:
                    data = json.loads(content)
                    if 'articles' in data and isinstance(data['articles'], list):
                        return data
                except json.JSONDecodeError:
                    pass
                    
                return None
                
        except Exception as e:
            logger.debug(f"JSON repair exception: {e}")
            return None
