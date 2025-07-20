#!/usr/bin/env python3
"""
Bulk Filtering Agent for Daily AI Times Swarm Intelligence System
Enhanced with Multi-Dimensional Scoring System (Phase 1.1)

MAJOR IMPROVEMENTS in this version:
- Migrated from manual aiohttp to official Groq Python library for cleaner, more reliable API interactions
- Implemented JSON Mode for guaranteed structured output, eliminating complex text parsing
- Added Pydantic models (ArticleScores, BatchScoringResponse) for type-safe JSON validation
- Leveraged Groq's built-in retry logic and error handling, reducing custom rate limiting complexity
- Improved error handling with specific Groq exception types
- Enhanced logging and monitoring integration
- Maintained backward compatibility with existing scoring systems

Key Benefits:
1. Much more reliable JSON parsing with Pydantic validation
2. Cleaner code with less manual HTTP handling
3. Better error recovery with Groq's built-in retry mechanisms
4. Type safety and validation for all API responses
5. Easier maintenance and debugging
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

class BatchScoringResponse(BaseModel):
    """Complete batch response with all article scores."""
    articles: List[ArticleScores] = Field(description="Scores for each article in batch")

class BulkFilteringAgent:
    """Individual agent in the bulk intelligence swarm."""
    
    def __init__(self, model_name: str, agent_config: Dict[str, Any], api_key: Optional[str] = None, agent_id: Optional[str] = None):
        """Initialize a bulk filtering agent with configuration."""
        self.model_name = model_name
        self.agent_id = agent_id or "Unknown"
        
        # Try to get API key from multiple sources
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        # If no API key found, try loading from .env files
        if not self.api_key:
            # Try to load environment variables
            try:
                from dotenv import load_dotenv
                load_dotenv('.env.local')
                load_dotenv()  # Also load .env if it exists
                self.api_key = os.getenv('GROQ_API_KEY')
            except ImportError:
                # Manual loading fallback
                env_files = ['.env.local', '.env']
                for env_file in env_files:
                    if os.path.exists(env_file):
                        with open(env_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#') and '=' in line:
                                    key, value = line.split('=', 1)
                                    if key.strip() == 'GROQ_API_KEY':
                                        self.api_key = value.strip()
                                        os.environ['GROQ_API_KEY'] = self.api_key
                                        break
                        if self.api_key:
                            break
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        # Configuration
        self.specialization = agent_config.get('specialization', 'general_filtering')
        self.batch_size = agent_config.get('batch_size', 50)
        self.consensus_weight = agent_config.get('consensus_weight', 0.35)
        self.expertise_domains = agent_config.get('expertise_domains', [])
        
        # Model-specific rate limits (from groq-limits.md - July 2025)
        self.model_limits = self._get_model_limits(model_name)
        self.tokens_per_minute = self.model_limits['tpm']
        self.requests_per_minute = self.model_limits['rpm']
        self.daily_tokens_limit = self.model_limits['daily_tokens']
        
        # Groq client configuration - SPEED OPTIMIZED
        self.timeout_seconds = 45  # Reduced timeout for faster failure detection
        self.max_retries = 3  # Built-in retry logic
        
        # Model-specific safety margins - OPTIMIZED FOR SPEED (GitHub Actions)
        if model_name == 'gemma2-9b-it':
            # Gemma2 has 15k tokens/min - maximize speed with large batches
            self.safety_margin = 0.90  # Aggressive but safe utilization
            self.min_request_delay = 3.0  # Much faster processing (20 batches/min)
        elif model_name in ['llama3-8b-8192', 'llama-3.1-8b-instant']:
            # 6k token models - optimize for speed while avoiding rate limits
            self.safety_margin = 0.85  # More aggressive for speed
            self.min_request_delay = 4.0  # 15 batches/min = 4s delay
        else:
            self.safety_margin = 0.80   # Balanced for unknown models
            self.min_request_delay = 5.0
            
        self.requests_per_minute_limit = int(self.requests_per_minute * self.safety_margin)
        self.tokens_per_minute_limit = int(self.tokens_per_minute * self.safety_margin)
        self.estimated_tokens_per_article = 250  # Reduced further for speed optimization
        
        # Daily token limit tracking
        self.daily_tokens_used = 0
        self.daily_start_time = time.time()
        self.tokens_per_day_limit = self.daily_tokens_limit if self.daily_tokens_limit != -1 else float('inf')
        
        # Simplified rate limiting state - STREAMLINED FOR SPEED
        self.current_minute_tokens = 0
        self.current_minute_start = time.time()
        self.last_request_time = 0
        self.min_request_delay = 60.0 / (self.requests_per_minute_limit * 1.2)  # Faster calculation
        self.jitter_factor = 0.1  # Minimal jitter for speed
        # self.agent_id = hash(model_name) % 1000  # Commented out - using agent_id from constructor
        
        # Enhanced rate limiting - OPTIMIZED FOR SPEED
        self.requests_this_minute = 0
        self.request_times = []  # Simplified tracking
        
        # Batch sizing - MAXIMIZED for speed with larger batches
        self.adaptive_batch_size = min(self.batch_size, self._calculate_optimal_batch_size())
        logger.info(f"{model_name}: Using batch size {self.adaptive_batch_size} (limit: {self.tokens_per_minute_limit} tpm)")
        
        # Additional constants - SPEED OPTIMIZED
        self.temperature = 0.1  # Lower for faster, more consistent processing
        self.top_p = 0.9
        self.max_tokens_per_article = 35  # Reduced for speed
        self.title_max_length = 70       # Reduced for speed
        self.description_max_length = 120 # Reduced for speed
        self.max_rate_limit_retries = 2
        self.rate_limit_retry_delay = 15.0  # Reduced significantly from 20.0
        self.backoff_multiplier = 1.5       # Reduced from 1.8 for faster recovery
        
        # Daily usage reset tracking
        self.daily_reset_hour = 0  # Assume resets at midnight UTC
        
        # Retry settings - SPEED OPTIMIZED
        self.max_retries = 3  # Use Groq's built-in retry logic
        self.base_retry_delay = 2.0  # Reduced for faster recovery
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
        logger.info(f"Model {model_name}: {limits['tpm']} tokens/min, {limits['rpm']} requests/min")
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
        """SIMPLIFIED token counter reset for speed."""
        current_time = time.time()
        if current_time - self.current_minute_start >= 60:
            self.current_minute_tokens = 0
            self.current_minute_start = current_time
            self.requests_this_minute = 0
            self.request_times = []  # Clear old requests
            
        # Reset daily counter if needed (simplified)
        if current_time - self.daily_start_time >= 86400:  # 24 hours
            self.daily_tokens_used = 0
            self.daily_start_time = current_time
            logger.info(f"Daily token counter reset for {self.model_name}")
    
    def _check_daily_token_limit(self, estimated_tokens: int) -> bool:
        """Check if we have enough daily tokens remaining."""
        remaining_tokens = self.tokens_per_day_limit - self.daily_tokens_used
        return remaining_tokens >= estimated_tokens
    
    def _estimate_batch_tokens(self, messages_or_articles) -> int:
        """Estimate tokens needed for a batch."""
        if isinstance(messages_or_articles, list) and len(messages_or_articles) > 0:
            if isinstance(messages_or_articles[0], dict) and 'role' in messages_or_articles[0]:
                total_content = ' '.join([msg.get('content', '') for msg in messages_or_articles])
                return int(len(total_content.split()) * 1.3)
            return len(messages_or_articles) * self.estimated_tokens_per_article
        return self.estimated_tokens_per_article
    
    async def _intelligent_rate_limit(self, estimated_tokens: int):
        """SIMPLIFIED rate limiting optimized for Groq client integration."""
        current_time = time.time()
        self._reset_token_counter_if_needed()
        
        # Check daily token limit first
        if not self._check_daily_token_limit(estimated_tokens):
            remaining_tokens = self.tokens_per_day_limit - self.daily_tokens_used
            logger.warning(f"Daily token limit approaching for {self.model_name}: {remaining_tokens} tokens remaining")
            if remaining_tokens <= 0:
                logger.error(f"Daily token limit exceeded for {self.model_name}")
                raise ValueError(f"Daily token limit exceeded for {self.model_name}")
        
        # SIMPLIFIED request rate limit check (Groq handles most rate limiting internally)
        if self.requests_this_minute >= self.requests_per_minute_limit:
            wait_time = 60 - (current_time - self.current_minute_start) + 0.2
            if wait_time > 0.2:
                jitter = random.uniform(0.9, 1.1)
                logger.info(f"Rate limiting: waiting {wait_time * jitter:.1f}s for {self.model_name}")
                await asyncio.sleep(wait_time * jitter)
                self._reset_token_counter_if_needed()
        
        # SIMPLIFIED token limit check
        if self.current_minute_tokens + estimated_tokens > self.tokens_per_minute_limit:
            wait_time = 60 - (current_time - self.current_minute_start) + 0.2
            if wait_time > 0.2:
                jitter = random.uniform(0.9, 1.1)
                logger.info(f"Token limit: waiting {wait_time * jitter:.1f}s for {self.model_name}")
                await asyncio.sleep(wait_time * jitter)
                self._reset_token_counter_if_needed()
        
        # Update counters (simplified since Groq handles the actual requests)
        self.current_minute_tokens += estimated_tokens
        self.daily_tokens_used += estimated_tokens
        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)
        self.requests_this_minute += 1
    
    def _get_filtering_prompt(self, article_count: int) -> str:
        """Get the OPTIMIZED multi-dimensional scoring prompt for JSON mode."""
        specialization_prompts = {
            'content_quality_analysis': "Focus on QUALITY and technical depth.",
            'relevance_scoring': "Focus on RELEVANCE to technology professionals.",
            'category_classification': "Focus on categorization and technology domains.",
            'trend_detection': "Focus on NOVELTY and IMPACT of emerging trends.",
            'innovation_curation': "Focus on NOVELTY and IMPACT of breakthrough tech."
        }
        
        # Model-specific prompt adjustments
        if self.model_name == 'gemma2-9b-it':
            base_prompt = f"""You are an expert tech news evaluator. You must analyze ALL {article_count} articles provided and return scores for every single one.

MANDATORY REQUIREMENT: Return exactly {article_count} score objects in the "articles" array. Count them: 1, 2, 3... up to {article_count}.

Score each article from 0.0 to 1.0 on these dimensions:
- relevance: How relevant to technology professionals
- quality: Content depth and technical accuracy  
- novelty: How new or innovative the content is
- impact: Importance to the tech community
- confidence: Your confidence in this assessment
- uncertainty: Your uncertainty about this assessment

Return only valid JSON in this exact format:
{{
  "articles": [
    {{"relevance": 0.8, "quality": 0.7, "novelty": 0.6, "impact": 0.9, "confidence": 0.8, "uncertainty": 0.2}}"""
            
            if article_count > 1:
                base_prompt += f""",
    {{"relevance": 0.3, "quality": 0.4, "novelty": 0.2, "impact": 0.1, "confidence": 0.9, "uncertainty": 0.1}}"""
                
            if article_count > 2:
                # For gemma2, be very explicit about the count
                base_prompt += f"""
    ... (continue this pattern for ALL {article_count} articles - do not stop until you have {article_count} complete entries)"""
            
            base_prompt += f"""
  ]
}}

CRITICAL: Your response must have exactly {article_count} objects in the articles array. If there are {article_count} articles provided, there must be {article_count} score objects returned."""
        
        else:
            # Original prompt for other models
            base_prompt = f"""You are an expert tech news evaluator. You will analyze EXACTLY {article_count} articles and provide scores from 0.0 to 1.0 for each article.

CRITICAL: Your response must contain EXACTLY {article_count} articles in the "articles" array. No more, no less.

Score each article on these dimensions:
- RELEVANCE: How relevant to technology professionals (AI, software, cybersecurity, startups)
- QUALITY: Content depth, technical accuracy, and journalistic quality
- NOVELTY: How new, unique, or innovative the content is
- IMPACT: Importance and potential influence on the tech community
- CONFIDENCE: Your confidence in the assessment (0.0 = uncertain, 1.0 = very confident)
- UNCERTAINTY: Your uncertainty about the assessment (0.0 = very certain, 1.0 = very uncertain)

Respond with a JSON object containing an "articles" array with EXACTLY {article_count} entries. Each article should have scores as decimal numbers between 0.0 and 1.0.

Required JSON format:
{{
  "articles": [
    {{
      "relevance": 0.8,
      "quality": 0.7,
      "novelty": 0.6,
      "impact": 0.9,
      "confidence": 0.8,
      "uncertainty": 0.2
    }}"""
        
            # Add more examples if we have multiple articles
            if article_count > 1:
                base_prompt += """,
    {
      "relevance": 0.3,
      "quality": 0.4,
      "novelty": 0.2,
      "impact": 0.1,
      "confidence": 0.9,
      "uncertainty": 0.1
    }"""
                
            if article_count > 2:
                base_prompt += f"""
    ... (continue for all {article_count} articles)"""
            
            base_prompt += """
  ]
}

IMPORTANT: The "articles" array must contain EXACTLY """ + str(article_count) + """ entries, one for each article provided."""
        
        if self.specialization in specialization_prompts:
            base_prompt += f"\n\nSpecial focus: {specialization_prompts[self.specialization]}"
        
        return base_prompt
    
    def _prepare_articles_for_batch(self, articles: List[Dict[str, Any]]) -> str:
        """Prepare articles for batch processing - OPTIMIZED for speed."""
        return '\n'.join([
            f"{i+1}. {article.get('title', 'No title')[:self.title_max_length]}"
            f"{' - ' + article.get('description', '')[:self.description_max_length] if article.get('description', '').strip() else ''}"
            for i, article in enumerate(articles)
        ])
    
    async def _make_api_call(self, messages: List[Dict[str, Any]]) -> Optional[BatchScoringResponse]:
        """Make API call using Groq client with JSON mode for structured output."""
        if not self.groq_client:
            raise RuntimeError("Groq client not initialized. Use async context manager.")

        await self._intelligent_rate_limit(self._estimate_batch_tokens(messages))

        start_time = time.time()

        try:
            # Model-specific token limits
            if self.model_name == 'gemma2-9b-it':
                # Give gemma2 more tokens to complete the JSON response properly
                max_tokens = min(len(messages[1]['content'].split()) * 3, 2000)  # More generous limit
            else:
                max_tokens = min(self.max_tokens_per_article * 20, 1000)
                
            # Use Groq's JSON mode for guaranteed structured output
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
                return None
                
        except groq.RateLimitError as e:
            logger.warning(f"Rate limit error for {self.model_name}: {e}")
            # Groq client handles retries automatically
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
        """Get default scores based on agent specialization."""
        # Base defaults - more reasonable values that vary by agent
        agent_variance = hash(self.model_name) % 100 / 1000.0  # 0.0 to 0.099 variance
        
        defaults = {
            'relevance': 0.5 + agent_variance,
            'quality': 0.5 + agent_variance,
            'novelty': 0.4 + agent_variance,
            'impact': 0.4 + agent_variance,
            'confidence': 0.6 + agent_variance,
            'uncertainty': 0.3 + agent_variance
        }
        
        # Adjust based on specialization
        if self.specialization == 'content_quality_analysis':
            defaults['quality'] = 0.6 + agent_variance
        elif self.specialization == 'relevance_scoring':
            defaults['relevance'] = 0.6 + agent_variance
        elif self.specialization == 'trend_detection':
            defaults['novelty'] = 0.5 + agent_variance
            defaults['impact'] = 0.5 + agent_variance
        elif self.specialization == 'innovation_curation':
            defaults['novelty'] = 0.5 + agent_variance
            defaults['impact'] = 0.5 + agent_variance
        
        # Clamp all values to valid range
        for key in defaults:
            defaults[key] = max(0.0, min(1.0, defaults[key]))
        
        return defaults
    
    async def process_batch(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process a batch of articles and return decisions with confidence scores."""
        if not articles:
            return []
        
        messages = [
            {"role": "system", "content": self._get_filtering_prompt(len(articles))},
            {"role": "user", "content": self._prepare_articles_for_batch(articles)}
        ]
        
        batch_response = await self._make_api_call(messages)
        if not batch_response:
            logger.warning(f"API call failed for {self.model_name}, returning conservative decisions")
            return [(article, False, 0.0) for article in articles]
        
        # Enhanced logging for debugging
        logger.info(f"{self.model_name} processing {len(articles)} articles")
        logger.info(f"{self.model_name} received {len(batch_response.articles)} scores")
        
        # Process structured JSON response
        if batch_response.articles and len(batch_response.articles) >= len(articles) * 0.7:
            results = []
            accept_count = 0
            
            # Ensure we have exactly the right number of scores
            scores_to_use = batch_response.articles[:len(articles)]
            
            # Pad with default scores if needed
            while len(scores_to_use) < len(articles):
                defaults = self._get_default_scores()
                default_score = ArticleScores(**defaults)
                scores_to_use.append(default_score)
            
            for i, (article, article_scores) in enumerate(zip(articles, scores_to_use)):
                # Create multi-dimensional score
                md_score = create_multi_dimensional_score(
                    relevance=article_scores.relevance,
                    quality=article_scores.quality,
                    novelty=article_scores.novelty,
                    impact=article_scores.impact,
                    confidence_mean=article_scores.confidence,
                    confidence_std=article_scores.uncertainty,
                    agent_name=self.model_name,
                    model_name=self.model_name,
                    specialization=self.specialization
                )
                
                decision, confidence = md_score.get_legacy_format()
                results.append((article, decision, confidence))
                article['multi_dimensional_score'] = md_score.to_dict()
                
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
            logger.warning(f"Agent {self.model_name}: JSON parsing failed or insufficient scores, got {len(batch_response.articles) if batch_response.articles else 0} scores for {len(articles)} articles")
            
            # Fallback: use conservative decisions
            logger.info(f"Agent {self.model_name}: Using conservative fallback decisions")
            return [(article, False, 0.2) for article in articles]
    
    async def process_articles_with_rate_limiting(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles in batches with intelligent rate limiting."""
        if not articles:
            return []
        
        effective_batch_size = self.adaptive_batch_size
        # Reduced logging - only log the initial setup
        total_batches = (len(articles) + effective_batch_size - 1) // effective_batch_size
        
        all_results = []
        
        for i in range(0, len(articles), effective_batch_size):
            batch = articles[i:i + effective_batch_size]
            batch_count = (i // effective_batch_size) + 1
            
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Intelligent delay between batches - OPTIMIZED FOR SPEED
            if i + effective_batch_size < len(articles):
                # SPEED-OPTIMIZED delay strategy - much faster processing
                if self.model_name == 'gemma2-9b-it':
                    # Gemma2: More conservative approach due to JSON validation issues
                    base_delay = 5.0  # Increased delay to reduce stress on the model
                    jitter = random.uniform(0, base_delay * 0.1)  # Small jitter
                elif self.tokens_per_minute_limit <= 6000:
                    # 6k token models: Moderate batches, fast delays  
                    base_delay = 4.0  # 4s delay = 15 batches/min
                    jitter = random.uniform(0, base_delay * 0.05)
                else:
                    # Medium models: Balanced for speed
                    base_delay = 5.0  # 5s delay = 12 batches/min
                    jitter = random.uniform(0, base_delay * 0.05)
                
                total_delay = base_delay + jitter
                await asyncio.sleep(total_delay)
        
        return all_results
    
    async def process_articles(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles in batches (legacy method)."""
        return await self.process_articles_with_rate_limiting(articles)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for monitoring."""
        return {
            'model_name': self.model_name,
            'specialization': self.specialization,
            'batch_size': self.batch_size,
            'consensus_weight': self.consensus_weight,
            'expertise_domains': self.expertise_domains,
            'requests_per_minute': self.requests_per_minute,
            'tokens_per_minute': self.tokens_per_minute,
            'tokens_per_day_limit': self.tokens_per_day_limit,
            'daily_tokens_used': self.daily_tokens_used,
            'daily_tokens_remaining': self.tokens_per_day_limit - self.daily_tokens_used if self.tokens_per_day_limit != float('inf') else 'unlimited',
            'adaptive_batch_size': self.adaptive_batch_size,
            'safety_margin': self.safety_margin,
            'api_client': 'groq-python',  # Using official Groq library
            'json_mode_enabled': True,    # Structured JSON output
            'max_retries': self.max_retries
        }
