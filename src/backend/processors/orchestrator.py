#!/usr/bin/env python3
"""
Orchestrator - Optimized 4-Stage Pipeline

Central coordinator for the NewsXP.ai processing pipeline implementing the optimized
4-stage architecture from PIPELINE_IMPROVEMENTS.md:

Stage 1: Bulk Processing (meta-llama/llama-4-scout-17b-16e-instruct)
Stage 2: Compound Intelligence (deepseek-r1-distill-llama-70b)  
Stage 3: Expert Analysis (llama-3.3-70b-versatile)
Stage 4: Final Ranking (qwen/qwen3-32b)
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

# Import deduplication utilities
try:
    from .deduplication_utils import ArticleDeduplicator
except ImportError:
    try:
        # Fallback for standalone execution
        from deduplication_utils import ArticleDeduplicator
    except ImportError:
        # Final fallback if import fails
        ArticleDeduplicator = None

class NewsOrchestrator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the News Orchestrator with 4-stage pipeline."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.groq_available = False
        
        # Initialize deduplication utility
        if ArticleDeduplicator:
            self.deduplicator = ArticleDeduplicator()
        else:
            self.deduplicator = None
        
        # Import optimized 4-stage agents
        self.Stage1BulkFilter = None
        self.Stage2CompoundAgent = None
        self.Stage3ExpertAgent = None
        self.Stage4FinalAgent = None
        
        try:
            sys.path.append(os.path.dirname(__file__))
            
            # Load new 4-stage agents
            from stage1_bulk_filter import Stage1BulkFilter
            from stage2_compound_agent import Stage2CompoundAgent
            from stage3_expert_agent import Stage3ExpertAgent
            from stage4_final_agent import Stage4FinalAgent
            self.Stage1BulkFilter = Stage1BulkFilter
            self.Stage2CompoundAgent = Stage2CompoundAgent
            self.Stage3ExpertAgent = Stage3ExpertAgent
            self.Stage4FinalAgent = Stage4FinalAgent
            
            self.groq_available = bool(self.api_key)
            
            logger.info("4-Stage pipeline agents loaded successfully")
        except ImportError as e:
            logger.warning(f"AI agents not available: {e}")
            self.groq_available = False
        except Exception as e:
            logger.error(f"Error loading AI agents: {e}")
            self.groq_available = False
            
        # Load pipeline configuration from app.json
        self.pipeline_config = self._load_pipeline_config()
    
    def _load_pipeline_config(self):
        """Load pipeline configuration from app.json."""
        try:
            # Get the project root directory
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            config_file = project_root / "src" / "shared" / "config" / "app.json"
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Use development config by default (can be overridden by environment)
            env = os.getenv('NODE_ENV', 'development')
            env_config = config.get(env, config.get('development', {}))
            pipeline_config = env_config.get('pipeline', {})
            
            return {
                'stage1': pipeline_config.get('stage1', {
                    'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                    'max_articles': 100,
                    'target_pass_rate': 0.70,
                    'time_budget_minutes': 10
                }),
                'stage2': pipeline_config.get('stage2', {
                    'model': 'deepseek-r1-distill-llama-70b',
                    'target_pass_rate': 0.35,
                    'time_budget_minutes': 5
                }),
                'stage3': pipeline_config.get('stage3', {
                    'model': 'llama-3.3-70b-versatile',
                    'target_pass_rate': 0.15,
                    'time_budget_minutes': 5
                }),
                'stage4': pipeline_config.get('stage4', {
                    'model': 'qwen/qwen3-32b',
                    'final_count': 5,
                    'time_budget_minutes': 15
                })
            }
            
        except Exception as e:
            logger.warning(f"Failed to load pipeline config from app.json: {e}. Using defaults.")
            # Fallback to hardcoded defaults
            return {
                'stage1': {
                    'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                    'max_articles': 100,
                    'target_pass_rate': 0.70,
                    'time_budget_minutes': 10
                },
                'stage2': {
                    'model': 'deepseek-r1-distill-llama-70b',
                    'target_pass_rate': 0.35,
                    'time_budget_minutes': 5
                },
                'stage3': {
                    'model': 'llama-3.3-70b-versatile',
                    'target_pass_rate': 0.15,
                    'time_budget_minutes': 5
                },
                'stage4': {
                    'model': 'qwen/qwen3-32b',
                    'final_count': 5,
                    'time_budget_minutes': 15
                }
            }

    def _deduplicate_pipeline_stage(self, articles: List[Dict], stage_name: str, strategy: str = "enhanced") -> List[Dict]:
        """Deduplicate articles between pipeline stages."""
        if not articles:
            return articles
            
        original_count = len(articles)
        if self.deduplicator:
            deduplicated = self.deduplicator.deduplicate_articles(articles, strategy=strategy)
        else:
            # Simple fallback deduplication by URL
            deduplicated = []
            seen_urls = set()
            for article in articles:
                url = article.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated.append(article)
        
        if len(deduplicated) < original_count:
            removed_count = original_count - len(deduplicated)
            logger.info(f"{stage_name} deduplication: {original_count} -> {len(deduplicated)} articles ({removed_count} duplicates removed)")
        
        return deduplicated
    
    def _clean_and_validate_articles(self, articles: List[Dict], stage_name: str) -> List[Dict]:
        """Clean and validate articles after processing stage."""
        if not articles:
            return articles
        
        # Remove articles that are incomplete or invalid
        valid_articles = []
        for article in articles:
            # Check if article has minimum required fields
            if (article.get('title') and 
                article.get('url') and 
                article.get('description') and
                len(str(article.get('title', ''))) > 5):
                valid_articles.append(article)
            else:
                logger.debug(f"{stage_name}: Removed invalid article: {article.get('title', 'NO_TITLE')}")
        
        removed_count = len(articles) - len(valid_articles)
        if removed_count > 0:
            logger.info(f"{stage_name} validation: {len(articles)} -> {len(valid_articles)} articles ({removed_count} invalid articles removed)")
        
        return valid_articles

    async def process_articles(self, articles: List[Dict], target_count: int = 25) -> Dict[str, Any]:
        """
        Main processing method - uses optimized 4-stage pipeline.
        
        For backward compatibility, this method now calls the optimized pipeline.
        """
        logger.info("Using optimized 4-stage pipeline")
        return await self.process_articles_optimized(articles, target_count)
    
    async def process_articles_optimized(self, articles: List[Dict], target_count: int = 25) -> Dict[str, Any]:
        """
        Process articles through the optimized 4-stage pipeline.
        
        Pipeline: Raw Articles -> Stage 1 Bulk -> Stage 2 Compound -> Stage 3 Expert -> Stage 4 Final
        """
        start_time = time.time()
        
        pipeline_result = {
            'input_count': len(articles),
            'output_count': 0,
            'articles': [],
            'processing_time': 0,
            'pipeline_stages': {},
            'pipeline_version': '4-stage-optimized',
            'pipeline_config': self.pipeline_config,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            if not articles:
                logger.warning("No articles provided for processing")
                return pipeline_result

            logger.info(f"Starting optimized 4-stage pipeline for {len(articles)} articles (target: {target_count})")
            
            # Pre-pipeline deduplication and cleanup
            current_articles = self._deduplicate_pipeline_stage(articles.copy(), "Pre-pipeline", strategy="enhanced")
            current_articles = self._clean_and_validate_articles(current_articles, "Pre-pipeline")
            
            logger.info(f"Pre-pipeline cleanup: {len(articles)} -> {len(current_articles)} articles")
            
            # Stage 1: Bulk Processing Filter
            stage_start = time.time()
            if self.groq_available and self.Stage1BulkFilter and self.api_key:
                logger.info(f"Stage 1: Bulk filtering up to {self.pipeline_config['stage1']['max_articles']} articles")
                
                async with self.Stage1BulkFilter(self.api_key, self.pipeline_config['stage1']) as stage1_agent:
                    stage1_result = await stage1_agent.bulk_filter_articles(
                        current_articles, 
                        max_articles=self.pipeline_config['stage1']['max_articles']
                    )
                
                current_articles = stage1_result['articles']
                
                # Post-stage 1 deduplication and cleanup
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 1")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Post-Stage 1")
                
                pipeline_result['pipeline_stages']['stage1_bulk_filter'] = {
                    'model': stage1_result['model'],
                    'input_count': stage1_result['input_count'],
                    'output_count': len(current_articles),
                    'pass_rate': len(current_articles) / stage1_result['input_count'] if stage1_result['input_count'] > 0 else 0,
                    'target_pass_rate': stage1_result['target_pass_rate'],
                    'requests_made': stage1_result['requests_made'],
                    'processing_time': time.time() - stage_start,
                    'articles_per_second': stage1_result['articles_per_second']
                }
                
                logger.info(f"Stage 1 complete: {len(current_articles)} articles ({pipeline_result['pipeline_stages']['stage1_bulk_filter']['pass_rate']:.1%} pass rate)")
            else:
                logger.warning("Stage 1: Skipping bulk filtering (AI not available)")
                pipeline_result['pipeline_stages']['stage1_bulk_filter'] = {
                    'input_count': len(articles),
                    'output_count': len(articles),
                    'processing_time': time.time() - stage_start,
                    'skipped': True,
                    'reason': 'ai_not_available'
                }
            
            # Stage 2: Compound Intelligence
            stage_start = time.time()
            if self.groq_available and self.Stage2CompoundAgent and self.api_key:
                logger.info(f"Stage 2: Compound analysis of {len(current_articles)} articles")
                
                async with self.Stage2CompoundAgent(self.api_key, self.pipeline_config['stage2']) as stage2_agent:
                    stage2_result = await stage2_agent.compound_analyze_articles(
                        current_articles, 
                        max_articles=len(current_articles)
                    )
                
                current_articles = stage2_result['articles']
                
                # Post-stage 2 deduplication and cleanup
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 2")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Post-Stage 2")
                
                pipeline_result['pipeline_stages']['stage2_compound'] = {
                    'model': stage2_result['model'],
                    'input_count': stage2_result['input_count'],
                    'output_count': len(current_articles),
                    'pass_rate': len(current_articles) / stage2_result['input_count'] if stage2_result['input_count'] > 0 else 0,
                    'target_pass_rate': stage2_result['target_pass_rate'],
                    'requests_made': stage2_result['requests_made'],
                    'processing_time': time.time() - stage_start,
                    'articles_per_second': stage2_result['articles_per_second'],
                    'batch_size': stage2_result['batch_size'],
                    'batches_processed': stage2_result['batches_processed']
                }
                
                logger.info(f"Stage 2 complete: {len(current_articles)} articles ({pipeline_result['pipeline_stages']['stage2_compound']['pass_rate']:.1%} pass rate)")
            else:
                logger.info("Stage 2: Skipping compound analysis (AI not available)")
                # Apply target rate for consistency
                target_count_stage2 = int(len(current_articles) * self.pipeline_config['stage2']['target_pass_rate'])
                current_articles = current_articles[:target_count_stage2]
                
                # Still apply deduplication and cleanup even without AI
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 2 (No AI)")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Post-Stage 2 (No AI)")
                
                pipeline_result['pipeline_stages']['stage2_compound'] = {
                    'input_count': len(current_articles) if 'stage1_bulk_filter' not in pipeline_result['pipeline_stages'] 
                                   else pipeline_result['pipeline_stages']['stage1_bulk_filter']['output_count'],
                    'output_count': len(current_articles),
                    'processing_time': time.time() - stage_start,
                    'skipped': True,
                    'reason': 'ai_not_available'
                }
            
            # Stage 3: Expert Analysis
            stage_start = time.time()
            if self.groq_available and self.Stage3ExpertAgent and self.api_key:
                logger.info(f"Stage 3: Expert analysis of {len(current_articles)} articles")
                
                async with self.Stage3ExpertAgent(self.api_key, self.pipeline_config['stage3']) as stage3_agent:
                    stage3_result = await stage3_agent.expert_analyze_articles(
                        current_articles, 
                        max_articles=len(current_articles)
                    )
                
                current_articles = stage3_result['articles']
                
                # Post-stage 3 deduplication and cleanup
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 3")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Post-Stage 3")
                
                pipeline_result['pipeline_stages']['stage3_expert'] = {
                    'model': stage3_result['model'],
                    'input_count': stage3_result['input_count'],
                    'output_count': len(current_articles),
                    'pass_rate': len(current_articles) / stage3_result['input_count'] if stage3_result['input_count'] > 0 else 0,
                    'target_pass_rate': stage3_result['target_pass_rate'],
                    'requests_made': stage3_result['requests_made'],
                    'processing_time': time.time() - stage_start,
                    'articles_per_second': stage3_result['articles_per_second'],
                    'batch_size': stage3_result['batch_size'],
                    'batches_processed': stage3_result['batches_processed'],
                    'categories_found': stage3_result['categories_found'],
                    'quality_distribution': stage3_result['quality_distribution']
                }
                
                logger.info(f"Stage 3 complete: {len(current_articles)} articles ({pipeline_result['pipeline_stages']['stage3_expert']['pass_rate']:.1%} pass rate)")
            else:
                logger.info("Stage 3: Skipping expert analysis (AI not available)")
                # Apply target rate for consistency
                target_count_stage3 = int(len(current_articles) * self.pipeline_config['stage3']['target_pass_rate'])
                current_articles = current_articles[:target_count_stage3]
                
                # Still apply deduplication and cleanup even without AI
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 3 (No AI)")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Post-Stage 3 (No AI)")
                
                pipeline_result['pipeline_stages']['stage3_expert'] = {
                    'input_count': pipeline_result['pipeline_stages']['stage2_compound']['output_count'],
                    'output_count': len(current_articles),
                    'processing_time': time.time() - stage_start,
                    'skipped': True,
                    'reason': 'ai_not_available'
                }
            
            # Stage 4: Final Ranking
            stage_start = time.time()
            if self.groq_available and self.Stage4FinalAgent and self.api_key:
                logger.info(f"Stage 4: Final ranking of {len(current_articles)} articles to {target_count}")
                
                async with self.Stage4FinalAgent(self.api_key, self.pipeline_config['stage4']) as stage4_agent:
                    stage4_result = await stage4_agent.final_rank_articles(
                        current_articles, 
                        target_count=target_count
                    )
                
                current_articles = stage4_result['articles']
                
                # Final deduplication and cleanup (strict mode)
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 4")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Final", strategy="strict")
                
                pipeline_result['pipeline_stages']['stage4_final'] = {
                    'model': stage4_result['model'],
                    'input_count': stage4_result['input_count'],
                    'output_count': len(current_articles),
                    'target_count': target_count,
                    'requests_made': stage4_result['requests_made'],
                    'successful_evaluations': stage4_result['successful_evaluations'],
                    'failed_evaluations': stage4_result['failed_evaluations'],
                    'processing_time': time.time() - stage_start,
                    'articles_per_second': stage4_result['articles_per_second'],
                    'acceptance_rate': stage4_result['acceptance_rate'],
                    'average_score': stage4_result['average_score'],
                    'score_distribution': stage4_result['score_distribution'],
                    'priority_distribution': stage4_result['priority_distribution'],
                    'quality_metrics': stage4_result['quality_metrics']
                }
                
                logger.info(f"Stage 4 complete: {len(current_articles)} final articles selected")
            else:
                logger.info("Stage 4: Skipping final ranking (AI not available)")
                # Ensure we don't exceed target count
                current_articles = current_articles[:target_count]
                
                # Final deduplication and cleanup even without AI
                current_articles = self._clean_and_validate_articles(current_articles, "Stage 4 (No AI)")
                current_articles = self._deduplicate_pipeline_stage(current_articles, "Final (No AI)", strategy="strict")
                
                pipeline_result['pipeline_stages']['stage4_final'] = {
                    'input_count': pipeline_result['pipeline_stages']['stage3_expert']['output_count'],
                    'output_count': len(current_articles),
                    'target_count': target_count,
                    'processing_time': time.time() - stage_start,
                    'skipped': True,
                    'reason': 'ai_not_available'
                }
            
            # Ensure proper category distribution (10 research + 15 regular = 25 total)
            current_articles = self._ensure_category_distribution(current_articles, target_count)
            
            # Final results
            pipeline_result['articles'] = current_articles
            pipeline_result['output_count'] = len(current_articles)
            pipeline_result['processing_time'] = time.time() - start_time
            
            # Calculate overall pipeline efficiency
            overall_pass_rate = len(current_articles) / len(articles) if articles else 0
            pipeline_result['overall_pass_rate'] = overall_pass_rate
            
            logger.info(f"4-Stage pipeline complete: {len(current_articles)}/{len(articles)} articles ({overall_pass_rate:.1%}) in {pipeline_result['processing_time']:.2f}s")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"4-Stage pipeline processing failed: {e}")
            pipeline_result['error'] = str(e)
            pipeline_result['processing_time'] = time.time() - start_time
            return pipeline_result
    
    def _calculate_fallback_score(self, article: Dict) -> float:
        """Calculate a simple fallback score when AI filtering is unavailable."""
        score = 50.0  # Base score
        
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        combined_text = f"{title} {description}"
        
        # AI keywords scoring
        ai_keywords = {
            'artificial intelligence': 15, 'machine learning': 12, 'deep learning': 12,
            'neural network': 10, 'transformer': 10, 'llm': 12, 'gpt': 10, 'claude': 8,
            'openai': 10, 'anthropic': 8, 'breakthrough': 12, 'research': 6, 'arxiv': 7,
            'open source': 8, 'safety': 7, 'agi': 10, 'multimodal': 8, 'robotics': 7
        }
        
        keyword_score = 0
        for keyword, points in ai_keywords.items():
            if keyword in combined_text:
                keyword_score += points
        
        score += min(30, keyword_score)
        
        # Source quality bonus
        source = article.get('source', '').lower()
        quality_sources = ['arxiv', 'nature', 'science', 'mit', 'stanford', 'openai', 'anthropic']
        if any(qs in source for qs in quality_sources):
            score += 15
        
        # Recent articles bonus
        try:
            from dateutil import parser as date_parser
            pub_date = date_parser.parse(article.get('published_date', ''))
            hours_old = (datetime.now(timezone.utc) - pub_date).total_seconds() / 3600
            if hours_old < 24:
                score += 10
        except:
            pass
        
        return min(100.0, max(0.0, score))
    
    def get_pipeline_summary(self, pipeline_result: Dict) -> Dict[str, Any]:
        """Generate a comprehensive summary of the pipeline execution."""
        stages = pipeline_result.get('pipeline_stages', {})
        
        total_processing_time = pipeline_result.get('processing_time', 0)
        
        stage_summary = {}
        for stage_name, stage_data in stages.items():
            stage_summary[stage_name] = {
                'duration': stage_data.get('processing_time', 0),
                'input_count': stage_data.get('input_count', 0),
                'output_count': stage_data.get('output_count', 0),
                'success': not stage_data.get('error') and not stage_data.get('skipped', False)
            }
        
        return {
            'pipeline_execution': {
                'total_time': total_processing_time,
                'input_articles': pipeline_result.get('input_count', 0),
                'output_articles': pipeline_result.get('output_count', 0),
                'pipeline_version': pipeline_result.get('pipeline_version', 'unknown'),
                'overall_pass_rate': pipeline_result.get('overall_pass_rate'),
                'success': not pipeline_result.get('error')
            },
            'stage_performance': stage_summary,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def format_for_api(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format pipeline results for API consumption (latest.json format).
        
        Args:
            pipeline_result: Result from process_articles_optimized()
            
        Returns:
            Dict formatted for API consumption
        """
        articles = pipeline_result.get('articles', [])
        
        return {
            "generated_at": pipeline_result.get('generated_at', datetime.now(timezone.utc).isoformat()),
            "collection_time_seconds": pipeline_result.get('processing_time', 0),
            "count": len(articles),
            "articles": articles,
            "pipeline_info": {
                "version": pipeline_result.get('pipeline_version', '4-stage-optimized'),
                "input_count": pipeline_result.get('input_count', 0),
                "output_count": pipeline_result.get('output_count', 0),
                "overall_pass_rate": pipeline_result.get('overall_pass_rate', 0),
                "processing_time": pipeline_result.get('processing_time', 0),
                "stages": {
                    stage_name: {
                        "input_count": stage_data.get('input_count', 0),
                        "output_count": stage_data.get('output_count', 0),
                        "pass_rate": stage_data.get('pass_rate', 0),
                        "processing_time": stage_data.get('processing_time', 0),
                        "model": stage_data.get('model', 'unknown'),
                        "skipped": stage_data.get('skipped', False)
                    }
                    for stage_name, stage_data in pipeline_result.get('pipeline_stages', {}).items()
                }
            }
        }

    def save_to_api_directories(self, pipeline_result: Dict[str, Any], project_root: str) -> bool:
        """
        Save pipeline results to API directories for web consumption.
        
        Args:
            pipeline_result: Result from process_articles_optimized()
            project_root: Path to project root directory
            
        Returns:
            bool: Success status
        """
        try:
            project_path = Path(project_root)
            api_data = self.format_for_api(pipeline_result)
            
            # Save to backend API directory
            backend_api_dir = project_path / "src" / "backend" / "api"
            backend_api_dir.mkdir(parents=True, exist_ok=True)
            backend_latest_path = backend_api_dir / "latest.json"
            
            with open(backend_latest_path, "w") as f:
                json.dump(api_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved processed articles to {backend_latest_path}")
            
            # Save to frontend API directory (for GitHub Pages)
            frontend_api_dir = project_path / "src" / "frontend" / "api"
            frontend_api_dir.mkdir(parents=True, exist_ok=True)
            frontend_latest_path = frontend_api_dir / "latest.json"
            
            with open(frontend_latest_path, "w") as f:
                json.dump(api_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved processed articles to {frontend_latest_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save to API directories: {e}")
            return False

    def _ensure_category_distribution(self, articles: List[Dict], target_count: int = 25) -> List[Dict]:
        """
        Ensure proper distribution: 10 research articles + 15 regular articles.
        The frontend expects exactly this split and will filter to these amounts.
        
        Args:
            articles: List of articles to distribute
            target_count: Total target count (should be 25)
            
        Returns:
            List of articles with proper distribution
        """
        if not articles:
            return articles
        
        # Separate research and regular articles
        research_articles = [a for a in articles if a.get('category') == 'Research']
        regular_articles = [a for a in articles if a.get('category') != 'Research']
        
        # Sort by quality/priority (highest first)
        research_articles.sort(key=lambda x: (
            x.get('final_score', 0),
            x.get('stage3_analysis', {}).get('quality_score', 0),
            x.get('stage2_score', 0)
        ), reverse=True)
        
        regular_articles.sort(key=lambda x: (
            x.get('final_score', 0),
            x.get('stage3_analysis', {}).get('quality_score', 0),
            x.get('stage2_score', 0)
        ), reverse=True)
        
        # Target distribution: 10 research + 15 regular = 25 total
        target_research = 10
        target_regular = 15
        
        # Take the best articles from each category
        selected_research = research_articles[:target_research]
        selected_regular = regular_articles[:target_regular]
        
        # If we don't have enough in one category, fill from the other
        total_selected = len(selected_research) + len(selected_regular)
        
        if total_selected < target_count:
            # Fill remaining slots with best available articles
            remaining_slots = target_count - total_selected
            
            # If we're short on research, try to get more research articles
            if len(selected_research) < target_research:
                additional_research_needed = min(remaining_slots, target_research - len(selected_research))
                additional_research = research_articles[len(selected_research):len(selected_research) + additional_research_needed]
                selected_research.extend(additional_research)
                remaining_slots -= len(additional_research)
            
            # If we're short on regular articles, try to get more regular articles
            if len(selected_regular) < target_regular and remaining_slots > 0:
                additional_regular_needed = min(remaining_slots, target_regular - len(selected_regular))
                additional_regular = regular_articles[len(selected_regular):len(selected_regular) + additional_regular_needed]
                selected_regular.extend(additional_regular)
                remaining_slots -= len(additional_regular)
            
            # If we still have slots, fill with any remaining articles
            if remaining_slots > 0:
                all_remaining = research_articles[len(selected_research):] + regular_articles[len(selected_regular):]
                all_remaining.sort(key=lambda x: (
                    x.get('final_score', 0),
                    x.get('stage3_analysis', {}).get('quality_score', 0),
                    x.get('stage2_score', 0)
                ), reverse=True)
                
                additional_articles = all_remaining[:remaining_slots]
                # Add to the appropriate lists based on category
                for article in additional_articles:
                    if article.get('category') == 'Research':
                        selected_research.append(article)
                    else:
                        selected_regular.append(article)
        
        # Combine and sort final articles by score for consistent ordering
        final_articles = selected_research + selected_regular
        final_articles.sort(key=lambda x: (
            x.get('final_score', 0),
            x.get('stage3_analysis', {}).get('quality_score', 0),
            x.get('stage2_score', 0)
        ), reverse=True)
        
        # Log distribution
        final_research = len([a for a in final_articles if a.get('category') == 'Research'])
        final_regular = len([a for a in final_articles if a.get('category') != 'Research'])
        
        logger.info(f"Category distribution: {final_research} research + {final_regular} regular = {len(final_articles)} total")
        
        return final_articles[:target_count]

async def main():
    """Main entry point for the orchestrator."""
    
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <news_json_file>")
        return
    
    with open(sys.argv[1], 'r') as f:
        news_data = json.load(f)
    
    articles = news_data.get('articles', [])
    
    orchestrator = NewsOrchestrator()
    pipeline_result = await orchestrator.process_articles(articles, target_count=25)
    
    print("Pipeline Results:")
    print(f"Input: {pipeline_result['input_count']} articles")
    print(f"Output: {pipeline_result['output_count']} articles")
    print(f"Processing time: {pipeline_result['processing_time']:.2f}s")
    print(f"Pipeline version: {pipeline_result.get('pipeline_version', 'unknown')}")
    
    if pipeline_result.get('overall_pass_rate'):
        print(f"Overall pass rate: {pipeline_result['overall_pass_rate']:.1%}")
    
    if pipeline_result.get('collection_summary'):
        print(f"\nCollection Summary:")
        summary = pipeline_result['collection_summary']
        print(f"Summary: {summary.get('summary', 'N/A')}")
        print(f"Key themes: {', '.join(summary.get('key_themes', []))}")
    
    summary = orchestrator.get_pipeline_summary(pipeline_result)
    print(f"\nPipeline Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save results to API directories for web consumption
    project_root = Path(__file__).parent.parent.parent.parent
    success = orchestrator.save_to_api_directories(pipeline_result, str(project_root))
    
    if success:
        print("✅ Results saved to API directories")
    else:
        print("❌ Failed to save results to API directories")
        
    # Also save to test_output for workflow artifacts
    test_output_dir = project_root / "test_output"
    test_output_dir.mkdir(exist_ok=True)
    
    # Save processed articles
    processed_file = test_output_dir / "processed_articles.json"
    with open(processed_file, 'w') as f:
        json.dump(pipeline_result, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved full pipeline result to {processed_file}")


if __name__ == "__main__":
    asyncio.run(main())
