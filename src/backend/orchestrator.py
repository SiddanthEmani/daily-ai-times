#!/usr/bin/env python3
"""
News Processing Pipeline Orchestrator for Daily AI Times
Orchestrates: Collection → Swarm Scoring → Consensus Filtering
"""

import os, json, logging, asyncio, time, sys, traceback, random
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console

from dotenv import load_dotenv
load_dotenv('.env.local')
load_dotenv()

# Add the project root to Python path to enable proper imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modules using the proper src.* paths
from src.backend.processors.bulk_agent import BulkFilteringAgent
from src.backend.processors.consensus_engine import ConsensusEngine
from src.backend.processors.deep_intelligence_agent import DeepIntelligenceAgent
from src.backend.processors.final_consensus_engine import FinalConsensusEngine
from src.backend.collectors.collectors import NewsCollector
from src.shared.config.config_loader import ConfigLoader, get_swarm_config
from src.shared.utils.logging_config import log_warning, log_error, log_step
from google import genai
from google.genai import types
import base64
import wave

# Image extraction imports
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

# Constants for processing delays (TPM values now come from swarm configuration)
SUPPRESSED_LOGGERS = [
    "httpx", "httpcore", "backend.processors.bulk_agent", 
    "urllib3", "asyncio", "rich", "backend.orchestrator",
    "backend.collectors", "backend.processors", "shared.utils",
    "backend.processors.deep_intelligence_agent"
]

@contextmanager
def suppress_logging():
    """Context manager to suppress verbose logging during processing."""
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    original_handler_levels = [handler.level for handler in root_logger.handlers]
    original_levels = {}
    
    # Suppress root logger and handlers
    for handler in root_logger.handlers:
        handler.setLevel(logging.CRITICAL + 1)
    root_logger.setLevel(logging.CRITICAL + 1)
    
    # Suppress specific loggers
    for logger_name in SUPPRESSED_LOGGERS:
        log_instance = logging.getLogger(logger_name)
        original_levels[logger_name] = log_instance.level
        log_instance.setLevel(logging.CRITICAL + 1)
    
    try:
        yield
    finally:
        # Restore original levels
        root_logger.setLevel(original_root_level)
        for i, handler in enumerate(root_logger.handlers):
            if i < len(original_handler_levels):
                handler.setLevel(original_handler_levels[i])
        
        for logger_name, original_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)

class NewsProcessingPipeline:
    """Orchestrates the complete news processing pipeline. Sources are loaded from a single sources.yaml file."""
    
    @staticmethod
    def _get_current_timestamp() -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def _get_scripts_test_output_dir() -> Path:
        """Get the scripts/test_output directory path."""
        return Path(__file__).parent.parent.parent / "scripts" / "test_output"
    
    @staticmethod
    def _create_progress_bar(description: str, total: int, transient: bool = True) -> Progress:
        """Create a standardized progress bar configuration."""
        return Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
            console=Console(stderr=True, quiet=False), transient=transient,
            refresh_per_second=10, disable=False, expand=False
        )
    
    def _save_json_file(self, data: Dict[str, Any], filename: str) -> bool:
        """Save data as JSON file with error handling."""
        try:
            directory = self._get_scripts_test_output_dir()
            directory.mkdir(parents=True, exist_ok=True)
            output_file = directory / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {filename} to: {output_file}")
            return True
        except Exception as e:
            log_warning(logger, f"Failed to save {filename}: {e}")
            return False
    
    @staticmethod
    def _get_article_id(article: Dict[str, Any]) -> str:
        """Extract article ID with fallback to URL."""
        return article.get('id', article.get('url', 'unknown'))
    
    def __init__(self, config_path: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the news processing pipeline."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self._load_configs()
        self.agents = {}
        self.deep_intelligence_agents = {}
        self._initialize_agents()
        self.news_collector = NewsCollector()
        self.consensus_engine = ConsensusEngine(self.consensus_config, self.swarm_config)
        self.final_consensus_engine = FinalConsensusEngine(self.final_consensus_config)
        
    def _load_configs(self) -> None:
        """Load and extract all configurations."""
        self.app_config = ConfigLoader.load_config("app")
        self.swarm_config = get_swarm_config()
        
        agents_config = self.swarm_config.get('agents', {})
        self.bulk_swarm_config = agents_config.get('bulk_intelligence_swarm', {})
        self.deep_intelligence_swarm_config = agents_config.get('deep_intelligence_swarm', {})
        
        if not self.bulk_swarm_config:
            raise ValueError("bulk_intelligence_swarm configuration not found")
        
        self.enable_deep_intelligence = bool(self.deep_intelligence_swarm_config)
        self.consensus_config = self.swarm_config.get('consensus', {})
        
        default_final_consensus = {
            'deep_intelligence_weight': 0.6, 'initial_consensus_weight': 0.4,
            'min_deep_intelligence_confidence': 0.4, 'min_combined_score': 0.4,
            'consensus_method': 'weighted_combination', 'enable_quality_gates': True,
            'min_fact_check_confidence': 0.4, 'max_bias_tolerance': 0.7, 'min_credibility_score': 0.5
        }
        self.final_consensus_config = {**default_final_consensus, **self.swarm_config.get('final_consensus', {})}
        
        batch_config = self.swarm_config.get('processing', {}).get('batch_processing', {})
        self.max_batch_size = batch_config.get('max_batch_size', 100)
        self.min_batch_size = batch_config.get('min_batch_size', 10)
        self.inter_agent_delay = batch_config.get('inter_agent_delay', 8)
        
        app_collection_config = self.app_config.get('collection', {})
        self.default_max_articles = app_collection_config.get('max_articles_to_collect', 100)
    
    def _initialize_agents(self):
        """Initialize all bulk filtering and deep intelligence agents."""
        # Initialize bulk filtering agents
        for model_name, agent_config in self.bulk_swarm_config.get('agents', {}).items():
            try:
                self.agents[model_name] = BulkFilteringAgent(
                    model_name, agent_config, self.api_key, f"Agent {len(self.agents) + 1}")
            except Exception as e:
                log_error(logger, f"Failed to initialize bulk agent {model_name}: {e}")
        
        if not self.agents:
            raise ValueError("No bulk agents were successfully initialized")
        
        # Initialize deep intelligence agents
        if self.enable_deep_intelligence:
            enhanced_config_defaults = {
                'enable_fact_checking': True, 'enable_bias_detection': True,
                'enable_impact_analysis': True, 'enable_credibility_scoring': True,
                'analysis_depth': 'comprehensive', 'temperature': 0.3, 'max_tokens': 4000
            }
            
            for model_name, agent_config in self.deep_intelligence_swarm_config.get('agents', {}).items():
                try:
                    enhanced_config = {**agent_config, **enhanced_config_defaults}
                    if not self.api_key:
                        raise ValueError("API key is required for deep intelligence agents")
                    self.deep_intelligence_agents[model_name] = DeepIntelligenceAgent(
                        model_name, enhanced_config, self.api_key)
                except Exception as e:
                    log_error(logger, f"Failed to initialize deep intelligence agent {model_name}: {e}")
            
            if not self.deep_intelligence_agents:
                log_warning(logger, "No deep intelligence agents were successfully initialized")
                self.enable_deep_intelligence = False
        
        # Single concise initialization summary
        total_agents = len(self.agents) + len(self.deep_intelligence_agents)
        logger.info(f"Pipeline ready: {total_agents} agents")
    
    def _get_article_image(self, url: str, timeout: int = 10) -> Optional[str]:
        """Extract article image URL from meta tags"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ImageExtractor/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try Open Graph image first (most reliable)
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return urljoin(url, og_image['content'])
            
            # Try Twitter Card image
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                return urljoin(url, twitter_image['content'])
            
            # Try other common meta tags
            meta_image = soup.find('meta', attrs={'name': 'image'})
            if meta_image and meta_image.get('content'):
                return urljoin(url, meta_image['content'])
            
            # Try article:image
            article_image = soup.find('meta', property='article:image')
            if article_image and article_image.get('content'):
                return urljoin(url, article_image['content'])
            
            # Try to find the first large image in the article
            images = soup.find_all('img')
            for img in images:
                src = img.get('src')
                if src:
                    # Skip small images, icons, avatars
                    if any(skip in src.lower() for skip in ['icon', 'avatar', 'logo', 'thumb']):
                        continue
                    # Look for larger images (check width/height attributes)
                    width = img.get('width')
                    height = img.get('height')
                    if width and height:
                        try:
                            if int(width) > 200 and int(height) > 200:
                                return urljoin(url, src)
                        except ValueError:
                            pass
                    # If no size info, just take the first non-icon image
                    return urljoin(url, src)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting image from {url}: {e}")
            return None

    def _download_image(self, image_url: str, output_path: Path, timeout: int = 30) -> bool:
        """Download image from URL to specified path"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ImageDownloader/1.0)',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(image_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Determine file extension from content type or URL
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Try to get extension from URL
                parsed_url = urlparse(image_url)
                path = parsed_url.path.lower()
                if path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    ext = Path(path).suffix
                else:
                    ext = '.jpg'  # Default
            
            # Ensure output path has correct extension
            if not output_path.suffix:
                output_path = output_path.with_suffix(ext)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            logger.debug(f"Error downloading image from {image_url}: {e}")
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove extra spaces and dots
        filename = re.sub(r'\s+', ' ', filename).strip()
        filename = filename.strip('.')
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        return filename

    def _extract_article_id_from_url(self, url: str) -> str:
        """Extract a unique identifier from the URL"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        if path:
            # Use the last part of the path as ID
            return path.split('/')[-1]
        else:
            # Fallback to domain
            return parsed.netloc.replace('.', '_')

    def _extract_article_images(self, articles: List[Dict[str, Any]], output_dir: Path, delay: float = 1.0) -> Dict[str, List[str]]:
        """Extract and download images from articles"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'successful': [],
            'failed': [],
            'no_image': []
        }
        
        logger.info(f"Extracting images from {len(articles)} articles")
        
        for i, article in enumerate(articles, 1):
            url = article.get('url')
            title = article.get('title', 'Untitled')
            article_id = article.get('article_id', self._extract_article_id_from_url(url))
            
            if not url:
                logger.debug(f"Article {i}: No URL found")
                results['failed'].append(f"Article {i}: No URL")
                continue
            
            logger.debug(f"Processing article {i}/{len(articles)}: {title[:50]}...")
            
            # Extract image URL
            image_url = self._get_article_image(url)
            
            if not image_url:
                logger.debug(f"No image found for: {url}")
                results['no_image'].append(url)
                continue
            
            # Create filename
            safe_title = self._sanitize_filename(title)
            filename = f"{article_id}_{safe_title}"
            image_path = output_dir / filename
            
            # Download image
            if self._download_image(image_url, image_path):
                logger.debug(f"Downloaded: {image_path.name}")
                results['successful'].append(url)
                
                # Add image path to article metadata
                article['image_path'] = str(image_path.relative_to(project_root))
            else:
                logger.debug(f"Failed to download: {image_url}")
                results['failed'].append(url)
            
            # Add delay to be respectful to servers
            if delay > 0 and i < len(articles):
                time.sleep(delay)
        
        # Log summary
        successful_count = len(results['successful'])
        failed_count = len(results['failed'])
        no_image_count = len(results['no_image'])
        total_processed = successful_count + failed_count + no_image_count
        
        logger.info(f"Image extraction complete: {successful_count} successful, {failed_count} failed, {no_image_count} no image")
        
        return results


    
    def _distribute_articles_by_tpm(self, articles: List[Dict[str, Any]], agents: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Distribute articles across agents based on their TPM (tokens per minute) limits from swarm configuration."""
        if not articles or not agents:
            return {}
        
        agent_names = list(agents.keys())
        total_tpm = 0
        agent_tpm_map = {}
        
        for agent_name in agent_names:
            agent = agents[agent_name]
            agent_tpm = agent.tokens_per_minute
            agent_tpm_map[agent_name] = agent_tpm
            total_tpm += agent_tpm
        
        logger.info(f"Distributing {len(articles)} articles based on TPM capacity: total {total_tpm} TPM across {len(agent_names)} agents")
        
        # Distribute articles proportionally based on TPM capacity
        agent_article_assignments = {}
        start_idx = 0
        
        for i, agent_name in enumerate(agent_names):
            agent_tpm = agent_tpm_map[agent_name]
            proportion = agent_tpm / total_tpm
            
            if i == len(agent_names) - 1:
                # Last agent gets remaining articles
                num_articles_for_agent = len(articles) - start_idx
            else:
                num_articles_for_agent = max(1, round(len(articles) * proportion))
                num_articles_for_agent = min(num_articles_for_agent, len(articles) - start_idx)
            
            end_idx = start_idx + num_articles_for_agent
            agent_article_assignments[agent_name] = articles[start_idx:end_idx]
            
            logger.info(f"Agent {agent_name}: {agent_tpm} TPM ({proportion*100:.1f}%) -> {num_articles_for_agent} articles")
            start_idx = end_idx
        
        return agent_article_assignments

    def _get_agent_delays(self, agent_name: str) -> Tuple[float, float]:
        """Get delay and timeout settings for agent based on TPM limits from swarm configuration."""
        # Get TPM from swarm configuration
        agent_tpm = 6000  # Default fallback
        
        # Check bulk intelligence swarm first
        if agent_name in self.bulk_swarm_config.get('agents', {}):
            agent_config = self.bulk_swarm_config['agents'][agent_name]
            agent_tpm = agent_config.get('tokens_per_minute', 6000)
        # Check deep intelligence swarm
        elif agent_name in self.deep_intelligence_swarm_config.get('agents', {}):
            agent_config = self.deep_intelligence_swarm_config['agents'][agent_name]
            agent_tpm = agent_config.get('tokens_per_minute', 6000)
        else:
            logger.warning(f"Agent {agent_name} not found in swarm config, using default TPM for delays")
        
        # Fixed timeout logic: Higher capacity models get longer timeouts
        if 'llama-4-scout' in agent_name:
            return 1.5, 60.0  # 30K TPM → longest timeout for highest capacity
        elif agent_name == 'llama-3.3-70b-versatile':
            return 2.0, 45.0  # 12K TPM → medium timeout
        elif agent_name == 'qwen/qwen3-32b':
            return 1.0, 30.0  # 6K TPM → shorter timeout but still reasonable
        elif agent_tpm >= 15000:
            return 1.5, 45.0  # High capacity models get longer timeouts
        elif agent_tpm >= 12000:
            return 2.0, 40.0  # Medium capacity models
        else:
            return 2.0, 30.0  # Lower capacity models get shorter timeouts

    async def _process_deep_intelligence_agent(self, agent_name: str, assigned_articles: List[Dict[str, Any]], 
                                             agent_index: int, task_id, progress) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles with a deep intelligence agent using optimized batch processing."""
        if not assigned_articles:
            return []
        
        agent = self.deep_intelligence_agents[agent_name]
        _, batch_timeout = self._get_agent_delays(agent_name)
        
        try:
            async with agent:
                if agent_index > 0:
                    await asyncio.sleep(agent_index * 2.0)  # Reduced stagger delay for batch processing
                
                # Create progress callback for intermediate updates
                def progress_callback(completed_count: int, total_count: int, batch_info: str = ""):
                    """Update progress bar during internal batch processing."""
                    accepted_count = getattr(progress_callback, 'accepted_count', 0)
                    rejected_count = completed_count - accepted_count
                    
                    progress.update(task_id, completed=completed_count,
                                  description=f"[cyan]{agent_name} ({accepted_count}, {rejected_count}, {completed_count}/{total_count}) - {batch_info}")
                
                # Function to update acceptance counts
                def update_acceptance_count(accepted: int):
                    progress_callback.accepted_count = accepted
                
                # Use batch processing for dramatically improved efficiency (5x fewer API calls)
                batch_timeout_total = batch_timeout * len(assigned_articles) // 5 + 120  # Scale timeout for batch size
                
                try:
                    all_results = await asyncio.wait_for(
                        agent.process_articles_batch(assigned_articles, progress_callback, update_acceptance_count),
                        timeout=batch_timeout_total
                    )
                    
                    # Update progress tracking for batch processing
                    total_accepted = sum(1 for _, accepted, _ in all_results if accepted)
                    total_rejected = len(all_results) - total_accepted
                    
                    progress.update(task_id, completed=len(assigned_articles),
                                  description=f"[green]{agent_name} ✓ ({total_accepted}, {total_rejected}, {len(assigned_articles)}) - batch completed")
                    
                    return all_results
                    
                except RuntimeError as e:
                    if "falling back to individual processing" in str(e):
                        logger.warning(f"Batch processing failed for {agent_name}, falling back to individual processing")
                        
                        # Fallback to individual article processing
                        individual_results = []
                        for i, article in enumerate(assigned_articles):
                            try:
                                # Update progress for individual processing
                                progress.update(task_id, completed=i,
                                              description=f"[yellow]{agent_name} (individual {i+1}/{len(assigned_articles)}) - fallback mode")
                                
                                # Process individual article with shorter timeout
                                result = await asyncio.wait_for(
                                    agent.analyze_article(article),
                                    timeout=60.0  # 1 minute per article
                                )
                                individual_results.append(result)
                                
                                # Small delay between individual calls
                                await asyncio.sleep(1.0)
                                
                            except Exception as individual_error:
                                logger.error(f"Individual processing failed for article {i+1}: {individual_error}")
                                # Default to accept for individual failures
                                enriched_article = article.copy()
                                enriched_article["deep_intelligence_analysis"] = {
                                    "error": f"Individual processing failed: {type(individual_error).__name__}",
                                    "fallback_failure": True,
                                    "model": agent_name,
                                    "default_recommendation": "ACCEPT",
                                    "reason": "Pre-filtered article defaulted to accept on individual processing failure"
                                }
                                individual_results.append((enriched_article, True, 0.5))
                        
                        # Update final progress
                        total_accepted = sum(1 for _, accepted, _ in individual_results if accepted)
                        total_rejected = len(individual_results) - total_accepted
                        
                        progress.update(task_id, completed=len(assigned_articles),
                                      description=f"[yellow]{agent_name} ⚠ ({total_accepted}, {total_rejected}, {len(assigned_articles)}) - individual fallback")
                        
                        return individual_results
                    else:
                        raise  # Re-raise if it's a different RuntimeError
                    
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.warning(f"Batch processing timed out for {agent_name}, using default acceptance")
                error_type = "timeout - defaulted to accept"
                color = "yellow"
                symbol = "⚠"
            else:
                error_type = "timeout" if "timeout" in str(e).lower() else (
                    "rate_limit" if "rate limit" in str(e).lower() else f"error: {type(e).__name__}")
                color = "red"
                symbol = "✗"
            
            progress.update(task_id, completed=len(assigned_articles),
                          description=f"[{color}]{agent_name} {symbol} ({error_type}) - {len(assigned_articles)} articles")
            
            # Default to ACCEPT for pre-filtered articles on any failure
            results = []
            for article in assigned_articles:
                enriched_article = article.copy()
                enriched_article["deep_intelligence_analysis"] = {
                    "error": f"Agent processing error: {type(e).__name__}",
                    "agent_failure": True,
                    "agent": agent_name,
                    "default_recommendation": "ACCEPT",
                    "reason": "Pre-filtered article defaulted to accept on processing error",
                    "fact_verification": {
                        "fact_check_confidence": 0.7
                    },
                    "bias_detection": {
                        "bias_detection_score": 0.3  # Low bias (good)
                    },
                    "credibility_assessment": {
                        "credibility_score": 0.7
                    }
                }
                results.append((enriched_article, True, 0.6))  # Accept with reasonable confidence
            return results

    async def _process_articles_with_deep_intelligence(self, consensus_filtered_articles: List[Dict[str, Any]]) -> Dict[str, List[Tuple[Dict[str, Any], bool, float]]]:
        """Process consensus-filtered articles with deep intelligence agents."""
        if not self.enable_deep_intelligence or not consensus_filtered_articles:
            logger.info("Deep intelligence processing skipped - no articles or disabled")
            return {}
        
        logger.info(f"Processing {len(consensus_filtered_articles)} articles with {len(self.deep_intelligence_agents)} deep intelligence agents")
        agent_article_assignments = self._distribute_articles_by_tpm(consensus_filtered_articles, self.deep_intelligence_agents)
        
        with suppress_logging():
            with self._create_progress_bar(f"[magenta]Distributed Deep Intelligence Processing", len(self.deep_intelligence_agents), transient=False) as progress:
                agent_tasks = {}
                for agent_name in self.deep_intelligence_agents.keys():
                    assigned_count = len(agent_article_assignments[agent_name])
                    agent_tasks[agent_name] = progress.add_task(
                        f"[magenta]{agent_name} (0, 0, 0) - {assigned_count} articles", total=assigned_count)
                
                progress.refresh()
                
                tasks = []
                for i, (agent_name, assigned_articles) in enumerate(agent_article_assignments.items()):
                    task_id = agent_tasks[agent_name]
                    task = asyncio.create_task(self._process_deep_intelligence_agent(agent_name, assigned_articles, i, task_id, progress))
                    tasks.append((agent_name, task))
                
                await asyncio.sleep(0.1)
                
                # Agents now have their own optimized timeouts, so we run without orchestrator timeout
                # Each agent calculates its own realistic timeout based on model capacity
                try:
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                except Exception as e:
                    logger.error(f"Deep intelligence processing failed: {e}")
                    results = [Exception("Processing failed") for _ in tasks]
                
                deep_intelligence_results = {}
                for i, ((agent_name, _), result) in enumerate(zip(tasks, results)):
                    if isinstance(result, Exception):
                        assigned_articles = agent_article_assignments[agent_name]
                        deep_intelligence_results[agent_name] = [(article, False, 0.0) for article in assigned_articles]
                    else:
                        deep_intelligence_results[agent_name] = result
        
        total_articles_processed = sum(len(results) for results in deep_intelligence_results.values())
        total_accepted = sum(sum(1 for _, accepted, _ in results if accepted) for results in deep_intelligence_results.values())
        logger.info(f"Distributed deep intelligence complete: {total_accepted}/{total_articles_processed} articles processed")
        
        return deep_intelligence_results

    def _calculate_deep_intelligence_timeout(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate realistic timeout for deep intelligence processing based on article count and model capacities."""
        
        if not articles:
            return 600.0  # 10 minutes minimum
        
        num_articles = len(articles)
        
        # Get the slowest agent (determines overall timeout)
        # Based on token limits: Scout (30k) > Versatile (12k) > Qwen (6k)
        slowest_agent_time = 0.0
        
        for agent_name in self.deep_intelligence_agents.keys():
            if 'llama-4-scout' in agent_name.lower():
                # High capacity: ~3 batches/min, batch size 15
                articles_for_agent = num_articles // len(self.deep_intelligence_agents)
                batches_needed = (articles_for_agent + 14) // 15  # 15 per batch
                time_needed = batches_needed * 20.0  # 20 seconds per batch
            elif 'llama-3.3-70b-versatile' in agent_name.lower():
                # Medium capacity: ~1 batch/min, batch size 8
                articles_for_agent = num_articles // len(self.deep_intelligence_agents)
                batches_needed = (articles_for_agent + 7) // 8   # 8 per batch
                time_needed = batches_needed * 60.0  # 60 seconds per batch
            elif 'qwen' in agent_name.lower():
                # Low capacity: ~0.6 batches/min, batch size 3
                articles_for_agent = num_articles // len(self.deep_intelligence_agents)
                batches_needed = (articles_for_agent + 2) // 3   # 3 per batch
                time_needed = batches_needed * 100.0  # 100 seconds per batch
            else:
                # Default conservative estimate
                articles_for_agent = num_articles // len(self.deep_intelligence_agents)
                batches_needed = (articles_for_agent + 4) // 5   # 5 per batch
                time_needed = batches_needed * 45.0  # 45 seconds per batch
            
            slowest_agent_time = max(slowest_agent_time, time_needed)
        
        # Add 50% buffer and ensure reasonable bounds
        timeout_with_buffer = slowest_agent_time * 1.5
        realistic_timeout = max(900.0, min(timeout_with_buffer, 3600.0))  # 15 min to 60 min
        
        logger.info(f"Calculated deep intelligence timeout: {realistic_timeout/60:.1f} minutes for {num_articles} articles")
        
        return realistic_timeout
    
    async def process_news_pipeline(self, num_articles: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete news processing pipeline."""
        articles_count = num_articles or self.default_max_articles
        
        # Step 1: Collect articles with progress bar
        logger.info(f"Step 1: Collecting articles from sources (max: {articles_count})")
        
        # Initialize collection stats tracking
        collection_start_time = time.time()
        
        with self._create_progress_bar(f"[green]Collecting from {len(self.news_collector.sources)} sources", 
                                     len(self.news_collector.sources), transient=True) as collection_progress:
            collection_task = collection_progress.add_task(f"[green]Collecting articles", total=len(self.news_collector.sources))
            
            def update_collection_progress(completed_sources: int, total_sources_count: int):
                collection_progress.update(collection_task, completed=completed_sources,
                                         description=f"[green]Collecting articles ({completed_sources}/{total_sources_count} sources)")
            
            collected_articles = await self.news_collector.collect_all(
                max_articles=articles_count, progress_callback=update_collection_progress)
        
        if not collected_articles:
            raise ValueError("No articles collected. Check source configuration and connectivity.")
        
        collection_duration = time.time() - collection_start_time
        
        # Use collector's comprehensive stats instead of recalculating
        collector_stats = self.news_collector.stats
        
        # Create comprehensive collection stats using collector's data
        collection_stats = {
            'generated_at': self._get_current_timestamp(),
            'collection_stats': {
                'total_articles': len(collected_articles),
                'total_sources': len(self.news_collector.sources),
                'successful_sources': collector_stats.successful,
                'failed_sources': collector_stats.failed,
                'empty_sources': collector_stats.empty,
                'processing_time': round(collection_duration, 2),
                'success_rate': round((collector_stats.successful / len(self.news_collector.sources)) * 100, 1) if len(self.news_collector.sources) > 0 else 0,
                'failure_details': collector_stats.failure_reasons,
                'collection_config': {
                    'max_articles': articles_count,
                    'max_age_days': getattr(self.news_collector, 'max_age_days', 7),
                    'total_configured_sources': len(self.news_collector.sources)
                },
                # Add deduplication stats from collector
                'deduplication_stats': {
                    'original_articles': collector_stats.original_articles,
                    'duplicates_removed': collector_stats.duplicates_removed,
                    'deduplication_time': collector_stats.deduplication_time
                }
            }
        }
        
        logger.info(f"Collected {len(collected_articles)} articles from {collector_stats.successful}/{len(self.news_collector.sources)} sources")
        
        # Save collected articles
        collection_data = {
            'timestamp': self._get_current_timestamp(),
            'total_articles': len(collected_articles),
            'articles': collected_articles
        }
        self._save_json_file(collection_data, f"collected_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Step 2: Bulk intelligence scoring
        start_time = time.time()
        
        # Score articles using the bulk intelligence swarm with TPM-based distribution
        if not collected_articles:
            agent_results = {}
        else:
            logger.info(f"Starting bulk intelligence scoring: {len(collected_articles)} articles distributed across {len(self.agents)} agents")
            agent_article_assignments = self._distribute_articles_by_tpm(collected_articles, self.agents)
            
            with suppress_logging():
                with self._create_progress_bar(f"[yellow]Distributed Bulk Intelligence Processing", len(self.agents), transient=True) as progress:
                    agent_tasks = {}
                    for agent_name in self.agents.keys():
                        assigned_count = len(agent_article_assignments[agent_name])
                        agent_tasks[agent_name] = progress.add_task(
                            f"[cyan]{agent_name} (0, 0, 0) - {assigned_count} articles", total=assigned_count)
                    
                    progress.refresh()
                    
                    # Create async tasks for each agent - let each agent handle its own batch processing
                    async def process_agent_articles(agent_name: str, assigned_articles: List[Dict[str, Any]], 
                                                   agent_index: int, task_id) -> List[Tuple[Dict[str, Any], bool, float]]:
                        """Process articles with a bulk agent using the agent's optimized batch processing."""
                        if not assigned_articles:
                            return []
                        
                        agent = self.agents[agent_name]
                        try:
                            async with agent:
                                # Stagger agent starts to avoid simultaneous API calls
                                if agent_index > 0:
                                    await asyncio.sleep(agent_index * 1.0)
                                
                                # Process in batches for progress updates, but let agent handle rate limiting
                                all_results = []
                                processed_count = total_accepted = total_rejected = 0
                                effective_batch_size = agent.adaptive_batch_size
                                
                                for i in range(0, len(assigned_articles), effective_batch_size):
                                    batch = assigned_articles[i:i + effective_batch_size]
                                    # Agent handles all rate limiting internally in process_batch()
                                    batch_results = await agent.process_batch(batch)
                                    all_results.extend(batch_results)
                                    processed_count += len(batch)
                                    
                                    # Update progress per batch
                                    batch_accepted = sum(1 for _, accepted, _ in batch_results if accepted)
                                    total_accepted += batch_accepted
                                    total_rejected += len(batch_results) - batch_accepted
                                    
                                    progress.update(task_id, completed=processed_count,
                                                  description=f"[cyan]{agent_name} ({total_accepted}, {total_rejected}, {processed_count}) - {len(assigned_articles)} articles")
                                    
                                    # No manual delays - agent handles inter-batch timing in process_batch()
                                
                                # Final progress update
                                progress.update(task_id, completed=len(assigned_articles),
                                              description=f"[green]{agent_name} ✓ ({total_accepted}, {total_rejected}, {len(assigned_articles)}) - completed")
                                return all_results
                                
                        except Exception as e:
                            progress.update(task_id, completed=len(assigned_articles),
                                          description=f"[red]{agent_name} ✗ (error: {type(e).__name__}) - {len(assigned_articles)} articles")
                            return [(article, False, 0.1) for article in assigned_articles]
                    
                    # Create and run all agent tasks
                    tasks = []
                    for i, (agent_name, assigned_articles) in enumerate(agent_article_assignments.items()):
                        task_id = agent_tasks[agent_name]
                        task = asyncio.create_task(process_agent_articles(agent_name, assigned_articles, i, task_id))
                        tasks.append((agent_name, task))
                    
                    await asyncio.sleep(0.1)
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                    
                    # Process results
                    agent_results = {}
                    for i, ((agent_name, _), result) in enumerate(zip(tasks, results)):
                        if isinstance(result, Exception):
                            assigned_articles = agent_article_assignments[agent_name]
                            agent_results[agent_name] = [(article, False, 0.1) for article in assigned_articles]
                        else:
                            agent_results[agent_name] = result
            
            total_accepted = sum(sum(1 for _, accepted, _ in results if accepted) for results in agent_results.values())
            total_processed = sum(len(results) for results in agent_results.values())
            logger.info(f"Distributed bulk intelligence complete: {total_processed} articles processed, {total_accepted} accepted")
            
            self._save_bulk_agent_results(agent_results, collected_articles)
        
        # Calculate bulk stage stats
        bulk_stats = self._calculate_bulk_stage_stats(agent_results)
        
        # Step 3: Initial consensus
        consensus_results = self.consensus_engine.apply_consensus(agent_results)
        logger.info(f"Initial consensus: {len(consensus_results)} decisions processed")
        
        # Calculate consensus stage stats
        consensus_stats = self.consensus_engine.get_consensus_stats(agent_results, consensus_results)
        
        # Save early stats for frontend (in case pipeline is interrupted)
        self._save_collection_stats(collection_stats, bulk_stats, consensus_stats, {}, {})
        logger.info("📊 Pipeline stats saved (collection + bulk + consensus stages)")
        
        # Step 4: Filter by confidence for deep intelligence
        consensus_filtered_articles = self.consensus_engine.filter_by_confidence(
            consensus_results, optimize_for_deep_intelligence=True)
        logger.info(f"Consensus filtering: {len(consensus_filtered_articles)} articles selected for deep intelligence")
        
        # Step 5: Deep intelligence processing with realistic timeouts
        deep_intelligence_timeout = self._calculate_deep_intelligence_timeout(consensus_filtered_articles)
        
        try:
            deep_intelligence_results = await asyncio.wait_for(
                self._process_articles_with_deep_intelligence(consensus_filtered_articles),
                timeout=deep_intelligence_timeout)
        except asyncio.TimeoutError:
            logger.error(f"Deep intelligence processing timed out after {deep_intelligence_timeout/60:.1f} minutes")
            deep_intelligence_results = {}
        
        # Calculate deep intelligence stage stats
        deep_intelligence_stats = self._calculate_deep_intelligence_stats(deep_intelligence_results, consensus_filtered_articles)
        
        # Step 6: Final consensus
        final_consensus_results = self.final_consensus_engine.apply_final_consensus(
            [(article, True, article.get('consensus_confidence', 0.5)) for article in consensus_filtered_articles],
            deep_intelligence_results)
        
        accepted_in_final = sum(1 for _, accept, _ in final_consensus_results if accept)
        logger.info(f"Final consensus: {accepted_in_final} articles accepted")
        
        # Calculate final consensus stage stats
        final_consensus_stats = self.final_consensus_engine.get_final_consensus_stats(final_consensus_results)
        
        # Step 7: Extract final articles
        final_articles = [article for article, accept, _ in final_consensus_results if accept]
        
        # Step 8: Extract article images
        logger.info(f"Step 8: Extracting images from {len(final_articles)} articles")
        images_output_dir = project_root / "src" / "frontend" / "assets" / "images" / "articles"
        image_extraction_results = self._extract_article_images(final_articles, images_output_dir, delay=1.0)
        
        # Step 9: Post-classification into content types
        classified_content = self.classify_and_allocate_content(final_articles)
        
        processing_duration = time.time() - start_time
        logger.info(f"Pipeline complete: {len(final_articles)} articles processed in {processing_duration:.1f}s")
        
        # Log categorized content distribution
        categories = classified_content.get('categories', {})
        total_categorized = sum(len(articles) for articles in categories.values())
        category_summary = ', '.join([f"{len(articles)} {category}" for category, articles in categories.items()])
        logger.info(f"Categorized content: {total_categorized} articles ({category_summary})")
        
        # Step 10: Save comprehensive stats for frontend consumption
        self._save_collection_stats(collection_stats, bulk_stats, consensus_stats, deep_intelligence_stats, final_consensus_stats)
        
        # Step 11: Generate API files for frontend consumption
        pipeline_info = self.get_pipeline_info()
        api_saved = self._save_api_files(classified_content, pipeline_info, processing_duration)
        if api_saved:
            logger.info("✅ API files generated successfully for frontend deployment")
            self.generate_audio()
        else:
            log_warning(logger, "⚠️ Failed to generate API files - frontend may not update")
        
        return classified_content
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get essential pipeline information for monitoring."""
        return {
            'pipeline_version': '3.0_with_deep_intelligence',
            'components': ['collection', 'bulk_scoring', 'initial_consensus', 
                          'deep_intelligence' if self.enable_deep_intelligence else None, 'final_consensus', 'image_extraction'],
            'agents': {
                'bulk_agents': len(self.agents),
                'deep_intelligence_agents': len(self.deep_intelligence_agents) if self.enable_deep_intelligence else 0
            },
            'configuration': {
                'max_articles': self.default_max_articles,
                'deep_intelligence_enabled': self.enable_deep_intelligence
            }
        }
    
    def _save_bulk_agent_results(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]], 
                                original_articles: List[Dict[str, Any]]) -> None:
        """Save bulk agent stage results with processed articles and decisions."""
        summary_stats = {}
        processed_articles_by_agent = {}
        
        for agent_name, results in agent_results.items():
            accepted_count = sum(1 for _, accepted, _ in results if accepted)
            summary_stats[agent_name] = {
                'total_articles': len(results),
                'accepted_articles': accepted_count,
                'acceptance_rate': (accepted_count / len(results) * 100) if results else 0.0
            }
            
            # Save processed articles with decisions and scores
            processed_articles_by_agent[agent_name] = []
            for article, accepted, confidence in results:
                article_result = {
                    'article_id': self._get_article_id(article),
                    'title': article.get('title', '')[:100],  # Truncate for readability
                    'source': article.get('source', ''),
                    'category': article.get('category', ''),
                    'published_date': article.get('published_date', ''),
                    'decision': accepted,
                    'confidence': confidence,
                    'multi_dimensional_score': article.get('multi_dimensional_score', {}),
                    'url': article.get('url', '')
                }
                processed_articles_by_agent[agent_name].append(article_result)
        
        bulk_agent_output = {
            'timestamp': self._get_current_timestamp(),
            'stage': 'bulk_agent_processing',
            'processing_info': {
                'total_agents': len(agent_results),
                'total_articles_processed': len(original_articles),
                'total_decisions': sum(len(results) for results in agent_results.values()),
                'overall_acceptance_rate': (
                    sum(sum(1 for _, accepted, _ in results if accepted) for results in agent_results.values()) /
                    sum(len(results) for results in agent_results.values()) * 100
                ) if agent_results else 0.0
            },
            'summary_statistics': summary_stats,
            'processed_articles_by_agent': processed_articles_by_agent,
            'agent_processing_details': {
                agent_name: {
                    'articles_assigned': len(results),
                    'accepted': sum(1 for _, accepted, _ in results if accepted),
                    'rejected': sum(1 for _, accepted, _ in results if not accepted),
                    'avg_confidence': sum(confidence for _, _, confidence in results) / len(results) if results else 0.0,
                    'confidence_range': {
                        'min': min(confidence for _, _, confidence in results) if results else 0.0,
                        'max': max(confidence for _, _, confidence in results) if results else 0.0
                    }
                }
                for agent_name, results in agent_results.items()
            }
        }
        
        filename = f"bulk_agent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_json_file(bulk_agent_output, filename)

    def _is_recent_article(self, published_date: str, hours: int = 48) -> bool:
        """Check if article was published within the specified hours."""
        try:
            if not published_date:
                return False
            
            # Parse the published date
            pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            time_diff = current_time - pub_date
            
            return time_diff.total_seconds() <= (hours * 3600)
        except Exception:
            return False

    def _is_headline_candidate(self, article: Dict[str, Any]) -> bool:
        """Identify articles suitable as headlines."""
        # Time-based: Recent articles (last 48 hours)
        pub_date = article.get('published_date', '')
        if not self._is_recent_article(pub_date, hours=48):
            return False
        
        # High impact + quality scores
        scores = article.get('consensus_multi_dimensional_score', {})
        impact_score = scores.get('impact_score', 0.0)
        quality_score = scores.get('quality_score', 0.0)
        overall_score = scores.get('overall_score', 0.0)
        
        # Deep intelligence score
        deep_intel_score = article.get('deep_intelligence_score', 0.0)
        
        # Major news sources (broader list)
        source = article.get('source', '').lower()
        major_sources = ['techcrunch', 'reuters', 'bloomberg', 'wsj', 'bbc', 'cnn', 'verge', 'wired', 'ars_technica']
        is_major_source = any(major in source for major in major_sources)
        
        # Breaking news indicators in title
        title = article.get('title', '').lower()
        breaking_keywords = ['breaking', 'announces', 'launches', 'acquires', 'releases', 'unveils']
        has_breaking_keywords = any(keyword in title for keyword in breaking_keywords)
        
        # Composite headline score
        headline_score = (impact_score * 0.4 + quality_score * 0.3 + 
                         overall_score * 0.2 + deep_intel_score * 0.1)
        
        return (headline_score > 0.65 and 
                (is_major_source or has_breaking_keywords) and
                impact_score > 0.6)

    def _is_scientific_paper(self, article: Dict[str, Any]) -> bool:
        """
        Identify genuine scientific research papers vs news about research.
        Phase 1: Enhanced detection with multi-tier validation.
        """
        source = article.get('source', '').lower()
        title = article.get('title', '').lower()
        category = article.get('category', '').lower()
        url = article.get('url', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        author = article.get('author', '').lower()
        
        # Tier 1: Explicit scientific publication sources (highest confidence)
        scientific_sources = [
            'arxiv', 'nature', 'science', 'jair', 'distill', 'acm_ai_news',
            'nature_machine_learning', 'science_ai', 'distill_pub'
        ]
        
        # Tier 2: Academic domains and URLs (high confidence)
        academic_urls = [
            'arxiv.org', 'nature.com', 'science.org', 'jair.org', 'distill.pub',
            'acm.org', 'ieee.org', 'aaai.org', 'papers.nips.cc', 'aclweb.org'
        ]
        
        # Tier 3: Scientific paper structure indicators
        paper_structure_indicators = [
            'abstract:', 'doi:', 'methodology', 'experimental', 'conclusion',
            'references', 'bibliography', 'volume', 'issue', 'pages'
        ]
        
        # Tier 4: Academic writing patterns
        academic_patterns = [
            'et al.', 'university', 'laboratory', 'institute', 'department',
            'faculty', 'professor', 'phd', 'ph.d.'
        ]
        
        # EXCLUSIONS: Filter out news/blog content about research
        news_exclusions = [
            'announces', 'launches', 'releases', 'introduces', 'unveils',
            'reports', 'says', 'claims', 'according to', 'breaking',
            'techcrunch', 'blog', 'news', 'press release', 'company'
        ]
        
        # Corporate research blogs (not academic papers)
        corporate_sources = [
            'google_research_blog', 'deepmind_research', 'amazon_science',
            'google_ai_blog', 'microsoft_ai_blog', 'openai_blog', 'marktechpost'
        ]
        
        # Government announcements (not research papers)
        government_sources = [
            'nih_ai_news', 'nist_ai_news', 'darpa_ai_research'
        ]
        
        # Industry/media sources (not academic)
        industry_media_sources = [
            'techcrunch_ai', 'venturebeat', 'wired', 'the_verge'
        ]
        
        # Apply exclusions first
        if (source in corporate_sources or 
            source in government_sources or 
            source in industry_media_sources or
            any(exclusion in title or exclusion in description for exclusion in news_exclusions)):
            return False
        
        # Check for explicit scientific sources
        if source in scientific_sources or category.lower() == 'research':
            return self._validate_paper_structure(article)
        
        # Check for academic URLs
        if any(academic_url in url for academic_url in academic_urls):
            return self._validate_paper_structure(article)
        
        # Check for scientific paper structure
        structure_score = sum(1 for indicator in paper_structure_indicators 
                            if indicator in description or indicator in content)
        
        # Check for academic patterns
        academic_score = sum(1 for pattern in academic_patterns 
                           if pattern in author or pattern in description)
        
        # Require strong evidence for classification as research paper
        # This prevents news articles about research from being classified as papers
        return (structure_score >= 2 and academic_score >= 1) or structure_score >= 3
    
    def _validate_paper_structure(self, article: Dict[str, Any]) -> bool:
        """
        Phase 3: Validate academic paper structure and quality.
        Enhanced validation for articles from research sources.
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        url = article.get('url', '').lower()
        author = article.get('author', '').lower()
        
        # DOI validation (strong indicator)
        has_doi = ('doi:' in description or 'doi:' in content or 
                  'doi.org/' in url or '/doi/' in url)
        
        # Journal/conference metadata
        has_publication_metadata = any(indicator in description or indicator in content for indicator in [
            'volume', 'issue', 'pages', 'published in', 'journal of', 
            'proceedings of', 'conference on', 'symposium on'
        ])
        
        # Academic structure
        has_academic_structure = any(structure in description or structure in content for structure in [
            'abstract', 'introduction', 'methodology', 'results', 
            'conclusion', 'references', 'bibliography'
        ])
        
        # Author credentials (multiple authors, institutional affiliations)
        has_academic_authors = (
            'et al.' in author or 
            len(author.split(',')) > 1 or  # Multiple authors
            any(institution in author for institution in ['university', 'institute', 'laboratory'])
        )
        
        # Exclude obvious news articles even from research sources
        news_indicators = [
            'announces', 'launches', 'reports', 'according to', 'breaking',
            'company', 'startup', 'funding', 'investment', 'ipo'
        ]
        
        has_news_indicators = any(indicator in title or indicator in description 
                                for indicator in news_indicators)
        
        if has_news_indicators:
            return False
        
        # Scoring system for paper validation
        score = 0
        score += 3 if has_doi else 0
        score += 2 if has_publication_metadata else 0
        score += 2 if has_academic_structure else 0
        score += 1 if has_academic_authors else 0
        
        # Require minimum score for validation
        return score >= 3
    
    def _get_research_quality_score(self, article: Dict[str, Any]) -> float:
        """
        Calculate research paper quality score for ranking.
        Phase 3: Quality validation component.
        """
        if not self._is_scientific_paper(article):
            return 0.0
        
        source = article.get('source', '').lower()
        url = article.get('url', '').lower()
        description = article.get('description', '').lower()
        
        # Tier-based scoring
        score = 0.5  # Base score for validated research paper
        
        # Top-tier journals and conferences
        if any(top_source in source for top_source in ['nature', 'science', 'arxiv']):
            score += 0.3
        elif any(good_source in source for good_source in ['jair', 'acm', 'ieee']):
            score += 0.2
        
        # Quality indicators
        if 'doi:' in description or 'doi.org/' in url:
            score += 0.1
        
        if any(quality_indicator in description for quality_indicator in [
            'peer-reviewed', 'impact factor', 'cited by'
        ]):
            score += 0.1
        
        return min(score, 1.0)

    def _is_research_paper(self, article: Dict[str, Any]) -> bool:
        """
        Legacy method wrapper for backward compatibility.
        Phase 4: Updated to use new scientific paper detection.
        """
        return self._is_scientific_paper(article)

    def _calculate_composite_score(self, article: Dict[str, Any]) -> float:
        """Calculate composite quality score for ranking articles."""
        consensus = article.get('consensus_multi_dimensional_score', {})
        deep_intel = article.get('deep_intelligence_score', 0.5)
        deep_confidence = article.get('deep_intelligence_confidence', 0.5)
        
        # Weighted combination of scores
        consensus_score = consensus.get('overall_score', 0.5)
        quality_score = consensus.get('quality_score', 0.5)
        relevance_score = consensus.get('relevance_score', 0.5)
        
        return (consensus_score * 0.4 + 
                deep_intel * 0.3 + 
                quality_score * 0.2 + 
                relevance_score * 0.1)

    def _classify_articles(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify articles into headlines, regular articles, and research papers.
        Phase 4: Enhanced with scientific paper validation and logging.
        """
        headlines = []
        regular_articles = []
        research_papers = []
        
        # Counters for classification tracking
        classification_stats = {
            'total': len(articles),
            'headlines': 0,
            'research_papers': 0,
            'regular_articles': 0,
            'research_rejected': 0,  # Articles that looked like research but were rejected
        }
        
        for article in articles:
            source = article.get('source', '')
            title = article.get('title', '')
            
            if self._is_headline_candidate(article):
                headlines.append(article)
                classification_stats['headlines'] += 1
                logger.debug(f"Classified as HEADLINE: {title[:60]}... (source: {source})")
            
            elif self._is_scientific_paper(article):
                research_papers.append(article)
                classification_stats['research_papers'] += 1
                
                # Log research paper details for monitoring
                quality_score = self._get_research_quality_score(article)
                logger.info(f"Classified as RESEARCH (quality: {quality_score:.2f}): {title[:60]}... (source: {source})")
                
                # Add quality score to article for ranking
                article['research_quality_score'] = quality_score
                
            else:
                regular_articles.append(article)
                classification_stats['regular_articles'] += 1
                
                # Check if this looked like research but was rejected
                source_lower = source.lower()
                title_lower = title.lower()
                if ('research' in source_lower or 'research' in title_lower or 
                    article.get('category', '').lower() == 'research'):
                    classification_stats['research_rejected'] += 1
                    logger.debug(f"Research-like but REJECTED: {title[:60]}... (source: {source})")
        
        # Log classification summary
        logger.info(f"Article classification complete:")
        logger.info(f"  Headlines: {classification_stats['headlines']}")
        logger.info(f"  Research papers: {classification_stats['research_papers']}")
        logger.info(f"  Regular articles: {classification_stats['regular_articles']}")
        logger.info(f"  Research-like rejected: {classification_stats['research_rejected']}")
        
        return headlines, regular_articles, research_papers

    def _select_best_articles(self, articles: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """
        Select the best articles based on composite scoring.
        Phase 3: Enhanced with research quality scoring.
        """
        if not articles:
            return []
        
        def get_article_score(article):
            """Calculate score with research quality enhancement."""
            base_score = self._calculate_composite_score(article)
            
            # Enhance research papers with quality scoring
            if self._is_scientific_paper(article):
                research_quality = self._get_research_quality_score(article)
                # Boost research papers with high quality scores
                return base_score * 0.7 + research_quality * 0.3
            
            return base_score
        
        # Sort by enhanced composite score (highest first)
        sorted_articles = sorted(articles, key=get_article_score, reverse=True)
        return sorted_articles[:count]

    def _categorize_articles_by_content(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize articles by content category using probability distributions.
        Returns a dictionary with categories as keys and lists of articles as values.
        """
        categories = {
            'ai': [],
            'entertainment': [],
            'sports': [],
            'health': []
        }
        
        for article in articles:
            # Get category probabilities from different sources
            category_probs = self._get_article_category_probabilities(article)
            
            # Assign article to category with highest probability
            best_category = max(category_probs.items(), key=lambda x: x[1])[0]
            
            # Add category information to article
            article['assigned_category'] = best_category
            article['category_probabilities'] = category_probs
            article['category_confidence'] = category_probs[best_category]
            
            categories[best_category].append(article)
        
        # Log categorization results
        for category, article_list in categories.items():
            if article_list:
                avg_confidence = sum(a.get('category_confidence', 0) for a in article_list) / len(article_list)
                logger.info(f"Category {category.upper()}: {len(article_list)} articles (avg confidence: {avg_confidence:.2f})")
            else:
                logger.warning(f"Category {category.upper()}: No articles assigned")
        
        return categories

    def _get_article_category_probabilities(self, article: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract and combine category probabilities from different agent analyses.
        """
        # Default probabilities (AI-focused pipeline)
        default_probs = {'ai': 0.7, 'entertainment': 0.1, 'sports': 0.1, 'health': 0.1}
        
        # Get probabilities from bulk agent (multi-dimensional score)
        bulk_probs = None
        md_score = article.get('multi_dimensional_score', {})
        if isinstance(md_score, dict) and 'category_probabilities' in md_score:
            bulk_probs = md_score['category_probabilities']
        
        # Get probabilities from consensus engine
        consensus_probs = None
        consensus_score = article.get('consensus_multi_dimensional_score', {})
        if isinstance(consensus_score, dict) and 'category_probabilities' in consensus_score:
            consensus_probs = consensus_score['category_probabilities']
        
        # Get probabilities from deep intelligence agent
        deep_probs = None
        deep_analysis = article.get('deep_intelligence_analysis', {})
        if isinstance(deep_analysis, dict):
            synthesis = deep_analysis.get('synthesis', {})
            if isinstance(synthesis, dict) and 'category_probabilities' in synthesis:
                deep_probs = synthesis['category_probabilities']
        
        # Combine probabilities with weighted average
        # Deep intelligence gets highest weight, then consensus, then bulk
        combined_probs = default_probs.copy()
        
        weights = []
        prob_sources = []
        
        if bulk_probs and isinstance(bulk_probs, dict):
            weights.append(0.3)
            prob_sources.append(bulk_probs)
        
        if consensus_probs and isinstance(consensus_probs, dict):
            weights.append(0.4)
            prob_sources.append(consensus_probs)
        
        if deep_probs and isinstance(deep_probs, dict):
            weights.append(0.6)
            prob_sources.append(deep_probs)
        
        if prob_sources:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            for category in combined_probs.keys():
                weighted_sum = 0.0
                for i, probs in enumerate(prob_sources):
                    prob_value = probs.get(category, 0.25)  # Default if category missing
                    weighted_sum += prob_value * normalized_weights[i]
                combined_probs[category] = weighted_sum
            
            # Normalize to sum to 1.0
            total = sum(combined_probs.values())
            if total > 0:
                for category in combined_probs:
                    combined_probs[category] = combined_probs[category] / total
        
        return combined_probs

    def _normalize_articles_for_category(self, articles: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
        """
        Normalize articles for a specific category with category-specific metadata.
        """
        normalized_articles = []
        
        for article in articles:
            # Start with base normalization
            normalized_article = self._normalize_single_article(article)
            
            # Add category-specific metadata
            normalized_article['category'] = category.upper()
            normalized_article['assigned_category'] = category
            normalized_article['category_confidence'] = article.get('category_confidence', 0.5)
            normalized_article['category_probabilities'] = article.get('category_probabilities', {})
            
            normalized_articles.append(normalized_article)
        
        return normalized_articles

    def _normalize_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a single article for frontend compatibility.
        """
        # Create a copy to avoid modifying original
        normalized_article = article.copy()
        
        # Ensure article_id exists
        if not normalized_article.get('article_id') and normalized_article.get('id'):
            normalized_article['article_id'] = normalized_article['id']
        elif not normalized_article.get('article_id'):
            # Generate a simple ID from title/URL hash if missing
            import hashlib
            title = normalized_article.get('title', '')
            url = normalized_article.get('url', '')
            id_source = f"{title}_{url}"
            normalized_article['article_id'] = hashlib.md5(id_source.encode()).hexdigest()[:8]
        
        # Ensure description field exists
        if not normalized_article.get('description'):
            content = normalized_article.get('content', '')
            if content:
                normalized_article['description'] = content[:500] + ('...' if len(content) > 500 else '')
            else:
                normalized_article['description'] = normalized_article.get('title', 'No description available')
        
        # Ensure category field exists (will be overridden in _normalize_articles_for_category)
        if not normalized_article.get('category'):
            normalized_article['category'] = normalized_article.get('assigned_category', 'AI').upper()
        
        # Ensure author field exists
        if not normalized_article.get('author'):
            normalized_article['author'] = ''
        
        return normalized_article

    def classify_and_allocate_content(self, final_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Post-process final articles into category-based structure.
        Target: Best articles for each category (AI, Entertainment, Sports, Health)
        """
        # Step 1: Categorize articles by content category
        categorized_articles = self._categorize_articles_by_content(final_articles)
        
        # Step 2: Select best articles for each category
        category_results = {}
        total_selected = 0
        
        for category, articles in categorized_articles.items():
            # Select top articles for this category (target ~8-12 per category)
            target_count = min(12, max(5, len(articles) // 2))  # Adaptive based on available articles
            selected = self._select_best_articles(articles, target_count)
            category_results[category] = selected
            total_selected += len(selected)
            
            logger.info(f"Category {category.upper()}: {len(articles)} candidates → {len(selected)} selected")
        
        # Create category distribution stats
        category_distribution = {
            category: len(articles) for category, articles in category_results.items()
        }
        
        logger.info(f"Total categorized content: {total_selected} articles across {len(category_results)} categories")
        logger.info(f"Category distribution: {category_distribution}")
        
        return {
            'categories': category_results,
            'classification_metadata': {
                'total_processed': len(final_articles),
                'total_selected': total_selected,
                'category_distribution': category_distribution,
                'categories_available': list(categorized_articles.keys()),
                'processing_method': 'category_based'
            }
        }

    def _save_api_files(self, classified_content: Dict[str, Any], pipeline_info: Dict[str, Any], processing_time: float) -> bool:
        """Save category-based API files for frontend consumption."""
        try:
            project_root = Path(__file__).parent.parent.parent
            
            # Create API directories
            backend_api_dir = project_root / "src" / "backend" / "api"
            frontend_api_dir = project_root / "src" / "frontend" / "api"
            backend_api_dir.mkdir(parents=True, exist_ok=True)
            frontend_api_dir.mkdir(parents=True, exist_ok=True)
            
            # Create categories subdirectory
            categories_dir = frontend_api_dir / "categories"
            categories_dir.mkdir(exist_ok=True)
            
            # Get categorized articles
            categories = classified_content.get('categories', {})
            
            # Save individual category files
            saved_files = []
            for category, articles in categories.items():
                if articles:  # Only create files for categories with articles
                    # Normalize articles for this category
                    normalized_articles = self._normalize_articles_for_category(articles, category)
                    
                    # Create category API response
                    category_response = {
                        'generated_at': self._get_current_timestamp(),
                        'category': category,
                        'articles': normalized_articles,
                        'count': len(normalized_articles),
                        'pipeline_info': {
                            'version': pipeline_info.get('pipeline_version', '4.0_category_based'),
                            'processing_time': processing_time,
                            'category_focus': category
                        }
                    }
                    
                    # Save category file
                    category_file = categories_dir / f"{category}.json"
                    with open(category_file, 'w', encoding='utf-8') as f:
                        json.dump(category_response, f, indent=2, ensure_ascii=False)
                    saved_files.append(category_file)
                    
                    logger.info(f"✅ Saved {category} category: {len(normalized_articles)} articles")
            
            # Extract all articles for main API (aggregate all categories)
            all_articles = []
            for articles in categories.values():
                all_articles.extend(articles)
            
            # **FIX**: Normalize articles for frontend compatibility
            normalized_articles = self._normalize_articles_for_frontend(all_articles, classified_content)
            
            # Create main API response (aggregate view)
            api_response = {
                'generated_at': self._get_current_timestamp(),
                'articles': normalized_articles,
                'count': len(normalized_articles),
                'pipeline_info': {
                    'version': pipeline_info.get('pipeline_version', '4.0_category_based'),
                    'processing_time': processing_time,
                    'components': pipeline_info.get('components', []),
                    'agents': pipeline_info.get('agents', {}),
                    'category_breakdown': classified_content.get('classification_metadata', {}).get('category_distribution', {}),
                    'classification_metadata': classified_content.get('classification_metadata', {}),
                    'available_categories': list(categories.keys()),
                    'processing_method': 'category_based'
                }
            }
            
            # Save latest.json to both directories
            latest_file_backend = backend_api_dir / "latest.json"
            latest_file_frontend = frontend_api_dir / "latest.json"
            
            with open(latest_file_backend, 'w', encoding='utf-8') as f:
                json.dump(api_response, f, indent=2, ensure_ascii=False)
            
            with open(latest_file_frontend, 'w', encoding='utf-8') as f:
                json.dump(api_response, f, indent=2, ensure_ascii=False)
            
            # Create widget.json with top stories
            widget_data = {
                'updated': self._get_current_timestamp(),
                'top_stories': normalized_articles[:5],  # Top 5 articles for widget
                'total_count': len(normalized_articles),
                'pipeline_version': pipeline_info.get('pipeline_version', '3.0_with_deep_intelligence')
            }
            
            widget_file_frontend = frontend_api_dir / "widget.json"
            with open(widget_file_frontend, 'w', encoding='utf-8') as f:
                json.dump(widget_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ API files saved:")
            logger.info(f"   - Backend: {latest_file_backend}")
            logger.info(f"   - Frontend: {latest_file_frontend}")
            logger.info(f"   - Widget: {widget_file_frontend}")
            
            return True
            
        except Exception as e:
            log_error(logger, f"Failed to save API files: {e}")
            return False

    def _normalize_articles_for_frontend(self, articles: List[Dict[str, Any]], classified_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize articles for frontend compatibility with category-based structure.
        """
        normalized_articles = []
        
        for article in articles:
            # Use the single article normalization method
            normalized_article = self._normalize_single_article(article)
            
            # Add category metadata if available
            if article.get('assigned_category'):
                normalized_article['assigned_category'] = article['assigned_category']
                normalized_article['category_confidence'] = article.get('category_confidence', 0.5)
                normalized_article['category_probabilities'] = article.get('category_probabilities', {})
            
            normalized_articles.append(normalized_article)
        
        logger.info(f"✅ Normalized {len(normalized_articles)} articles for frontend compatibility")
        return normalized_articles

    def generate_audio(self):
        """Generate podcast audio using Gemini 2.5 TTS with correct format."""
        # Check if GEMINI_API_KEY is available
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.info("ℹ️ GEMINI_API_KEY not available - skipping audio generation")
            logger.info("   💡 For local development: Set GEMINI_API_KEY in .env.local")
            logger.info("   🚀 Audio generation works automatically in GitHub Actions")
            return
        
        def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
            """Create a proper WAV file with correct headers."""
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(pcm)
        
        try:
            # Read the latest articles
            with open('src/backend/api/latest.json', 'r') as f:
                data = json.load(f)
            
            # Extract content from articles, limiting to avoid token limits
            articles = data.get('articles', [])
            if not articles:
                logger.warning("No articles found for audio generation")
                return
            
            # Create rich content for TTS (use top 5 articles with full content for 2.5 Pro's 250k TPM)
            story_summaries = []
            for i, article in enumerate(articles[:5]):
                title = article.get('title', '')
                description = article.get('description', article.get('content', ''))[:500]  # More content for richer script
                source = article.get('source', '')
                category = article.get('category', '')
                story_summaries.append(f"Story {i+1}: {title}\nSource: {source} | Category: {category}\nSummary: {description}")
            
            content_text = '\n\n'.join(story_summaries)
            
            # Create Gemini client using API key only
            client = genai.Client(api_key=gemini_api_key)
            
            # Generate podcast script (optimized for TTS model's 10k TPM limit)
            script_prompt = f"""Create a concise 2-3 minute news podcast script with two speakers alternating. 
            Format as dialogue between Jane and Joe alternating speakers. Keep it engaging but brief.
            
            Structure:
            Jane: Welcome to today's AI news update. I'm Jane with the latest developments.
            Joe: And I'm Joe. Let's dive into today's top stories.
            
            Cover 3 key stories, alternating speakers:
            - Each speaker gets 1-2 sentences per story
            - Keep explanations clear and concise
            - Make transitions smooth
            
            End with:
            Jane: That's today's AI update.
            Joe: Thanks for listening. See you next time.
            
            Keep the total script under 200 words to fit TTS model's 10k TPM limit.
            Focus on the most important stories and key insights.
            
            Content to cover:
            {content_text}"""
            
            script_response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=script_prompt
            )
            
            if not script_response or not hasattr(script_response, 'text'):
                raise ValueError("Invalid script response from Gemini")
                
            script = script_response.text
            logger.info("✅ Podcast script generated successfully")
            logger.debug(f"Generated script preview: {script[:1000]}...")
            
            # Format the script for TTS with correct prompt format
            tts_prompt = f"""TTS the following conversation between Jane and Joe:
{script}"""
            
            # Generate TTS audio from script using correct format
            tts_response = client.models.generate_content(
                model='gemini-2.5-flash-preview-tts',
                contents=tts_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(
                                    speaker='Jane',
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Kore'
                                        )
                                    )
                                ),
                                types.SpeakerVoiceConfig(
                                    speaker='Joe',
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name='Puck'
                                        )
                                    )
                                ),
                            ]
                        )
                    )
                )
            )
            
            # Extract PCM data from response
            if not tts_response or not tts_response.candidates:
                raise ValueError("No TTS response received")
            
            candidate = tts_response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("No audio content in TTS response")
            
            audio_part = candidate.content.parts[0]
            if not hasattr(audio_part, 'inline_data') or not audio_part.inline_data.data:
                raise ValueError("No inline audio data in TTS response")
            
            # Get the PCM data (base64 decode the response data)
            pcm_data = audio_part.inline_data.data
            
            # Ensure audio directory exists
            audio_dir = Path('src/frontend/assets/audio')
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as proper WAV file using wave module
            audio_file = audio_dir / 'latest-podcast.wav'
            wave_file(str(audio_file), pcm_data)
            
            # Verify file was created and has reasonable size
            if not audio_file.exists():
                raise ValueError("Audio file was not created")
            
            file_size = audio_file.stat().st_size
            if file_size < 1000:
                raise ValueError(f"Generated audio file too small: {file_size} bytes")
            
            logger.info(f"✅ Podcast audio generated successfully: {audio_file} ({file_size:,} bytes)")
            
        except Exception as e:
            log_error(logger, f"Failed to generate podcast audio: {e}")
            logger.debug(f"TTS generation error details: {traceback.format_exc()}")

    def _save_collection_stats(self, collection_stats: Dict[str, Any], bulk_stats: Optional[Dict[str, Any]] = None, consensus_stats: Optional[Dict[str, Any]] = None, deep_intelligence_stats: Optional[Dict[str, Any]] = None, final_consensus_stats: Optional[Dict[str, Any]] = None) -> None:
        """Save comprehensive pipeline statistics to a JSON file."""
        try:
            project_root = Path(__file__).parent.parent.parent
            
            # Create the comprehensive stats structure
            comprehensive_stats = {
                'generated_at': collection_stats['generated_at'],
                'collection_stats': {
                    'total_articles': collection_stats['collection_stats']['total_articles'],
                    'total_sources': collection_stats['collection_stats']['total_sources'],
                    'successful_sources': collection_stats['collection_stats']['successful_sources'],
                    'failed_sources': collection_stats['collection_stats']['failed_sources'],
                    'empty_sources': collection_stats['collection_stats']['empty_sources'],
                    'processing_time': collection_stats['collection_stats']['processing_time'],
                    'success_rate': collection_stats['collection_stats']['success_rate'],
                    'category_distribution': collection_stats['collection_stats']['category_distribution'],
                    'failure_details': collection_stats['collection_stats'].get('failure_details', {}),
                    'collection_config': collection_stats['collection_stats']['collection_config']
                }
            }
            
            # Add bulk stage stats if available
            if bulk_stats:
                comprehensive_stats['bulk_stage_stats'] = bulk_stats
            
            # Add consensus stage stats if available
            if consensus_stats:
                comprehensive_stats['consensus_stage_stats'] = consensus_stats
            
            # Add deep intelligence stage stats if available
            if deep_intelligence_stats:
                comprehensive_stats['deep_intelligence_stage_stats'] = deep_intelligence_stats
            
            # Add final consensus stage stats if available
            if final_consensus_stats:
                comprehensive_stats['final_consensus_stage_stats'] = final_consensus_stats
            
            # Save to test_output for archival
            output_dir = project_root / "scripts" / "test_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"comprehensive_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
            
            # Save to frontend API as stats.json for immediate access
            frontend_api_dir = project_root / "src" / "frontend" / "api"
            frontend_api_dir.mkdir(parents=True, exist_ok=True)
            with open(frontend_api_dir / "stats.json", 'w', encoding='utf-8') as f:
                json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comprehensive statistics saved to: {output_dir / filename} and {frontend_api_dir / 'stats.json'}")
        except Exception as e:
            log_error(logger, f"Failed to save comprehensive statistics: {e}")

    def _calculate_bulk_stage_stats(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> Dict[str, Any]:
        """Calculate comprehensive bulk stage statistics."""
        if not agent_results:
            return {}
        
        total_articles = sum(len(results) for results in agent_results.values())
        total_accepted = sum(sum(1 for _, accepted, _ in results if accepted) for results in agent_results.values())
        
        agent_stats = {}
        for agent_name, results in agent_results.items():
            accepted_count = sum(1 for _, accepted, _ in results if accepted)
            agent_stats[agent_name] = {
                'total_processed': len(results),
                'accepted': accepted_count,
                'acceptance_rate': round((accepted_count / len(results) * 100), 1) if results else 0.0
            }
        
        return {
            'total_agents': len(agent_results),
            'total_articles_processed': total_articles,
            'total_accepted': total_accepted,
            'overall_acceptance_rate': round((total_accepted / total_articles * 100), 1) if total_articles > 0 else 0.0,
            'agents': agent_stats
        }

    def _calculate_deep_intelligence_stats(self, deep_intelligence_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]], input_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive deep intelligence stage statistics."""
        if not deep_intelligence_results:
            return {
                'enabled': self.enable_deep_intelligence,
                'total_agents': len(self.deep_intelligence_agents) if self.enable_deep_intelligence else 0,
                'total_articles_processed': 0,
                'total_accepted': 0,
                'overall_acceptance_rate': 0.0,
                'agents': {}
            }
        
        total_articles = sum(len(results) for results in deep_intelligence_results.values())
        total_accepted = sum(sum(1 for _, accepted, _ in results if accepted) for results in deep_intelligence_results.values())
        
        agent_stats = {}
        for agent_name, results in deep_intelligence_results.items():
            accepted_count = sum(1 for _, accepted, _ in results if accepted)
            confidences = [conf for _, _, conf in results if conf > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            agent_stats[agent_name] = {
                'total_processed': len(results),
                'accepted': accepted_count,
                'acceptance_rate': round((accepted_count / len(results) * 100), 1) if results else 0.0,
                'average_confidence': round(avg_confidence, 2)
            }
        
        return {
            'enabled': self.enable_deep_intelligence,
            'total_agents': len(deep_intelligence_results),
            'input_articles': len(input_articles),
            'total_articles_processed': total_articles,
            'total_accepted': total_accepted,
            'overall_acceptance_rate': round((total_accepted / total_articles * 100), 1) if total_articles > 0 else 0.0,
            'coverage_rate': round((total_articles / len(input_articles) * 100), 1) if input_articles else 0.0,
            'agents': agent_stats
        }


async def main():
    """Main function to run the news processing pipeline from command line."""
    from src.shared.utils.logging_config import setup_logging, create_progress_logger, log_step, log_error
    
    setup_logging(level="DEBUG", quiet_mode=False, show_progress=True)
    logger = create_progress_logger(__name__)
    
    try:
        log_step(logger, "Initializing News Processing Pipeline")
        pipeline = NewsProcessingPipeline()
        
        pipeline_info = pipeline.get_pipeline_info()
        source_count = len(pipeline.news_collector.sources)
        
        logger.info(f"Sources: {source_count}")
        
        log_step(logger, "Starting Pipeline Processing")
        start_time = time.time()
        classified_content = await pipeline.process_news_pipeline()
        duration = time.time() - start_time
        
        await _save_and_display_results(pipeline, classified_content, duration, logger)
        return 0
        
    except KeyboardInterrupt:
        log_error(logger, "Pipeline interrupted by user")
        return 1
    except Exception as e:
        log_error(logger, f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


async def _save_and_display_results(pipeline, classified_content, duration, logger):
    """Save results and display summary."""
    logger.info(f"\n{'='*50}")
    logger.info("PIPELINE RESULTS")
    logger.info('='*50)
    
    # Extract counts from classified content
    headline_count = len(classified_content.get('headline', []))
    articles_count = len(classified_content.get('articles', []))
    research_count = len(classified_content.get('research_papers', []))
    total_count = headline_count + articles_count + research_count
    
    logger.info(f"✓ Content classified: {headline_count} headline, {articles_count} articles, {research_count} research papers")
    logger.info(f"✓ Total content: {total_count} items")
    logger.info(f"✓ Processing time: {duration:.1f}s")
    
    # Save results
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'pipeline_info': pipeline.get_pipeline_info(),
        'results_summary': {
            'content_breakdown': {
                'headline': headline_count,
                'articles': articles_count,
                'research_papers': research_count,
                'total': total_count
            },
            'processing_time_seconds': duration,
            'timestamp': NewsProcessingPipeline._get_current_timestamp()
        },
        'classified_content': classified_content
    }
    
    output_file = output_dir / f"orchestrator_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_file}")
    
    # Show sample content from each category
    logger.info(f"\n📰 Sample Content:")
    
    # Show headline
    if classified_content.get('headline'):
        headline = classified_content['headline'][0]
        logger.info(f"  HEADLINE: {headline['title']}")
        logger.info(f"            Source: {headline.get('source', 'Unknown')} | Category: {headline.get('category', 'Uncategorized')}")
    
    # Show sample articles
    if classified_content.get('articles'):
        logger.info(f"\n  ARTICLES (showing 3 of {articles_count}):")
        for i, article in enumerate(classified_content['articles'][:3]):
            logger.info(f"    {i+1}. {article['title']}")
            source = article.get('source', 'Unknown source')
            category = article.get('category', 'Uncategorized')
            logger.info(f"       Source: {source} | Category: {category}")
    
    # Show sample research papers
    if classified_content.get('research_papers'):
        logger.info(f"\n  RESEARCH PAPERS (showing 3 of {research_count}):")
        for i, paper in enumerate(classified_content['research_papers'][:3]):
            logger.info(f"    {i+1}. {paper['title']}")
            source = paper.get('source', 'Unknown source')
            category = paper.get('category', 'Uncategorized')
            logger.info(f"       Source: {source} | Category: {category}")


def run():
    """Entry point for running the orchestrator."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
