#!/usr/bin/env python3
"""
News Processing Pipeline Orchestrator for NewsXP.ai
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

try:
    from .processors.bulk_agent import BulkFilteringAgent
    from .processors.consensus_engine import ConsensusEngine
    from .processors.deep_intelligence_agent import DeepIntelligenceAgent
    from .processors.final_consensus_engine import FinalConsensusEngine
    from .collectors.collectors import NewsCollector
    from ..shared.config.config_loader import ConfigLoader, get_swarm_config
    from ..shared.utils.logging_config import log_warning, log_error, log_step
    from google import genai
    from google.genai import types
    import base64
    import wave
except ImportError:
    from backend.processors.bulk_agent import BulkFilteringAgent
    from backend.processors.consensus_engine import ConsensusEngine
    from backend.processors.deep_intelligence_agent import DeepIntelligenceAgent
    from backend.processors.final_consensus_engine import FinalConsensusEngine
    from backend.collectors.collectors import NewsCollector
    from shared.config.config_loader import ConfigLoader, get_swarm_config
    from shared.utils.logging_config import log_warning, log_error, log_step
    from google import genai
    from google.genai import types
    import base64
    import wave

logger = logging.getLogger(__name__)

# Constants for model TPM limits and processing delays
MODEL_TPM_LIMITS = {
    'meta-llama/llama-4-scout-17b-16e-instruct': 30000,
    'meta-llama/llama-4-maverick-17b-128e-instruct': 6000,
    'llama-3.3-70b-versatile': 12000,
    'llama3-70b-8192': 6000,
    'qwen/qwen3-32b': 6000,
    'qwen-qwq-32b': 6000,
    'gemma2-9b-it': 15000,
    'llama-3.1-8b-instant': 6000,
    'llama3-8b-8192': 6000
}

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
    """Orchestrates the complete news processing pipeline."""
    
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
            refresh_per_second=2, disable=False, expand=False
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
            'consensus_method': 'weighted_combination', 'enable_quality_gates': True
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
        
        logger.info(f"Initialized {len(self.agents)} bulk agents, {len(self.deep_intelligence_agents)} deep intelligence agents")
    


    async def _process_bulk_agent(self, agent_name: str, assigned_articles: List[Dict[str, Any]], 
                                 agent_index: int, task_id, progress) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Process articles with a bulk agent."""
        if not assigned_articles:
            return []
        
        agent = self.agents[agent_name]
        try:
            async with agent:
                if agent_index > 0:
                    await asyncio.sleep(agent_index * 1.0)
                
                all_results = []
                processed_count = total_accepted = total_rejected = 0
                effective_batch_size = agent.adaptive_batch_size
                
                for i in range(0, len(assigned_articles), effective_batch_size):
                    batch = assigned_articles[i:i + effective_batch_size]
                    batch_results = await agent.process_batch(batch)
                    all_results.extend(batch_results)
                    processed_count += len(batch)
                    
                    batch_accepted = sum(1 for _, accepted, _ in batch_results if accepted)
                    total_accepted += batch_accepted
                    total_rejected += len(batch_results) - batch_accepted
                    
                    progress.update(task_id, completed=processed_count,
                                  description=f"[cyan]{agent_name} ({total_accepted}, {total_rejected}, {processed_count}) - {len(assigned_articles)} articles")
                    
                    if i + effective_batch_size < len(assigned_articles):
                        base_delay = 3.0 if agent.model_name == 'gemma2-9b-it' else (
                            2.0 if agent.tokens_per_minute_limit <= 6000 else 2.5)
                        await asyncio.sleep(base_delay + random.uniform(0, base_delay * 0.1))
                
                progress.update(task_id, completed=len(assigned_articles),
                              description=f"[green]{agent_name} ✓ ({total_accepted}, {total_rejected}, {len(assigned_articles)}) - completed")
                return all_results
                
        except Exception as e:
            progress.update(task_id, completed=len(assigned_articles),
                          description=f"[red]{agent_name} ✗ (error: {type(e).__name__}) - {len(assigned_articles)} articles")
            return [(article, False, 0.1) for article in assigned_articles]

    async def _score_articles_with_swarm(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Tuple[Dict[str, Any], bool, float]]]:
        """Score articles using the bulk intelligence swarm with distributed processing."""
        if not articles:
            return {}
        
        logger.info(f"Starting bulk intelligence scoring: {len(articles)} articles distributed across {len(self.agents)} agents")
        agent_article_assignments = self._distribute_articles_by_tpm(articles, self.agents)
        
        with suppress_logging():
            with self._create_progress_bar(f"[yellow]Distributed Bulk Intelligence Processing", len(self.agents), transient=True) as progress:
                agent_tasks = {}
                for agent_name in self.agents.keys():
                    assigned_count = len(agent_article_assignments[agent_name])
                    agent_tasks[agent_name] = progress.add_task(
                        f"[cyan]{agent_name} (0, 0, 0) - {assigned_count} articles", total=assigned_count)
                
                progress.refresh()
                
                tasks = []
                for i, (agent_name, assigned_articles) in enumerate(agent_article_assignments.items()):
                    task_id = agent_tasks[agent_name]
                    task = asyncio.create_task(self._process_bulk_agent(agent_name, assigned_articles, i, task_id, progress))
                    tasks.append((agent_name, task))
                
                await asyncio.sleep(0.1)
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                agent_results = {}
                for i, ((agent_name, _), result) in enumerate(zip(tasks, results)):
                    if isinstance(result, Exception):
                        assigned_articles = agent_article_assignments[agent_name]
                        agent_results[agent_name] = [(article, False, 0.1) for article in assigned_articles]
                    else:
                        agent_results[agent_name] = result
        
        total_accepted = sum(sum(1 for _, accepted, _ in results if accepted) for results in agent_results.values())
        total_processed = sum(len(results) for results in agent_results.values())
        logger.info(f"Distributed bulk intelligence complete: {total_accepted}/{total_processed} articles processed")
        
        self._save_bulk_agent_results(agent_results, articles)
        return agent_results
    
    def _distribute_articles_by_tpm(self, articles: List[Dict[str, Any]], agents: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Distribute articles across agents based on their TPM (tokens per minute) limits."""
        if not articles or not agents:
            return {}
        
        agent_names = list(agents.keys())
        
        # Calculate total TPM capacity across all agents
        total_tpm = 0
        agent_tpm_map = {}
        
        for agent_name in agent_names:
            # For bulk agents, get TPM from agent object
            if hasattr(agents[agent_name], 'tokens_per_minute_limit'):
                agent_tpm = agents[agent_name].tokens_per_minute_limit
            else:
                # Fallback to MODEL_TPM_LIMITS for deep intelligence or unknown agents
                agent_tpm = MODEL_TPM_LIMITS.get(agent_name, 6000)
            
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
        """Get delay and timeout settings for agent based on TPM limits."""
        agent_tpm = MODEL_TPM_LIMITS.get(agent_name, 6000)
        
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
        
        # Step 1: Collect articles
        logger.info(f"Step 1: Collecting articles from sources (max: {articles_count})")
        
        with self._create_progress_bar(f"[green]Collecting from {len(self.news_collector.sources)} sources", 
                                     len(self.news_collector.sources), transient=True) as collection_progress:
            collection_task = collection_progress.add_task(f"[green]Collecting articles", total=len(self.news_collector.sources))
            
            def update_collection_progress(completed_sources: int, total_sources_count: int):
                collection_progress.update(collection_task, completed=completed_sources,
                                         description=f"[green]Collecting articles ({completed_sources}/{total_sources_count} sources)")
            
            collected_articles = await self.news_collector.collect_all_with_progress(
                max_articles=articles_count, progress_callback=update_collection_progress)
            
        if not collected_articles:
            raise ValueError("No articles collected. Check source configuration and connectivity.")
        
        logger.info(f"Collected {len(collected_articles)} articles")
        
        # Save collected articles
        collection_data = {
            'timestamp': self._get_current_timestamp(),
            'total_articles': len(collected_articles),
            'articles': collected_articles
        }
        self._save_json_file(collection_data, f"collected_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Step 2: Bulk intelligence scoring
        start_time = time.time()
        agent_results = await self._score_articles_with_swarm(collected_articles)
        
        # Step 3: Initial consensus
        consensus_results = self.consensus_engine.apply_consensus(agent_results)
        logger.info(f"Initial consensus: {len(consensus_results)} decisions processed")
        
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
        
        # Step 6: Final consensus
        final_consensus_results = self.final_consensus_engine.apply_final_consensus(
            [(article, True, article.get('consensus_confidence', 0.5)) for article in consensus_filtered_articles],
            deep_intelligence_results)
        
        accepted_in_final = sum(1 for _, accept, _ in final_consensus_results if accept)
        logger.info(f"Final consensus: {accepted_in_final} articles accepted")
        
        # Step 7: Extract final articles
        final_articles = [article for article, accept, _ in final_consensus_results if accept]
        
        # Step 8: Post-classification into content types
        classified_content = self.classify_and_allocate_content(final_articles)
        
        processing_duration = time.time() - start_time
        logger.info(f"Pipeline complete: {len(final_articles)} articles processed in {processing_duration:.1f}s")
        logger.info(f"Classified content: {len(classified_content['headline'])} headline, "
                   f"{len(classified_content['articles'])} articles, "
                   f"{len(classified_content['research_papers'])} research papers")
        
        # Step 9: Generate API files for frontend consumption
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
                          'deep_intelligence' if self.enable_deep_intelligence else None, 'final_consensus'],
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
        """Save bulk agent stage results."""
        summary_stats = {}
        for agent_name, results in agent_results.items():
            accepted_count = sum(1 for _, accepted, _ in results if accepted)
            summary_stats[agent_name] = {
                'total_articles': len(results),
                'accepted_articles': accepted_count,
                'acceptance_rate': (accepted_count / len(results) * 100) if results else 0.0
            }
        
        bulk_agent_output = {
            'timestamp': self._get_current_timestamp(),
            'stage': 'bulk_agent_processing',
            'processing_info': {
                'total_agents': len(agent_results),
                'total_articles_processed': len(original_articles)
            },
            'summary_statistics': summary_stats
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

    def _is_research_paper(self, article: Dict[str, Any]) -> bool:
        """Identify research papers and academic content."""
        source = article.get('source', '').lower()
        title = article.get('title', '').lower()
        category = article.get('category', '').lower()
        url = article.get('url', '').lower()
        
        # Research source indicators
        research_sources = ['arxiv', 'research', 'academic', 'scholar', 'mit', 'stanford', 'university']
        research_keywords = ['study', 'research', 'paper', 'analysis', 'methodology', 'experiment', 'survey']
        research_urls = ['arxiv.org', 'research.', 'papers.', '.edu/', 'scholar.']
        
        # Check if it's from a research source
        is_research_source = (any(rs in source for rs in research_sources) or
                             category == 'research' or
                             any(ru in url for ru in research_urls))
        
        # Check for research keywords in title
        has_research_keywords = any(kw in title for kw in research_keywords)
        
        return is_research_source or has_research_keywords

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
        """Classify articles into headlines, regular articles, and research papers."""
        headlines = []
        regular_articles = []
        research_papers = []
        
        for article in articles:
            if self._is_headline_candidate(article):
                headlines.append(article)
            elif self._is_research_paper(article):
                research_papers.append(article)
            else:
                regular_articles.append(article)
        
        return headlines, regular_articles, research_papers

    def _select_best_articles(self, articles: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """Select the best articles based on composite scoring."""
        if not articles:
            return []
        
        # Sort by composite score (highest first)
        sorted_articles = sorted(articles, key=self._calculate_composite_score, reverse=True)
        return sorted_articles[:count]

    def classify_and_allocate_content(self, final_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Post-process final articles into structured content types.
        Target: 1 headline + 14 articles + 10 research papers
        """
        # Step 1: Classify articles by type
        headlines, regular_articles, research_papers = self._classify_articles(final_articles)
        
        # Step 2: Select best of each type
        selected_headline = self._select_best_articles(headlines, 1)
        selected_articles = self._select_best_articles(regular_articles, 14)
        selected_research = self._select_best_articles(research_papers, 10)
        
        logger.info(f"Content classification: {len(headlines)} headline candidates → {len(selected_headline)} selected")
        logger.info(f"Content classification: {len(regular_articles)} article candidates → {len(selected_articles)} selected")
        logger.info(f"Content classification: {len(research_papers)} research candidates → {len(selected_research)} selected")
        
        return {
            'headline': selected_headline,
            'articles': selected_articles,
            'research_papers': selected_research,
            'classification_metadata': {
                'total_processed': len(final_articles),
                'candidates': {
                    'headlines': len(headlines),
                    'articles': len(regular_articles),
                    'research_papers': len(research_papers)
                },
                'selected': {
                    'headlines': len(selected_headline),
                    'articles': len(selected_articles),
                    'research_papers': len(selected_research)
                }
            }
        }

    def _save_api_files(self, classified_content: Dict[str, Any], pipeline_info: Dict[str, Any], processing_time: float) -> bool:
        """Save API files for frontend consumption in required directories."""
        try:
            project_root = Path(__file__).parent.parent.parent
            
            # Create API directories
            backend_api_dir = project_root / "src" / "backend" / "api"
            frontend_api_dir = project_root / "src" / "frontend" / "api"
            backend_api_dir.mkdir(parents=True, exist_ok=True)
            frontend_api_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract all articles for main API
            all_articles = []
            all_articles.extend(classified_content.get('headline', []))
            all_articles.extend(classified_content.get('articles', []))
            all_articles.extend(classified_content.get('research_papers', []))
            
            # **FIX**: Normalize articles for frontend compatibility
            normalized_articles = self._normalize_articles_for_frontend(all_articles, classified_content)
            
            # Create main API response
            api_response = {
                'generated_at': self._get_current_timestamp(),
                'articles': normalized_articles,
                'count': len(normalized_articles),
                'pipeline_info': {
                    'version': pipeline_info.get('pipeline_version', '3.0_with_deep_intelligence'),
                    'processing_time': processing_time,
                    'components': pipeline_info.get('components', []),
                    'agents': pipeline_info.get('agents', {}),
                    'content_breakdown': {
                        'headline': len(classified_content.get('headline', [])),
                        'articles': len(classified_content.get('articles', [])),
                        'research_papers': len(classified_content.get('research_papers', []))
                    },
                    'classification_metadata': classified_content.get('classification_metadata', {})
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
        Normalize articles for frontend compatibility.
        Ensures all required fields exist and are in the expected format.
        **FIX**: Now adds article type metadata to preserve classification.
        """
        normalized_articles = []
        
        # Create lookup for article types based on original classification
        headline_articles = {article.get('article_id', ''): 'headline' for article in classified_content.get('headline', [])}
        regular_articles = {article.get('article_id', ''): 'article' for article in classified_content.get('articles', [])}
        research_articles = {article.get('article_id', ''): 'research' for article in classified_content.get('research_papers', [])}
        
        # Combine all article type mappings
        article_type_map = {**headline_articles, **regular_articles, **research_articles}
        
        for article in articles:
            # Create a copy to avoid modifying original
            normalized_article = article.copy()
            
            # **FIX**: Ensure required frontend fields exist
            
            # 1. Ensure article_id exists (use 'id' as fallback, generate if missing)
            if not normalized_article.get('article_id') and normalized_article.get('id'):
                normalized_article['article_id'] = normalized_article['id']
            elif not normalized_article.get('article_id'):
                # Generate a simple ID from title/URL hash if missing
                import hashlib
                title = normalized_article.get('title', '')
                url = normalized_article.get('url', '')
                id_source = f"{title}_{url}"
                normalized_article['article_id'] = hashlib.md5(id_source.encode()).hexdigest()[:8]
            
            # **FIX**: Add article type metadata for frontend classification
            article_id = normalized_article.get('article_id', '')
            if article_id in article_type_map:
                normalized_article['article_type'] = article_type_map[article_id]
            else:
                # Fallback: try to determine type from category
                category = normalized_article.get('category', '').lower()
                if category == 'research':
                    normalized_article['article_type'] = 'research'
                else:
                    normalized_article['article_type'] = 'article'  # Default to regular article
            
            # 2. Ensure description field exists (critical for frontend rendering)
            if not normalized_article.get('description'):
                # Use content field as fallback
                content = normalized_article.get('content', '')
                if content:
                    # Truncate content to reasonable description length
                    normalized_article['description'] = content[:500] + ('...' if len(content) > 500 else '')
                else:
                    normalized_article['description'] = normalized_article.get('title', 'No description available')
            
            # 3. Ensure category field exists (critical for article categorization)
            if not normalized_article.get('category'):
                # Try to infer category from source or default to Media
                source = normalized_article.get('source', '').lower()
                if 'arxiv' in source or 'research' in source:
                    normalized_article['category'] = 'Research'
                else:
                    normalized_article['category'] = 'Media'
            
            # 4. Ensure author field exists (optional but expected for research articles)
            if not normalized_article.get('author'):
                normalized_article['author'] = ''
            
            # 5. Validate and log missing critical fields
            required_fields = ['title', 'url', 'source', 'published_date', 'description', 'category']
            missing_fields = [field for field in required_fields if not normalized_article.get(field)]
            
            if missing_fields:
                logger.warning(f"Article missing required fields {missing_fields}: {normalized_article.get('title', 'Unknown')[:50]}")
                # Continue processing but log the issue
            
            normalized_articles.append(normalized_article)
        
        logger.info(f"✅ Normalized {len(normalized_articles)} articles for frontend compatibility with type metadata")
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


async def main():
    """Main function to run the news processing pipeline from command line."""
    try:
        from ..shared.utils.logging_config import setup_logging, create_progress_logger, log_step, log_error
    except ImportError:
        from shared.utils.logging_config import setup_logging, create_progress_logger, log_step, log_error
    
    setup_logging(level="DEBUG", quiet_mode=False, show_progress=True)
    logger = create_progress_logger(__name__)
    
    try:
        log_step(logger, "Initializing News Processing Pipeline")
        pipeline = NewsProcessingPipeline()
        
        pipeline_info = pipeline.get_pipeline_info()
        source_count = len(pipeline.news_collector.sources)
        total_agents = pipeline_info['agents']['bulk_agents'] + pipeline_info['agents']['deep_intelligence_agents']
        
        logger.info(f"Pipeline ready: {total_agents} agents, {source_count} sources")
        
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
