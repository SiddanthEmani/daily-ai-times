#!/usr/bin/env python3
"""
News Collection - Streamlined source collectors and parsers
"""

import asyncio
import aiohttp
import feedparser
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

from .core import DateUtils, TextUtils

logger = logging.getLogger(__name__)

class ArticleParser:
    """Streamlined RSS article parser."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parser with simple defaults."""
        self.config = config or {}
    
    def parse_article(self, source_id: str, config: Dict, entry: Any, max_age_days: int) -> Optional[Dict]:
        """Parse article from RSS entry with essential fields."""
        try:
            # Extract core fields
            title = getattr(entry, 'title', '').strip()
            url = getattr(entry, 'link', '').strip()
            
            if not (title and url):
                return None
            
            # Clean description
            description = TextUtils.clean_html(
                getattr(entry, 'summary', '') or getattr(entry, 'description', ''),
                config=self.config
            )
            
            # Parse date
            pub_date = DateUtils.parse_date(
                getattr(entry, 'published_parsed', None) or 
                getattr(entry, 'updated_parsed', None)
            )
            
            # Apply age filter
            if not DateUtils.is_recent(pub_date, max_age_days):
                return None
            
            # Generate ID
            item_id = getattr(entry, 'id', '') or getattr(entry, 'guid', '') or url
            article_id = hashlib.md5(f"{source_id}_{item_id}".encode()).hexdigest()
            
            return {
                'id': article_id,
                'source_id': source_id,
                'source': config.get('name', source_id),
                'category': config.get('category', 'Other'),
                'title': title,
                'url': url,
                'description': description,
                'published_date': pub_date,
                'collected_at': datetime.now(timezone.utc).isoformat(),
                'author': getattr(entry, 'author', ''),
                'source_priority': config.get('priority', 5)
            }
        except Exception as e:
            logger.error(f"Parse error for {source_id}: {e}")
            return None

class SourceCollector:
    """Streamlined RSS source collector."""
    
    def __init__(self, session: aiohttp.ClientSession, parser: ArticleParser):
        self.session = session
        self.parser = parser
    
    async def collect_source(self, source_id: str, config: Dict, max_age_days: int) -> Tuple[List[Dict], str]:
        """Collect articles from RSS source."""
        try:
            url = config['url']
            max_items = config.get('maxArticles', 20)
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return [], f"HTTP {response.status}"
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                if not hasattr(feed, 'entries') or not feed.entries:
                    return [], "no entries found"
                
                articles = []
                for entry in feed.entries[:max_items]:
                    article = self.parser.parse_article(source_id, config, entry, max_age_days)
                    if article:
                        articles.append(article)
                
                return articles, "success" if articles else f"all articles too old (>{max_age_days}d)"
                
        except asyncio.TimeoutError:
            return [], "timeout"
        except aiohttp.ClientError as e:
            return [], f"connection error: {e}"
        except KeyError as e:
            return [], f"missing config: {e}"
        except Exception as e:
            return [], f"error: {str(e)[:50]}"

class BatchCollector:
    """Parallel batch processing with progress tracking."""
    
    def __init__(self, session: aiohttp.ClientSession, config: Optional[Dict[str, Any]] = None, progress_callback=None):
        self.session = session
        self.config = config or {}
        self.progress_callback = progress_callback
        self.batch_size = self.config.get('batch_processing', {}).get('default_batch_size', 10)
        self.pause_between_batches = self.config.get('batch_processing', {}).get('pause_between_batches_seconds', 0.5)
        
        self.parser = ArticleParser(config)
        self.collector = SourceCollector(session, self.parser)
        self.stats = {'successful': 0, 'empty': 0, 'failed': 0, 'reasons': {}}
    
    async def collect_all(self, sources: Dict[str, Dict], max_age_days: int) -> List[Dict]:
        """Process all RSS sources in parallel batches."""
        all_articles = []
        source_items = list(sources.items())
        total_sources = len(source_items)
        completed_sources = 0
        
        # Initial progress update
        if self.progress_callback:
            self.progress_callback(completed_sources, total_sources)
        
        for i in range(0, len(source_items), self.batch_size):
            batch = source_items[i:i + self.batch_size]
            batch_articles = await self._process_batch(batch, max_age_days)
            all_articles.extend(batch_articles)
            
            # Update progress
            completed_sources += len(batch)
            if self.progress_callback:
                self.progress_callback(completed_sources, total_sources)
            
            # Pause between batches
            if i + self.batch_size < len(source_items) and self.pause_between_batches > 0:
                await asyncio.sleep(self.pause_between_batches)
        
        return all_articles
    
    async def _process_batch(self, batch: List[Tuple[str, Dict]], max_age_days: int) -> List[Dict]:
        """Process a batch of sources concurrently."""
        tasks = [
            self.collector.collect_source(source_id, config, max_age_days) 
            for source_id, config in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_articles = []
        
        for (source_id, _), result in zip(batch, results):
            if isinstance(result, Exception):
                self.stats['failed'] += 1
                self.stats['reasons'][source_id] = str(result)
            elif isinstance(result, tuple) and len(result) == 2:
                articles, reason = result
                if articles:
                    batch_articles.extend(articles)
                    self.stats['successful'] += 1
                else:
                    self.stats['empty'] += 1
                    self.stats['reasons'][source_id] = reason
            else:
                self.stats['failed'] += 1
                self.stats['reasons'][source_id] = "invalid response format"
        
        return batch_articles
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'successful_sources': self.stats['successful'],
            'empty_sources': self.stats['empty'],
            'failed_sources': self.stats['failed'],
            'failure_reasons': self.stats['reasons']
        }

class ArticleProcessor:
    """Streamlined article processing with deduplication."""
    
    def __init__(self, config):
        self.config = config
    
    def process(self, articles: List[Dict]) -> List[Dict]:
        """Process articles through deduplication and basic quality filtering."""
        if not articles:
            return articles
        
        # Deduplicate
        from ..processors.deduplication_utils import ArticleDeduplicator
        deduplicator = ArticleDeduplicator()
        unique_articles = deduplicator.deduplicate_articles(articles, 'enhanced')
        logger.info(f"Deduplication: {len(articles)} → {len(unique_articles)}")
        
        # Basic quality filter
        filtered_articles = [
            a for a in unique_articles 
            if all(a.get(field) for field in ['id', 'title', 'url', 'description'])
            and len(a.get('title', '')) >= self.config.min_title_length
            and len(a.get('description', '')) >= self.config.min_description_length
            and TextUtils.is_valid_url(a.get('url', ''))
        ]
        if len(filtered_articles) != len(unique_articles):
            removed = len(unique_articles) - len(filtered_articles)
            logger.info(f"Quality filter: removed {removed} articles")
        
        return filtered_articles
    



class NewsCollector:
    """Main news collection orchestrator."""
    
    def __init__(self):
        """Initialize collector with configuration and sources."""
        from .core import ConfigManager, CollectionStats
        self.config, self.sources = ConfigManager.load()
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = CollectionStats()
        logger.info(f"NewsCollector initialized with {len(self.sources)} sources")
    
    def get_sources_by_categories(self, categories: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get sources filtered by categories using the SourcesLoader.
        
        Args:
            categories: List of category names (e.g., ['government', 'industry'])
            
        Returns:
            Dictionary of sources that match the categories
        """
        try:
            # Import sources loader
            import sys
            from pathlib import Path
            
            # Add src directory to path if not already there
            src_dir = Path(__file__).parent.parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
                
            from shared.config.sources_loader import get_sources_loader
            
            loader = get_sources_loader()
            
            # Get available categories and normalize case
            available_categories = loader.get_categories()
            normalized_available = {cat.lower(): cat for cat in available_categories}
            
            # Find matching categories (case-insensitive)
            matched_categories = []
            for requested_cat in categories:
                normalized_requested = requested_cat.lower().strip()
                if normalized_requested in normalized_available:
                    matched_categories.append(normalized_available[normalized_requested])
                else:
                    logger.warning(f"Category '{requested_cat}' not found. Available: {available_categories}")
            
            if not matched_categories:
                logger.error(f"No valid categories found from: {categories}")
                logger.info(f"Available categories: {available_categories}")
                return {}
            
            # Collect sources from matching categories using existing loader methods
            filtered_sources = {}
            for category in matched_categories:
                category_sources = loader.get_sources_by_category(category)
                filtered_sources.update(category_sources)
                logger.info(f"Category '{category}': {len(category_sources)} sources loaded")
            
            logger.info(f"Total {len(filtered_sources)} sources selected from categories: {matched_categories}")
            return filtered_sources
            
        except Exception as e:
            logger.error(f"Failed to load sources by categories: {e}")
            return {}
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available source categories.
        
        Returns:
            List of available category names
        """
        try:
            # Import sources loader
            import sys
            from pathlib import Path
            
            # Add src directory to path if not already there
            src_dir = Path(__file__).parent.parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
                
            from shared.config.sources_loader import get_sources_loader
            
            loader = get_sources_loader()
            return loader.get_categories()
            
        except Exception as e:
            logger.error(f"Failed to get available categories: {e}")
            return []
    
    async def collect_all_with_progress(self, 
                                      source_ids: Optional[List[str]] = None, 
                                      max_age_days: Optional[int] = None, 
                                      max_articles: Optional[int] = None,
                                      progress_callback=None) -> List[Dict]:
        """
        Collect articles from news sources with progress tracking.
        
        Args:
            source_ids: Specific source IDs to collect from (None = all sources)
            max_age_days: Maximum age of articles in days (None = use config default)
            max_articles: Maximum number of articles to return (None = no limit)
            progress_callback: Callable that receives (completed_sources, total_sources)
            
        Returns:
            List of article dictionaries
        """
        start_time = time.time()
        
        # Prepare parameters
        sources_to_use = self.sources if source_ids is None else {k: v for k, v in self.sources.items() if k in source_ids}
        effective_max_age = max_age_days or self.config.max_age_days
        
        # Initialize session
        await self._init_session()
        
        try:
            # Collect articles
            batch_collector = BatchCollector(self.session, self.config.rss_config, progress_callback)
            articles = await batch_collector.collect_all(sources_to_use, effective_max_age)
            
            # Apply early limit before processing
            if max_articles and len(articles) > max_articles:
                articles = self._prioritize_articles(articles)[:max_articles]
                logger.info(f"Applied early limit: {len(articles)} articles")
            
            # Update stats
            batch_stats = batch_collector.get_stats()
            self.stats.successful = batch_stats['successful_sources']
            self.stats.empty = batch_stats['empty_sources']
            self.stats.failed = batch_stats['failed_sources']
            self.stats.failure_reasons = batch_stats['failure_reasons']
            
            # Process articles
            processor = ArticleProcessor(self.config)
            processed_articles = processor.process(articles)
            
            # Apply final limit
            if max_articles and len(processed_articles) > max_articles:
                processed_articles = self._prioritize_articles(processed_articles)[:max_articles]
            
            # Final stats
            self.stats.total_articles = len(processed_articles)
            self.stats.processing_time = time.time() - start_time
            
            self._log_summary()
            return processed_articles
            
        finally:
            await self._cleanup_session()
    
    async def _init_session(self):
        """Initialize HTTP session."""
        if self.session is not None:
            return
        
        connector = aiohttp.TCPConnector(
            limit=50, limit_per_host=20, ttl_dns_cache=300,
            use_dns_cache=True, keepalive_timeout=30
        )
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'NewsXP.ai/3.0 RSS Reader (+https://newsxp.ai)'}
        )
    
    async def _cleanup_session(self):
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    

    def _prioritize_articles(self, articles: List[Dict]) -> List[Dict]:
        """Sort articles by priority and recency."""
        def calculate_score(article):
            priority = article.get('source_priority', 5)
            
            try:
                pub_date = datetime.fromisoformat(article.get('published_date', ''))
                hours_old = (datetime.now(timezone.utc) - pub_date).total_seconds() / 3600
                recency_bonus = max(0, 24 - hours_old) / 24
            except (ValueError, TypeError):
                recency_bonus = 0
                
            return priority + recency_bonus
        
        return sorted(articles, key=calculate_score, reverse=True)
    
    def _log_summary(self):
        """Log collection summary."""
        s = self.stats
        logger.info(f"Collection: ✓{s.successful} ○{s.empty} ✗{s.failed} "
                   f"→ {s.total_articles} articles ({s.processing_time:.2f}s)")
        
        if s.failure_reasons:
            logger.warning(f"Failed sources: {len(s.failure_reasons)}")


