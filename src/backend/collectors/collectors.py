#!/usr/bin/env python3
"""Ultra-Fast News Collection - Optimized for 1500+ articles in under 10 seconds"""

import asyncio
import aiohttp
import feedparser
import hashlib
import logging
import re
import socket
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class Config:
    max_articles: int = 500
    max_age_days: int = 7
    timeout_seconds: int = 12
    min_title_length: int = 5
    min_description_length: int = 10
    rss_config: Dict[str, Any] = field(default_factory=dict)
    # Deduplication settings
    enable_deduplication: bool = True
    min_title_length_dedup: int = 10
    min_description_length_dedup: int = 20

@dataclass
class Stats:
    successful: int = 0
    empty: int = 0
    failed: int = 0
    total_articles: int = 0
    processing_time: float = 0.0
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    # Deduplication stats
    original_articles: int = 0
    duplicates_removed: int = 0
    deduplication_time: float = 0.0

class ArticleDeduplicator:
    """
    Clean, fast article deduplication optimized for large datasets.
    Achieves O(n) complexity using hash-based lookups.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the deduplicator."""
        self.config = config or Config()
        self.seen_hashes: Set[str] = set()
    
    @lru_cache(maxsize=1024)
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        
        # Single-pass normalization
        normalized = title.lower().strip()
        
        # Remove common prefixes
        prefixes = ['breaking:', 'news:', 'update:', 'exclusive:', 'new:', 'latest:',
                   'ai breakthrough:', 'research:', 'study:', 'report:']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
        
        # Clean special chars and normalize whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @lru_cache(maxsize=1024)
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        if not url:
            return ""
        
        # Single-pass URL normalization
        normalized = url.lower().split('?')[0].split('#')[0].rstrip('/')
        
        # Remove www. prefix
        if normalized.startswith('www.'):
            normalized = normalized[4:]
        
        return normalized
    
    def _get_article_hash(self, article: Dict) -> str:
        """Generate hash for article deduplication."""
        title = self._normalize_title(article.get('title', ''))
        url = self._normalize_url(article.get('url', ''))
        
        # Use title and URL for deduplication
        content_string = f"{title}|{url}"
        return hashlib.md5(content_string.encode()).hexdigest()
    
    def _is_valid_article(self, article: Dict) -> bool:
        """Check if article has required fields and minimum content."""
        # Check required fields
        if not all(article.get(field) for field in ['title', 'url']):
            return False
        
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Check minimum lengths
        if len(title) < self.config.min_title_length_dedup or len(description) < self.config.min_description_length_dedup:
            return False
        
        # Skip articles with only processed_at field
        if len(article.keys()) <= 2 and 'processed_at' in article:
            return False
        
        return True
    
    def deduplicate(self, articles: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Deduplicate articles using hash-based lookups.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (unique_articles, stats)
        """
        if not articles:
            return [], {'original_count': 0, 'final_count': 0, 'removed_count': 0, 'duration': 0.0}
        
        start_time = time.time()
        logger.info(f"Starting deduplication of {len(articles)} articles")
        
        # Clean and validate articles
        valid_articles = [article for article in articles if self._is_valid_article(article)]
        logger.debug(f"Valid articles: {len(articles)} -> {len(valid_articles)}")
        
        # Deduplicate using hash-based lookups
        unique_articles = []
        seen_hashes = set()
        
        for article in valid_articles:
            article_hash = self._get_article_hash(article)
            
            if article_hash not in seen_hashes:
                seen_hashes.add(article_hash)
                unique_articles.append(article)
        
        duration = time.time() - start_time
        removed_count = len(valid_articles) - len(unique_articles)
        
        stats = {
            'original_count': len(articles),
            'valid_count': len(valid_articles),
            'final_count': len(unique_articles),
            'removed_count': removed_count,
            'duration': duration
        }
        
        logger.info(f"Deduplication complete: {len(valid_articles)} -> {len(unique_articles)} "
                   f"({removed_count} duplicates removed) in {duration:.3f}s")
        
        return unique_articles, stats

# Convenience function for easy imports
def deduplicate_articles(articles: List[Dict], config: Optional[Config] = None) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Convenience function to deduplicate articles.
    
    Args:
        articles: List of article dictionaries
        config: Optional configuration
    
    Returns:
        Tuple of (unique_articles, stats)
    """
    deduplicator = ArticleDeduplicator(config)
    return deduplicator.deduplicate(articles)

class ArticleParser:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._html_cleanup = re.compile(r'<[^>]+>')
        self._whitespace_cleanup = re.compile(r'\s+')
        self.max_age_days = self.config.get('max_age_days', 7)
    
    def parse_article(self, source_id: str, config: Dict, entry: Any) -> Optional[Dict]:
        try:
            title = getattr(entry, 'title', '').strip()
            url = getattr(entry, 'link', '').strip()
            if not (title and url and len(title) >= 5):
                return None
                
            description = self._clean_text(getattr(entry, 'summary', '') or getattr(entry, 'description', ''))
            if len(description) < 10:
                return None
                
            item_id = getattr(entry, 'id', '') or getattr(entry, 'guid', '') or url
            article_id = hashlib.md5(f"{source_id}_{item_id}".encode()).hexdigest()
            
            # Extract published_date from entry
            published_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    published_date = datetime(*entry.published_parsed[:6]).replace(tzinfo=timezone.utc).isoformat()
                except (ValueError, TypeError):
                    pass
            
            if not published_date and hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                try:
                    published_date = datetime(*entry.updated_parsed[:6]).replace(tzinfo=timezone.utc).isoformat()
                except (ValueError, TypeError):
                    pass
            
            if not published_date:
                published_date = datetime.now(timezone.utc).isoformat()
            
            # Check if article is within max_age_days
            try:
                pub_datetime = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - pub_datetime).days
                if age_days > self.max_age_days:
                    return None
            except (ValueError, TypeError):
                # If we can't parse the date, skip the article
                return None
            
            return {
                'id': article_id, 'source': config.get('name', source_id),
                'title': title, 'url': url, 'description': description, 
                'published_date': published_date, 'author': getattr(entry, 'author', '')
            }
        except Exception:
            return None
    
    def _clean_text(self, content: str, max_length: int = 500) -> str:
        if not content:
            return ''
        if isinstance(content, dict):
            content = content.get('rendered', str(content))
        
        cleaned = self._whitespace_cleanup.sub(' ', self._html_cleanup.sub(' ', str(content))).strip()
        if len(cleaned) > max_length:
            truncated = cleaned[:max_length].rsplit(' ', 1)[0]
            return truncated + '...' if truncated else cleaned[:max_length]
        return cleaned

class SourceCollector:
    def __init__(self, session: aiohttp.ClientSession, parser: ArticleParser):
        self.session = session
        self.parser = parser
    
    async def collect_source(self, source_id: str, config: Dict) -> Tuple[List[Dict], str]:
        url, max_items = config['url'], config.get('maxArticles', 10)
        
        for timeout_val in [7, 12]:
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_val, connect=3, sock_read=timeout_val-1)
                async with self.session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        return [], f"HTTP {response.status}"
                    
                    feed = feedparser.parse(await response.text())
                    if not hasattr(feed, 'entries') or not feed.entries:
                        return [], "no entries found"
                    
                    articles = [article for entry in feed.entries[:max_items] 
                              if (article := self.parser.parse_article(source_id, config, entry))]
                    return articles, "success" if articles else "no recent articles"
            except Exception as e:
                last_error = str(e)
        return [], f"error after retries: {last_error}"

class BatchCollector:
    def __init__(self, session: aiohttp.ClientSession, config: Optional[Dict[str, Any]] = None, 
                 progress_callback=None):
        self.session = session
        self.config = config or {}
        self.progress_callback = progress_callback
        self.max_concurrent = 200
        self.collector = SourceCollector(session, ArticleParser(self.config))
        self.stats = {'successful': 0, 'empty': 0, 'failed': 0, 'reasons': {}}
    
    async def collect_all(self, sources: Dict[str, Dict]) -> List[Dict]:
        source_items = list(sources.items())
        if self.progress_callback:
            self.progress_callback(0, len(source_items))
        
        all_articles, failed_sources = await self._process_batch(source_items)
        
        if failed_sources:
            retry_items = [(sid, sources[sid]) for sid in failed_sources]
            retry_articles, still_failed = await self._process_batch(retry_items, is_retry=True)
            all_articles.extend(retry_articles)
            self.stats['failed'] += len(still_failed)
        
        return all_articles
    
    async def _process_batch(self, batch: List[Tuple[str, Dict]], is_retry: bool = False) -> Tuple[List[Dict], List[str]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        failed_sources, batch_articles, completed_count = [], [], 0
        
        async def collect_with_semaphore(source_id: str, config: Dict) -> Tuple[str, List[Dict], str]:
            nonlocal completed_count
            async with semaphore:
                articles, reason = await self.collector.collect_source(source_id, config)
                completed_count += 1
                if self.progress_callback:
                    self.progress_callback(completed_count, len(batch))
                return source_id, articles, reason
        
        results = await asyncio.gather(*[collect_with_semaphore(sid, config) for sid, config in batch], return_exceptions=True)
        
        for idx, result in enumerate(results):
            source_id = batch[idx][0]
            if isinstance(result, Exception):
                self.stats['reasons'][source_id] = str(result)
                failed_sources.append(source_id)
            elif isinstance(result, tuple) and len(result) == 3:
                sid, articles, reason = result
                if articles:
                    batch_articles.extend(articles)
                    self.stats['successful'] += 1
                else:
                    if not is_retry:
                        self.stats['empty'] += 1
                    self.stats['reasons'][sid] = reason
                    failed_sources.append(sid)
            else:
                self.stats['reasons'][source_id] = "invalid response format"
                failed_sources.append(source_id)
        
        return batch_articles, failed_sources
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'successful_sources': self.stats['successful'],
            'empty_sources': self.stats['empty'],
            'failed_sources': self.stats['failed'],
            'failure_reasons': self.stats['reasons']
        }

class NewsCollector:
    def __init__(self):
        self.config, self.sources = self._load_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = Stats()
        logger.info(f"NewsCollector: {len(self.sources)} sources")
    
    def _load_config(self) -> Tuple[Config, Dict[str, Dict]]:
        try:
            import sys
            from pathlib import Path
            
            src_dir = Path(__file__).parent.parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            from shared.config.config_loader import ConfigLoader, get_collectors_config
            from shared.config.sources_loader import get_sources_loader
            
            collection_data = ConfigLoader.get('collection', {}, "app")
            collectors_config = get_collectors_config()
            
            config = Config(
                max_articles=collection_data.get('max_articles_to_collect', 500),
                max_age_days=collection_data.get('max_age_days', 7),
                timeout_seconds=collection_data.get('performance', {}).get('timeout_seconds', 12),
                min_title_length=collection_data.get('quality_filters', {}).get('min_title_length', 5),
                min_description_length=collection_data.get('quality_filters', {}).get('min_description_length', 10),
                rss_config=collectors_config
            )
            
            sources_loader = get_sources_loader()
            sources = sources_loader.get_sources()
            logger.info(f"Loaded {len(sources)} sources")
            
            return config, sources
        except Exception as e:
            logger.warning(f"Config loader not available: {e}")
            return Config(), {}
    
    async def collect_all(self, source_ids: Optional[List[str]] = None, 
                         max_age_days: Optional[int] = None, 
                         max_articles: Optional[int] = None,
                         progress_callback=None) -> List[Dict]:
        start_time = time.time()
        
        sources_to_use = (self.sources if source_ids is None 
                         else {k: v for k, v in self.sources.items() if k in source_ids})
        
        await self._init_session()
        
        try:
            batch_config = {**self.config.rss_config, 'max_age_days': self.config.max_age_days}
            batch_collector = BatchCollector(self.session, batch_config, progress_callback)
            articles = await batch_collector.collect_all(sources_to_use)
            
            articles = self._process_articles(articles)
            if max_articles and len(articles) > max_articles:
                articles = self._prioritize_articles(articles)[:max_articles]
            
            batch_stats = batch_collector.get_stats()
            self.stats.successful = batch_stats['successful_sources']
            self.stats.empty = batch_stats['empty_sources']
            self.stats.failed = batch_stats['failed_sources']
            self.stats.failure_reasons = batch_stats['failure_reasons']
            self.stats.total_articles = len(articles)
            self.stats.processing_time = time.time() - start_time
            
            self._log_summary()
            return articles
        finally:
            await self._cleanup_session()
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        if not articles:
            return articles
        
        # Store original count for stats
        original_count = len(articles)
        
        # Apply deduplication if enabled
        if self.config.enable_deduplication:
            unique_articles, dedup_stats = deduplicate_articles(articles, self.config)
            self.stats.original_articles = dedup_stats['original_count']
            self.stats.duplicates_removed = dedup_stats['removed_count']
            self.stats.deduplication_time = dedup_stats['duration']
        else:
            unique_articles = articles
            self.stats.original_articles = original_count
            self.stats.duplicates_removed = 0
            self.stats.deduplication_time = 0.0
        
        # Apply final quality filters
        return [article for article in unique_articles
                if (article.get('title') and article.get('url') and article.get('description') and
                    len(article.get('title', '')) >= self.config.min_title_length and
                    len(article.get('description', '')) >= self.config.min_description_length and
                    article.get('url', '').startswith('http'))]
    
    def _prioritize_articles(self, articles: List[Dict]) -> List[Dict]:
        def calculate_score(article: Dict) -> float:
            try:
                pub_date = datetime.fromisoformat(article.get('published_date', ''))
                hours_old = (datetime.now(timezone.utc) - pub_date).total_seconds() / 3600
                recency_bonus = max(0, 24 - hours_old) / 24
            except (ValueError, TypeError):
                recency_bonus = 0
            return recency_bonus
        
        return sorted(articles, key=calculate_score, reverse=True)
    
    async def _init_session(self):
        if self.session is not None:
            return
        
        connector = aiohttp.TCPConnector(
            limit=500, limit_per_host=100, ttl_dns_cache=300,
            use_dns_cache=True, keepalive_timeout=30,
            enable_cleanup_closed=True, force_close=False, family=socket.AF_INET
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds, connect=3, sock_read=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout,
            headers={
                'User-Agent': 'Daily AI Times/Professional RSS Reader (+https://daily-ai-times.ai)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
    
    async def _cleanup_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    def _log_summary(self):
        s = self.stats
        logger.info(f"Collection: ✓{s.successful} ○{s.empty} ✗{s.failed} "
                   f"→ {s.total_articles} articles ({s.processing_time:.2f}s)")
        
        if s.original_articles > 0 and s.duplicates_removed > 0:
            logger.info(f"Deduplication: {s.original_articles} → {s.total_articles} "
                       f"({s.duplicates_removed} removed, {s.deduplication_time:.3f}s)")
        
        if s.failure_reasons:
            logger.warning(f"Failed sources: {len(s.failure_reasons)}")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    setup_logging()
    
    async def run_collection():
        collector = NewsCollector()
        articles = await collector.collect_all(max_articles=collector.config.max_articles)
        print("=" * 40)
        print(f"Collected {len(articles)} articles")
        print(f"Sources: ✓{collector.stats.successful} ○{collector.stats.empty} ✗{collector.stats.failed}")
        print(f"Processing time: {collector.stats.processing_time:.2f}s")
    
    asyncio.run(run_collection())

__all__ = ['Config', 'Stats', 'ArticleDeduplicator', 'deduplicate_articles', 'ArticleParser', 'SourceCollector', 'BatchCollector', 'NewsCollector', 'main', 'setup_logging']

if __name__ == "__main__":
    main()
