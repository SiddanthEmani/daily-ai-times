#!/usr/bin/env python3
"""
Ultra-Fast News Collection - Optimized for 1500+ articles in under 10 seconds
"""

import asyncio
import aiohttp
import feedparser
import hashlib
import json
import logging
import time
import socket
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re

logger = logging.getLogger(__name__)

class ArticleParser:
    """RSS article parser with minimal processing."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.html_cleanup = re.compile(r'<[^>]+>')
        self.whitespace_cleanup = re.compile(r'\s+')
        self.special_chars = re.compile(r'[^\w\s]')

    def parse_article(self, source_id: str, config: Dict, entry: Any, max_age_days: int) -> Optional[Dict]:
        try:
            title = getattr(entry, 'title', '').strip()
            url = getattr(entry, 'link', '').strip()
            if not (title and url and len(title) > 5):
                return None
            description = self._fast_clean_text(
                getattr(entry, 'summary', '') or getattr(entry, 'description', '')
            )
            if len(description) < 10:
                return None
            pub_date = datetime.now(timezone.utc).isoformat()
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
        except Exception:
            return None

    def _fast_clean_text(self, content: str) -> str:
        if not content:
            return ''
        cleaned = self.html_cleanup.sub(' ', content)
        cleaned = self.whitespace_cleanup.sub(' ', cleaned).strip()
        if len(cleaned) > 500:
            cleaned = cleaned[:500].rsplit(' ', 1)[0] + '...'
        return cleaned

class SourceCollector:
    """RSS source collector with retry logic and reasonable timeouts."""
    def __init__(self, session: aiohttp.ClientSession, parser: ArticleParser):
        self.session = session
        self.parser = parser

    async def collect_source(self, source_id: str, config: Dict, max_age_days: int) -> Tuple[List[Dict], str]:
        url = config['url']
        max_items = config.get('maxArticles', 10)
        timeouts = [7, 12]
        last_error = None
        for attempt, timeout_val in enumerate(timeouts):
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_val, connect=3, sock_read=timeout_val-1)
                async with self.session.get(url, timeout=timeout) as response:
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
                    return articles, "success" if articles else "no recent articles"
            except Exception as e:
                last_error = str(e)
        return [], f"error after retries: {last_error}"

class BatchCollector:
    """Parallel batch processing with retry for failed sources."""
    def __init__(self, session: aiohttp.ClientSession, config: Optional[Dict[str, Any]] = None, progress_callback=None):
        self.session = session
        self.config = config or {}
        self.progress_callback = progress_callback
        self.batch_size = 100
        self.max_concurrent_requests = 200
        self.parser = ArticleParser(config)
        self.collector = SourceCollector(session, self.parser)
        self.stats = {'successful': 0, 'empty': 0, 'failed': 0, 'reasons': {}}

    async def collect_all(self, sources: Dict[str, Dict], max_age_days: int) -> List[Dict]:
        source_items = list(sources.items())
        total_sources = len(source_items)
        if self.progress_callback:
            self.progress_callback(0, total_sources)
        all_articles, failed_sources = await self._process_batch_with_failures(source_items, max_age_days)
        if failed_sources:
            retry_items = [(sid, sources[sid]) for sid in failed_sources]
            retry_articles, still_failed = await self._process_batch_with_failures(retry_items, max_age_days, is_retry=True)
            all_articles.extend(retry_articles)
            for sid in still_failed:
                self.stats['failed'] += 1
        if self.progress_callback:
            self.progress_callback(total_sources, total_sources)
        return all_articles

    async def _process_batch_with_failures(self, batch, max_age_days, is_retry=False):
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        failed_sources = []
        batch_articles = []
        async def collect_with_semaphore(source_id, config):
            async with semaphore:
                articles, reason = await self.collector.collect_source(source_id, config, max_age_days)
                return source_id, articles, reason
        tasks = [collect_with_semaphore(source_id, config) for source_id, config in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
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

class ArticleProcessor:
    """Article processing with deduplication and quality filtering."""
    def __init__(self, config):
        self.config = config
    def process(self, articles: List[Dict]) -> List[Dict]:
        if not articles:
            return articles
        from ..processors.deduplication_utils import UltraFastDeduplicator
        deduplicator = UltraFastDeduplicator()
        print("\n[Processing] Starting deduplication and filtering...")
        start_time = time.time()
        unique_articles = deduplicator.deduplicate_articles_ultra_fast(articles)
        dedup_time = time.time() - start_time
        print(f"[Processing] ✓ Deduplication: {len(articles)} → {len(unique_articles)} articles in {dedup_time:.3f}s")
        filtered_articles = []
        for article in unique_articles:
            if (article.get('title') and article.get('url') and article.get('description') and
                len(article.get('title', '')) >= 5 and
                len(article.get('description', '')) >= 10 and
                article.get('url', '').startswith('http')):
                filtered_articles.append(article)
        filter_time = time.time() - start_time - dedup_time
        print(f"[Processing] ✓ Quality filter: {len(unique_articles)} → {len(filtered_articles)} articles in {filter_time:.3f}s")
        return filtered_articles

class NewsCollector:
    """Main news collection orchestrator."""
    def __init__(self):
        from .core import ConfigManager, CollectionStats
        self.config, self.sources = ConfigManager.load()
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = CollectionStats()
        logger.info(f"NewsCollector initialized with {len(self.sources)} sources")
    async def collect_all(self, 
                       source_ids: Optional[List[str]] = None, 
                       max_age_days: Optional[int] = None, 
                       max_articles: Optional[int] = None,
                       progress_callback=None) -> List[Dict]:
        start_time = time.time()
        sources_to_use = self.sources if source_ids is None else {k: v for k, v in self.sources.items() if k in source_ids}
        effective_max_age = max_age_days or self.config.max_age_days
        await self._init_session()
        try:
            batch_collector = BatchCollector(self.session, self.config.rss_config, progress_callback)
            articles = await batch_collector.collect_all(sources_to_use, effective_max_age)
            if max_articles and len(articles) > max_articles:
                articles = self._prioritize_articles(articles)[:max_articles]
            processor = ArticleProcessor(self.config)
            articles = processor.process(articles)
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
    async def _init_session(self):
        if self.session is not None:
            return
        connector = aiohttp.TCPConnector(
            limit=500,
            limit_per_host=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,
            family=socket.AF_INET
        )
        timeout = aiohttp.ClientTimeout(total=12, connect=3, sock_read=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
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
    def _prioritize_articles(self, articles: List[Dict]) -> List[Dict]:
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
        s = self.stats
        logger.info(f"Collection: ✓{s.successful} ○{s.empty} ✗{s.failed} "
                    f"→ {s.total_articles} articles ({s.processing_time:.2f}s)")
        if s.failure_reasons:
            logger.warning(f"Failed sources: {len(s.failure_reasons)}")
