#!/usr/bin/env python3
"""
News Collection System

A high-performance news collector for AI-related content from RSS feeds and APIs.
Focuses solely on collection and parsing - filtering is handled by dedicated agents.
"""

import asyncio
import aiohttp
import feedparser
import json
import logging
import hashlib
import re
import time
import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from functools import lru_cache

# Enhanced logging setup with performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optimized connection pool configuration
CONNECTOR_CONFIG = {
    'limit': 100,
    'limit_per_host': 30,
    'ttl_dns_cache': 300,
    'use_dns_cache': True,
    'keepalive_timeout': 30,
    'enable_cleanup_closed': True
}

# Common field mappings for optimization
TITLE_FIELDS = ('title', 'headline', 'name', 'display_name', 'full_name')
URL_FIELDS = ('url', 'link', 'permalink', 'html_url', 'web_url', 'url_abs', 'url_pdf', 'guid')
DESC_FIELDS = ('summary', 'description', 'content', 'subtitle', 'body', 'excerpt')
DATE_FIELDS = ('published', 'date', 'publishedAt', 'created_at', 'updated_at', 'submission_date')


class NewsCollector:
    def __init__(self, sources_file='../../shared/config/sources.json'):
        self.sources_file = sources_file
        self.session = None
        self.sources = {}
        self.load_config()
    
    def load_config(self):
        """Load sources configuration with error handling."""
        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.sources = config.get('sources', {})
            logger.info(f"Loaded {len(self.sources)} total sources")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    @lru_cache(maxsize=128)
    def _get_article_signature(self, title: str, url: str) -> str:
        """Generate cached article signature for deduplication."""
        title_norm = re.sub(r'[^\w\s]', '', title.lower())
        url_norm = url.lower().split('?')[0]
        return hashlib.md5(f"{title_norm}_{url_norm}".encode()).hexdigest()
    
    def _extract_field_value(self, item: Any, fields: tuple, default: str = '') -> str:
        """Extract field value using multiple fallback options."""
        for field in fields:
            if hasattr(item, field):
                value = getattr(item, field, default)
            elif isinstance(item, dict) and field in item:
                value = item[field]
            else:
                continue
            
            if value:
                if isinstance(value, list) and value:
                    value = value[0].get('value', str(value[0])) if isinstance(value[0], dict) else str(value[0])
                return str(value).strip()
        return default
    
    def _clean_html_description(self, description: str, max_length: int = 500) -> str:
        """Clean HTML from description and truncate."""
        if not description:
            return ''
        if isinstance(description, dict):
            description = description.get('rendered', str(description))
        cleaned = BeautifulSoup(str(description), 'html.parser').get_text()
        return cleaned[:max_length].strip()
    
    def _parse_article_date(self, date_value: Any) -> str:
        """Parse various date formats into ISO format."""
        if not date_value:
            return datetime.now(timezone.utc).isoformat()
        
        try:
            if hasattr(date_value, 'timetuple'):
                return date_value.isoformat()
            elif isinstance(date_value, (tuple, list)) and len(date_value) >= 6:
                from time import mktime
                time_tuple = tuple(date_value[:9]) if len(date_value) >= 9 else tuple(list(date_value) + [0] * (9 - len(date_value)))
                return datetime.fromtimestamp(mktime(time_tuple), timezone.utc).isoformat()
            elif isinstance(date_value, str):
                return date_parser.parse(date_value).isoformat()
        except Exception:
            pass
        
        return datetime.now(timezone.utc).isoformat()
    
    def _apply_age_filter(self, pub_date: str, max_age_days: int) -> bool:
        """Check if article passes age filter."""
        if max_age_days <= 0:
            return True
        
        try:
            article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            if article_date.tzinfo is None:
                article_date = article_date.replace(tzinfo=timezone.utc)
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            return article_date >= cutoff_date
        except Exception:
            return True  # Keep article if date parsing fails
    
    async def collect_all(self, source_ids: Optional[List[str]] = None, max_age_days: int = 14, max_articles: Optional[int] = None) -> List[Dict]:
        """Collect articles from all or specified sources with optimized processing."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'NewsXP.ai/2.0 (+https://newsxp.ai)'},
            connector=aiohttp.TCPConnector(**CONNECTOR_CONFIG)
        )
        
        try:
            # Filter enabled sources
            sources_to_use = self._get_enabled_sources(source_ids)
            logger.info(f"Collecting from {len(sources_to_use)} sources")
            
            # Collect articles with improved error tracking
            all_articles, stats = await self._collect_from_sources(sources_to_use, max_age_days, max_articles)
            self._log_collection_stats(stats, max_age_days)
            
            # Enhanced deduplication with validation
            logger.info(f"Applying enhanced deduplication to {len(all_articles)} articles")
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Additional quality check after deduplication
            final_articles = []
            for article in unique_articles:
                # Ensure article has basic required structure
                if (article.get('id') and 
                    article.get('title') and 
                    article.get('url') and 
                    article.get('description') and
                    len(article.get('title', '')) > 5 and
                    len(article.get('description', '')) > 10):
                    final_articles.append(article)
                else:
                    logger.debug(f"Removed low-quality article: {article.get('title', 'NO_TITLE')}")
            
            quality_removed = len(unique_articles) - len(final_articles)
            if quality_removed > 0:
                logger.info(f"Quality filter: removed {quality_removed} low-quality articles")
            
            return final_articles
            
        finally:
            if self.session:
                await self.session.close()
    
    def _get_enabled_sources(self, source_ids: Optional[List[str]]) -> Dict[str, Dict]:
        """Get filtered and enabled sources."""
        if source_ids:
            return {k: v for k, v in self.sources.items() 
                   if k in source_ids and v.get('enabled', True)}
        return {k: v for k, v in self.sources.items() if v.get('enabled', True)}
    
    async def _collect_from_sources(self, sources: Dict[str, Dict], max_age_days: int, max_articles: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """Collect from all sources with source diversification and return articles with statistics."""
        all_articles = []
        stats = {'successful': 0, 'empty': 0, 'failed': 0, 'reasons': {}}
        
        # If max_articles is set, implement source diversification strategy
        if max_articles:
            return await self._collect_with_diversification(sources, max_age_days, max_articles)
        
        # Process sources in batches to avoid overwhelming the system
        batch_size = 10
        source_items = list(sources.items())
        
        for i in range(0, len(source_items), batch_size):
            batch = source_items[i:i + batch_size]
            tasks = [
                self._collect_from_single_source(source_id, config, max_age_days)
                for source_id, config in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (source_id, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    stats['failed'] += 1
                    logger.error(f"Failed to collect from {source_id}: {result}")
                elif isinstance(result, (tuple, list)) and len(result) == 2:
                    try:
                        articles, reason = result
                        all_articles.extend(articles)
                        
                        if articles:
                            stats['successful'] += 1
                            logger.debug(f"Collected {len(articles)} articles from {source_id}")
                        else:
                            stats['empty'] += 1
                            stats['reasons'][source_id] = reason
                            logger.debug(f"Collected 0 articles from {source_id} - {reason}")
                    except (TypeError, ValueError) as e:
                        stats['failed'] += 1
                        logger.error(f"Invalid result format from {source_id}: {e}")
                else:
                    stats['failed'] += 1
                    logger.error(f"Unexpected result type from {source_id}: {type(result)}")
        
        return all_articles, stats
    
    async def _collect_with_diversification(self, sources: Dict[str, Dict], max_age_days: int, max_articles: int) -> Tuple[List[Dict], Dict]:
        """Collect articles with source diversification to ensure variety."""
        all_articles = []
        stats = {'successful': 0, 'empty': 0, 'failed': 0, 'reasons': {}}
        articles_per_source = {}
        
        # Group sources by category for better diversification
        sources_by_category = {}
        for source_id, config in sources.items():
            category = config.get('category', 'Other')
            if category not in sources_by_category:
                sources_by_category[category] = []
            sources_by_category[category].append((source_id, config))
        
        logger.info(f"   Diversifying across {len(sources_by_category)} categories: {list(sources_by_category.keys())}")
        
        # Calculate target articles per category
        articles_per_category = max(1, max_articles // len(sources_by_category))
        logger.info(f"   Target: ~{articles_per_category} articles per category")
        
        # Collect from each category
        for category, category_sources in sources_by_category.items():
            if len(all_articles) >= max_articles:
                break
                
            category_articles = []
            remaining_for_category = min(articles_per_category, max_articles - len(all_articles))
            
            # Try sources in this category until we get enough articles
            for source_id, config in category_sources:
                if len(category_articles) >= remaining_for_category:
                    break
                    
                try:
                    articles, reason = await self._collect_from_single_source(source_id, config, max_age_days)
                    
                    if articles:
                        # Limit articles from this source
                        needed = remaining_for_category - len(category_articles)
                        if len(articles) > needed:
                            articles = articles[:needed]
                            logger.debug(f"   Limited {source_id} to {len(articles)} articles for diversity")
                        
                        category_articles.extend(articles)
                        articles_per_source[source_id] = len(articles)
                        stats['successful'] += 1
                        logger.debug(f"   Collected {len(articles)} articles from {source_id} ({category})")
                    else:
                        stats['empty'] += 1
                        stats['reasons'][source_id] = reason
                        
                except Exception as e:
                    stats['failed'] += 1
                    logger.error(f"   Failed to collect from {source_id}: {e}")
            
            all_articles.extend(category_articles)
            logger.info(f"   Category '{category}': {len(category_articles)} articles collected")
        
        # If we still need more articles, collect from remaining sources
        if len(all_articles) < max_articles:
            remaining_needed = max_articles - len(all_articles)
            logger.info(f"   Need {remaining_needed} more articles, collecting from remaining sources")
            
            for source_id, config in sources.items():
                if len(all_articles) >= max_articles:
                    break
                if source_id in articles_per_source:  # Already collected from this source
                    continue
                    
                try:
                    articles, reason = await self._collect_from_single_source(source_id, config, max_age_days)
                    
                    if articles:
                        needed = max_articles - len(all_articles)
                        if len(articles) > needed:
                            articles = articles[:needed]
                        
                        all_articles.extend(articles)
                        stats['successful'] += 1
                        logger.debug(f"   Additional: {len(articles)} articles from {source_id}")
                    else:
                        stats['empty'] += 1
                        stats['reasons'][source_id] = reason
                        
                except Exception as e:
                    stats['failed'] += 1
                    logger.error(f"   Failed to collect from {source_id}: {e}")
        
        logger.info(f"   Final collection: {len(all_articles)} articles from {len(articles_per_source)} sources")
        return all_articles, stats
    
    async def _collect_from_single_source(self, source_id: str, config: Dict, max_age_days: int) -> Tuple[List[Dict], str]:
        """Collect from a single source with optimized error handling."""
        try:
            source_type = config.get('type', 'rss').lower()
            
            if source_type == 'rss':
                return await self._collect_rss_optimized(source_id, config, max_age_days)
            elif source_type == 'api':
                return await self._collect_api_optimized(source_id, config, max_age_days)
            elif source_type in ['scrape', 'webhook']:
                return [], f"feature not implemented ({source_type})"
            else:
                return [], f"unknown source type '{source_type}'"
        except Exception as e:
            logger.error(f"Source collection error for {source_id}: {e}")
            return [], f"collection error: {str(e)[:50]}"
    
    def _log_collection_stats(self, stats: Dict, max_age_days: int):
        """Log collection statistics."""
        logger.info(f"Collection summary: {stats['successful']} sources with articles, "
                   f"{stats['empty']} sources empty, {stats['failed']} sources failed")
        if max_age_days > 0:
            logger.info(f"Age filter applied: only articles newer than {max_age_days} days included")
    
    async def collect_from_source(self, source_id: str, config: Dict, max_age_days: int = 14) -> tuple[List[Dict], str]:
        """Legacy method - redirects to optimized version."""
        return await self._collect_from_single_source(source_id, config, max_age_days)
    
    async def _collect_rss_optimized(self, source_id: str, config: Dict, max_age_days: int = 14) -> Tuple[List[Dict], str]:
        """Optimized RSS collection with better error handling."""
        if not self.session:
            return [], "no session available"
        
        try:
            url = config['url']
            async with self.session.get(url) as response:
                if response.status != 200:
                    return [], f"HTTP {response.status} error"
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                # Validate feed
                if not hasattr(feed, 'entries') or not feed.entries:
                    return [], self._get_empty_feed_reason(config)
                
                # Process entries efficiently
                articles = []
                max_articles = config.get('maxArticles', 20)
                
                for entry in feed.entries[:max_articles]:
                    article = self._parse_entry_optimized(source_id, config, feed, entry, max_age_days)
                    if article:
                        articles.append(article)
                
                return articles, "success" if articles else "entries could not be parsed"
                
        except Exception as e:
            logger.error(f"RSS collection failed for {source_id}: {e}")
            return [], f"collection error: {str(e)[:50]}"
    
    async def _collect_api_optimized(self, source_id: str, config: Dict, max_age_days: int = 7) -> Tuple[List[Dict], str]:
        """Optimized API collection with better error handling."""
        if not self.session:
            return [], "no session available"
        
        try:
            url = config['url']
            headers = {'Accept': 'application/json'}
            if 'apiKey' in config:
                headers['Authorization'] = f"Bearer {config['apiKey']}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return [], self._get_api_error_reason(response.status)
                
                # Parse response efficiently
                try:
                    data = await response.json()
                except Exception:
                    return [], "failed to parse JSON response"
                
                # Extract articles array
                articles_data = self._extract_articles_from_api_response(data)
                if not articles_data:
                    return [], self._get_empty_api_reason(config)
                
                # Process articles
                articles = []
                max_articles = config.get('maxArticles', 20)
                
                for item in articles_data[:max_articles]:
                    article = self._parse_api_item_optimized(source_id, config, item, max_age_days)
                    if article:
                        articles.append(article)
                
                return articles, "success" if articles else "items could not be parsed"
                
        except Exception as e:
            logger.error(f"API collection failed for {source_id}: {e}")
            return [], f"collection error: {str(e)[:50]}"
    
    def _get_empty_feed_reason(self, config: Dict) -> str:
        """Get reason for empty RSS feed."""
        update_freq = config.get('updateFrequency', 'unknown')
        current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
        current_hour = datetime.now().hour
        
        if update_freq == 'daily' and current_day in [5, 6]:
            return "weekend (daily updates)"
        elif update_freq == 'weekdays' and current_day in [5, 6]:
            return "weekend (weekdays only)"
        elif 'midnight' in config.get('note', '').lower() and current_hour < 8:
            return "early morning (updates at midnight)"
        else:
            return "feed empty or unavailable"
    
    def _get_api_error_reason(self, status_code: int) -> str:
        """Get API error reason based on status code."""
        error_map = {
            400: "HTTP 400 error - bad request",
            403: "HTTP 403 error - access forbidden",
            429: "HTTP 429 error - rate limit exceeded"
        }
        return error_map.get(status_code, f"HTTP {status_code} error")
    
    def _get_empty_api_reason(self, config: Dict) -> str:
        """Get reason for empty API response."""
        update_freq = config.get('updateFrequency', 'unknown')
        current_day = datetime.now().weekday()
        
        if update_freq == 'daily' and current_day in [5, 6]:
            return "weekend (daily updates)"
        elif update_freq == 'weekdays' and current_day in [5, 6]:
            return "weekend (weekdays only)"
        else:
            return "API returned empty results"
    
    def _extract_articles_from_api_response(self, data: Any) -> List[Dict]:
        """Extract articles array from API response."""
        if isinstance(data, list):
            return data
        
        if isinstance(data, dict):
            for key in ['articles', 'items', 'results', 'data', 'entries']:
                if key in data and isinstance(data[key], list):
                    return data[key]
        
        return []
    
    def _parse_entry_optimized(self, source_id: str, config: Dict, feed: Any, entry: Any, max_age_days: int) -> Optional[Dict]:
        """Optimized RSS entry parsing."""
        try:
            # Extract basic fields using optimized field extraction
            title = self._extract_field_value(entry, TITLE_FIELDS)
            url = self._extract_field_value(entry, URL_FIELDS)
            
            if not title or not url:
                return None
            
            # Extract and clean description
            description = self._extract_field_value(entry, DESC_FIELDS)
            description = self._clean_html_description(description)
            
            # Parse and validate date
            pub_date = self._parse_article_date(
                getattr(entry, 'published_parsed', None) or 
                getattr(entry, 'updated_parsed', None)
            )
            
            if not self._apply_age_filter(pub_date, max_age_days):
                return None
            
            # Generate unique ID
            entry_id = getattr(entry, 'id', url) or str(hash(title))
            article_id = hashlib.md5(f"{source_id}_{entry_id}".encode()).hexdigest()
            
            return {
                'id': article_id,
                'source_id': source_id,
                'source': getattr(feed.feed, 'title', source_id) if hasattr(feed, 'feed') and feed.feed else source_id,
                'category': config.get('category', 'Other'),
                'title': title,
                'url': url,
                'description': description,
                'published_date': pub_date,
                'collected_at': datetime.now(timezone.utc).isoformat(),
                'author': getattr(entry, 'author', '').strip(),
                'source_priority': config.get('priority', 5)
            }
            
        except Exception as e:
            logger.error(f"Error parsing RSS entry from {source_id}: {e}")
            return None
    
    def _parse_api_item_optimized(self, source_id: str, config: Dict, item: Dict, max_age_days: int) -> Optional[Dict]:
        """Optimized API item parsing."""
        try:
            # Extract basic fields
            title = self._extract_field_value(item, TITLE_FIELDS)
            url = self._extract_field_value(item, URL_FIELDS)
            
            # Special handling for different API formats
            if not title or not url:
                title, url = self._handle_special_api_formats(source_id, item, title, url)
            
            if not title or not url:
                return None
            
            # Extract description
            description = self._extract_field_value(item, DESC_FIELDS)
            description = self._clean_html_description(description)
            
            # Parse date
            pub_date = self._parse_article_date(None)
            for field in DATE_FIELDS:
                if field in item and item[field]:
                    pub_date = self._parse_article_date(item[field])
                    break
            
            if not self._apply_age_filter(pub_date, max_age_days):
                return None
            
            # Generate unique ID
            item_id = item.get('id', item.get('guid', url))
            article_id = hashlib.md5(f"{source_id}_{item_id}".encode()).hexdigest()
            
            return {
                'id': article_id,
                'source_id': source_id,
                'source': config.get('name', source_id),
                'category': config.get('category', 'Other'),
                'title': title.strip(),
                'url': url,
                'description': description,
                'published_date': pub_date,
                'collected_at': datetime.now(timezone.utc).isoformat(),
                'author': str(item.get('author', '')).strip(),
                'source_priority': config.get('priority', 5)
            }
            
        except Exception as e:
            logger.error(f"Error parsing API item from {source_id}: {e}")
            return None
    
    def _handle_special_api_formats(self, source_id: str, item: Dict, title: str, url: str) -> Tuple[str, str]:
        """Handle special API formats like GitHub, ArXiv, etc."""
        # GitHub API format
        if source_id == 'github_openai':
            if not title and item.get('type') == 'PushEvent':
                commits = item.get('payload', {}).get('commits', [])
                if commits:
                    repo_name = item.get('repo', {}).get('name', 'repository')
                    title = f"New commits to {repo_name}: {commits[0].get('message', '')[:100]}"
            elif not title and item.get('type') == 'ReleaseEvent':
                release = item.get('payload', {}).get('release', {})
                title = f"Release: {release.get('name', release.get('tag_name', 'New Release'))}"
            
            if not url:
                repo_name = item.get('repo', {}).get('name', '')
                if repo_name:
                    event_type = item.get('type', '')
                    if event_type == 'PushEvent':
                        url = f"https://github.com/{repo_name}/commits"
                    elif event_type == 'ReleaseEvent':
                        url = f"https://github.com/{repo_name}/releases"
                    else:
                        url = f"https://github.com/{repo_name}"
        
        # ArXiv paper formats
        elif 'arxiv_id' in item and item['arxiv_id']:
            url = f"https://arxiv.org/abs/{item['arxiv_id']}"
        elif 'paper' in item and isinstance(item['paper'], dict):
            paper_id = item['paper'].get('id')
            if paper_id:
                url = f"https://arxiv.org/abs/{paper_id}"
        
        return title, url
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Optimized deduplication using enhanced strategy."""
        try:
            # Try to import the deduplication utility
            import sys
            import os
            processors_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processors')
            sys.path.insert(0, processors_path)
            
            from deduplication_utils import ArticleDeduplicator #type: ignore[import]
            
            deduplicator = ArticleDeduplicator()
            unique_articles = deduplicator.deduplicate_articles(articles, strategy="enhanced")
            
            removed_count = len(articles) - len(unique_articles)
            logger.info(f"Collection deduplication: {len(articles)} -> {len(unique_articles)} articles ({removed_count} duplicates removed)")
            return unique_articles
            
        except ImportError as e:
            logger.warning(f"Enhanced deduplication not available: {e}. Using basic deduplication.")
            # Fallback to basic deduplication
            from typing import Set
            seen: Set[str] = set()
            unique = []
            
            for article in articles:
                signature = self._get_article_signature(
                    article.get('title', ''), 
                    article.get('url', '')
                )
                
                if signature not in seen:
                    seen.add(signature)
                    unique.append(article)
            
            logger.info(f"Basic deduplication: {len(articles)} -> {len(unique)} articles")
            return unique
    
    # Legacy methods maintained for compatibility
    async def collect_rss(self, source_id: str, config: Dict, max_age_days: int = 7) -> Tuple[List[Dict], str]:
        """Legacy RSS collection method - redirects to optimized version."""
        return await self._collect_rss_optimized(source_id, config, max_age_days)
    
    async def collect_api(self, source_id: str, config: Dict, max_age_days: int = 7) -> Tuple[List[Dict], str]:
        """Legacy API collection method - redirects to optimized version."""
        return await self._collect_api_optimized(source_id, config, max_age_days)
    
    def parse_rss_entry(self, source_id: str, config: Dict, feed: Any, entry: Any, max_age_days: int = 7) -> Optional[Dict]:
        """Legacy RSS parsing method - redirects to optimized version."""
        return self._parse_entry_optimized(source_id, config, feed, entry, max_age_days)
    
    def parse_api_item(self, source_id: str, config: Dict, item: Dict, max_age_days: int = 7) -> Optional[Dict]:
        """Legacy API parsing method - redirects to optimized version."""
        return self._parse_api_item_optimized(source_id, config, item, max_age_days)
    
    def parse_date(self, date_tuple) -> str:
        """Legacy date parsing method - redirects to optimized version."""
        return self._parse_article_date(date_tuple)
    
    def deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Legacy deduplication method - redirects to optimized version."""
        return self._deduplicate_articles(articles)
    
    def score_articles(self, articles: List[Dict]) -> List[Dict]:
        """Legacy scoring method - now returns articles unchanged."""
        logger.warning("Legacy score_articles method called - scoring should be done by filtering agents")
        return articles


async def main():
    """Main execution function for news collection."""
    parser = argparse.ArgumentParser(description='Collect AI news from configured sources')
    parser.add_argument('--sources', help='Comma-separated list of source IDs')
    parser.add_argument('--output', default='data/news.json', help='Output file path')
    parser.add_argument('--archive', action='store_true', help='Save to daily archive')
    parser.add_argument('--config', default='../../shared/config/sources.json', help='Sources config file')
    parser.add_argument('--max-articles', type=int, default=1000, help='Max articles to collect')
    parser.add_argument('--max-age-days', type=int, default=7, help='Max age of articles in days')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh (enables verbose logging)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose or args.force_refresh:
        logging.getLogger().setLevel(logging.DEBUG)
        if args.force_refresh:
            logger.info("Force refresh enabled - using verbose logging")
    
    try:
        # Initialize collector
        collector = NewsCollector(args.config)
        
        # Parse sources
        sources = None
        if args.sources and args.sources.lower() != 'all':
            sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        
        # Collect articles with timing
        start_time = time.time()
        logger.info("Starting optimized news collection...")
        
        articles = await collector.collect_all(sources, max_age_days=args.max_age_days, max_articles=args.max_articles)
        collection_time = time.time() - start_time
        
        logger.info(f"Collection completed in {collection_time:.2f} seconds")
        
        # Sort by date for consistent output
        articles.sort(key=lambda x: x.get('published_date', ''), reverse=True)
        
        # Prepare optimized output
        output_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'collection_time_seconds': round(collection_time, 2),
            'count': len(articles),
            'articles': articles
        }
        
        # Efficient file operations
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with optimized JSON serialization
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, separators=(',', ':'))
        
        logger.info(f"Saved {len(articles)} articles to {output_path}")
        
        # Archive if requested
        if args.archive:
            archive_dir = output_path.parent / 'archive'
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            daily_archive = archive_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"
            with open(daily_archive, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, separators=(',', ':'))
            
            logger.info(f"Archived to {daily_archive}")
        
        # Success output
        result = {
            'success': True,
            'articles_collected': len(articles),
            'collection_time': round(collection_time, 2),
            'output_file': str(output_path),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        print(json.dumps(result, separators=(',', ':')))
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        print(json.dumps(error_result, separators=(',', ':')))
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())
