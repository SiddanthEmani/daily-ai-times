#!/usr/bin/env python3
"""
Simplified AI News Collection System

A streamlined news collector for AI-related content from RSS feeds and APIs.
"""

import asyncio
import json
import logging
import hashlib
import re
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news_collection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Simplified AI keywords for scoring
AI_KEYWORDS = {
    'artificial intelligence': 15, 'machine learning': 12, 'deep learning': 12,
    'neural network': 10, 'transformer': 10, 'llm': 12, 'gpt': 10, 'claude': 8,
    'openai': 10, 'anthropic': 8, 'breakthrough': 12, 'research': 6, 'arxiv': 7,
    'open source': 8, 'safety': 7, 'agi': 10, 'multimodal': 8, 'robotics': 7
}


class NewsCollector:
    def __init__(self, sources_file='config/sources.json'):
        self.sources_file = sources_file
        self.session = None
        self.load_config()
    
    def load_config(self):
        """Load sources configuration."""
        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.sources = config.get('sources', {})
            logger.info(f"Loaded {len(self.sources)} sources")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    async def collect_all(self, source_ids: Optional[List[str]] = None, max_age_days: int = 7) -> List[Dict]:
        """Collect articles from all or specified sources."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'NewsXP.ai/2.0 (+https://newsxp.ai)'}
        )
        
        try:
            # Determine which sources to collect from
            if source_ids:
                sources_to_use = {k: v for k, v in self.sources.items() 
                                if k in source_ids and v.get('enabled', True)}
            else:
                sources_to_use = {k: v for k, v in self.sources.items() 
                                if v.get('enabled', True)}
            
            logger.info(f"Collecting from {len(sources_to_use)} sources")
            
            # Collect articles
            all_articles = []
            successful_sources = 0
            failed_sources = 0
            empty_sources = 0
            source_reasons = {}  # Track reasons for empty sources
            
            for source_id, config in sources_to_use.items():
                try:
                    articles, reason = await self.collect_from_source(source_id, config, max_age_days)
                    all_articles.extend(articles)
                    
                    if len(articles) > 0:
                        successful_sources += 1
                        logger.info(f"Collected {len(articles)} articles from {source_id}")
                    else:
                        empty_sources += 1
                        source_reasons[source_id] = reason
                        logger.info(f"Collected 0 articles from {source_id} - {reason}")
                        
                except Exception as e:
                    failed_sources += 1
                    logger.error(f"Failed to collect from {source_id}: {e}")
            
            # Log collection summary
            logger.info(f"Collection summary: {successful_sources} sources with articles, {empty_sources} sources empty, {failed_sources} sources failed")
            if max_age_days > 0:
                logger.info(f"Age filter applied: only articles newer than {max_age_days} days included")
            
            # Process articles
            unique_articles = self.deduplicate(all_articles)
            scored_articles = self.score_articles(unique_articles)
            scored_articles.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            logger.info(f"Final result: {len(scored_articles)} articles")
            return scored_articles
            
        finally:
            if self.session:
                await self.session.close()
    
    async def collect_from_source(self, source_id: str, config: Dict, max_age_days: int = 7) -> tuple[List[Dict], str]:
        """Collect articles from a single source."""
        source_type = config.get('type', 'rss').lower()
        
        if source_type == 'rss':
            return await self.collect_rss(source_id, config, max_age_days)
        elif source_type == 'api':
            return await self.collect_api(source_id, config, max_age_days)
        elif source_type == 'scrape':
            logger.warning(f"Source type 'scrape' for {source_id} not implemented yet - web scraping functionality planned for future release")
            return [], "feature not implemented (web scraping)"
        elif source_type == 'webhook':
            logger.warning(f"Source type 'webhook' for {source_id} not implemented yet - webhook functionality planned for future release")
            return [], "feature not implemented (webhook)"
        else:
            logger.error(f"Unknown source type '{source_type}' for {source_id} - supported types: rss, api")
            return [], f"unknown source type '{source_type}'"
    
    async def collect_rss(self, source_id: str, config: Dict, max_age_days: int = 7) -> tuple[List[Dict], str]:
        """Collect articles from RSS feed."""
        articles = []
        reason = "unknown"
        
        try:
            if not self.session:
                logger.error(f"No session available for {source_id}")
                return [], "no session available"
                
            url = config['url']
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for RSS {source_id} - {url}")
                    return [], f"HTTP {response.status} error"
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                # Check if feed was parsed successfully
                if hasattr(feed, 'bozo') and feed.bozo:
                    bozo_reason = getattr(feed, 'bozo_exception', 'Unknown parsing error')
                    logger.warning(f"RSS feed parsing issues for {source_id}: {bozo_reason}")
                
                # Check if feed has entries
                if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                    update_freq = config.get('updateFrequency', 'unknown')
                    note = config.get('note', '')
                    
                    if update_freq in ['daily'] and datetime.now().weekday() in [5, 6]:  # Saturday/Sunday
                        reason = "weekend (daily updates)"
                        logger.info(f"No articles from {source_id} - likely due to weekend (updates {update_freq}). {note}")
                    elif update_freq in ['weekdays'] and datetime.now().weekday() in [5, 6]:
                        reason = "weekend (weekdays only)"
                        logger.info(f"No articles from {source_id} - weekdays only source checked on weekend")
                    elif 'midnight' in note.lower() and datetime.now().hour < 8:
                        reason = "early morning (updates at midnight)"
                        logger.info(f"No articles from {source_id} - may not have updated yet (updates at midnight)")
                    elif hasattr(feed, 'bozo') and feed.bozo:
                        reason = "RSS parsing error"
                        logger.warning(f"No articles available from {source_id} - feed has parsing issues")
                    else:
                        reason = "feed empty or unavailable"
                        logger.warning(f"No articles available from {source_id} - feed may be empty or experiencing issues (updates {update_freq})")
                    return [], reason
                
                max_articles = config.get('maxArticles', 20)
                total_entries = len(feed.entries)
                articles_before_age_filter = 0
                
                for entry in feed.entries[:max_articles]:
                    # Count articles before age filtering for logging
                    articles_before_age_filter += 1
                    article = self.parse_rss_entry(source_id, config, feed, entry, max_age_days)
                    if article:
                        articles.append(article)
                
                # Log age filtering results if any articles were filtered
                if max_age_days > 0 and articles_before_age_filter > len(articles):
                    filtered_count = articles_before_age_filter - len(articles)
                    logger.debug(f"RSS {source_id}: filtered {filtered_count} articles older than {max_age_days} days")
                
                # Log detailed info about article parsing
                if len(articles) == 0 and total_entries > 0:
                    reason = "entries could not be parsed"
                    logger.warning(f"RSS {source_id} had {total_entries} entries but none could be parsed into valid articles")
                    return [], reason
                        
        except Exception as e:
            logger.error(f"RSS collection failed for {source_id}: {e}")
            return [], f"collection error: {str(e)[:50]}"
        
        return articles, "success"
    
    def parse_rss_entry(self, source_id: str, config: Dict, feed: Any, entry: Any, max_age_days: int = 7) -> Optional[Dict]:
        """Parse RSS entry into article format."""
        try:
            # Extract basic fields with more fallback options
            title = getattr(entry, 'title', '').strip()
            url = getattr(entry, 'link', '') or getattr(entry, 'guid', '')
            
            # Extract and clean description
            description = ''
            for desc_field in ['summary', 'description', 'content', 'subtitle']:
                if hasattr(entry, desc_field):
                    desc_content = getattr(entry, desc_field)
                    if desc_content:
                        if isinstance(desc_content, list) and len(desc_content) > 0:
                            desc_content = desc_content[0].get('value', '') if isinstance(desc_content[0], dict) else str(desc_content[0])
                        description = BeautifulSoup(str(desc_content), 'html.parser').get_text()
                        break
            description = description[:500].strip()
            
            # Debug logging for failed parsing
            if not title:
                available_attrs = [attr for attr in dir(entry) if not attr.startswith('_')]
                logger.debug(f"RSS {source_id}: Entry missing title. Available attrs: {available_attrs}")
                if logger.isEnabledFor(logging.DEBUG):
                    # Show first few attributes with their values
                    sample_data = {attr: getattr(entry, attr) for attr in available_attrs[:10] if hasattr(entry, attr)}
                    logger.debug(f"RSS {source_id}: Sample entry data: {sample_data}")
            
            if not url:
                logger.debug(f"RSS {source_id}: Entry missing URL. Available attrs: {[attr for attr in dir(entry) if not attr.startswith('_')]}")
            
            # Parse date
            pub_date = self.parse_date(
                getattr(entry, 'published_parsed', None) or 
                getattr(entry, 'updated_parsed', None)
            )
            
            # Apply age filter
            if max_age_days > 0:
                try:
                    article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    if article_date.tzinfo is None:
                        article_date = article_date.replace(tzinfo=timezone.utc)
                    
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                    if article_date < cutoff_date:
                        logger.debug(f"RSS {source_id}: Entry filtered due to age: {article_date} < {cutoff_date}")
                        return None  # Article is too old
                except Exception:
                    # If date parsing fails, keep the article
                    pass
            
            # Generate unique ID
            entry_id = getattr(entry, 'id', url) or str(hash(title))
            article_id = hashlib.md5(f"{source_id}_{entry_id}".encode()).hexdigest()
            
            # Validate required fields
            if not title or not url:
                logger.debug(f"RSS {source_id}: Entry validation failed - title: '{title}', url: '{url}'")
                return None
            
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
    
    async def collect_api(self, source_id: str, config: Dict, max_age_days: int = 7) -> tuple[List[Dict], str]:
        """Collect articles from API endpoint."""
        articles = []
        reason = "unknown"
        
        try:
            if not self.session:
                logger.error(f"No session available for {source_id}")
                return [], "no session available"
                
            url = config['url']
            headers = {'Accept': 'application/json'}
            if 'apiKey' in config:
                headers['Authorization'] = f"Bearer {config['apiKey']}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 403:
                    logger.error(f"HTTP 403 for API {source_id} - Access forbidden. API key may be invalid or expired")
                    return [], "HTTP 403 error - access forbidden"
                elif response.status == 400:
                    logger.error(f"HTTP 400 for API {source_id} - Bad request. Check API parameters or endpoint URL")
                    return [], "HTTP 400 error - bad request"
                elif response.status == 429:
                    logger.error(f"HTTP 429 for API {source_id} - Rate limit exceeded. Try again later")
                    return [], "HTTP 429 error - rate limit exceeded"
                elif response.status != 200:
                    logger.error(f"HTTP {response.status} for API {source_id} - {url}")
                    return [], f"HTTP {response.status} error"
                
                # Check content type before parsing
                content_type = response.headers.get('content-type', '').lower()
                
                try:
                    if 'application/json' in content_type:
                        data = await response.json()
                    elif 'application/atom+xml' in content_type or 'application/xml' in content_type:
                        # Handle XML/Atom feeds that were misconfigured as API endpoints
                        logger.warning(f"API {source_id} returned XML/Atom content instead of JSON. Consider changing type to 'rss'")
                        content = await response.text()
                        feed = feedparser.parse(content)
                        if hasattr(feed, 'entries') and len(feed.entries) > 0:
                            logger.info(f"Successfully parsed XML content for {source_id} using RSS parser")
                            max_articles = config.get('maxArticles', 20)
                            for entry in feed.entries[:max_articles]:
                                article = self.parse_rss_entry(source_id, config, feed, entry, max_age_days)
                                if article:
                                    articles.append(article)
                            return articles, "success"
                        else:
                            logger.warning(f"XML content from {source_id} contained no parseable entries")
                            return [], "XML content contained no parseable entries"
                    else:
                        logger.error(f"API {source_id} returned unexpected content type: {content_type}")
                        return [], f"unexpected content type: {content_type}"
                        
                except Exception as parse_error:
                    logger.error(f"Failed to parse response from API {source_id}: {parse_error}")
                    return [], f"parse error: {str(parse_error)[:50]}"
                
                # Handle different API response formats
                articles_data = data
                if isinstance(data, dict):
                    for key in ['articles', 'items', 'results', 'data', 'entries']:
                        if key in data and isinstance(data[key], list):
                            articles_data = data[key]
                            break
                
                if not isinstance(articles_data, list):
                    logger.error(f"Unexpected API response format for {source_id} - expected array or object with articles array")
                    return [], "unexpected API response format"
                
                # Check if API returned empty results
                if len(articles_data) == 0:
                    update_freq = config.get('updateFrequency', 'unknown')
                    if update_freq in ['daily'] and datetime.now().weekday() in [5, 6]:
                        reason = "weekend (daily updates)"
                        logger.info(f"No articles from API {source_id} - likely due to weekend (updates {update_freq})")
                    elif update_freq in ['weekdays'] and datetime.now().weekday() in [5, 6]:
                        reason = "weekend (weekdays only)"
                        logger.info(f"No articles from API {source_id} - weekdays only source checked on weekend")
                    else:
                        reason = "API returned empty results"
                        logger.info(f"No articles available from API {source_id} - endpoint returned empty results")
                    return [], reason
                
                # Parse articles
                max_articles = config.get('maxArticles', 20)
                total_items = len(articles_data)
                items_before_age_filter = 0
                
                for item in articles_data[:max_articles]:
                    # Count items before age filtering for logging
                    items_before_age_filter += 1
                    article = self.parse_api_item(source_id, config, item, max_age_days)
                    if article:
                        articles.append(article)
                
                # Log age filtering results if any articles were filtered
                if max_age_days > 0 and items_before_age_filter > len(articles):
                    filtered_count = items_before_age_filter - len(articles)
                    logger.debug(f"API {source_id}: filtered {filtered_count} articles older than {max_age_days} days")
                
                # Log detailed info about article parsing
                if len(articles) == 0 and total_items > 0:
                    reason = "items could not be parsed"
                    logger.warning(f"API {source_id} returned {total_items} items but none could be parsed into valid articles")
                    return [], reason
                elif len(articles) > 0:
                    reason = "success"
                        
        except Exception as e:
            logger.error(f"API collection failed for {source_id}: {e}")
            return [], f"collection error: {str(e)[:50]}"
        
        return articles, reason
    
    def parse_api_item(self, source_id: str, config: Dict, item: Dict, max_age_days: int = 7) -> Optional[Dict]:
        """Parse API item into article format."""
        try:
            # Extract basic fields with more fallback options
            title = (item.get('title') or item.get('headline') or item.get('name') or 
                    item.get('display_name') or item.get('full_name') or '')
            
            # Special handling for GitHub API
            if not title and source_id == 'github_openai':
                if item.get('type') == 'PushEvent':
                    commits = item.get('payload', {}).get('commits', [])
                    if commits:
                        title = f"New commits to {item.get('repo', {}).get('name', 'repository')}: {commits[0].get('message', '')[:100]}"
                elif item.get('type') == 'ReleaseEvent':
                    release = item.get('payload', {}).get('release', {})
                    title = f"Release: {release.get('name', release.get('tag_name', 'New Release'))}"
                elif item.get('type'):
                    title = f"{item['type']} in {item.get('repo', {}).get('name', 'repository')}"
            
            url = (item.get('url') or item.get('link') or item.get('permalink') or 
                  item.get('html_url') or item.get('web_url') or 
                  item.get('url_abs') or item.get('url_pdf') or '')
            
            # Special handling for different API formats
            if not url and 'paper' in item:
                if isinstance(item['paper'], dict):
                    # HuggingFace papers API format
                    paper_id = item['paper'].get('id')
                    if paper_id:
                        url = f"https://arxiv.org/abs/{paper_id}"
                        logger.debug(f"API {source_id}: Constructed ArXiv URL from paper ID: {url}")
                    else:
                        paper_url = item['paper'].get('url') or item['paper'].get('link')
                        if paper_url:
                            url = paper_url
                elif isinstance(item['paper'], str) and item['paper'].startswith('http'):
                    url = item['paper']
            
            # Papers with Code API format
            if not url and 'arxiv_id' in item and item['arxiv_id']:
                url = f"https://arxiv.org/abs/{item['arxiv_id']}"
                logger.debug(f"API {source_id}: Constructed ArXiv URL from arxiv_id: {url}")
            
            # GitHub API format - construct URLs from repository info
            if not url and source_id == 'github_openai':
                repo_name = item.get('repo', {}).get('name', '')
                if repo_name:
                    if item.get('type') == 'PushEvent':
                        url = f"https://github.com/{repo_name}/commits"
                    elif item.get('type') == 'ReleaseEvent':
                        url = f"https://github.com/{repo_name}/releases"
                    elif item.get('type') == 'CreateEvent':
                        url = f"https://github.com/{repo_name}"
                    elif item.get('type') == 'IssuesEvent':
                        issue_num = item.get('payload', {}).get('issue', {}).get('number')
                        url = f"https://github.com/{repo_name}/issues/{issue_num}" if issue_num else f"https://github.com/{repo_name}/issues"
                    elif item.get('type') == 'PullRequestEvent':
                        pr_num = item.get('payload', {}).get('pull_request', {}).get('number')
                        url = f"https://github.com/{repo_name}/pull/{pr_num}" if pr_num else f"https://github.com/{repo_name}/pulls"
                    else:
                        # Default to repository URL for other event types
                        url = f"https://github.com/{repo_name}"
                elif 'html_url' in item:
                    url = item['html_url']
            
            description = (item.get('description') or item.get('summary') or 
                          item.get('content') or item.get('body') or 
                          item.get('excerpt') or '')
            
            # Special handling for GitHub API
            if not description and source_id == 'github_openai':
                if item.get('type') == 'PushEvent':
                    commits = item.get('payload', {}).get('commits', [])
                    if commits:
                        description = '\n'.join([f"• {commit.get('message', '')}" for commit in commits[:3]])
                elif item.get('type') == 'ReleaseEvent':
                    release = item.get('payload', {}).get('release', {})
                    description = release.get('body', release.get('notes', ''))
                elif 'payload' in item:
                    description = str(item['payload'])[:200]
            
            # Debug logging for failed parsing
            if not title:
                logger.debug(f"API {source_id}: Item missing title. Available keys: {list(item.keys())}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"API {source_id}: Sample item data: {json.dumps(item, indent=2)[:500]}")
            
            if not url:
                logger.debug(f"API {source_id}: Item missing URL after all extraction attempts. Available keys: {list(item.keys())}")
            
            # Clean HTML from description
            if isinstance(description, dict):
                description = description.get('rendered', str(description))
            if description:
                description = BeautifulSoup(description, 'html.parser').get_text()[:500].strip()
            
            # Parse date with more options
            pub_date = datetime.now(timezone.utc).isoformat()
            for date_field in ['published', 'date', 'publishedAt', 'created_at', 'updated_at', 'submission_date']:
                if date_field in item and item[date_field]:
                    try:
                        pub_date = date_parser.parse(str(item[date_field])).isoformat()
                        break
                    except:
                        continue
            
            # Special date handling for different APIs
            if pub_date == datetime.now(timezone.utc).isoformat():  # No date found yet
                if source_id == 'github_openai' and 'created_at' in item:
                    try:
                        pub_date = date_parser.parse(item['created_at']).isoformat()
                    except:
                        pass
            
            # Apply age filter
            if max_age_days > 0:
                try:
                    article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    if article_date.tzinfo is None:
                        article_date = article_date.replace(tzinfo=timezone.utc)
                    
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                    if article_date < cutoff_date:
                        logger.debug(f"API {source_id}: Item filtered due to age: {article_date} < {cutoff_date}")
                        return None  # Article is too old
                except Exception:
                    # If date parsing fails, keep the article
                    pass
            
            # Generate unique ID
            item_id = item.get('id', item.get('guid', url))
            article_id = hashlib.md5(f"{source_id}_{item_id}".encode()).hexdigest()
            
            # Validate required fields
            if not title or not url:
                logger.debug(f"API {source_id}: Item validation failed - title: '{title}', url: '{url}'")
                return None
            
            return {
                'id': article_id,
                'source_id': source_id,
                'source': config.get('name', source_id) if config else source_id,
                'category': config.get('category', 'Other') if config else 'Other',
                'title': title.strip(),
                'url': url,
                'description': description,
                'published_date': pub_date,
                'collected_at': datetime.now(timezone.utc).isoformat(),
                'author': str(item.get('author', '')).strip(),
                'source_priority': config.get('priority', 5) if config else 5
            }
            
        except Exception as e:
            logger.error(f"Error parsing API item from {source_id}: {e}")
            return None
    
    def parse_date(self, date_tuple) -> str:
        """Parse date tuple or string into ISO format."""
        if date_tuple:
            try:
                if hasattr(date_tuple, 'timetuple'):
                    return date_tuple.isoformat()
                elif isinstance(date_tuple, (tuple, list)) and len(date_tuple) >= 6:
                    from time import mktime
                    # Convert to proper time tuple format for mktime
                    time_tuple = tuple(date_tuple[:9]) if len(date_tuple) >= 9 else tuple(list(date_tuple) + [0] * (9 - len(date_tuple)))
                    return datetime.fromtimestamp(mktime(time_tuple), timezone.utc).isoformat()
                elif isinstance(date_tuple, str):
                    return date_parser.parse(date_tuple).isoformat()
            except:
                pass
        return datetime.now(timezone.utc).isoformat()
    
    def deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles."""
        seen = set()
        unique = []
        
        for article in articles:
            # Create signature based on title and URL
            title_norm = re.sub(r'[^\w\s]', '', article.get('title', '').lower())
            url_norm = article.get('url', '').lower().split('?')[0]
            
            signature = hashlib.md5(f"{title_norm}_{url_norm}".encode()).hexdigest()
            
            if signature not in seen:
                seen.add(signature)
                unique.append(article)
        
        logger.info(f"Deduplicated: {len(articles)} -> {len(unique)} articles")
        return unique
    
    def score_articles(self, articles: List[Dict]) -> List[Dict]:
        """Score articles based on relevance and recency."""
        now = datetime.now(timezone.utc)
        
        for article in articles:
            score = 50  # Base score
            
            # Recency score (max 30 points)
            try:
                pub_date = datetime.fromisoformat(article.get('published_date', '').replace('Z', '+00:00'))
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                
                hours_old = (now - pub_date).total_seconds() / 3600
                recency_score = max(0, 30 * (0.95 ** (hours_old / 24)))
                score += recency_score
            except:
                pass
            
            # Source quality score (max 20 points)
            priority = article.get('source_priority', 5)
            score += max(0, 15 - (priority * 1.5))
            
            # Content relevance score (max 30 points)
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            combined_text = f"{title} {description}"
            
            keyword_score = 0
            for keyword, points in AI_KEYWORDS.items():
                if keyword in combined_text:
                    keyword_score += points
            
            score += min(30, keyword_score)
            
            # Quality bonuses
            if article.get('author'):
                score += 2
            if len(article.get('description', '')) > 100:
                score += 3
            
            article['score'] = max(0, min(100, round(score)))
        
        return articles


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Collect AI news from configured sources')
    parser.add_argument('--sources', help='Comma-separated list of source IDs')
    parser.add_argument('--output', default='data/news.json', help='Output file path')
    parser.add_argument('--archive', action='store_true', help='Save to daily archive')
    parser.add_argument('--config', default='config/sources.json', help='Sources config file')
    parser.add_argument('--max-articles', type=int, default=1000, help='Max articles to save')
    parser.add_argument('--min-score', type=float, default=0, help='Min score threshold')
    parser.add_argument('--max-age-days', type=int, default=7, help='Max age of articles in days (default: 7)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize collector
        collector = NewsCollector(args.config)
        
        # Parse sources
        sources = None
        if args.sources and args.sources.lower() != 'all':
            sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        
        # Collect articles
        start_time = time.time()
        logger.info("Starting news collection...")
        
        articles = await collector.collect_all(sources, max_age_days=args.max_age_days)
        
        collection_time = time.time() - start_time
        logger.info(f"Collection completed in {collection_time:.2f} seconds")
        
        # Apply filters
        if args.min_score > 0:
            articles = [a for a in articles if a.get('score', 0) >= args.min_score]
            logger.info(f"After score filtering: {len(articles)} articles")
        
        # Limit articles
        if len(articles) > args.max_articles:
            articles = articles[:args.max_articles]
            logger.info(f"Limited to top {args.max_articles} articles")

        # Sort articles by latest date
        articles = sorted(articles, key=lambda x: x.get('published_date', ''), reverse=True)

        # Prepare output
        output_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'collection_time_seconds': round(collection_time, 2),
            'count': len(articles),
            'articles': articles
        }
        
        # Save main output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {output_path}")
        
        # Save archive if requested
        if args.archive:
            archive_dir = Path('data/archive')
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            daily_archive = archive_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"
            with open(daily_archive, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Archived to {daily_archive}")
        
        # Print summary for GitHub Actions
        print(json.dumps({
            'success': True,
            'articles_collected': len(articles),
            'collection_time': round(collection_time, 2),
            'output_file': str(output_path),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        print(json.dumps({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())
