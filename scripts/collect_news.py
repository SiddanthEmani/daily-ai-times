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
from datetime import datetime, timezone
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
    
    async def collect_all(self, source_ids: Optional[List[str]] = None) -> List[Dict]:
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
            for source_id, config in sources_to_use.items():
                try:
                    articles = await self.collect_from_source(source_id, config)
                    all_articles.extend(articles)
                    logger.info(f"Collected {len(articles)} articles from {source_id}")
                except Exception as e:
                    logger.error(f"Failed to collect from {source_id}: {e}")
            
            # Process articles
            unique_articles = self.deduplicate(all_articles)
            scored_articles = self.score_articles(unique_articles)
            scored_articles.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            logger.info(f"Final result: {len(scored_articles)} articles")
            return scored_articles
            
        finally:
            if self.session:
                await self.session.close()
    
    async def collect_from_source(self, source_id: str, config: Dict) -> List[Dict]:
        """Collect articles from a single source."""
        source_type = config.get('type', 'rss').lower()
        
        if source_type == 'rss':
            return await self.collect_rss(source_id, config)
        elif source_type == 'api':
            return await self.collect_api(source_id, config)
        else:
            logger.error(f"Unknown source type '{source_type}' for {source_id}")
            return []
    
    async def collect_rss(self, source_id: str, config: Dict) -> List[Dict]:
        """Collect articles from RSS feed."""
        articles = []
        
        try:
            if not self.session:
                logger.error(f"No session available for {source_id}")
                return []
                
            url = config['url']
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for {source_id}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                max_articles = config.get('maxArticles', 20)
                for entry in feed.entries[:max_articles]:
                    article = self.parse_rss_entry(source_id, config, feed, entry)
                    if article:
                        articles.append(article)
                        
        except Exception as e:
            logger.error(f"RSS collection failed for {source_id}: {e}")
        
        return articles
    
    def parse_rss_entry(self, source_id: str, config: Dict, feed: Any, entry: Any) -> Optional[Dict]:
        """Parse RSS entry into article format."""
        try:
            # Extract basic fields
            title = getattr(entry, 'title', '').strip()
            url = getattr(entry, 'link', '')
            
            # Extract and clean description
            description = ''
            if hasattr(entry, 'summary'):
                description = BeautifulSoup(entry.summary, 'html.parser').get_text()
            elif hasattr(entry, 'description'):
                description = BeautifulSoup(entry.description, 'html.parser').get_text()
            description = description[:500].strip()
            
            # Parse date
            pub_date = self.parse_date(
                getattr(entry, 'published_parsed', None) or 
                getattr(entry, 'updated_parsed', None)
            )
            
            # Generate unique ID
            entry_id = getattr(entry, 'id', url) or str(hash(title))
            article_id = hashlib.md5(f"{source_id}_{entry_id}".encode()).hexdigest()
            
            # Validate required fields
            if not title or not url:
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
    
    async def collect_api(self, source_id: str, config: Dict) -> List[Dict]:
        """Collect articles from API endpoint."""
        articles = []
        
        try:
            if not self.session:
                logger.error(f"No session available for {source_id}")
                return []
                
            url = config['url']
            headers = {'Accept': 'application/json'}
            if 'apiKey' in config:
                headers['Authorization'] = f"Bearer {config['apiKey']}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for API {source_id}")
                    return []
                
                data = await response.json()
                
                # Handle different API response formats
                articles_data = data
                if isinstance(data, dict):
                    for key in ['articles', 'items', 'results', 'data']:
                        if key in data and isinstance(data[key], list):
                            articles_data = data[key]
                            break
                
                if not isinstance(articles_data, list):
                    logger.error(f"Unexpected API response format for {source_id}")
                    return []
                
                # Parse articles
                max_articles = config.get('maxArticles', 20)
                for item in articles_data[:max_articles]:
                    article = self.parse_api_item(source_id, config, item)
                    if article:
                        articles.append(article)
                        
        except Exception as e:
            logger.error(f"API collection failed for {source_id}: {e}")
        
        return articles
    
    def parse_api_item(self, source_id: str, config: Dict, item: Dict) -> Optional[Dict]:
        """Parse API item into article format."""
        try:
            # Extract basic fields
            title = item.get('title') or item.get('headline') or item.get('name', '')
            url = item.get('url') or item.get('link') or item.get('permalink', '')
            description = item.get('description') or item.get('summary') or item.get('content', '')
            
            # Clean HTML from description
            if isinstance(description, dict):
                description = description.get('rendered', str(description))
            if description:
                description = BeautifulSoup(description, 'html.parser').get_text()[:500].strip()
            
            # Parse date
            pub_date = datetime.now(timezone.utc).isoformat()
            for date_field in ['published', 'date', 'publishedAt', 'created_at']:
                if date_field in item and item[date_field]:
                    try:
                        pub_date = date_parser.parse(str(item[date_field])).isoformat()
                        break
                    except:
                        continue
            
            # Generate unique ID
            item_id = item.get('id', item.get('guid', url))
            article_id = hashlib.md5(f"{source_id}_{item_id}".encode()).hexdigest()
            
            # Validate required fields
            if not title or not url:
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
    parser.add_argument('--max-articles', type=int, default=100, help='Max articles to save')
    parser.add_argument('--min-score', type=float, default=0, help='Min score threshold')
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
        
        articles = await collector.collect_all(sources)
        
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
