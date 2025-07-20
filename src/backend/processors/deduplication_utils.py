#!/usr/bin/env python3
"""
Deduplication Utilities

Shared utilities for deduplicating articles across the Daily AI Times pipeline.
Provides more robust deduplication than simple title+URL matching.
"""

import re
import hashlib
import logging
from typing import List, Dict, Set, Tuple
from functools import lru_cache
from datetime import datetime, timezone
import difflib

logger = logging.getLogger(__name__)

class ArticleDeduplicator:
    """
    Enhanced article deduplication with multiple strategies.
    """
    
    def __init__(self):
        """Initialize the deduplicator."""
        self.seen_signatures: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
        self.title_similarity_threshold = 0.85
        
    @lru_cache(maxsize=512)
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower().strip()
        
        # Remove common prefixes/suffixes that don't affect meaning
        prefixes_to_remove = [
            'breaking:', 'news:', 'update:', 'exclusive:', 'new:', 'latest:',
            'ai breakthrough:', 'research:', 'study:', 'report:'
        ]
        
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
        
        # Remove special characters and extra whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @lru_cache(maxsize=512)
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        if not url:
            return ""
        
        # Remove query parameters and fragments
        normalized = url.lower().split('?')[0].split('#')[0]
        
        # Remove trailing slashes
        normalized = normalized.rstrip('/')
        
        # Remove common tracking parameters
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'ref', 'source']
        if '?' in url:
            base_url, params = url.split('?', 1)
            param_pairs = params.split('&')
            filtered_params = [
                pair for pair in param_pairs 
                if not any(pair.startswith(f'{track}=') for track in tracking_params)
            ]
            if filtered_params:
                normalized = f"{base_url}?{'&'.join(filtered_params)}"
            else:
                normalized = base_url
        
        return normalized
    
    def _get_content_hash(self, article: Dict) -> str:
        """Generate content-based hash for deeper deduplication."""
        # Core content fields that define uniqueness
        title = self._normalize_title(article.get('title', ''))
        description = article.get('description', '').lower().strip()
        url = self._normalize_url(article.get('url', ''))
        
        # For very similar articles, also consider source and date
        source = article.get('source', '').lower()
        published_date = article.get('published_date', '')
        
        # Create content signature
        content_string = f"{title}|{description[:200]}|{url}|{source}|{published_date[:10]}"
        return hashlib.sha256(content_string.encode()).hexdigest()
    
    @lru_cache(maxsize=512)
    def _get_article_signature(self, title: str, url: str) -> str:
        """Generate article signature for basic deduplication."""
        title_norm = self._normalize_title(title)
        url_norm = self._normalize_url(url)
        return hashlib.md5(f"{title_norm}_{url_norm}".encode()).hexdigest()
    
    def _are_titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are semantically similar."""
        if not title1 or not title2:
            return False
        
        norm1 = self._normalize_title(title1)
        norm2 = self._normalize_title(title2)
        
        if norm1 == norm2:
            return True
        
        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= self.title_similarity_threshold
    
    def _is_valid_article(self, article: Dict) -> bool:
        """Check if an article has minimum required fields."""
        required_fields = ['title', 'url', 'description']
        
        for field in required_fields:
            if not article.get(field) or not str(article[field]).strip():
                return False
        
        # Check for minimum content quality
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Filter out very short or low-quality content
        if len(title) < 10 or len(description) < 20:
            return False
        
        # Filter out common non-article content
        non_article_indicators = [
            'error:', 'page not found', '404', 'access denied',
            'coming soon', 'under construction', 'placeholder'
        ]
        
        combined_text = f"{title} {description}".lower()
        if any(indicator in combined_text for indicator in non_article_indicators):
            return False
        
        return True
    
    def _clean_incomplete_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove articles that failed processing or are incomplete."""
        clean_articles = []
        
        for article in articles:
            # Skip articles that only have processed_at but no content
            if (len(article.keys()) <= 2 and 
                'processed_at' in article and 
                not article.get('title')):
                logger.debug(f"Removing incomplete article: {article}")
                continue
            
            # Skip articles that don't meet minimum quality requirements
            if not self._is_valid_article(article):
                logger.debug(f"Removing invalid article: {article.get('title', 'NO_TITLE')}")
                continue
            
            clean_articles.append(article)
        
        removed_count = len(articles) - len(clean_articles)
        if removed_count > 0:
            logger.info(f"Cleaned {removed_count} incomplete/invalid articles")
        
        return clean_articles
    
    def deduplicate_articles(self, articles: List[Dict], strategy: str = "enhanced") -> List[Dict]:
        """
        Deduplicate articles using specified strategy.
        
        Args:
            articles: List of article dictionaries
            strategy: "basic", "enhanced", or "strict"
                - basic: title + URL matching
                - enhanced: content-based with similarity matching
                - strict: enhanced + cross-reference checking
        
        Returns:
            List of unique articles
        """
        if not articles:
            return []
        
        logger.info(f"Deduplicating {len(articles)} articles using {strategy} strategy")
        
        # First, clean incomplete articles
        clean_articles = self._clean_incomplete_articles(articles)
        
        if strategy == "basic":
            return self._deduplicate_basic(clean_articles)
        elif strategy == "enhanced":
            return self._deduplicate_enhanced(clean_articles)
        elif strategy == "strict":
            return self._deduplicate_strict(clean_articles)
        else:
            logger.warning(f"Unknown deduplication strategy: {strategy}. Using enhanced.")
            return self._deduplicate_enhanced(clean_articles)
    
    def _deduplicate_basic(self, articles: List[Dict]) -> List[Dict]:
        """Basic deduplication using title + URL signatures."""
        seen_signatures: Set[str] = set()
        unique_articles = []
        
        for article in articles:
            signature = self._get_article_signature(
                article.get('title', ''), 
                article.get('url', '')
            )
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_articles.append(article)
        
        removed_count = len(articles) - len(unique_articles)
        logger.info(f"Basic deduplication: {len(articles)} -> {len(unique_articles)} ({removed_count} duplicates removed)")
        return unique_articles
    
    def _deduplicate_enhanced(self, articles: List[Dict]) -> List[Dict]:
        """Enhanced deduplication using content hashing and similarity."""
        seen_content_hashes: Set[str] = set()
        seen_titles: List[Tuple[str, int]] = []  # (normalized_title, index)
        unique_articles = []
        
        for i, article in enumerate(articles):
            # Check content hash first
            content_hash = self._get_content_hash(article)
            if content_hash in seen_content_hashes:
                logger.debug(f"Duplicate content hash: {article.get('title', 'NO_TITLE')}")
                continue
            
            # Check title similarity
            current_title = self._normalize_title(article.get('title', ''))
            is_similar = False
            
            for seen_title, _ in seen_titles:
                if self._are_titles_similar(current_title, seen_title):
                    logger.debug(f"Similar title found: '{current_title}' ~ '{seen_title}'")
                    is_similar = True
                    break
            
            if is_similar:
                continue
            
            # Article is unique
            seen_content_hashes.add(content_hash)
            seen_titles.append((current_title, i))
            unique_articles.append(article)
        
        removed_count = len(articles) - len(unique_articles)
        logger.info(f"Enhanced deduplication: {len(articles)} -> {len(unique_articles)} ({removed_count} duplicates removed)")
        return unique_articles
    
    def _deduplicate_strict(self, articles: List[Dict]) -> List[Dict]:
        """Strict deduplication with cross-reference checking."""
        # Start with enhanced deduplication
        enhanced_unique = self._deduplicate_enhanced(articles)
        
        # Additional strict checks
        strict_unique = []
        seen_urls = set()
        
        for article in enhanced_unique:
            url = self._normalize_url(article.get('url', ''))
            
            # Check for URL variations that might be the same article
            url_variations = [
                url,
                url.replace('www.', ''),
                f"www.{url}" if not url.startswith('www.') else url,
                url.replace('http://', 'https://'),
                url.replace('https://', 'http://')
            ]
            
            is_duplicate_url = any(variation in seen_urls for variation in url_variations)
            
            if not is_duplicate_url:
                for variation in url_variations:
                    seen_urls.add(variation)
                strict_unique.append(article)
            else:
                logger.debug(f"Duplicate URL variation: {url}")
        
        removed_count = len(enhanced_unique) - len(strict_unique)
        if removed_count > 0:
            logger.info(f"Strict deduplication: {len(enhanced_unique)} -> {len(strict_unique)} ({removed_count} additional duplicates removed)")
        
        return strict_unique
    
    def get_deduplication_stats(self, original_count: int, final_count: int) -> Dict:
        """Generate deduplication statistics."""
        removed_count = original_count - final_count
        removal_rate = (removed_count / original_count * 100) if original_count > 0 else 0
        
        return {
            'original_count': original_count,
            'final_count': final_count,
            'removed_count': removed_count,
            'removal_rate_percent': round(removal_rate, 2),
            'deduplication_efficiency': f"{final_count}/{original_count} articles kept"
        }

# Convenience function for easy imports
def deduplicate_articles(articles: List[Dict], strategy: str = "enhanced") -> List[Dict]:
    """
    Convenience function to deduplicate articles.
    
    Args:
        articles: List of article dictionaries
        strategy: "basic", "enhanced", or "strict"
    
    Returns:
        List of unique articles
    """
    deduplicator = ArticleDeduplicator()
    return deduplicator.deduplicate_articles(articles, strategy)
