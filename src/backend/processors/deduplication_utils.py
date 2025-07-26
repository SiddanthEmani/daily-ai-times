#!/usr/bin/env python3
"""
Ultra-Fast Deduplication Utilities

Optimized for processing 1500+ articles in under 10 seconds.
Uses hash-based lookups and minimal string operations for maximum performance.
"""

import re
import hashlib
import logging
from typing import List, Dict, Set, Tuple
from functools import lru_cache
from datetime import datetime, timezone
import time

logger = logging.getLogger(__name__)

class UltraFastDeduplicator:
    """
    Ultra-fast article deduplication optimized for large datasets.
    Achieves O(n) complexity using hash-based lookups.
    """
    
    def __init__(self):
        """Initialize the ultra-fast deduplicator."""
        self.seen_content_hashes: Set[str] = set()
        self.seen_url_hashes: Set[str] = set()
        self.seen_title_hashes: Set[str] = set()
        self.title_similarity_threshold = 0.85
        
    @lru_cache(maxsize=2048)
    def _fast_normalize_title(self, title: str) -> str:
        """Ultra-fast title normalization with minimal operations."""
        if not title:
            return ""
        
        # Single-pass normalization: lowercase + strip + basic cleanup
        normalized = title.lower().strip()
        
        # Remove common prefixes in one pass
        prefixes = ['breaking:', 'news:', 'update:', 'exclusive:', 'new:', 'latest:',
                   'ai breakthrough:', 'research:', 'study:', 'report:']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
        
        # Fast regex cleanup: remove special chars and normalize whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @lru_cache(maxsize=2048)
    def _fast_normalize_url(self, url: str) -> str:
        """Ultra-fast URL normalization."""
        if not url:
            return ""
        
        # Single-pass URL normalization
        normalized = url.lower().split('?')[0].split('#')[0].rstrip('/')
        
        # Remove www. prefix for consistency
        if normalized.startswith('www.'):
            normalized = normalized[4:]
        
        return normalized
    
    def _fast_content_hash(self, article: Dict) -> str:
        """Generate fast content hash using key fields only."""
        title = self._fast_normalize_title(article.get('title', ''))
        url = self._fast_normalize_url(article.get('url', ''))
        
        # Use only essential fields for speed
        content_string = f"{title}|{url}"
        return hashlib.md5(content_string.encode()).hexdigest()
    
    def _fast_title_hash(self, title: str) -> str:
        """Generate fast title hash for similarity detection."""
        normalized = self._fast_normalize_title(title)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _fast_url_hash(self, url: str) -> str:
        """Generate fast URL hash."""
        normalized = self._fast_normalize_url(url)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _is_valid_article_fast(self, article: Dict) -> bool:
        """Fast validity check with minimal operations."""
        # Quick field existence check
        if not all(article.get(field) for field in ['title', 'url']):
            return False
        
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Fast length checks
        if len(title) < 10 or len(description) < 20:
            return False
        
        return True
    
    def _clean_articles_fast(self, articles: List[Dict]) -> List[Dict]:
        """Ultra-fast article cleaning."""
        clean_articles = []
        
        for article in articles:
            # Skip articles with only processed_at field
            if len(article.keys()) <= 2 and 'processed_at' in article:
                continue
            
            # Fast validity check
            if self._is_valid_article_fast(article):
                clean_articles.append(article)
        
        return clean_articles
    
    def deduplicate_articles_ultra_fast(self, articles: List[Dict]) -> List[Dict]:
        """
        Ultra-fast deduplication using hash-based lookups.
        Achieves O(n) complexity for 1500+ articles in under 10 seconds.
        """
        if not articles:
            return []
        
        start_time = time.time()
        logger.info(f"Starting ultra-fast deduplication of {len(articles)} articles")
        
        # Step 1: Fast cleaning
        clean_articles = self._clean_articles_fast(articles)
        logger.debug(f"Cleaned {len(articles)} -> {len(clean_articles)} articles")
        
        # Step 2: Ultra-fast hash-based deduplication
        unique_articles = []
        seen_content_hashes = set()
        seen_url_hashes = set()
        seen_title_hashes = set()
        
        for article in clean_articles:
            # Generate fast hashes
            content_hash = self._fast_content_hash(article)
            url_hash = self._fast_url_hash(article.get('url', ''))
            title_hash = self._fast_title_hash(article.get('title', ''))
            
            # Check for duplicates using hash lookups (O(1) each)
            if (content_hash in seen_content_hashes or 
                url_hash in seen_url_hashes or 
                title_hash in seen_title_hashes):
                continue
            
            # Article is unique - add to seen sets and result
            seen_content_hashes.add(content_hash)
            seen_url_hashes.add(url_hash)
            seen_title_hashes.add(title_hash)
            unique_articles.append(article)
        
        duration = time.time() - start_time
        removed_count = len(clean_articles) - len(unique_articles)
        
        logger.info(f"Ultra-fast deduplication complete: {len(clean_articles)} -> {len(unique_articles)} "
                   f"({removed_count} duplicates removed) in {duration:.3f}s")
        
        return unique_articles

# Legacy class for backward compatibility
class ArticleDeduplicator:
    """
    Legacy deduplicator - now uses ultra-fast implementation internally.
    """
    
    def __init__(self):
        """Initialize the deduplicator."""
        self.seen_signatures: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
        self.title_similarity_threshold = 0.85
        self._ultra_fast_deduplicator = UltraFastDeduplicator()
        
    @lru_cache(maxsize=512)
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        return self._ultra_fast_deduplicator._fast_normalize_title(title)
    
    @lru_cache(maxsize=512)
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        return self._ultra_fast_deduplicator._fast_normalize_url(url)
    
    def _get_content_hash(self, article: Dict) -> str:
        """Generate content-based hash for deeper deduplication."""
        return self._ultra_fast_deduplicator._fast_content_hash(article)
    
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
        import difflib
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= self.title_similarity_threshold
    
    def _is_valid_article(self, article: Dict) -> bool:
        """Check if an article has minimum required fields."""
        return self._ultra_fast_deduplicator._is_valid_article_fast(article)
    
    def _clean_incomplete_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove articles that failed processing or are incomplete."""
        return self._ultra_fast_deduplicator._clean_articles_fast(articles)
    
    def deduplicate_articles(self, articles: List[Dict], strategy: str = "enhanced") -> List[Dict]:
        """
        Deduplicate articles using ultra-fast implementation.
        
        Args:
            articles: List of article dictionaries
            strategy: "basic", "enhanced", or "strict" (all use ultra-fast now)
        
        Returns:
            List of unique articles
        """
        # All strategies now use the ultra-fast implementation
        return self._ultra_fast_deduplicator.deduplicate_articles_ultra_fast(articles)
    
    def _deduplicate_basic(self, articles: List[Dict]) -> List[Dict]:
        """Basic deduplication using ultra-fast implementation."""
        return self._ultra_fast_deduplicator.deduplicate_articles_ultra_fast(articles)
    
    def _deduplicate_enhanced(self, articles: List[Dict]) -> List[Dict]:
        """Enhanced deduplication using ultra-fast implementation."""
        return self._ultra_fast_deduplicator.deduplicate_articles_ultra_fast(articles)
    
    def _deduplicate_strict(self, articles: List[Dict]) -> List[Dict]:
        """Strict deduplication using ultra-fast implementation."""
        return self._ultra_fast_deduplicator.deduplicate_articles_ultra_fast(articles)
    
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
    Convenience function to deduplicate articles using ultra-fast implementation.
    
    Args:
        articles: List of article dictionaries
        strategy: "basic", "enhanced", or "strict" (all use ultra-fast now)
    
    Returns:
        List of unique articles
    """
    deduplicator = UltraFastDeduplicator()
    return deduplicator.deduplicate_articles_ultra_fast(articles)
