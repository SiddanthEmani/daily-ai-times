#!/usr/bin/env python3
"""
Article repository for database operations
Handles CRUD operations with minimal schema and lifecycle management
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from .connection import DatabaseManager

logger = logging.getLogger(__name__)

class ArticleRepository:
    """Repository for article database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize article repository"""
        self.db_manager = db_manager
    
    def save_new_articles(self, articles: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Save only new articles (not already in database by URL) using optimized bulk operations
        Returns: (inserted_count, duplicate_count)
        """
        if not articles:
            return 0, 0
        
        # Pre-filter obvious duplicates within the batch
        articles = self._deduplicate_articles(articles)
        
        # Use bulk operations for better performance
        if len(articles) > 100:  # Use bulk method for large datasets
            return self._save_articles_bulk(articles)
        else:
            return self._save_articles_batch(articles)
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles within the batch based on URL"""
        seen_urls = set()
        deduplicated = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(article)
        
        if len(deduplicated) < len(articles):
            logger.info(f"Pre-filtered {len(articles) - len(deduplicated)} internal duplicates")
        
        return deduplicated
    
    def _save_articles_bulk(self, articles: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Optimized bulk save using ON CONFLICT for large article sets
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Prepare bulk data
                    article_data = []
                    for article in articles:
                        # Parse published_date if it's a string
                        published_date = article.get('published_date')
                        if isinstance(published_date, str):
                            try:
                                from datetime import datetime as dt
                                published_date = dt.fromisoformat(published_date.replace('Z', '+00:00'))
                            except ValueError:
                                published_date = None
                        
                        article_data.append((
                            article.get('url', ''),
                            article.get('title', '')[:1000],  # Truncate to fit VARCHAR(1000)
                            article.get('description', ''),
                            article.get('author', '')[:500] if article.get('author') else None,
                            published_date
                        ))
                    
                    # Get count before insertion for duplicate calculation
                    cursor.execute("SELECT COUNT(*) FROM articles WHERE url = ANY(%s)", 
                                 ([data[0] for data in article_data],))
                    existing_count = cursor.fetchone()[0]
                    
                    # Bulk insert with ON CONFLICT DO NOTHING for duplicate handling
                    insert_sql = """
                    INSERT INTO articles (url, title, description, author, published_date, created_at)
                    VALUES %s
                    ON CONFLICT (url) DO NOTHING
                    """
                    
                    # Use execute_values for efficient bulk insert
                    from psycopg2.extras import execute_values
                    from datetime import datetime, timezone
                    current_time = datetime.now(timezone.utc)
                    
                    execute_values(
                        cursor, 
                        insert_sql,
                        [(data[0], data[1], data[2], data[3], data[4], current_time) for data in article_data],
                        template=None,
                        page_size=1000
                    )
                    
                    inserted_count = cursor.rowcount
                    duplicate_count = len(articles) - inserted_count
                    
                    conn.commit()
                    logger.info(f"Bulk article save complete: {inserted_count} new, {duplicate_count} duplicates")
                    
        except Exception as e:
            logger.error(f"Failed to bulk save articles: {e}")
            # Fallback to batch method
            logger.info("Falling back to batch method...")
            return self._save_articles_batch(articles)
        
        return inserted_count, duplicate_count
    
    def _save_articles_batch(self, articles: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Optimized batch save using executemany for smaller article sets
        """
        inserted_count = 0
        duplicate_count = 0
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Prepare batch data
                    article_data = []
                    for article in articles:
                        # Parse published_date if it's a string
                        published_date = article.get('published_date')
                        if isinstance(published_date, str):
                            try:
                                from datetime import datetime as dt
                                published_date = dt.fromisoformat(published_date.replace('Z', '+00:00'))
                            except ValueError:
                                published_date = None
                        
                        article_data.append((
                            article.get('url', ''),
                            article.get('title', '')[:1000],
                            article.get('description', ''),
                            article.get('author', '')[:500] if article.get('author') else None,
                            published_date
                        ))
                    
                    # Batch insert with ON CONFLICT
                    insert_sql = """
                    INSERT INTO articles (url, title, description, author, published_date, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (url) DO NOTHING
                    """
                    
                    cursor.executemany(insert_sql, article_data)
                    inserted_count = cursor.rowcount
                    duplicate_count = len(articles) - inserted_count
                    
                    conn.commit()
                    logger.info(f"Batch article save complete: {inserted_count} new, {duplicate_count} duplicates")
                    
        except Exception as e:
            logger.error(f"Failed to batch save articles: {e}")
            raise
        
        return inserted_count, duplicate_count
    
    def get_unprocessed_articles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get articles that haven't been processed by bulk agents (is_ai_related IS NULL)"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                    SELECT id, url, title, description, author, published_date, created_at
                    FROM articles 
                    WHERE is_ai_related IS NULL
                    ORDER BY created_at ASC
                    """
                    
                    if limit:
                        sql += f" LIMIT {limit}"
                    
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    
                    articles = []
                    for row in rows:
                        articles.append({
                            'id': row[0],
                            'url': row[1],
                            'title': row[2],
                            'description': row[3],
                            'author': row[4],
                            'published_date': row[5].isoformat() if row[5] else None,
                            'created_at': row[6].isoformat() if row[6] else None
                        })
                    
                    logger.info(f"Retrieved {len(articles)} unprocessed articles")
                    return articles
                    
        except Exception as e:
            logger.error(f"Failed to get unprocessed articles: {e}")
            raise
    
    def update_bulk_results(self, article_updates: List[Tuple[int, bool]]) -> int:
        """
        Update articles with bulk processing results
        article_updates: List of (article_id, is_ai_related) tuples
        Returns: number of updated articles
        """
        if not article_updates:
            return 0
        
        updated_count = 0
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    update_sql = """
                    UPDATE articles 
                    SET is_ai_related = %s
                    WHERE id = %s
                    """
                    
                    for article_id, is_ai_related in article_updates:
                        try:
                            cursor.execute(update_sql, (is_ai_related, article_id))
                            if cursor.rowcount > 0:
                                updated_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to update article {article_id}: {e}")
                    
                    conn.commit()
                    logger.info(f"Updated {updated_count} articles with bulk results")
                    
        except Exception as e:
            logger.error(f"Failed to update bulk results: {e}")
            raise
        
        return updated_count
    
    def get_ai_related_articles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get articles marked as AI-related by bulk processing"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                    SELECT id, url, title, description, author, published_date, created_at, category
                    FROM articles 
                    WHERE is_ai_related = true
                    ORDER BY published_date DESC
                    """
                    
                    if limit:
                        sql += f" LIMIT {limit}"
                    
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    
                    articles = []
                    for row in rows:
                        articles.append({
                            'id': row[0],
                            'url': row[1],
                            'title': row[2],
                            'description': row[3],
                            'author': row[4],
                            'published_date': row[5].isoformat() if row[5] else None,
                            'created_at': row[6].isoformat() if row[6] else None,
                            'category': row[7]
                        })
                    
                    logger.info(f"Retrieved {len(articles)} AI-related articles")
                    return articles
                    
        except Exception as e:
            logger.error(f"Failed to get AI-related articles: {e}")
            raise

    def update_article_categories(self, updates: List[Tuple[int, str]]) -> int:
        """Batch update categories for articles. Expects list of (id, category)."""
        if not updates:
            return 0
        updated = 0
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                    UPDATE articles
                    SET category = %s
                    WHERE id = %s
                    """
                    for article_id, category in updates:
                        try:
                            cursor.execute(sql, (category, article_id))
                            if cursor.rowcount > 0:
                                updated += 1
                        except Exception as e:
                            logger.warning(f"Failed to update category for article {article_id}: {e}")
                    conn.commit()
            logger.info(f"Updated categories for {updated} articles")
            return updated
        except Exception as e:
            logger.error(f"Failed to update article categories: {e}")
            raise

    def cleanup_old_articles(self, days_old: int = 7) -> Dict[str, int]:
        """
        Delete articles older than specified days
        Returns: cleanup statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # First, count articles to be deleted
                    # Use published_date for age determination; fallback to created_at when published_date is NULL
                    count_sql = """
                    SELECT COUNT(*) FROM articles 
                    WHERE COALESCE(published_date, created_at) < NOW() - INTERVAL '%s days'
                    """
                    cursor.execute(count_sql, (days_old,))
                    articles_to_delete = cursor.fetchone()[0]
                    
                    if articles_to_delete == 0:
                        logger.info("No old articles to cleanup")
                        return {'deleted_count': 0, 'days_old': days_old}
                    
                    # Delete old articles
                    delete_sql = """
                    DELETE FROM articles 
                    WHERE COALESCE(published_date, created_at) < NOW() - INTERVAL '%s days'
                    """
                    cursor.execute(delete_sql, (days_old,))
                    deleted_count = cursor.rowcount
                    
                    conn.commit()
                    
                    cleanup_stats = {
                        'deleted_count': deleted_count,
                        'days_old': days_old
                    }
                    
                    logger.info(f"Cleanup complete: deleted {deleted_count} articles older than {days_old} days")
                    return cleanup_stats
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old articles: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    stats_sql = """
                    SELECT 
                        COUNT(*) as total_articles,
                        COUNT(CASE WHEN is_ai_related IS NULL THEN 1 END) as unprocessed,
                        COUNT(CASE WHEN is_ai_related = true THEN 1 END) as ai_related,
                        COUNT(CASE WHEN is_ai_related = false THEN 1 END) as not_ai_related
                    FROM articles
                    """
                    
                    cursor.execute(stats_sql)
                    row = cursor.fetchone()
                    
                    stats = {
                        'total_articles': row[0],
                        'unprocessed': row[1],
                        'ai_related': row[2],
                        'not_ai_related': row[3]
                    }
                    
                    return stats
                    
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise


class SourceRepository:
    """Repository for source (RSS) database operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def upsert_many(self, sources: List[Dict[str, Any]]) -> int:
        """Upsert many sources by unique constraints.
        Strategy: conflict target is URL; we update description/enabled, keep existing name stable.
        Returns inserted/updated count.
        """
        if not sources:
            return 0
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Normalize and deduplicate by URL (keep first occurrence's name)
                    url_to_row: Dict[str, Tuple[str, str, Optional[str], bool]] = {}
                    for src in sources:
                        name = (src.get('name') or '').strip()
                        url = (src.get('url') or '').strip()
                        if not name or not url:
                            continue
                        if url in url_to_row:
                            # Skip duplicates by URL to avoid ON CONFLICT churn
                            continue
                        description = (src.get('description') or None)
                        enabled = bool(src.get('enabled', True))
                        url_to_row[url] = (name, url, description, enabled)

                    rows: List[Tuple[str, str, Optional[str], bool]] = list(url_to_row.values())

                    if not rows:
                        return 0

                    # Upsert using ON CONFLICT on URL.
                    # Do not change name on conflict to preserve existing identifier.
                    sql = (
                        "INSERT INTO sources (name, url, description, enabled, created_at, updated_at) "
                        "VALUES %s "
                        "ON CONFLICT (url) DO UPDATE SET "
                        "description = EXCLUDED.description, enabled = EXCLUDED.enabled, updated_at = NOW()"
                    )

                    from psycopg2.extras import execute_values
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    execute_values(
                        cursor,
                        sql,
                        [(n, u, d, e, now, now) for (n, u, d, e) in rows],
                        template=None,
                        page_size=500
                    )
                    affected = cursor.rowcount
                    conn.commit()
                    logger.info(f"Upserted {affected} sources")
                    return affected
        except Exception as e:
            logger.error(f"Failed to upsert sources: {e}")
            raise

    def get_sources(self, enabled_only: bool = True) -> Dict[str, Dict[str, Any]]:
        """Return sources keyed by name in the shape collectors expect."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "SELECT name, url, description, enabled FROM sources"
                    if enabled_only:
                        sql += " WHERE enabled = TRUE"
                    sql += " ORDER BY name ASC"
                    cursor.execute(sql)
                    rows = cursor.fetchall()

                    result: Dict[str, Dict[str, Any]] = {}
                    for name, url, description, enabled in rows:
                        result[name] = {
                            'name': name,
                            'url': url,
                            'description': description or '',
                            'enabled': bool(enabled),
                        }
                    return result
        except Exception as e:
            logger.error(f"Failed to load sources: {e}")
            raise

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable/disable a source by name."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE sources SET enabled = %s, updated_at = NOW() WHERE name = %s",
                        (enabled, name),
                    )
                    updated = cursor.rowcount > 0
                    conn.commit()
                    return updated
        except Exception as e:
            logger.error(f"Failed to update source enabled state: {e}")
            raise
