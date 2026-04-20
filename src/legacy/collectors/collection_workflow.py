#!/usr/bin/env python3
"""
Standalone Article Collection Workflow
Runs independently every 4 hours to collect articles and store them in the database.
This workflow is separate from the main orchestrator and focuses only on collection.
"""

import sys
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to Python path to enable proper imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modules using the proper src.* paths
from src.backend.collectors.collectors import NewsCollector, Config as CollectorConfig
from src.backend.database.connection import DatabaseManager
from src.backend.database.repositories import ArticleRepository
from src.shared.config.config_loader import ConfigLoader
from src.shared.utils.logging_config import setup_logging
from src.shared.utils import (
    CollectionStats, 
    WorkflowStats, 
    get_current_timestamp,
    get_current_timestamp_for_filename,
    save_json_file
)

logger = logging.getLogger(__name__)

class ArticleCollectionWorkflow:
    """Standalone workflow for collecting articles and storing them in database."""
    
    def __init__(self):
        """Initialize the article collection workflow."""
        self.app_config = ConfigLoader.load_config("app")
        app_collection_config = self.app_config.get('collection', {})
        self.max_articles_to_collect = app_collection_config.get('max_articles_to_collect', 500)
        logger.info(f"Max {self.max_articles_to_collect} articles to be collected per run")

        # Initialize collector with configuration
        collector_config = CollectorConfig(
            max_articles=self.max_articles_to_collect,
            max_age_days=app_collection_config.get('max_age_days', 7),
            timeout_seconds=app_collection_config.get('performance', {}).get('timeout_seconds', 30),
            min_title_length=app_collection_config.get('quality_filters', {}).get('min_title_length', 5),
            min_description_length=app_collection_config.get('quality_filters', {}).get('min_description_length', 10)
        )
        
        self.news_collector = NewsCollector(config=collector_config)
        
        # Initialize workflow statistics
        self.workflow_stats = WorkflowStats()
        self.collection_stats = CollectionStats()
        
        # Initialize database (required for collection workflow)
        try:
            self.db_manager = DatabaseManager()
            self.article_repo = ArticleRepository(self.db_manager)
            # Try to create tables, but don't fail if they already exist
            try:
                self.db_manager.create_tables()
            except Exception as create_error:
                # Log the error but continue if it's just a table/constraint already exists issue
                if "already exists" in str(create_error).lower():
                    logger.warning(f"Database tables already exist: {create_error}")
                else:
                    raise create_error
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise ValueError("Database connection is required for collection workflow")
    
    async def collect_and_store_articles(self) -> Dict[str, Any]:
        """
        Main collection workflow: collect articles from sources and store in database.
        Returns collection statistics.
        """
        # Start workflow timing
        self.workflow_stats.start_workflow("article_collection")
        self.collection_stats.start_timing()

        # Step 1: Database cleanup
        logger.info("Step 1: Performing database cleanup")
        try:
            cleanup_stats = self.article_repo.cleanup_old_articles(days_old=7)
            if cleanup_stats['deleted_count'] > 0:
                logger.info(f"Database cleanup: removed {cleanup_stats['deleted_count']} articles older than 7 days")
            self.workflow_stats.articles_cleaned_up = cleanup_stats['deleted_count']
        except Exception as e:
            logger.warning(f"Database cleanup failed: {e}")
            self.workflow_stats.articles_cleaned_up = 0

        # Step 2: Collect articles from sources
        self.collection_stats.total_sources = len(self.news_collector.sources)
        logger.info(f"Step 2: Collecting articles from {self.collection_stats.total_sources} sources")
        
        try:
            def progress_callback(completed: int, total: int):
                """Progress callback for collection updates."""
                logger.debug(f"Collection progress: {completed}/{total} sources processed")

            collected_articles = await self.news_collector.collect_articles(
                max_articles=self.max_articles_to_collect,
                progress_callback=progress_callback
            )
            
            if not collected_articles:
                logger.warning("No articles collected from sources")
                self.workflow_stats.complete_workflow(success=False, error="No articles collected")
                return self._generate_final_summary()

            # Update collection statistics
            self.collection_stats.total_articles = len(collected_articles)
            self.collection_stats.stop_timing()
            
            # Save collected articles to JSON files
            await self._save_collection_results(collected_articles)
            
            logger.info(f"Article collection completed in {self.collection_stats.processing_time:.2f} seconds")
            logger.info(f"Collected {len(collected_articles)} articles")
            
        except Exception as e:
            logger.error(f"Article collection failed: {e}")
            self.workflow_stats.complete_workflow(success=False, error=str(e))
            return self._generate_final_summary()

        # Step 3: Store articles in database
        logger.info("Step 3: Storing articles in database")
        try:
            inserted_count, duplicate_count = self.article_repo.save_new_articles(collected_articles)
            logger.info(f"Database storage: {inserted_count} new articles, {duplicate_count} duplicates skipped")
            
            self.workflow_stats.articles_inserted = inserted_count
            self.workflow_stats.duplicates_skipped = duplicate_count
            
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            self.workflow_stats.complete_workflow(success=False, error=f"Database storage failed: {e}")
            return self._generate_final_summary()

        # Step 4: Complete workflow
        self.workflow_stats.complete_workflow(success=True)
        
        logger.info(f"✅ Collection workflow completed successfully in {self.workflow_stats.duration_seconds:.2f} seconds")
        logger.info(f"📊 Summary: {self.workflow_stats.articles_inserted} new articles added, "
                   f"{self.workflow_stats.duplicates_skipped} duplicates, "
                   f"{self.workflow_stats.articles_cleaned_up} old articles cleaned up")
        
        return self._generate_final_summary()
    
    async def _save_collection_results(self, collected_articles: List[Dict]) -> None:
        """Save collection results to JSON files."""
        try:
            # Save collected articles
            collection_data = {
                'timestamp': get_current_timestamp(),
                'total_articles': len(collected_articles),
                'articles': collected_articles
            }
            filename = f"collected_articles_{get_current_timestamp_for_filename()}.json"
            save_json_file(collection_data, filename)
            
            # Save collection statistics
            stats_data = {
                'timestamp': get_current_timestamp(),
                'collection_stats': self.collection_stats.to_dict()
            }
            stats_filename = f"collection_stats_{get_current_timestamp_for_filename()}.json"
            save_json_file(stats_data, stats_filename)
            
        except Exception as e:
            logger.warning(f"Failed to save collection results: {e}")
            
    def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate comprehensive collection workflow summary."""
        try:
            db_stats = self.article_repo.get_statistics()
        except Exception:
            db_stats = {'total_articles': 0, 'unprocessed': 0, 'ai_related': 0, 'not_ai_related': 0}
        
        self.workflow_stats.total_articles_in_db = db_stats['total_articles']
        self.workflow_stats.collection_stats = self.collection_stats
        
        summary = self.workflow_stats.to_dict()
        summary['database_stats'].update({
            'unprocessed_articles': db_stats['unprocessed'],
            'ai_related_articles': db_stats['ai_related'],
            'not_ai_related_articles': db_stats['not_ai_related']
        })
        
        return summary
    
async def main():
    """Main entry point for the collection workflow."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger.info("🔄 Article Collection Workflow Starting")

        collection_workflow = ArticleCollectionWorkflow()
        summary = await collection_workflow.collect_and_store_articles()

        if summary['workflow_info']['status'] == 'success':
            logger.info("✅ Collection workflow completed successfully")
        else:
            logger.error(f"❌ Collection workflow failed: {summary['workflow_info']['error']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Collection workflow crashed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
