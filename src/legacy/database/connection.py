#!/usr/bin/env python3
"""
Database connection manager for PostgreSQL (Aiven)
Minimal implementation for article storage with lifecycle management
"""

import os
import logging
import psycopg2
import psycopg2.pool
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Simple PostgreSQL connection manager for article storage"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database connection manager"""
        self.connection_string = connection_string or os.getenv('AIVEN_SERVICE_URI')
        if not self.connection_string:
            raise ValueError("Database connection string not found. Set AIVEN_SERVICE_URI environment variable.")
        
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool with optimized settings for bulk operations"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,      # Increased minimum connections
                maxconn=10,     # Increased maximum connections for better concurrency
                dsn=self.connection_string
            )
            logger.info("Database connection pool initialized successfully (2-10 connections)")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        connection = None
        try:
            if self.connection_pool:
                connection = self.connection_pool.getconn()
                yield connection
            else:
                raise RuntimeError("Connection pool not initialized")
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if connection and self.connection_pool:
                self.connection_pool.putconn(connection)
    
    def create_tables(self):
        """Create required tables with optimized schema and indexes"""
        create_articles_sql = """
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            url VARCHAR(2048) UNIQUE NOT NULL,
            title VARCHAR(1000) NOT NULL,
            description TEXT,
            author VARCHAR(500),
            published_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW(),
            is_ai_related BOOLEAN DEFAULT NULL
        );
        
        -- Create optimized indexes for performance
        CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_unique ON articles(url);
        CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_articles_is_ai_related ON articles(is_ai_related) WHERE is_ai_related IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_articles_published_date ON articles(published_date DESC);
        """
        
        # Additive, idempotent schema changes
        alter_articles_sql = """
        -- Add category column if missing
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='articles' AND column_name='category'
            ) THEN
                ALTER TABLE articles ADD COLUMN category TEXT NULL;
            END IF;
        END$$;

        -- Index on category for fast lookups
        CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
        """
        
        # Separate SQL for constraint management (handle existing constraints gracefully)
        constraint_sql = """
        -- Ensure URL constraint for duplicate prevention (only if not exists)
        DO $$ 
        BEGIN
            -- Drop existing constraint if it exists
            IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'articles_url_key') THEN
                ALTER TABLE articles DROP CONSTRAINT articles_url_key;
            END IF;
            
            -- Add unique constraint only if it doesn't exist
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'articles_url_unique') THEN
                ALTER TABLE articles ADD CONSTRAINT articles_url_unique UNIQUE (url);
            END IF;
        END $$;
        """

        # Minimal sources table (DB is the single source of truth for sources)
        create_sources_sql = """
        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            url TEXT NOT NULL UNIQUE,
            description TEXT,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_name_unique ON sources(name);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_url_unique ON sources(url);
        CREATE INDEX IF NOT EXISTS idx_sources_enabled ON sources(enabled);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Execute table and index creation
                    cursor.execute(create_articles_sql)
                    cursor.execute(alter_articles_sql)
                    cursor.execute(create_sources_sql)
                    # Execute constraint management
                    cursor.execute(constraint_sql)
                    conn.commit()
                    logger.info("Articles and sources tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create articles table: {e}")
            raise
    
    def close(self):
        """Close connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")
