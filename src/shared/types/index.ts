// Type definitions for NewsXP AI project

export interface Article {
  id: string;
  title: string;
  description: string;
  url: string;
  source: string;
  published_date: string;
  category: ArticleCategory;
  content_type: 'article' | 'research' | 'announcement';
  summary?: string;
  tags?: string[];
}

export type ArticleCategory = 'research' | 'industry' | 'open-source' | 'media';

export interface NewsData {
  articles: Article[];
  metadata: {
    generated_at: string;
    total_articles: number;
    categories: Record<ArticleCategory, number>;
    sources: string[];
  };
}

export interface SourceConfig {
  name: string;
  url: string;
  type: 'rss' | 'api';
  category: ArticleCategory;
  enabled: boolean;
  filters?: {
    keywords?: string[];
    exclude_keywords?: string[];
  };
}

export interface AppConfig {
  sources: Record<string, SourceConfig>;
  collection: {
    max_articles_per_source: number;
    update_interval_hours: number;
    archive_after_days: number;
  };
  api: {
    rate_limit: number;
    cache_duration_minutes: number;
  };
}

export interface PerformanceMetrics {
  mark_name: string;
  timestamp: number;
  duration?: number;
}

export interface CacheEntry<T = any> {
  data: T;
  timestamp: number;
  expires_at: number;
}
