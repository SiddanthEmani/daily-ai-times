// Minimal shared types describing the categories JSON in src/frontend/api/categories/

export type UiCategoryKey =
  | 'us'
  | 'world'
  | 'industry'
  | 'business'
  | 'research'
  | 'health'
  | 'ethics'
  | 'learn';

export interface CategoryArticle {
  id: number | string;
  url: string;
  title: string;
  description?: string;
  author?: string;
  published_date?: string; // ISO
  created_at?: string; // ISO
  category?: string; // backend category labels vary
  assigned_category?: string;
  image_path?: string; // relative path to asset
  source?: string; // optional; some feeds include source
  article_id?: number;
}

export interface CategoryFile {
  generated_at: string; // ISO
  category: string; // human label e.g., "Industry"
  articles: CategoryArticle[];
  count?: number;
  pipeline_info?: Record<string, unknown>;
}


