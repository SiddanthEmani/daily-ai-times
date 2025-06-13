#!/usr/bin/env python3
import json
import shutil
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib

def calculate_content_hash(content):
    """Calculate hash of content to check for changes"""
    return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()

def write_json_file(filepath, data, force=False):
    """Write JSON file only if content has changed"""
    content_hash = calculate_content_hash(data)
    hash_file = filepath.with_suffix('.hash')
    
    # Check if file exists and content is unchanged
    if not force and filepath.exists() and hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                existing_hash = f.read().strip()
            if existing_hash == content_hash:
                return False  # No changes needed
        except:
            pass  # Continue with write if hash check fails
    
    # Write the JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Write the hash file
    with open(hash_file, 'w') as f:
        f.write(content_hash)
    
    return True

def generate_category_file(category_data):
    """Generate a single category file"""
    category, articles = category_data
    filename = f"{category.lower().replace(' ', '-')}.json"
    filepath = Path('src/backend/api/categories') / filename
    
    data = {
        'category': category,
        'count': len(articles),
        'articles': articles,
        'updated': datetime.now().isoformat()
    }
    
    return filepath, write_json_file(filepath, data)

def generate_api_files():
    """Generate API files with optimizations"""
    print("Starting API file generation...")
    
    # Read latest news
    news_path = Path('src/backend/data/news.json')
    if not news_path.exists():
        print("Error: news.json not found")
        return
    
    with open(news_path, 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    # Ensure API directory exists
    api_dir = Path('src/backend/api')
    api_dir.mkdir(exist_ok=True)
    
    categories_dir = api_dir / 'categories'
    categories_dir.mkdir(exist_ok=True)
    
    # Copy latest.json only if source is newer
    latest_path = api_dir / 'latest.json'
    if not latest_path.exists() or news_path.stat().st_mtime > latest_path.stat().st_mtime:
        shutil.copy2(news_path, latest_path)
        print("✅ Updated latest.json")
    else:
        print("ℹ️  latest.json is up to date")
    
    # Categorize articles
    categories = {}
    for article in news_data.get('articles', []):
        cat = article.get('category', 'Other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(article)
    
    # Generate category files in parallel
    changed_files = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        category_results = list(executor.map(generate_category_file, categories.items()))
    
    for filepath, was_changed in category_results:
        if was_changed:
            changed_files.append(filepath.name)
            print(f"✅ Updated {filepath.name}")
        else:
            print(f"ℹ️  {filepath.name} is up to date")
    
    # Create widget endpoint
    widget_data = {
        'updated': news_data.get('generated_at'),
        'top_stories': news_data.get('articles', [])[:5],
        'categories': list(categories.keys()),
        'total_articles': len(news_data.get('articles', []))
    }
    
    widget_path = api_dir / 'widget.json'
    if write_json_file(widget_path, widget_data):
        changed_files.append('widget.json')
        print("✅ Updated widget.json")
    else:
        print("ℹ️  widget.json is up to date")
    
    # Create archive index
    archive_dir = Path('src/backend/data/archive')
    archives_data = []
    if archive_dir.exists():
        archives_data = sorted([f.stem for f in archive_dir.glob('*.json')])
    
    archives_path = api_dir / 'archives.json'
    if write_json_file(archives_path, archives_data):
        changed_files.append('archives.json')
        print("✅ Updated archives.json")
    else:
        print("ℹ️  archives.json is up to date")
    
    # Generate API documentation
    generate_api_documentation(api_dir, categories)
    
    print(f"✅ API generation complete. Changed files: {len(changed_files)}")
    if changed_files:
        print(f"   Changed: {', '.join(changed_files)}")

def generate_api_documentation(api_dir, categories):
    """Generate API documentation HTML"""
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NewsXP AI - API Documentation</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1e40af;
            --secondary: #64748b;
            --background: #f8fafc;
            --surface: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #059669;
            --warning: #d97706;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--background);
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: var(--surface);
            border-radius: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: var(--primary);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        
        .endpoints {{
            display: grid;
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .endpoint {{
            background: var(--surface);
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary);
        }}
        
        .endpoint h3 {{
            color: var(--primary-dark);
            margin-bottom: 0.5rem;
            font-size: 1.25rem;
        }}
        
        .endpoint code {{
            background: var(--background);
            padding: 0.25rem 0.75rem;
            border-radius: 0.375rem;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            color: var(--primary);
            font-weight: 600;
        }}
        
        .endpoint p {{
            margin-top: 0.75rem;
            color: var(--text-muted);
        }}
        
        .categories {{
            background: var(--surface);
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .categories h3 {{
            margin-bottom: 1rem;
            color: var(--primary-dark);
        }}
        
        .category-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .category-tag {{
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        footer {{
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NewsXP AI API</h1>
            <p class="subtitle">Real-time AI News Aggregation & Analysis</p>
        </header>
        
        <div class="endpoints">
            <div class="endpoint">
                <h3>Latest News</h3>
                <code>GET /api/latest.json</code>
                <p>Returns the most recent AI news articles with full metadata including scores, categories, and publication dates.</p>
            </div>
            
            <div class="endpoint">
                <h3>Widget Data</h3>
                <code>GET /api/widget.json</code>
                <p>Lightweight endpoint optimized for widgets and embedded displays. Contains top 5 stories and summary statistics.</p>
            </div>
            
            <div class="endpoint">
                <h3>Category News</h3>
                <code>GET /api/categories/{{category}}.json</code>
                <p>Get news articles filtered by category. Available categories are listed below.</p>
            </div>
            
            <div class="endpoint">
                <h3>Archives Index</h3>
                <code>GET /api/archives.json</code>
                <p>List of all available archive dates for historical news data retrieval.</p>
            </div>
        </div>
        
        <div class="categories">
            <h3>Available Categories</h3>
            <div class="category-list">
                {' '.join(f'<span class="category-tag">{cat}</span>' for cat in sorted(categories.keys()))}
            </div>
        </div>
        
        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | NewsXP AI v1.0</p>
        </footer>
    </div>
</body>
</html>'''
    
    with open(api_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Generated API documentation")

if __name__ == '__main__':
    generate_api_files()