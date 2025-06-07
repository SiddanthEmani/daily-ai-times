#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
from datetime import datetime

def generate_api_files():
    # Read latest news
    with open('data/news.json', 'r') as f:
        news_data = json.load(f)
    
    # Ensure API directory exists
    api_dir = Path('api')
    api_dir.mkdir(exist_ok=True)
    
    # Copy latest to API directory
    shutil.copy('data/news.json', 'api/latest.json')
    
    # Create category files
    categories_dir = api_dir / 'categories'
    categories_dir.mkdir(exist_ok=True)
    
    categories = {}
    for article in news_data['articles']:
        cat = article['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(article)
    
    for category, articles in categories.items():
        with open(categories_dir / f"{category.lower().replace(' ', '-')}.json", 'w') as f:
            json.dump({
                'category': category,
                'count': len(articles),
                'articles': articles
            }, f, indent=2)
    
    # Create widget endpoint
    widget_data = {
        'updated': news_data['generated_at'],
        'top_stories': news_data['articles'][:5],
        'categories': list(categories.keys())
    }
    
    with open(api_dir / 'widget.json', 'w') as f:
        json.dump(widget_data, f, indent=2)
    
    # Create archive index
    archive_dir = Path('data/archive')
    if archive_dir.exists():
        archives = sorted([f.stem for f in archive_dir.glob('*.json')])
        with open(api_dir / 'archives.json', 'w') as f:
            json.dump(archives, f, indent=2)
    
    # Create index.html
    with open(api_dir / 'index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>NewsXP AI - API</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        code { background: #e0e0e0; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>NewsXP AI API</h1>
    <p>AI News Aggregation API Endpoints:</p>
    
    <div class="endpoint">
        <h3>Latest News</h3>
        <code>GET /api/latest.json</code>
        <p>Returns the most recent AI news articles</p>
    </div>
    
    <div class="endpoint">
        <h3>Widget Data</h3>
        <code>GET /api/widget.json</code>
        <p>Lightweight endpoint for widgets (top 5 stories)</p>
    </div>
    
    <div class="endpoint">
        <h3>Categories</h3>
        <code>GET /api/categories/{category}.json</code>
        <p>Get news by category (research, industry, open-source)</p>
    </div>
    
    <div class="endpoint">
        <h3>Archives</h3>
        <code>GET /api/archives.json</code>
        <p>List of available archive dates</p>
    </div>
</body>
</html>''')
    
    print(f"Generated API files in {api_dir}")

if __name__ == '__main__':
    generate_api_files()