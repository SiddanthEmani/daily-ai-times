#!/usr/bin/env python3
"""
Extract Article Images Script

This script reads the latest.json file, visits each article URL,
extracts headline images using meta tags, and downloads them to
a test_output folder.
"""

import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
import re
from typing import Optional, Dict, List
import argparse

def get_article_image(url: str, timeout: int = 10) -> Optional[str]:
    """Extract article image URL from meta tags"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ImageExtractor/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try Open Graph image first (most reliable)
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return urljoin(url, og_image['content'])
        
        # Try Twitter Card image
        twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
        if twitter_image and twitter_image.get('content'):
            return urljoin(url, twitter_image['content'])
        
        # Try other common meta tags
        meta_image = soup.find('meta', attrs={'name': 'image'})
        if meta_image and meta_image.get('content'):
            return urljoin(url, meta_image['content'])
        
        # Try article:image
        article_image = soup.find('meta', property='article:image')
        if article_image and article_image.get('content'):
            return urljoin(url, article_image['content'])
        
        # Try to find the first large image in the article
        images = soup.find_all('img')
        for img in images:
            src = img.get('src')
            if src:
                # Skip small images, icons, avatars
                if any(skip in src.lower() for skip in ['icon', 'avatar', 'logo', 'thumb']):
                    continue
                # Look for larger images (check width/height attributes)
                width = img.get('width')
                height = img.get('height')
                if width and height:
                    try:
                        if int(width) > 200 and int(height) > 200:
                            return urljoin(url, src)
                    except ValueError:
                        pass
                # If no size info, just take the first non-icon image
                return urljoin(url, src)
        
        return None
        
    except Exception as e:
        print(f"Error extracting image from {url}: {e}")
        return None

def download_image(image_url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download image from URL to specified path"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ImageDownloader/1.0)',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(image_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Determine file extension from content type or URL
        content_type = response.headers.get('content-type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'png' in content_type:
            ext = '.png'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'webp' in content_type:
            ext = '.webp'
        else:
            # Try to get extension from URL
            parsed_url = urlparse(image_url)
            path = parsed_url.path.lower()
            if path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                ext = Path(path).suffix
            else:
                ext = '.jpg'  # Default
        
        # Ensure output path has correct extension
        if not output_path.suffix:
            output_path = output_path.with_suffix(ext)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces and dots
    filename = re.sub(r'\s+', ' ', filename).strip()
    filename = filename.strip('.')
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def extract_article_id_from_url(url: str) -> str:
    """Extract a unique identifier from the URL"""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if path:
        # Use the last part of the path as ID
        return path.split('/')[-1]
    else:
        # Fallback to domain
        return parsed.netloc.replace('.', '_')

def process_latest_json(json_path: str, output_dir: str, delay: float = 1.0) -> Dict[str, List[str]]:
    """Process latest.json and extract images from all articles"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}
    
    articles = data.get('articles', [])
    print(f"Found {len(articles)} articles to process")
    
    results = {
        'successful': [],
        'failed': [],
        'no_image': []
    }
    
    for i, article in enumerate(articles, 1):
        url = article.get('url')
        title = article.get('title', 'Untitled')
        article_id = article.get('article_id', extract_article_id_from_url(url))
        
        if not url:
            print(f"Article {i}: No URL found")
            results['failed'].append(f"Article {i}: No URL")
            continue
        
        print(f"Processing article {i}/{len(articles)}: {title[:50]}...")
        
        # Extract image URL
        image_url = get_article_image(url)
        
        if not image_url:
            print(f"  No image found for: {url}")
            results['no_image'].append(url)
            continue
        
        # Create filename
        safe_title = sanitize_filename(title)
        filename = f"{article_id}_{safe_title}"
        image_path = output_path / filename
        
        # Download image
        if download_image(image_url, image_path):
            print(f"  ✅ Downloaded: {image_path.name}")
            results['successful'].append(url)
        else:
            print(f"  ❌ Failed to download: {image_url}")
            results['failed'].append(url)
        
        # Add delay to be respectful to servers
        if delay > 0 and i < len(articles):
            time.sleep(delay)
    
    return results

def print_summary(results: Dict[str, List[str]]):
    """Print summary of results"""
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful downloads: {len(results['successful'])}")
    print(f"❌ Failed downloads: {len(results['failed'])}")
    print(f"📭 No image found: {len(results['no_image'])}")
    print(f"📊 Total processed: {sum(len(v) for v in results.values())}")
    
    if results['failed']:
        print(f"\nFailed URLs:")
        for url in results['failed']:
            print(f"  - {url}")
    
    if results['no_image']:
        print(f"\nURLs with no image:")
        for url in results['no_image']:
            print(f"  - {url}")

def main():
    parser = argparse.ArgumentParser(description='Extract article images from latest.json')
    parser.add_argument('--json-path', default='src/backend/api/latest.json', 
                       help='Path to latest.json file')
    parser.add_argument('--output-dir', default='test_output/article_images',
                       help='Output directory for images')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    print(f"Extracting images from: {args.json_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Delay between requests: {args.delay}s")
    print(f"Request timeout: {args.timeout}s")
    print()
    
    # Process the JSON file
    results = process_latest_json(args.json_path, args.output_dir, args.delay)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main() 