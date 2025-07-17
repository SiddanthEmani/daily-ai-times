#!/usr/bin/env python3
"""
Script to validate RSS feeds from Telugu news sources and generate telugu.yaml
"""

import re
import yaml
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Tuple
import feedparser
import os
import sys

def extract_rss_feeds_from_markdown(md_file: str) -> List[Tuple[str, str, str]]:
    """
    Extract RSS feed URLs, names, and descriptions from markdown file.
    Returns list of tuples: (source_name, url, description)
    """
    feeds = []
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match RSS feed entries
    # Looking for patterns like "RSS Feed: url" or "- RSS Feed: url"
    rss_pattern = r'(?:RSS Feed:|RSS:)\s*([^\n\r]+)'
    name_pattern = r'\*\*([^*]+)\*\*'
    
    lines = content.split('\n')
    current_source = None
    current_description = None
    
    for i, line in enumerate(lines):
        # Extract source name
        name_match = re.search(name_pattern, line)
        if name_match:
            current_source = name_match.group(1).strip()
            # Get description from the parentheses or following content
            if '(' in line and ')' in line:
                desc_match = re.search(r'\(([^)]+)\)', line)
                if desc_match:
                    current_description = f"{current_source} - {desc_match.group(1)}"
                else:
                    current_description = current_source
            else:
                current_description = current_source
        
        # Extract RSS feed URL
        rss_match = re.search(rss_pattern, line)
        if rss_match and current_source:
            url = rss_match.group(1).strip()
            
            # Clean up URL - remove extra text
            url = url.split()[0] if ' ' in url else url
            url = url.replace('(created via Feed43)', '').strip()
            url = url.replace('Alternative:', '').strip()
            
            # Skip if it says "Not found" or similar
            if 'not found' in url.lower() or 'not directly accessible' in url.lower():
                continue
                
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Skip obviously incomplete URLs
            if len(url) < 10 or '..' in url:
                continue
                
            feeds.append((current_source, url, current_description))
    
    return feeds

def validate_rss_feed(url: str, timeout: int = 3) -> Tuple[bool, str]:
    """
    Validate if an RSS feed URL is accessible and contains valid RSS content.
    Returns (is_valid, error_message)
    """
    try:
        # Create request with user agent
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        # Make request with short timeout
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() != 200:
                return False, f"HTTP {response.getcode()}"
            
            # Read only first 50KB to avoid large downloads
            content = response.read(50000)
        
        # Try to parse as RSS/XML using feedparser
        feed = feedparser.parse(content)
        
        if feed.bozo and feed.bozo_exception:
            # Allow some common parsing issues for XML feeds
            if "not well-formed" not in str(feed.bozo_exception).lower():
                return False, f"Invalid RSS: {str(feed.bozo_exception)}"
        
        if not feed.entries:
            return False, "No entries found in feed"
        
        return True, "Valid RSS feed"
        
    except urllib.error.HTTPError as e:
        return False, f"HTTP error: {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL error: timeout or connection failed"
    except Exception as e:
        return False, f"Parse error: {str(e)[:50]}"

def generate_yaml_key(source_name: str) -> str:
    """Generate a valid YAML key from source name"""
    # Remove special characters and convert to lowercase
    key = re.sub(r'[^a-zA-Z0-9\s]', '', source_name)
    key = re.sub(r'\s+', '_', key.strip())
    key = key.lower()
    
    # Handle specific cases
    if 'tv' in key and 'news' in key:
        key = key.replace('news', '').replace('tv', 'tv_news')
    elif key.endswith('_tv'):
        key = key.replace('_tv', '_tv_news')
    
    return key

def main():
    """Main function to validate RSS feeds and generate telugu.yaml"""
    
    md_file = 'src/shared/config/sources/new_telugu_sources.md'
    output_file = 'src/shared/config/sources/telugu.yaml'
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found")
        sys.exit(1)
    
    print("Extracting RSS feeds from markdown file...")
    feeds = extract_rss_feeds_from_markdown(md_file)
    print(f"Found {len(feeds)} potential RSS feeds")
    
    valid_sources = {}
    validation_results = []
    
    print("\nValidating RSS feeds...")
    for i, (source_name, url, description) in enumerate(feeds, 1):
        print(f"[{i}/{len(feeds)}] Validating {source_name}: {url}")
        
        is_valid, message = validate_rss_feed(url)
        validation_results.append((source_name, url, is_valid, message))
        
        if is_valid:
            key = generate_yaml_key(source_name)
            
            # Ensure unique keys
            original_key = key
            counter = 1
            while key in valid_sources:
                key = f"{original_key}_{counter}"
                counter += 1
            
            valid_sources[key] = {
                'url': url,
                'type': 'rss',
                'category': 'Telugu Media',
                'enabled': True,
                'description': description,
                'legal_note': 'Please check individual source for republication rights'
            }
            print(f"  ✓ Valid: {message}")
        else:
            print(f"  ✗ Invalid: {message}")
        
        # Small delay to be respectful
        time.sleep(0.1)
    
    # Generate YAML content
    yaml_content = {
        'sources': valid_sources
    }
    
    print(f"\nGenerating {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
    
    # Print summary
    valid_count = len(valid_sources)
    invalid_count = len(feeds) - valid_count
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total feeds found: {len(feeds)}")
    print(f"Valid feeds: {valid_count}")
    print(f"Invalid feeds: {invalid_count}")
    print(f"Success rate: {valid_count/len(feeds)*100:.1f}%")
    
    print(f"\nValid sources written to: {output_file}")
    
    # Print invalid feeds for reference
    if invalid_count > 0:
        print(f"\nInvalid feeds:")
        for source_name, url, is_valid, message in validation_results:
            if not is_valid:
                print(f"  - {source_name}: {url} ({message})")

if __name__ == "__main__":
    main() 