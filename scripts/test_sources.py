#!/usr/bin/env python3
"""
RSS Feed Validation Script

This script validates RSS feed URLs for validity and errors.
It can extract URLs from markdown files, perform batch processing,
and compare against existing sources to identify new valid feeds.

Usage:
    python test_sources.py --url <single_url>
    python test_sources.py --file <markdown_file>
    python test_sources.py --yaml <yaml_file>
    python test_sources.py --existing-sources <path_to_sources_dir>
    python test_sources.py --all  # Test all sources
"""

import asyncio
import aiohttp
import argparse
import re
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from datetime import datetime
import json

@dataclass
class FeedValidationResult:
    """Result of RSS feed validation"""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    feed_title: Optional[str] = None
    feed_description: Optional[str] = None
    item_count: Optional[int] = None
    last_updated: Optional[str] = None
    content_type: Optional[str] = None
    response_time_ms: Optional[float] = None

class RSSValidator:
    """Validates RSS feeds with async batch processing"""
    
    def __init__(self, max_concurrent: int = 10, timeout: int = 30):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'RSS-Validator/1.0 (https://github.com/daily-ai-times)'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def validate_feed(self, url: str) -> FeedValidationResult:
        """Validate a single RSS feed"""
        start_time = time.time()
        
        try:
            async with self.session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                content = await response.text()
                content_type = response.headers.get('content-type', '')
                
                if response.status != 200:
                    return FeedValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=response.status,
                        error_message=f"HTTP {response.status}",
                        response_time_ms=response_time,
                        content_type=content_type
                    )
                
                # Try to parse as XML
                try:
                    root = ET.fromstring(content)
                    
                    # Check if it's RSS or Atom
                    feed_info = self._extract_feed_info(root, content)
                    
                    return FeedValidationResult(
                        url=url,
                        is_valid=True,
                        status_code=response.status,
                        feed_title=feed_info['title'],
                        feed_description=feed_info['description'],
                        item_count=feed_info['item_count'],
                        last_updated=feed_info['last_updated'],
                        content_type=content_type,
                        response_time_ms=response_time
                    )
                    
                except ET.ParseError as e:
                    return FeedValidationResult(
                        url=url,
                        is_valid=False,
                        status_code=response.status,
                        error_message=f"XML Parse Error: {str(e)}",
                        content_type=content_type,
                        response_time_ms=response_time
                    )
                    
        except asyncio.TimeoutError:
            return FeedValidationResult(
                url=url,
                is_valid=False,
                error_message="Timeout",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return FeedValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def _extract_feed_info(self, root: ET.Element, content: str) -> Dict:
        """Extract information from RSS/Atom feed"""
        info = {
            'title': None,
            'description': None,
            'item_count': 0,
            'last_updated': None
        }
        
        # RSS 2.0
        if root.tag == 'rss':
            channel = root.find('channel')
            if channel is not None:
                title_elem = channel.find('title')
                desc_elem = channel.find('description')
                items = channel.findall('item')
                
                info['title'] = title_elem.text if title_elem is not None else None
                info['description'] = desc_elem.text if desc_elem is not None else None
                info['item_count'] = len(items)
                
                # Try to get last build date
                last_build = channel.find('lastBuildDate')
                if last_build is not None:
                    info['last_updated'] = last_build.text
        
        # Atom
        elif 'atom' in root.tag.lower():
            title_elem = root.find('{http://www.w3.org/2005/Atom}title')
            subtitle_elem = root.find('{http://www.w3.org/2005/Atom}subtitle')
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            info['title'] = title_elem.text if title_elem is not None else None
            info['description'] = subtitle_elem.text if subtitle_elem is not None else None
            info['item_count'] = len(entries)
            
            updated_elem = root.find('{http://www.w3.org/2005/Atom}updated')
            if updated_elem is not None:
                info['last_updated'] = updated_elem.text
        
        return info
    
    async def validate_feeds_batch(self, urls: List[str]) -> List[FeedValidationResult]:
        """Validate multiple feeds concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def validate_with_semaphore(url):
            async with semaphore:
                return await self.validate_feed(url)
        
        tasks = [validate_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

class SourceManager:
    """Manages existing sources and identifies new ones"""
    
    def __init__(self, sources_dir: Path):
        self.sources_dir = Path(sources_dir)
        self.existing_urls: Set[str] = set()
        self.existing_sources: Dict[str, Dict] = {}
        
    def load_existing_sources(self):
        """Load existing sources from YAML files"""
        for yaml_file in self.sources_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'sources' in data:
                        for source_name, source_info in data['sources'].items():
                            url = source_info.get('url')
                            if url:
                                self.existing_urls.add(url)
                                self.existing_sources[url] = {
                                    'name': source_name,
                                    'file': yaml_file.name,
                                    **source_info
                                }
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
    
    def extract_urls_from_markdown(self, md_file: Path) -> List[Tuple[str, str]]:
        """Extract URLs from markdown file, returning (url, source_name) tuples"""
        urls = []
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern to match table rows with URLs
            # Looking for | Source Name | URL |
            table_pattern = r'\|([^|]+)\|([^|]*https?://[^\s|]+[^|]*)\|'
            matches = re.findall(table_pattern, content, re.MULTILINE)
            
            for source_name, url_cell in matches:
                # Extract URL from the cell (might have extra text)
                url_match = re.search(r'https?://[^\s|]+', url_cell.strip())
                if url_match:
                    url = url_match.group().strip()
                    source_name = source_name.strip()
                    urls.append((url, source_name))
            
            # Also look for plain URLs in the text
            url_pattern = r'https?://[^\s\)]+(?:rss|feed|xml|atom)[^\s\)]*'
            plain_urls = re.findall(url_pattern, content, re.IGNORECASE)
            
            for url in plain_urls:
                if not any(url == existing_url for existing_url, _ in urls):
                    urls.append((url, "Extracted from text"))
                    
        except Exception as e:
            print(f"Error reading markdown file: {e}")
        
        return urls
    
    def find_new_sources(self, validated_results: List[FeedValidationResult]) -> List[FeedValidationResult]:
        """Find new valid sources not in existing configuration"""
        new_sources = []
        
        for result in validated_results:
            if result.is_valid and result.url not in self.existing_urls:
                new_sources.append(result)
                
        return new_sources

def print_validation_report(results: List[FeedValidationResult], show_all: bool = True):
    """Print a formatted validation report"""
    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = len(results) - valid_count
    
    print(f"\n{'='*80}")
    print(f"RSS FEED VALIDATION REPORT")
    print(f"{'='*80}")
    print(f"Total feeds tested: {len(results)}")
    print(f"✅ Valid feeds: {valid_count}")
    print(f"❌ Invalid feeds: {invalid_count}")
    print(f"Success rate: {(valid_count/len(results)*100):.1f}%")
    
    # Always show invalid feeds if there are any
    if invalid_count > 0:
        print(f"\n{'FAILED FEEDS':-^80}")
        for result in results:
            if not result.is_valid:
                print(f"❌ {result.url}")
                print(f"   Error: {result.error_message}")
                if result.status_code:
                    print(f"   Status: {result.status_code}")
                if result.response_time_ms:
                    print(f"   Response time: {result.response_time_ms:.0f}ms")
                print()
    else:
        print(f"\n🎉 All feeds are valid!")
    
    # Only show valid feeds if explicitly requested with show_all
    if show_all and valid_count > 0:
        print(f"\n{'VALID FEEDS (showing details)':-^80}")
        for result in results:
            if result.is_valid:
                print(f"✅ {result.url}")
                if result.feed_title:
                    print(f"   Title: {result.feed_title}")
                if result.item_count is not None:
                    print(f"   Items: {result.item_count}")
                if result.response_time_ms:
                    print(f"   Response time: {result.response_time_ms:.0f}ms")
                print()
    elif valid_count > 0:
        print(f"✅ Valid sources saved to test_output/valid_sources.yaml")

def extract_urls_from_yaml(yaml_file: Path) -> List[str]:
    """Extract URLs from YAML file with sources structure"""
    urls = []
    
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'sources' in data:
            for source_name, source_info in data['sources'].items():
                url = source_info.get('url')
                if url:
                    urls.append(url)
        else:
            print(f"Warning: No 'sources' key found in {yaml_file}")
            
    except Exception as e:
        print(f"Error reading YAML file {yaml_file}: {e}")
    
    return urls

def create_valid_sources_yaml(valid_results: List[FeedValidationResult], new_sources: List[FeedValidationResult]):
    """Create YAML file with valid sources in test_output folder"""
    # Create test_output directory if it doesn't exist
    test_output_dir = Path("test_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # Generate source name from URL
    def generate_source_name(url: str, title: str = None) -> str:
        """Generate a source name from URL and title"""
        if title:
            # Clean title for use as key
            name = re.sub(r'[^a-zA-Z0-9_\s]', '', title.lower())
            name = re.sub(r'\s+', '_', name.strip())
            if name and len(name) > 3:
                return name
        
        # Fallback to domain-based name
        try:
            domain = urlparse(url).netloc
            name = domain.replace('.', '_').replace('-', '_')
            return name
        except:
            return f"source_{hash(url) % 10000}"
    
    # Build sources dictionary
    sources_dict = {}
    for result in valid_results:
        source_name = generate_source_name(result.url, result.feed_title)
        
        # Ensure unique names
        counter = 1
        original_name = source_name
        while source_name in sources_dict:
            source_name = f"{original_name}_{counter}"
            counter += 1
        
        sources_dict[source_name] = {
            'url': result.url,
            'description': result.feed_title or result.feed_description or 'RSS Feed'
        }
    
    # Create YAML structure
    yaml_data = {
        'sources': sources_dict
    }
    
    # Write to file
    output_file = test_output_dir / "valid_sources.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=True)
    
    print(f"✅ Valid sources saved to {output_file}")
    print(f"   Total valid sources: {len(valid_results)}")
    if new_sources:
        print(f"   New sources: {len(new_sources)}")

def save_results_json(results: List[FeedValidationResult], filename: str):
    """Save results to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'total_tested': len(results),
        'valid_count': sum(1 for r in results if r.is_valid),
        'results': [asdict(result) for result in results]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description='RSS Feed Validator')
    parser.add_argument('--url', help='Single URL to test')
    parser.add_argument('--file', help='Markdown file to extract URLs from')
    parser.add_argument('--yaml', help='YAML file to extract URLs from (sources format)')
    parser.add_argument('--existing-sources', help='Directory containing existing source YAML files')
    parser.add_argument('--all', action='store_true', help='Test all existing sources')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum concurrent requests')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Show only summary')
    parser.add_argument('--verbose', action='store_true', help='Show details for valid feeds too')
    
    args = parser.parse_args()
    
    # Default to sources directory if not specified
    sources_dir = Path(args.existing_sources) if args.existing_sources else Path('../src/shared/config/sources')
    
    # Initialize source manager
    source_manager = SourceManager(sources_dir)
    source_manager.load_existing_sources()
    
    urls_to_test = []
    
    # Collect URLs to test
    if args.url:
        urls_to_test.append(args.url)
    
    if args.file:
        extracted_urls = source_manager.extract_urls_from_markdown(Path(args.file))
        urls_to_test.extend([url for url, _ in extracted_urls])
        print(f"Extracted {len(extracted_urls)} URLs from {args.file}")
    
    if args.yaml:
        yaml_urls = extract_urls_from_yaml(Path(args.yaml))
        urls_to_test.extend(yaml_urls)
        print(f"Extracted {len(yaml_urls)} URLs from {args.yaml}")
    
    if args.all:
        urls_to_test.extend(source_manager.existing_urls)
        print(f"Testing {len(source_manager.existing_urls)} existing sources")
    
    if not urls_to_test:
        parser.print_help()
        return
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls_to_test:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    print(f"Testing {len(unique_urls)} unique URLs...")
    
    # Validate feeds
    start_time = time.time()
    async with RSSValidator(max_concurrent=args.max_concurrent, timeout=args.timeout) as validator:
        results = await validator.validate_feeds_batch(unique_urls)
    
    total_time = time.time() - start_time
    print(f"Validation completed in {total_time:.1f} seconds")
    
    # Print report (show failed feeds always, valid feeds only with --verbose)
    print_validation_report(results, show_all=args.verbose)
    
    # Find new sources
    new_sources = source_manager.find_new_sources(results)
    if new_sources:
        print(f"\n{'NEW VALID SOURCES':-^80}")
        print(f"Found {len(new_sources)} new valid sources")
    
    # Create YAML file with valid sources in test_output folder
    valid_results = [r for r in results if r.is_valid]
    if valid_results:
        create_valid_sources_yaml(valid_results, new_sources)
    
    # Save results if requested
    if args.output:
        save_results_json(results, args.output)

if __name__ == "__main__":
    asyncio.run(main()) 