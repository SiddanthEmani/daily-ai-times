#!/usr/bin/env python3
"""
Test script for ultra-fast news collection performance.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.collectors.collectors import UltraFastNewsCollector

async def test_ultra_fast_collection():
    """Test the ultra-fast collection performance."""
    print("🚀 Testing Ultra-Fast News Collection")
    print("=" * 50)
    
    # Initialize the ultra-fast collector
    collector = UltraFastNewsCollector()
    print(f"📊 Loaded {len(collector.sources)} sources")
    
    # Test collection with timing
    start_time = time.time()
    
    try:
        articles = await collector.collect_all_ultra_fast(max_articles=1500)
        
        duration = time.time() - start_time
        
        print(f"\n✅ Ultra-Fast Collection Results:")
        print(f"   📰 Articles collected: {len(articles)}")
        print(f"   ⏱️  Total time: {duration:.2f} seconds")
        print(f"   🚀 Speed: {len(articles)/duration:.1f} articles/second")
        print(f"   📊 Sources: ✓{collector.stats.successful} ○{collector.stats.empty} ✗{collector.stats.failed}")
        
        if duration < 10:
            print(f"🎉 SUCCESS: Collection completed in {duration:.2f}s (target: <10s)")
        else:
            print(f"⚠️  WARNING: Collection took {duration:.2f}s (target: <10s)")
        
        # Show sample articles
        if articles:
            print(f"\n📋 Sample Articles:")
            for i, article in enumerate(articles[:3]):
                print(f"   {i+1}. {article.get('title', 'No title')[:60]}...")
                print(f"      Source: {article.get('source', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ultra_fast_collection()) 