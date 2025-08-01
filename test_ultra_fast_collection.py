#!/usr/bin/env python3
"""
Test script to validate _distribute_articles_by_tpm method with swarm configuration.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backend.orchestrator import NewsProcessingPipeline

def test_distribute_articles_by_tpm():
    """Test the _distribute_articles_by_tpm method with swarm configuration."""
    try:
        # Initialize pipeline
        pipeline = NewsProcessingPipeline()
        
        # Create test articles
        test_articles = [
            {'title': f'Test Article {i}', 'url': f'https://example.com/{i}', 'source': 'test'} 
            for i in range(20)
        ]
        
        # Test with bulk agents
        print("Testing with bulk agents...")
        bulk_assignments = pipeline._distribute_articles_by_tpm(test_articles, pipeline.agents)
        
        print(f"Bulk agent assignments:")
        for agent_name, articles in bulk_assignments.items():
            print(f"  {agent_name}: {len(articles)} articles")
        
        # Test with deep intelligence agents (if enabled)
        if pipeline.enable_deep_intelligence and pipeline.deep_intelligence_agents:
            print("\nTesting with deep intelligence agents...")
            deep_assignments = pipeline._distribute_articles_by_tpm(test_articles, pipeline.deep_intelligence_agents)
            
            print(f"Deep intelligence agent assignments:")
            for agent_name, articles in deep_assignments.items():
                print(f"  {agent_name}: {len(articles)} articles")
        
        # Test agent delays
        print("\nTesting agent delays...")
        for agent_name in list(pipeline.agents.keys()) + list(pipeline.deep_intelligence_agents.keys()):
            delay, timeout = pipeline._get_agent_delays(agent_name)
            print(f"  {agent_name}: delay={delay}s, timeout={timeout}s")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distribute_articles_by_tpm()
    sys.exit(0 if success else 1) 