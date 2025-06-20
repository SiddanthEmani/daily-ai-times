#!/usr/bin/env python3
"""
Simple Backend Pipeline Test Script

Tests the NewsXP.ai backend pipeline with simplified workflow.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "src" / "backend"

# Add backend paths
sys.path.insert(0, str(BACKEND_DIR / "collectors"))
sys.path.insert(0, str(BACKEND_DIR / "processors"))

# Load environment variables from .env.local
load_dotenv(PROJECT_ROOT / ".env.local")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleBackendTester:
    """Simple tester for NewsXP.ai backend pipeline."""
    
    def __init__(self):
        """Initialize the tester."""
        self.api_key = os.getenv('GROQ_API_KEY')
        self.output_dir = PROJECT_ROOT / "test_output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration settings
        self.config = self._load_config()
        
        if not self.api_key:
            logger.warning("No GROQ_API_KEY found in .env.local")
        else:
            logger.info("✓ API key loaded from .env.local")
    
    def _load_config(self):
        """Load configuration from app.json."""
        try:
            config_file = PROJECT_ROOT / "src" / "shared" / "config" / "app.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Use development config by default
            dev_config = config.get('development', {})
            collection_config = dev_config.get('collection', {})
            pipeline_config = dev_config.get('pipeline', {})
            
            settings = {
                'max_articles_to_collect': collection_config.get('max_articles_to_collect', 2988),
                'target_articles_count': collection_config.get('target_articles_count', 25),
                'pipeline_version': pipeline_config.get('version', '4-stage-optimized'),
                'stage1_max_articles': pipeline_config.get('stage1', {}).get('max_articles', 2988),
                'stage1_target_pass_rate': pipeline_config.get('stage1', {}).get('target_pass_rate', 0.60)
            }
            
            logger.info(f"✓ Config loaded - Pipeline: {settings['pipeline_version']}")
            logger.info(f"  Max articles: {settings['max_articles_to_collect']}, Target: {settings['target_articles_count']}")
            return settings
            
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'max_articles_to_collect': 25,
                'target_articles_count': 5,
                'pipeline_version': '4-stage-optimized',
                'stage1_max_articles': 25,
                'stage1_target_pass_rate': 0.60
            }
    
    async def test_collection(self, max_articles_to_process=None):
        """Test news collection."""
        if max_articles_to_process is None:
            max_articles_to_process = self.config['max_articles_to_collect']
            
        logger.info("🧪 Testing news collection...")
        
        try:
            # Import and initialize collector
            sys.path.insert(0, str(BACKEND_DIR))
            from collectors.collect_news import NewsCollector
            
            # Get sources file path
            sources_file = PROJECT_ROOT / "src" / "shared" / "config" / "sources.json"
            
            # Create collector with correct parameter
            collector = NewsCollector(sources_file=str(sources_file))
            
            # Collect news from all sources but limit total articles collected
            logger.info(f"   Using max_articles limit: {max_articles_to_process}")
            articles = await collector.collect_all(max_age_days=14, max_articles=max_articles_to_process)

            # No need for additional limiting since collector respects max_articles
            
            result = {
                'articles': articles,
                'total_collected': len(articles),
                'timestamp': asyncio.get_event_loop().time()
            }
            
            logger.info(f"✅ Collected {len(articles)} articles")
            
            # Save results
            with open(self.output_dir / "collected_articles.json", "w") as f:
                json.dump(result, f, indent=2)
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Collection failed: {e}")
            return None
    
    async def test_processing(self, articles, target_articles=None):
        """Test article processing."""
        if target_articles is None:
            target_articles = self.config['target_articles_count']
            
        if not articles:
            logger.warning("No articles to process")
            return None
            
        logger.info("🧪 Testing article processing...")
        logger.info(f"   Target articles count: {target_articles}")
        
        try:
            # Import orchestrator
            sys.path.insert(0, str(BACKEND_DIR))
            from processors.orchestrator import NewsOrchestrator 
            
            # Create orchestrator
            orchestrator = NewsOrchestrator(api_key=self.api_key)
            
            # Process articles with target count
            result = await orchestrator.process_articles(articles=articles, target_count=target_articles)
            
            logger.info(f"✅ Processed {result.get('output_count', 0)} articles")
            
            # Save results
            with open(self.output_dir / "processed_articles.json", "w") as f:
                json.dump(result, f, indent=2)
                
            return result, orchestrator
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return None, None
    
    async def run_test(self, max_articles_to_process=None, final_article_count=None):
        """Run the complete test."""
        if max_articles_to_process is None:
            max_articles_to_process = self.config['max_articles_to_collect']
        if final_article_count is None:
            final_article_count = self.config['target_articles_count']
            
        logger.info("🚀 Starting optimized 4-stage pipeline test...")
        logger.info(f"   Enhanced deduplication and cleanup enabled")
        logger.info(f"   Stage 1 Target: {int(max_articles_to_process * self.config['stage1_target_pass_rate'])} articles ({self.config['stage1_target_pass_rate']:.0%})")
        
        # Test collection
        collection_result = await self.test_collection(max_articles_to_process)
        if not collection_result:
            logger.error("❌ Test failed at collection stage")
            return False
        
        # Test processing - only if we have articles
        articles = collection_result.get('articles', [])
        if articles:
            test_result = await self.test_processing(articles, final_article_count)
            if not test_result or test_result[0] is None:
                logger.error("❌ Test failed at processing stage")
                return False
            
            # Unpack the result
            processing_result, orchestrator = test_result
            
            # Save processed articles to API endpoints for GitHub Pages
            if orchestrator and hasattr(orchestrator, 'save_to_api_directories'):
                orchestrator.save_to_api_directories(processing_result, str(PROJECT_ROOT))
            else:
                # Fallback to local method
                self.save_to_api_endpoints(processing_result)
            
            # Report pipeline performance
            logger.info("📊 Pipeline Performance Summary:")
            if processing_result and 'pipeline_stages' in processing_result:
                stages = processing_result['pipeline_stages']
                for stage_name, stage_data in stages.items():
                    if not stage_data.get('skipped', False):
                        input_count = stage_data.get('input_count', 0)
                        output_count = stage_data.get('output_count', 0)
                        pass_rate = (output_count / input_count * 100) if input_count > 0 else 0
                        processing_time = stage_data.get('processing_time', 0)
                        
                        logger.info(f"   {stage_name}: {input_count} → {output_count} ({pass_rate:.1f}%) in {processing_time:.2f}s")
                        
                        if 'model' in stage_data:
                            logger.info(f"     Model: {stage_data['model']}")
                        if 'requests_made' in stage_data:
                            logger.info(f"     API Requests: {stage_data['requests_made']}")
        else:
            logger.info("ℹ️  No articles to process, but collection test passed")
        
        logger.info("🎉 All tests completed successfully!")
        return True
    
    def save_usage_report(self):
        """Save Groq API usage report."""
        try:
            # Import the usage tracker
            sys.path.insert(0, str(BACKEND_DIR / "processors"))
            from groq_usage_tracker import usage_tracker # type: ignore
            
            # Save usage report to test output
            usage_report_path = self.output_dir / "groq_api_usage.json"
            usage_tracker.save_usage_report(usage_report_path)
            
            # Also create a human-readable summary
            summary = usage_tracker.get_usage_summary()
            human_readable_path = self.output_dir / "groq_usage_summary.txt"
            
            with open(human_readable_path, 'w') as f:
                f.write("=== GROQ API USAGE SUMMARY ===\n\n")
                f.write(f"Session Duration: {summary.total_duration:.2f} seconds\n")
                f.write(f"Total API Calls: {summary.total_calls}\n")
                f.write(f"Successful Calls: {summary.successful_calls}\n")
                f.write(f"Failed Calls: {summary.failed_calls}\n")
                
                if summary.total_calls > 0:
                    f.write(f"Success Rate: {(summary.successful_calls/summary.total_calls*100):.1f}%\n\n")
                else:
                    f.write(f"Success Rate: N/A (no calls made)\n\n")
                
                f.write(f"Token Usage:\n")
                f.write(f"  - Total Tokens: {summary.total_tokens:,}\n")
                f.write(f"  - Input Tokens: {summary.total_request_tokens:,}\n")
                f.write(f"  - Output Tokens: {summary.total_response_tokens:,}\n")
                
                if summary.successful_calls > 0:
                    f.write(f"  - Avg Tokens/Call: {summary.total_tokens/summary.successful_calls:.1f}\n\n")
                else:
                    f.write(f"  - Avg Tokens/Call: N/A\n\n")
                
                f.write(f"Performance:\n")
                f.write(f"  - Avg Processing Time: {summary.average_processing_time:.2f}s\n")
                
                if summary.total_duration > 0:
                    f.write(f"  - Tokens/Second: {summary.total_tokens/summary.total_duration:.1f}\n")
                    f.write(f"  - Calls/Minute: {summary.total_calls/(summary.total_duration/60):.1f}\n\n")
                else:
                    f.write(f"  - Tokens/Second: N/A\n")
                    f.write(f"  - Calls/Minute: N/A\n\n")
                
                if summary.models_used:
                    f.write(f"Models Used:\n")
                    for model, count in summary.models_used.items():
                        f.write(f"  - {model}: {count} calls\n")
                    f.write("\n")
                else:
                    f.write(f"Models Used: None (no API calls made)\n\n")
                
                if summary.agents_used:
                    f.write(f"Agents Used:\n")
                    for agent, count in summary.agents_used.items():
                        f.write(f"  - {agent}: {count} calls\n")
                    f.write("\n")
                else:
                    f.write(f"Agents Used: None (no API calls made)\n\n")
                
                f.write(f"Estimated Cost: ${summary.estimated_cost:.6f}\n")
                f.write(f"Rate Limit Hits: {summary.rate_limit_hits}\n")
                
                # Add detailed breakdown by agent and model
                if summary.total_calls > 0:
                    f.write("\n=== DETAILED BREAKDOWN ===\n\n")
                    
                    # Group calls by agent
                    agent_breakdown = {}
                    model_breakdown = {}
                    
                    for call in summary.calls:
                        if call.success:  # Only count successful calls
                            agent = call.agent or "unknown"
                            model = call.model
                            
                            # Agent breakdown
                            if agent not in agent_breakdown:
                                agent_breakdown[agent] = {
                                    'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                                    'processing_time': 0.0, 'models': {}
                                }
                            
                            agent_breakdown[agent]['calls'] += 1
                            agent_breakdown[agent]['input_tokens'] += call.request_tokens
                            agent_breakdown[agent]['output_tokens'] += call.response_tokens
                            agent_breakdown[agent]['total_tokens'] += call.total_tokens
                            agent_breakdown[agent]['processing_time'] += call.processing_time
                            
                            if model not in agent_breakdown[agent]['models']:
                                agent_breakdown[agent]['models'][model] = 0
                            agent_breakdown[agent]['models'][model] += 1
                            
                            # Model breakdown
                            if model not in model_breakdown:
                                model_breakdown[model] = {
                                    'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                                    'processing_time': 0.0, 'agents': {}
                                }
                            
                            model_breakdown[model]['calls'] += 1
                            model_breakdown[model]['input_tokens'] += call.request_tokens
                            model_breakdown[model]['output_tokens'] += call.response_tokens
                            model_breakdown[model]['total_tokens'] += call.total_tokens
                            model_breakdown[model]['processing_time'] += call.processing_time
                            
                            if agent not in model_breakdown[model]['agents']:
                                model_breakdown[model]['agents'][agent] = 0
                            model_breakdown[model]['agents'][agent] += 1
                    
                    # Write agent breakdown
                    f.write("Usage by Agent:\n")
                    for agent, data in agent_breakdown.items():
                        f.write(f"  {agent}:\n")
                        f.write(f"    - Calls: {data['calls']}\n")
                        f.write(f"    - Input Tokens: {data['input_tokens']:,}\n")
                        f.write(f"    - Output Tokens: {data['output_tokens']:,}\n")
                        f.write(f"    - Total Tokens: {data['total_tokens']:,}\n")
                        f.write(f"    - Avg Processing Time: {data['processing_time']/data['calls']:.2f}s\n")
                        f.write(f"    - Models Used: {', '.join(f'{m}({c})' for m, c in data['models'].items())}\n")
                        f.write("\n")
                    
                    # Write model breakdown
                    f.write("Usage by Model:\n")
                    for model, data in model_breakdown.items():
                        f.write(f"  {model}:\n")
                        f.write(f"    - Calls: {data['calls']}\n")
                        f.write(f"    - Input Tokens: {data['input_tokens']:,}\n")
                        f.write(f"    - Output Tokens: {data['output_tokens']:,}\n")
                        f.write(f"    - Total Tokens: {data['total_tokens']:,}\n")
                        f.write(f"    - Avg Processing Time: {data['processing_time']/data['calls']:.2f}s\n")
                        f.write(f"    - Agents Using: {', '.join(f'{a}({c})' for a, c in data['agents'].items())}\n")
                        f.write("\n")
                
                if summary.total_calls == 0:
                    f.write(f"\nNOTE: No Groq API calls were made during this test.\n")
                    f.write(f"This may indicate that Groq agents are not properly loaded or configured.\n")
            
            logger.info(f"✓ Usage reports saved to {self.output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save usage report: {e}")
        
        return True
    
    def save_to_api_endpoints(self, processing_result):
        """Save processed articles to API endpoints for GitHub Pages."""
        if not processing_result or not processing_result.get('articles'):
            logger.warning("No processed articles to save to API endpoints")
            return False
            
        try:
            # Prepare the API data structure similar to collection result
            api_data = {
                "generated_at": processing_result.get('generated_at', datetime.now(timezone.utc).isoformat()),
                "collection_time_seconds": processing_result.get('processing_time', 0),
                "count": processing_result.get('output_count', len(processing_result.get('articles', []))),
                "articles": processing_result.get('articles', []),
                "pipeline_info": {
                    "version": processing_result.get('pipeline_version', '4-stage-optimized'),
                    "input_count": processing_result.get('input_count', 0),
                    "output_count": processing_result.get('output_count', 0),
                    "overall_pass_rate": processing_result.get('overall_pass_rate', 0),
                    "processing_time": processing_result.get('processing_time', 0)
                }
            }
            
            # Save to backend API directory
            backend_api_dir = PROJECT_ROOT / "src" / "backend" / "api"
            backend_api_dir.mkdir(exist_ok=True)
            backend_latest_path = backend_api_dir / "latest.json"
            
            with open(backend_latest_path, "w") as f:
                json.dump(api_data, f, indent=2)
            logger.info(f"✓ Saved processed articles to {backend_latest_path}")
            
            # Save to frontend API directory (for GitHub Pages)
            frontend_api_dir = PROJECT_ROOT / "src" / "frontend" / "api"
            frontend_api_dir.mkdir(exist_ok=True)
            frontend_latest_path = frontend_api_dir / "latest.json"
            
            with open(frontend_latest_path, "w") as f:
                json.dump(api_data, f, indent=2)
            logger.info(f"✓ Saved processed articles to {frontend_latest_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save to API endpoints: {e}")
            return False
    


async def main():
    """Main function."""
    tester = SimpleBackendTester()
    
    # Run test with settings from configuration
    success = await tester.run_test()
    
    # Save usage report
    tester.save_usage_report()
    
    if success:
        print("\n✅ Backend pipeline test completed successfully!")
        print("📊 Check test_output/ for detailed API usage reports")
        sys.exit(0)
    else:
        print("\n❌ Backend pipeline test failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        sys.exit(1)
