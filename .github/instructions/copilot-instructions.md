# Daily AI Times - Copilot Instructions

## Architecture Overview

This is a **serverless AI-powered news aggregation platform** with a unique two-tier architecture:
- **Backend**: Python-based AI processing pipeline running on GitHub Actions (every 4 hours)
- **Frontend**: Static vanilla JavaScript site deployed to GitHub Pages consuming JSON APIs

### Key Architectural Principles
- **Serverless-first**: Zero infrastructure costs using GitHub's free tier
- **AI Swarm Intelligence**: Multi-agent consensus-driven content curation
- **Static API Generation**: Backend generates JSON files consumed by frontend
- **Separation of Concerns**: Backend processes, frontend consumes

## Critical Development Workflows

### Local Development
```bash
# Run complete AI pipeline locally
./orchestrator.run
# Or directly: python src/backend/orchestrator.py

# Frontend development server
npm run dev  # Serves on localhost:8000

# Test specific components
python scripts/test_sources.py
```
Note: Always activate or check for virtual environment before running commands.

### Deployment Pipeline
- **Automatic**: Every 4 hours via GitHub Actions workflow
- **Manual**: Trigger via GitHub Actions UI with custom parameters
- **Local Testing**: Use `./orchestrator.run` before pushing changes

## AI Swarm Architecture (Core Innovation)

### Multi-Agent Processing Flow
1. **Collection**: `NewsCollector` gathers from 30+ configured sources
2. **Bulk Intelligence**: Parallel processing with 3 fast models (`llama-3.1-8b-instant`, `gemma2-9b-it`, `llama3-8b-8192`)
3. **Consensus Engine**: Votes and confidence scoring across agents
4. **Deep Intelligence**: Advanced analysis with slower, more capable models
5. **Final Consensus**: Weighted combination producing top 25 articles

### Configuration Files (Critical)
- `src/shared/config/app.yaml`: Pipeline settings, model lists, processing limits
- `src/shared/config/swarm.yaml`: AI agent configurations, TPM limits, consensus weights
- `src/shared/config/sources/`: Modular YAML files per news category

### Agent Rate Limiting
Each AI model has specific TPM (Tokens Per Minute) limits defined in `swarm.yaml`. The orchestrator automatically batches requests and manages delays. **Never modify TPM values without understanding Groq API limits**.

## Project-Specific Patterns

### Import Structure
```python
# Always use src.* imports from project root
from src.backend.processors.bulk_agent import BulkFilteringAgent
from src.shared.config.config_loader import ConfigLoader
```

### Error Handling Pattern
All processors implement graceful degradation with fallback models and timeout handling. See `orchestrator.py` for the canonical error recovery pattern.

### Frontend Data Flow
```javascript
// Components follow this loading pattern:
// 1. Check cache validity (appVersion-based)
// 2. Fetch from /api/{category}.json
// 3. Render with performance monitoring
```

### Static API Generation
Backend generates these JSON endpoints consumed by frontend:
- `/api/ai.json`, `/api/entertainment.json`, etc.
- `/api/stats.json` for pipeline metrics
- All generated in `src/frontend/api/` during processing

## Integration Points

### GitHub Actions Integration
- Secrets: `GROQ_API_KEY` (required), `GOOGLE_ANALYTICS_ID` (optional)
- Artifact handling: Pipeline results stored as workflow artifacts
- Pages deployment: Automatic via GitHub Actions built-in functionality

### AI Model Integration
- **Groq API**: Primary AI provider via OpenAI-compatible endpoints
- **Rate Limiting**: Sophisticated TPM management in `bulk_agent.py` and `deep_intelligence_agent.py`
- **Fallback Strategy**: Multiple models per processing stage for resilience

## Code Conventions

### File Organization
- **Processors**: Self-contained AI agents in `src/backend/processors/`
- **Collectors**: News source handlers in `src/backend/collectors/`
- **Shared**: Configuration and utilities in `src/shared/`
- **Frontend**: Pure vanilla JavaScript modules in `src/frontend/components/`

### Configuration Management
All configuration is YAML-based and environment-aware. Use `ConfigLoader` for accessing settings - never hardcode values.

### Performance Patterns
- Frontend uses lazy loading and performance monitoring
- Backend implements batching and parallel processing
- All network requests include timeout and retry logic

## Critical Files to Understand
- `src/backend/orchestrator.py`: Main pipeline coordinator (2000+ lines)
- `src/shared/config/swarm.yaml`: AI agent configuration and limits
- `src/frontend/components/app.js`: Frontend application controller
- `.github/workflows/collect-news.yml`: Serverless deployment pipeline

## Common Gotchas
- **Path issues**: Always run scripts from project root due to import structure
- **API limits**: Respect TPM settings in swarm.yaml to avoid rate limiting
- **Cache invalidation**: Frontend uses appVersion for cache busting
- **Environment variables**: Use `.env.local` for local development secrets

When modifying AI processing logic, always test locally with `./orchestrator.run` before pushing to avoid breaking the automated pipeline.
