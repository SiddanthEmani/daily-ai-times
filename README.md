# NewsXP AI

Serverless AI-powered news aggregation platform with automated content curation and 4-stage processing pipeline.

## Live Demo

**Production Site**: [https://siddanthemani.github.io/newsxp-ai](https://siddanthemani.github.io/newsxp-ai)

## Architecture

![Architecture Diagram](./docs/diagrams/Architecture.excalidraw.svg)

**Serverless Backend (GitHub Actions)** → **Static Frontend (GitHub Pages)**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  GitHub Actions  │    │  GitHub Pages   │
│                 │    │  (Serverless     │    │  (Static        │
│ • RSS Feeds     │───▶│   Backend)       │───▶│   Frontend)     │
│ • News APIs     │    │                  │    │                 │
│ • Research APIs │    │ • Data Collection│    │ • Static Site   │
└─────────────────┘    │ • AI Processing  │    │ • JSON APIs     │
                       │ • API Generation │    │ • Asset Serving │
                       └──────────────────┘    └─────────────────┘
```

### Backend Pipeline (Every 4 Hours)
1. **Data Collection** - Fetch from 30+ configured news sources
2. **4-Stage AI Processing** - Groq LLM filtering, deduplication, and categorization
3. **API Generation** - Create static JSON endpoints for frontend consumption
4. **Deployment** - Automatic push to GitHub Pages

### Frontend Features
- **Responsive Design** - Newspaper-style layout optimized for all devices
- **Real-time Loading** - Dynamic content updates from JSON APIs
- **Offline Support** - Service worker caching for offline access
- **Performance Optimized** - Minimal JavaScript, fast loading

### Benefits
- **Zero hosting costs** - GitHub Pages + Actions free tier
- **Global CDN** - Automatic scaling and edge distribution
- **Secure & reliable** - GitHub's enterprise infrastructure
- **Fast deployment** - Changes live in minutes
- **Built-in monitoring** - Workflow status and analytics

## Quick Start

### For Users
Visit the live site - no installation required.

### For Developers

```bash
git clone https://github.com/SiddanthEmani/newsxp-ai.git
cd newsxp-ai

# Install Python dependencies
pip install -r src/backend/requirements.txt

# Test the complete pipeline locally
python scripts/test_backend.py
```

### For Contributors
1. **Fork the repository**
2. **Configure GitHub Secrets**:
   - `GROQ_API_KEY` - Required for AI processing
   - `GOOGLE_ANALYTICS_ID` - Optional analytics
3. **Enable GitHub Pages** with "GitHub Actions" source
4. **Test locally** before pushing changes
5. **Push changes** - Automatic deployment via workflow

## Features

- **AI-Curated Content** - Multi-stage Groq LLM filtering and scoring
- **Multiple Sources** - Research papers, industry news, open-source updates
- **Smart Categorization** - Automatic content classification by topic
- **Fresh Content** - Updates every 4 hours automatically
- **Performance Monitoring** - Built-in analytics and pipeline health checks
- **Deduplication** - Advanced content deduplication and cleanup

## Development Commands

| Command | Description |
|---------|-------------|
| `python scripts/test_backend.py` | Run complete pipeline test locally |
| `npm run dev` | Start development server |
| `npm run build` | Build production assets |
| `npm run mcp-server` | Start MCP server |

## Pipeline Configuration

The 4-stage processing pipeline can be configured via:

**Sources**: `src/shared/config/sources.json` - Configure news sources and RSS feeds  
**Settings**: `src/shared/config/app.json` - Pipeline parameters and processing limits

### Pipeline Settings
- **Collection**: Max articles to collect (default: 2988)
- **Target Output**: Final article count (default: 25)
- **Stage 1**: Bulk filtering with 60% target pass rate
- **Stages 2-4**: Compound, expert, and final agents for content refinement

## Workflow Automation

The GitHub Actions workflow (`collect-news.yml`) runs every 4 hours and includes:

1. **Fresh News Pipeline** - Complete backend processing with retry logic
2. **Content Deployment** - Automatic deployment to GitHub Pages
3. **Monitoring & Reports** - Usage tracking and performance analytics

Manual triggers available with options for:
- Custom source selection
- Force refresh
- Debug mode
- Skip deployment (testing)

## License

MIT License

## License

MIT License
