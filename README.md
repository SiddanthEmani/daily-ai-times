# Daily AI Times

Modern serverless AI-powered news aggregation platform with advanced multi-agent swarm intelligence and consensus-driven content curation.

## Live Demo

**Production Site**: [https://www.dailyai.wtf](https://www.dailyai.wtf)

![Demo](./src/frontend/assets/images/demo.png)

## Architecture

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
1. **Collection** - Intelligent gathering from 30+ configured news sources
2. **Bulk Intelligence Swarm** - Multi-agent parallel processing with specialized AI models
3. **Initial Consensus** - Advanced consensus algorithms for content filtering
4. **Deep Intelligence Analysis** - Sophisticated fact-checking, bias detection, and impact analysis
5. **Final Consensus** - Weighted combination of initial and deep intelligence results
6. **Content Classification** - Automatic categorization into headlines, articles, and research papers
7. **API Generation** - Dynamic creation of optimized JSON endpoints
8. **Deployment** - Seamless push to GitHub Pages with validation

### Frontend Features
- **Responsive Design** - Newspaper-style layout optimized for all devices
- **Real-time Loading** - Dynamic content updates from JSON APIs
- **Offline Support** - Service worker caching for offline access
- **Performance Optimized** - Minimal JavaScript, fast loading

#### Sidebar data charts
The sidebar bar charts pull from small, independently-refreshed JSON feeds under `src/frontend/api/`, each with a static in-code fallback so the page never breaks if a feed is missing:

- **AI Data Center Buildout** (`capex.json`, via `capex_collector.py`, refreshed by the **Update Benchmark Leaderboard** workflow alongside the leaderboard) — annualized data center / AI-infrastructure spend, `$B`. A **hybrid** feed with no API key required:
  - **Public companies** (Amazon, Microsoft, Alphabet, Meta, Oracle) — trailing-12-month capital expenditure fetched live from the keyless [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) XBRL API.
  - **Private companies** (OpenAI, Anthropic) — file nothing with the SEC, so their figures come from `src/backend/collectors/capex_curated.json`, a committed, source-cited file. These are flagged as estimates (a `*` on the chart).
  - Note: no filing breaks out *data-center-only* capex, so public figures are total company capex (overwhelmingly AI/data-center spend for these firms). To show a data-center-only estimate for any company, add a `value` + `source`/`basis` override to that company in `capex_curated.json`.
- **Benchmark Leaderboard** (`leaderboard.json`) and the masthead **Ticker** (`ticker.json`) follow the same collector → JSON → fallback pattern.

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
git clone https://github.com/SiddanthEmani/daily-ai-times.git
cd daily-ai-times

# Install Python dependencies
pip install -r src/backend/requirements.txt

# Test the complete pipeline locally
./orchestrator.run

# Or run with Python directly
python src/backend/orchestrator.py
```

### For Contributors
1. **Fork the repository**
2. **Configure GitHub Secrets**:
   - `GROQ_API_KEY` - Required for AI processing
   - `GOOGLE_ANALYTICS_ID` - Optional analytics
   - `ALPHA_VANTAGE_API_KEY` - Optional, powers the live masthead ticker (falls back to static placeholder quotes if unset)
   - `ARTIFICIAL_ANALYSIS_API_KEY` - Optional, powers the live benchmark leaderboard chart via the [Artificial Analysis API](https://artificialanalysis.ai/api) (falls back to static placeholder figures if unset)
   - _No secret is needed for the **Data Center Capex** chart — it uses the keyless [SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) plus a committed curated file (see below)._
3. **Enable GitHub Pages** with "GitHub Actions" source
4. **Test locally** before pushing changes
5. **Push changes** - Automatic deployment via workflow

## Features

### 🤖 Advanced AI Processing
- **Multi-Agent Swarm Intelligence** - Distributed processing with specialized AI models
- **Consensus Algorithms** - Advanced voting and confidence-based filtering
- **Deep Intelligence Analysis** - Fact-checking, bias detection, and credibility scoring
- **Adaptive Batch Processing** - Dynamic optimization for model rate limits

### 📰 Content Management
- **Smart Classification** - Automatic sorting into headlines, articles, and research papers
- **Multi-Source Integration** - Research papers, industry news, open-source updates
- **Advanced Deduplication** - Intelligent content cleanup and similarity detection
- **Quality Gates** - Multi-tier confidence scoring and validation

### ⚡ Performance & Reliability
- **Fresh Content** - Updates every 4 hours with modern orchestrator
- **Robust Error Handling** - Built-in timeouts, retries, and graceful degradation
- **Performance Monitoring** - Detailed pipeline metrics and usage tracking
- **Serverless Architecture** - Zero infrastructure management with GitHub Actions

## Development Commands

| Command | Description |
|---------|-------------|
| `./orchestrator.run` | Run complete AI processing pipeline locally |
| `python src/backend/orchestrator.py` | Run orchestrator directly with Python |
| `npm run mcp-server` | Start MCP server for development tools |
| Manual triggers via GitHub Actions | Test pipeline with custom parameters |

## Pipeline Configuration

The modern AI pipeline is configured via YAML files in `src/shared/config/`:

**Sources**: `sources/*.yaml` - Individual source configurations for each category  
**Application**: `app.yaml` - General pipeline parameters and collection limits  
**AI Swarm**: `swarm.yaml` - Multi-agent configuration, consensus rules, and model settings

### AI Swarm Configuration
- **Bulk Intelligence Agents** - Multiple specialized models for parallel processing
- **Deep Intelligence Agents** - Advanced analysis models with enhanced capabilities
- **Consensus Engine** - Voting algorithms and confidence thresholds
- **Final Consensus** - Weighted combination rules and quality gates
- **Rate Limiting** - Intelligent model-specific TPM management and batching

## Workflow Automation

The GitHub Actions workflow (`collect-news.yml`) runs every 4 hours with modern serverless architecture:

### Automated Processing
1. **AI News Processing Pipeline** - Single orchestrator handles complete workflow
2. **Multi-Agent Swarm Intelligence** - Distributed AI processing with consensus algorithms  
3. **Automatic API Generation** - Dynamic creation of frontend-ready JSON endpoints
4. **GitHub Pages Deployment** - Seamless content delivery with validation
5. **Pipeline Monitoring** - Detailed metrics, usage tracking, and performance analytics

### Manual Triggers
Available via GitHub Actions interface with advanced options:
- **Source Selection** - Choose specific sources or process all
- **Force Refresh** - Override caching and force complete refresh
- **Debug Mode** - Enable detailed deployment structure logging
- **Skip Deployment** - Run processing without deploying (useful for testing)
- **Pipeline Metrics** - Real-time processing statistics and agent performance

## License

MIT License
