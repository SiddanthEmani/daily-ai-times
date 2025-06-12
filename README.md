# NewsXP AI

AI-curated daily news aggregation platform featuring the latest developments in technology, research, and industry insights.

## Architecture

![Architecture Diagram](./docs/diagrams/Architecture.excalidraw.svg)

The system follows a three-stage pipeline:
1. **Data Collection** - RSS feeds and API content extraction
2. **AI Processing** - LLM-powered filtering, categorization, and summarization  
3. **Web Interface** - Responsive newspaper-style layout

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16.0+

### Installation & Setup
```bash
git clone https://github.com/SiddanthEmani/newsxp-ai.git
cd newsxp-ai

# Install all dependencies
npm install
npm run install-backend
npm run install-mcp
```

### Usage
```bash
# Start development server
npm run dev

# Collect latest news
npm run collect

# Build and serve
npm start
```

## Features

- **AI-Curated Content** - Automated filtering and categorization
- **Multiple Sources** - Research papers, industry news, open-source updates  
- **Newspaper Layout** - Classic, responsive design
- **Real-time Updates** - Fresh content on schedule
- **MCP Integration** - Model Context Protocol server for external tools

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run collect` | Collect latest news |
| `npm run generate-api` | Generate API files |
| `npm run build` | Full build process |
| `npm start` | Build and serve |
| `npm run mcp-server` | Start MCP server |

## Configuration

News sources can be configured in `src/shared/config/sources.json`.

## License

MIT License
