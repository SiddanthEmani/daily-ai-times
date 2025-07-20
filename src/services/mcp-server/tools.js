import { Octokit } from '@octokit/rest';
import { 
  config, 
  cache, 
  rateLimiter, 
  httpGet, 
  validators, 
  createErrorResponse, 
  createSuccessResponse, 
  withRetry, 
  getDateRange, 
  getTopItems 
} from './config.js';

// GitHub client with enhanced configuration
const octokit = new Octokit({
  auth: config.github.token,
  userAgent: 'Daily AI Times MCP Server/1.0',
  previews: ['v3'],
  throttle: {
    onRateLimit: (retryAfter, options) => {
      console.warn(`Rate limit hit for ${options.method} ${options.url}. Retrying after ${retryAfter}s`);
      return true;
    },
    onAbuseLimit: (retryAfter, options) => {
      console.warn(`Abuse detection for ${options.method} ${options.url}. Retrying after ${retryAfter}s`);
      return true;
    }
  }
});

// Tool definitions
export const toolDefinitions = [
  {
    name: 'trigger_news_collection',
    description: 'Trigger GitHub Action to collect latest AI news',
    inputSchema: {
      type: 'object',
      properties: {
        sources: {
          type: 'array',
          items: { type: 'string' },
          description: 'Specific sources to collect from (optional)'
        },
        force_refresh: {
          type: 'boolean',
          default: false,
          description: 'Force refresh even if recent data exists'
        }
      }
    }
  },
  {
    name: 'get_latest_news',
    description: 'Get the most recent AI news articles',
    inputSchema: {
      type: 'object',
      properties: {
        category: {
          type: 'string',
          enum: ['all', 'Research', 'Industry', 'Open Source', 'Startups', 'Government', 'Media'],
          default: 'all'
        },
        limit: {
          type: 'integer',
          minimum: 1,
          maximum: 50,
          default: 10
        },
        min_score: {
          type: 'number',
          description: 'Minimum relevance score (0-100)',
          default: 0
        }
      }
    }
  },
  {
    name: 'search_news',
    description: 'Search through collected news articles',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query'
        },
        date_from: {
          type: 'string',
          format: 'date',
          description: 'Start date (YYYY-MM-DD)'
        },
        date_to: {
          type: 'string',
          format: 'date',
          description: 'End date (YYYY-MM-DD)'
        }
      },
      required: ['query']
    }
  },
  {
    name: 'analyze_trends',
    description: 'Analyze trends in AI news topics',
    inputSchema: {
      type: 'object',
      properties: {
        days: {
          type: 'integer',
          enum: [7, 30, 90],
          default: 7,
          description: 'Number of days to analyze'
        },
        top_n: {
          type: 'integer',
          default: 10,
          description: 'Number of top items to return'
        }
      }
    }
  },
  {
    name: 'add_news_source',
    description: 'Add a new RSS/API source to monitor',
    inputSchema: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Unique identifier for the source'
        },
        url: {
          type: 'string',
          description: 'RSS feed or API URL'
        },
        type: {
          type: 'string',
          enum: ['rss', 'api'],
          default: 'rss'
        },
        category: {
          type: 'string',
          enum: ['Research', 'Industry', 'Open Source', 'Startups', 'Government', 'Media', 'Other'],
          default: 'Other'
        },
        priority: {
          type: 'integer',
          minimum: 1,
          maximum: 10,
          default: 5,
          description: 'Source priority (1=highest, 10=lowest)'
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Tags for categorizing the source'
        },
        reliability: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: 0.8,
          description: 'Reliability score (0-1)'
        },
        updateFrequency: {
          type: 'string',
          enum: ['hourly', 'daily', 'weekly'],
          default: 'daily',
          description: 'How often to check this source'
        },
        language: {
          type: 'string',
          default: 'en',
          description: 'Primary language of the source'
        },
        description: {
          type: 'string',
          description: 'Human-readable description of the source'
        }
      },
      required: ['name', 'url']
    }
  },
  {
    name: 'generate_summary_report',
    description: 'Generate a summary report of recent AI news',
    inputSchema: {
      type: 'object',
      properties: {
        format: {
          type: 'string',
          enum: ['markdown', 'json', 'html'],
          default: 'markdown'
        },
        include_trends: {
          type: 'boolean',
          default: true
        }
      }
    }
  },
  {
    name: 'get_sources_config',
    description: 'Get current sources configuration',
    inputSchema: {
      type: 'object',
      properties: {
        category: {
          type: 'string',
          enum: ['Research', 'Industry', 'Open Source', 'Startups', 'Government', 'Media'],
          description: 'Filter by category (optional)'
        },
        enabled_only: {
          type: 'boolean',
          default: true,
          description: 'Only return enabled sources'
        }
      }
    }
  },
  {
    name: 'update_source_config',
    description: 'Update configuration for an existing source',
    inputSchema: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Source name to update'
        },
        enabled: {
          type: 'boolean',
          description: 'Enable/disable the source'
        },
        priority: {
          type: 'integer',
          minimum: 1,
          maximum: 10,
          description: 'Update source priority'
        },
        reliability: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          description: 'Update reliability score'
        }
      },
      required: ['name']
    }
  },
  {
    name: 'validate_sources',
    description: 'Validate all configured sources for accessibility and format',
    inputSchema: {
      type: 'object',
      properties: {
        sources: {
          type: 'array',
          items: { type: 'string' },
          description: 'Specific sources to validate (optional, validates all if not provided)'
        }
      }
    }
  }
];

// Tool implementations with enhanced error handling and optimization
export const toolImplementations = {

  async trigger_news_collection(args) {
    try {
      // Rate limiting for GitHub API
      rateLimiter.check('github', config.rateLimit.github);

      // Trigger workflow with retry logic
      const workflow = await withRetry(async () => {
        return await octokit.actions.createWorkflowDispatch({
          owner: config.github.owner,
          repo: config.github.repo,
          workflow_id: config.github.workflowId,
          ref: 'main',
          inputs: {
            sources: args.sources ? args.sources.join(',') : 'all',
            force_refresh: String(args.force_refresh || false)
          }
        });
      });

      // Enhanced polling with exponential backoff
      let attempts = 0;
      let status = 'queued';
      let workflowRun = null;

      while (attempts < config.server.maxPollingAttempts && !['completed', 'failure', 'cancelled'].includes(status)) {
        const delay = Math.min(config.server.pollingInterval * Math.pow(1.5, attempts), 10000);
        await new Promise(resolve => setTimeout(resolve, delay));

        try {
          const runs = await octokit.actions.listWorkflowRuns({
            owner: config.github.owner,
            repo: config.github.repo,
            workflow_id: config.github.workflowId,
            per_page: 1
          });

          if (runs.data.workflow_runs.length > 0) {
            workflowRun = runs.data.workflow_runs[0];
            status = workflowRun.status;
            
            if (workflowRun.conclusion) {
              status = workflowRun.conclusion;
            }
          }
        } catch (error) {
          console.warn(`Polling attempt ${attempts + 1} failed:`, error.message);
        }

        attempts++;
      }

      return createSuccessResponse({
        status: status,
        attempts: attempts,
        workflow_url: workflowRun?.html_url,
        conclusion: workflowRun?.conclusion
      }, { operation: 'trigger_news_collection' });

    } catch (error) {
      return createErrorResponse(`Failed to trigger news collection: ${error.message}`);
    }
  },

  async get_latest_news(args) {
    try {
      // Input validation
      if (args.category) {
        validators.enum(args.category, 'category', ['all', 'research', 'industry', 'open-source', 'community']);
      }
      if (args.limit) {
        validators.integer(args.limit, 'limit', { min: 1, max: 50 });
      }

      const url = `https://${config.github.owner}.github.io/${config.github.repo}/api/latest.json`;
      const data = await httpGet(url, { cacheTTL: 180000 }); // 3-minute cache

      let articles = data.articles || [];

      // Apply filters efficiently
      if (args.category && args.category !== 'all') {
        articles = articles.filter(a => 
          a.category && a.category.toLowerCase() === args.category.toLowerCase()
        );
      }

      if (args.min_score > 0) {
        articles = articles.filter(a => (a.score || 0) >= args.min_score);
      }

      // Limit results
      const limit = args.limit || 10;
      const limitedArticles = articles.slice(0, limit);

      return createSuccessResponse({
        articles: limitedArticles,
        total_found: articles.length,
        total_available: data.articles?.length || 0,
        filters_applied: {
          category: args.category || 'all',
          min_score: args.min_score || 0,
          limit: limit
        }
      }, { 
        operation: 'get_latest_news',
        last_updated: data.generated_at,
        cache_used: cache.has(`http:${url}:{}`)
      });

    } catch (error) {
      return createErrorResponse(`Failed to fetch latest news: ${error.message}`);
    }
  },

  async search_news(args) {
    try {
      // Input validation
      validators.required(args.query, 'query');
      validators.string(args.query, 'query', { minLength: 2, maxLength: 100 });

      if (args.date_from) validators.date(args.date_from, 'date_from');
      if (args.date_to) validators.date(args.date_to, 'date_to');

      // Cache key for search results
      const cacheKey = `search:${JSON.stringify(args)}`;
      if (cache.has(cacheKey)) {
        return createSuccessResponse(cache.get(cacheKey), { 
          operation: 'search_news',
          cache_used: true 
        });
      }

      rateLimiter.check('github', config.rateLimit.github);

      // Enhanced search with better error handling
      const searchQuery = `${args.query} repo:${config.github.owner}/${config.github.repo} path:data/archive`;
      
      const searchResults = await withRetry(async () => {
        return await octokit.search.code({
          q: searchQuery,
          per_page: 20
        });
      });

      const matches = [];
      const processedFiles = new Set();

      // Process search results with batching
      for (const file of searchResults.data.items.slice(0, 10)) {
        if (processedFiles.has(file.path)) continue;
        processedFiles.add(file.path);

        try {
          const content = await octokit.repos.getContent({
            owner: config.github.owner,
            repo: config.github.repo,
            path: file.path
          });

          if ('content' in content.data) {
            const decoded = Buffer.from(content.data.content, 'base64').toString();
            const articles = JSON.parse(decoded);

            // Efficient filtering
            const filtered = articles.filter(article => {
              const articleText = JSON.stringify(article).toLowerCase();
              const queryMatch = articleText.includes(args.query.toLowerCase());

              if (!queryMatch) return false;

              if (args.date_from || args.date_to) {
                const articleDate = new Date(article.published_date);
                if (args.date_from && articleDate < new Date(args.date_from)) return false;
                if (args.date_to && articleDate > new Date(args.date_to)) return false;
              }

              return true;
            });

            matches.push(...filtered.slice(0, 3));
          }
        } catch (error) {
          console.warn(`Failed to process file ${file.path}:`, error.message);
        }
      }

      const result = {
        query: args.query,
        matches_found: matches.length,
        results: matches.slice(0, 50), // Limit total results
        search_metadata: {
          files_searched: processedFiles.size,
          date_range: args.date_from || args.date_to ? {
            from: args.date_from,
            to: args.date_to
          } : null
        }
      };

      // Cache results for 10 minutes
      cache.set(cacheKey, result, 600000);

      return createSuccessResponse(result, { operation: 'search_news' });

    } catch (error) {
      return createErrorResponse(`Search failed: ${error.message}`);
    }
  },

  async analyze_trends(args) {
    try {
      const days = validators.integer(args.days || 7, 'days');
      const topN = validators.integer(args.top_n || 10, 'top_n', { min: 1, max: 50 });

      const cacheKey = `trends:${days}:${topN}`;
      if (cache.has(cacheKey)) {
        return createSuccessResponse(cache.get(cacheKey), { 
          operation: 'analyze_trends',
          cache_used: true 
        });
      }

      const { startDate, endDate } = getDateRange(days);
      const articles = [];
      const fetchPromises = [];

      // Parallel fetching of archived data
      for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
        const dateStr = d.toISOString().split('T')[0];
        const url = `https://${config.github.owner}.github.io/${config.github.repo}/api/archive/${dateStr}.json`;
        
        fetchPromises.push(
          httpGet(url, { skipCache: false })
            .then(data => data.articles || [])
            .catch(() => []) // Ignore missing dates
        );
      }

      const results = await Promise.allSettled(fetchPromises);
      results.forEach(result => {
        if (result.status === 'fulfilled') {
          articles.push(...result.value);
        }
      });

      // Enhanced trend analysis
      const analysis = {
        categories: {},
        sources: {},
        keywords: {},
        domains: {},
        daily_counts: {}
      };

      articles.forEach(article => {
        // Daily distribution
        const date = article.published_date?.split('T')[0];
        if (date) {
          analysis.daily_counts[date] = (analysis.daily_counts[date] || 0) + 1;
        }

        // Category trends
        if (article.category) {
          analysis.categories[article.category] = (analysis.categories[article.category] || 0) + 1;
        }

        // Source trends
        if (article.source) {
          analysis.sources[article.source] = (analysis.sources[article.source] || 0) + 1;
        }

        // Domain analysis
        if (article.url) {
          try {
            const domain = new URL(article.url).hostname;
            analysis.domains[domain] = (analysis.domains[domain] || 0) + 1;
          } catch (e) {
            // Invalid URL
          }
        }

        // Enhanced keyword extraction
        const text = `${article.title} ${article.description || ''}`.toLowerCase();
        const words = text.split(/\W+/).filter(word => word.length > 3);
        const aiKeywords = ['artificial', 'intelligence', 'machine', 'learning', 'neural', 'model', 'training', 'llm', 'gpt', 'transformer', 'ai'];
        
        words.forEach(word => {
          if (aiKeywords.includes(word) || word.length > 6) {
            analysis.keywords[word] = (analysis.keywords[word] || 0) + 1;
          }
        });
      });

      const result = {
        period: `${days} days`,
        date_range: {
          start: startDate.toISOString().split('T')[0],
          end: endDate.toISOString().split('T')[0]
        },
        total_articles: articles.length,
        daily_average: Math.round(articles.length / days * 10) / 10,
        top_categories: getTopItems(analysis.categories, topN),
        top_sources: getTopItems(analysis.sources, topN),
        trending_keywords: getTopItems(analysis.keywords, topN),
        top_domains: getTopItems(analysis.domains, Math.min(topN, 15)),
        daily_distribution: analysis.daily_counts
      };

      // Cache for 30 minutes
      cache.set(cacheKey, result, 1800000);

      return createSuccessResponse(result, { operation: 'analyze_trends' });

    } catch (error) {
      return createErrorResponse(`Trend analysis failed: ${error.message}`);
    }
  },

  async add_news_source(args) {
    try {
      // Input validation
      validators.required(args.name, 'name');
      validators.required(args.url, 'url');
      validators.string(args.name, 'name', { minLength: 3, maxLength: 50 });
      validators.string(args.url, 'url', { 
        pattern: /^https?:\/\/.+/,
        minLength: 10 
      });

      if (args.type) {
        validators.enum(args.type, 'type', ['rss', 'api']);
      }

      rateLimiter.check('github', config.rateLimit.github);

      // Create branch with better naming
      const sanitizedName = args.name.toLowerCase().replace(/[^a-z0-9]/g, '-');
      const branch = `add-source-${sanitizedName}-${Date.now()}`;

      const operations = await withRetry(async () => {
        // Get main branch reference
        const mainRef = await octokit.git.getRef({
          owner: config.github.owner,
          repo: config.github.repo,
          ref: 'heads/main'
        });

        // Create new branch
        await octokit.git.createRef({
          owner: config.github.owner,
          repo: config.github.repo,
          ref: `refs/heads/${branch}`,
          sha: mainRef.data.object.sha
        });

        // Get current sources file
        const sourcesFile = await octokit.repos.getContent({
          owner: config.github.owner,
          repo: config.github.repo,
          path: 'config/sources.json'
        });

        const sources = JSON.parse(
          Buffer.from(sourcesFile.data.content, 'base64').toString()
        );

        // Check for duplicate names
        if (sources.sources && sources.sources[args.name]) {
          throw new Error(`Source with name "${args.name}" already exists`);
        }

        // Add new source with enhanced metadata
        if (!sources.sources) sources.sources = {};
        sources.sources[args.name] = {
          url: args.url,
          type: args.type || 'rss',
          category: args.category || 'Other',
          enabled: true,
          priority: args.priority || 5,
          updateFrequency: args.updateFrequency || 'daily',
          description: args.description || `${args.type || 'RSS'} feed for ${args.name}`,
          tags: args.tags || [],
          language: args.language || 'en',
          reliability: args.reliability || 0.8,
          added_date: new Date().toISOString(),
          added_by: 'mcp-server'
        };

        // Update file
        await octokit.repos.createOrUpdateFileContents({
          owner: config.github.owner,
          repo: config.github.repo,
          path: 'config/sources.json',
          message: `Add news source: ${args.name}`,
          content: Buffer.from(JSON.stringify(sources, null, 2)).toString('base64'),
          sha: sourcesFile.data.sha,
          branch: branch
        });

        // Create PR with enhanced description
        const pr = await octokit.pulls.create({
          owner: config.github.owner,
          repo: config.github.repo,
          title: `Add news source: ${args.name}`,
          head: branch,
          base: 'main',
          body: `## Adding New News Source

**Source Details:**
- **Name**: \`${args.name}\`
- **URL**: ${args.url}
- **Type**: ${args.type || 'rss'}
- **Category**: ${args.category || 'Other'}

**Validation Checklist:**
- [ ] URL is accessible
- [ ] Feed format is valid
- [ ] Category is appropriate
- [ ] No duplicate sources

_This PR was created automatically by the MCP server._`
        });

        return { branch, pr };
      });

      return createSuccessResponse({
        source_name: args.name,
        branch: operations.branch,
        pull_request: {
          number: operations.pr.data.number,
          url: operations.pr.data.html_url,
          title: operations.pr.data.title
        }
      }, { operation: 'add_news_source' });

    } catch (error) {
      return createErrorResponse(`Failed to add news source: ${error.message}`);
    }
  },

  async generate_summary_report(args) {
    try {
      if (args.format) {
        validators.enum(args.format, 'format', ['markdown', 'json', 'html']);
      }

      const format = args.format || 'markdown';
      const includeTrends = args.include_trends !== false;

      // Fetch data in parallel
      const [newsData, trendsData] = await Promise.all([
        this.get_latest_news({ limit: 20 }),
        includeTrends ? this.analyze_trends({ days: 7 }) : Promise.resolve(null)
      ]);

      const news = JSON.parse(newsData.content[0].text).data;
      const trends = trendsData ? JSON.parse(trendsData.content[0].text).data : null;

      let report;

      if (format === 'markdown') {
        report = generateMarkdownReport(news, trends, includeTrends);
      } else if (format === 'json') {
        report = JSON.stringify({
          generated_at: new Date().toISOString(),
          summary: {
            total_articles: news.total_found,
            articles_shown: news.articles.length,
            last_updated: news.metadata?.last_updated
          },
          top_stories: news.articles.slice(0, 5),
          trends: includeTrends ? trends : null
        }, null, 2);
      } else {
        report = generateHTMLReport(news, trends, includeTrends);
      }

      return createSuccessResponse({
        report: report,
        format: format,
        includes_trends: includeTrends
      }, { 
        operation: 'generate_summary_report',
        generated_at: new Date().toISOString()
      });

    } catch (error) {
      return createErrorResponse(`Failed to generate report: ${error.message}`);
    }
  },

  async get_sources_config(args) {
    try {
      const cacheKey = `sources_config:${args.category || 'all'}:${args.enabled_only}`;
      if (cache.has(cacheKey)) {
        return createSuccessResponse(cache.get(cacheKey), { 
          operation: 'get_sources_config',
          cache_used: true 
        });
      }

      rateLimiter.check('github', config.rateLimit.github);

      // Get sources configuration file
      const sourcesFile = await withRetry(async () => {
        return await octokit.repos.getContent({
          owner: config.github.owner,
          repo: config.github.repo,
          path: 'config/sources.json'
        });
      });

      const sourcesConfig = JSON.parse(
        Buffer.from(sourcesFile.data.content, 'base64').toString()
      );

      let sources = sourcesConfig.sources || {};
      
      // Apply filters
      if (args.category && args.category !== 'all') {
        sources = Object.fromEntries(
          Object.entries(sources).filter(([, source]) => source.category === args.category)
        );
      }

      if (args.enabled_only !== false) {
        sources = Object.fromEntries(
          Object.entries(sources).filter(([, source]) => source.enabled !== false)
        );
      }

      const result = {
        metadata: sourcesConfig.metadata,
        categories: sourcesConfig.categories,
        sources: sources,
        quality_filters: sourcesConfig.quality_filters,
        summary: {
          total_sources: Object.keys(sources).length,
          by_category: {},
          by_type: {},
          average_reliability: 0
        }
      };

      // Calculate summary statistics
      const sourcesList = Object.values(sources);
      if (sourcesList.length > 0) {
        sourcesList.forEach(source => {
          result.summary.by_category[source.category] = (result.summary.by_category[source.category] || 0) + 1;
          result.summary.by_type[source.type] = (result.summary.by_type[source.type] || 0) + 1;
        });

        const reliabilityScores = sourcesList.filter(s => s.reliability).map(s => s.reliability);
        if (reliabilityScores.length > 0) {
          result.summary.average_reliability = reliabilityScores.reduce((a, b) => a + b, 0) / reliabilityScores.length;
        }
      }

      // Cache for 10 minutes
      cache.set(cacheKey, result, 600000);

      return createSuccessResponse(result, { operation: 'get_sources_config' });

    } catch (error) {
      return createErrorResponse(`Failed to get sources config: ${error.message}`);
    }
  },

  async update_source_config(args) {
    try {
      validators.required(args.name, 'name');

      rateLimiter.check('github', config.rateLimit.github);

      const sanitizedName = args.name.toLowerCase().replace(/[^a-z0-9]/g, '-');
      const branch = `update-source-${sanitizedName}-${Date.now()}`;

      const operations = await withRetry(async () => {
        // Get main branch reference
        const mainRef = await octokit.git.getRef({
          owner: config.github.owner,
          repo: config.github.repo,
          ref: 'heads/main'
        });

        // Create new branch
        await octokit.git.createRef({
          owner: config.github.owner,
          repo: config.github.repo,
          ref: `refs/heads/${branch}`,
          sha: mainRef.data.object.sha
        });

        // Get current sources file
        const sourcesFile = await octokit.repos.getContent({
          owner: config.github.owner,
          repo: config.github.repo,
          path: 'config/sources.json'
        });

        const sources = JSON.parse(
          Buffer.from(sourcesFile.data.content, 'base64').toString()
        );

        // Check if source exists
        if (!sources.sources || !sources.sources[args.name]) {
          throw new Error(`Source "${args.name}" not found`);
        }

        // Update source configuration
        const updates = {};
        if (args.enabled !== undefined) {
          sources.sources[args.name].enabled = args.enabled;
          updates.enabled = args.enabled;
        }
        if (args.priority !== undefined) {
          sources.sources[args.name].priority = args.priority;
          updates.priority = args.priority;
        }
        if (args.reliability !== undefined) {
          sources.sources[args.name].reliability = args.reliability;
          updates.reliability = args.reliability;
        }

        // Add last updated timestamp
        sources.sources[args.name].last_updated = new Date().toISOString();

        // Update file
        await octokit.repos.createOrUpdateFileContents({
          owner: config.github.owner,
          repo: config.github.repo,
          path: 'config/sources.json',
          message: `Update source config: ${args.name}`,
          content: Buffer.from(JSON.stringify(sources, null, 2)).toString('base64'),
          sha: sourcesFile.data.sha,
          branch: branch
        });

        // Create PR
        const pr = await octokit.pulls.create({
          owner: config.github.owner,
          repo: config.github.repo,
          title: `Update source configuration: ${args.name}`,
          head: branch,
          base: 'main',
          body: `## Update Source Configuration

**Source**: \`${args.name}\`

**Changes:**
${Object.entries(updates).map(([key, value]) => `- **${key}**: ${value}`).join('\n')}

_This PR was created automatically by the MCP server._`
        });

        return { branch, pr, updates };
      });

      return createSuccessResponse({
        source_name: args.name,
        updates: operations.updates,
        branch: operations.branch,
        pull_request: {
          number: operations.pr.data.number,
          url: operations.pr.data.html_url,
          title: operations.pr.data.title
        }
      }, { operation: 'update_source_config' });

    } catch (error) {
      return createErrorResponse(`Failed to update source config: ${error.message}`);
    }
  },

  async validate_sources(args) {
    try {
      // Get sources configuration
      const sourcesResult = await this.get_sources_config({ enabled_only: false });
      const sourcesData = JSON.parse(sourcesResult.content[0].text).data;
      
      let sourcesToValidate = Object.keys(sourcesData.sources);
      
      // Filter to specific sources if provided
      if (args.sources && args.sources.length > 0) {
        sourcesToValidate = args.sources.filter(name => sourcesData.sources[name]);
      }

      const validationResults = [];
      const batchSize = 5; // Validate in batches to avoid overwhelming servers

      for (let i = 0; i < sourcesToValidate.length; i += batchSize) {
        const batch = sourcesToValidate.slice(i, i + batchSize);
        
        const batchPromises = batch.map(async (sourceName) => {
          const source = sourcesData.sources[sourceName];
          const result = {
            name: sourceName,
            url: source.url,
            type: source.type,
            category: source.category,
            enabled: source.enabled,
            status: 'unknown',
            response_time: null,
            error: null,
            content_type: null,
            last_checked: new Date().toISOString()
          };

          try {
            const startTime = Date.now();
            
            const response = await fetch(source.url, {
              method: 'HEAD',
              timeout: 10000,
              headers: {
                'User-Agent': 'Daily AI Times Source Validator/1.0'
              }
            });

            result.response_time = Date.now() - startTime;
            result.status = response.ok ? 'accessible' : 'error';
            result.content_type = response.headers.get('content-type');
            
            if (!response.ok) {
              result.error = `HTTP ${response.status}: ${response.statusText}`;
            }

          } catch (error) {
            result.status = 'error';
            result.error = error.message;
          }

          return result;
        });

        const batchResults = await Promise.allSettled(batchPromises);
        validationResults.push(...batchResults.map(r => r.status === 'fulfilled' ? r.value : {
          name: 'unknown',
          status: 'error',
          error: r.reason?.message || 'Validation failed'
        }));

        // Small delay between batches
        if (i + batchSize < sourcesToValidate.length) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }

      const summary = {
        total_validated: validationResults.length,
        accessible: validationResults.filter(r => r.status === 'accessible').length,
        errors: validationResults.filter(r => r.status === 'error').length,
        average_response_time: null
      };

      const accessibleSources = validationResults.filter(r => r.response_time);
      if (accessibleSources.length > 0) {
        summary.average_response_time = Math.round(
          accessibleSources.reduce((sum, r) => sum + r.response_time, 0) / accessibleSources.length
        );
      }

      return createSuccessResponse({
        summary,
        results: validationResults,
        validation_timestamp: new Date().toISOString()
      }, { operation: 'validate_sources' });

    } catch (error) {
      return createErrorResponse(`Failed to validate sources: ${error.message}`);
    }
  }
};

// Report generation helpers
function generateMarkdownReport(news, trends, includeTrends) {
  let report = `# AI News Summary Report
Generated: ${new Date().toISOString()}

## Overview
- **Total articles found**: ${news.total_found}
- **Articles shown**: ${news.articles.length}
- **Last updated**: ${news.metadata?.last_updated || 'Unknown'}

## Top Stories
`;

  news.articles.slice(0, 5).forEach((article, index) => {
    report += `
### ${index + 1}. ${article.title}
- **Source**: ${article.source || 'Unknown'}
- **Category**: ${article.category || 'Uncategorized'}
- **Published**: ${article.published_date || 'Unknown'}
- **Score**: ${article.score || 'N/A'}
${article.summary || article.description ? `- **Summary**: ${article.summary || article.description}` : ''}
- **[Read more](${article.url})**
`;
  });

  if (includeTrends && trends) {
    report += `
## Trends Analysis (${trends.period})
- **Daily average**: ${trends.daily_average} articles

### Top Categories
${trends.top_categories.map(cat => `- **${cat.name}**: ${cat.count} articles`).join('\n')}

### Trending Keywords
${trends.trending_keywords.slice(0, 10).map(kw => `- **${kw.name}**: ${kw.count} mentions`).join('\n')}

### Most Active Sources
${trends.top_sources.slice(0, 8).map(src => `- **${src.name}**: ${src.count} articles`).join('\n')}

### Top Domains
${trends.top_domains.slice(0, 5).map(domain => `- **${domain.name}**: ${domain.count} articles`).join('\n')}
`;
  }

  report += `
---
*Report generated by Daily AI Times MCP Server*`;

  return report;
}

function generateHTMLReport(news, trends, includeTrends) {
  return `<!DOCTYPE html>
<html>
<head>
    <title>AI News Summary Report</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .article { border-left: 4px solid #007acc; padding-left: 16px; margin-bottom: 20px; }
        .trends { background: #f9f9f9; padding: 15px; border-radius: 5px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI News Summary Report</h1>
        <p><strong>Generated:</strong> ${new Date().toISOString()}</p>
        <p><strong>Total articles:</strong> ${news.total_found} | <strong>Shown:</strong> ${news.articles.length}</p>
    </div>
    
    <h2>Top Stories</h2>
    ${news.articles.slice(0, 5).map((article, index) => `
        <div class="article">
            <h3>${index + 1}. ${article.title}</h3>
            <p><strong>Source:</strong> ${article.source} | <strong>Category:</strong> ${article.category}</p>
            <p><strong>Published:</strong> ${article.published_date}</p>
            ${article.summary ? `<p>${article.summary}</p>` : ''}
            <p><a href="${article.url}" target="_blank">Read more →</a></p>
        </div>
    `).join('')}
    
    ${includeTrends && trends ? `
    <div class="trends">
        <h2>Trends (${trends.period})</h2>
        <div class="grid">
            <div>
                <h3>Top Categories</h3>
                <ul>
                ${trends.top_categories.map(cat => `<li>${cat.name}: ${cat.count}</li>`).join('')}
                </ul>
            </div>
            <div>
                <h3>Trending Keywords</h3>
                <ul>
                ${trends.trending_keywords.slice(0, 8).map(kw => `<li>${kw.name}: ${kw.count}</li>`).join('')}
                </ul>
            </div>
        </div>
    </div>
    ` : ''}
    
    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p><em>Report generated by Daily AI Times MCP Server</em></p>
    </footer>
</body>
</html>`;
}