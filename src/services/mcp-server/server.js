import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { config, validateEnvironment, createErrorResponse, getSourceCategories } from './config.js';
import { toolDefinitions, toolImplementations } from './tools.js';

// Validate environment before starting
try {
  validateEnvironment();
} catch (error) {
  console.error('Environment validation failed:', error.message);
  process.exit(1);
}

// Initialize MCP server with optimized configuration
const server = new Server(
  {
    name: config.server.name,
    version: config.server.version,
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Tool list handler
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: toolDefinitions
}));

// Tool call handler with enhanced error handling and logging
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const startTime = Date.now();

  try {
    // Validate tool exists
    if (!toolImplementations[name]) {
      return createErrorResponse(`Unknown tool: ${name}`, {
        available_tools: Object.keys(toolImplementations)
      });
    }

    // Execute tool with proper context binding
    const result = await toolImplementations[name].call(toolImplementations, args || {});
    
    // Add execution metadata
    const executionTime = Date.now() - startTime;
    if (result.content && result.content[0] && result.content[0].text) {
      try {
        const parsed = JSON.parse(result.content[0].text);
        if (parsed.metadata) {
          parsed.metadata.execution_time_ms = executionTime;
          result.content[0].text = JSON.stringify(parsed, null, 2);
        }
      } catch (e) {
        // Not JSON, skip metadata injection
      }
    }

    console.error(`Tool ${name} executed in ${executionTime}ms`);
    return result;

  } catch (error) {
    const executionTime = Date.now() - startTime;
    console.error(`Tool ${name} failed after ${executionTime}ms:`, error.message);
    
    return createErrorResponse(
      `Tool execution failed: ${error.message}`,
      {
        tool: name,
        execution_time_ms: executionTime,
        args: args
      }
    );
  }
});

// Graceful shutdown handling
process.on('SIGINT', () => {
  console.error('Received SIGINT, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.error('Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

// Start server
async function startServer() {
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error(`${config.server.name} v${config.server.version} started successfully`);
    console.error(`Environment: GitHub Owner=${config.github.owner}, Repo=${config.github.repo}`);
    
    // Log sources configuration info
    if (config.sources) {
      const categories = getSourceCategories();
      const totalSources = Object.keys(config.sources.sources || {}).length;
      console.error(`Sources: ${totalSources} total, ${categories.length} categories: ${categories.join(', ')}`);
    } else {
      console.warn('No sources configuration loaded');
    }
  } catch (error) {
    console.error('Failed to start server:', error.message);
    process.exit(1);
  }
}

startServer();