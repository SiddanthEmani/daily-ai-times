import fetch from 'node-fetch';
import { Agent } from 'http';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load sources configuration
let sourcesConfig = null;
try {
  const sourcesPath = join(__dirname, '..', 'config', 'sources.json');
  sourcesConfig = JSON.parse(readFileSync(sourcesPath, 'utf8'));
} catch (error) {
  console.warn('Could not load sources.json configuration:', error.message);
}

// Configuration
export const config = {
  github: {
    owner: process.env.GITHUB_OWNER,
    repo: 'newsxp-ai',
    token: process.env.GITHUB_TOKEN,
    workflowId: 'collect-news.yml',
    apiVersion: '2022-11-28'
  },
  server: {
    name: 'newsxp-ai-mcp',
    version: '1.0.0',
    maxPollingAttempts: 30,
    pollingInterval: 2000,
    requestTimeout: 30000
  },
  cache: {
    defaultTTL: 300000, // 5 minutes
    maxSize: 100
  },
  rateLimit: {
    github: { requests: 5000, window: 3600000 }, // 5000/hour
    general: { requests: 100, window: 60000 }     // 100/minute
  },
  sources: sourcesConfig
};

// HTTP Agent with connection pooling
const httpAgent = new Agent({
  keepAlive: true,
  maxSockets: 10,
  timeout: config.server.requestTimeout
});

// Simple in-memory cache with TTL
class Cache {
  constructor() {
    this.data = new Map();
    this.timers = new Map();
  }

  set(key, value, ttl = config.cache.defaultTTL) {
    // Clear existing timer
    if (this.timers.has(key)) {
      clearTimeout(this.timers.get(key));
    }

    // Evict oldest if at max size
    if (this.data.size >= config.cache.maxSize && !this.data.has(key)) {
      const firstKey = this.data.keys().next().value;
      this.delete(firstKey);
    }

    this.data.set(key, value);
    
    // Set expiration timer
    const timer = setTimeout(() => this.delete(key), ttl);
    this.timers.set(key, timer);
  }

  get(key) {
    return this.data.get(key);
  }

  has(key) {
    return this.data.has(key);
  }

  delete(key) {
    if (this.timers.has(key)) {
      clearTimeout(this.timers.get(key));
      this.timers.delete(key);
    }
    return this.data.delete(key);
  }

  clear() {
    this.timers.forEach(timer => clearTimeout(timer));
    this.timers.clear();
    this.data.clear();
  }
}

export const cache = new Cache();

// Rate limiter
class RateLimiter {
  constructor() {
    this.limits = new Map();
  }

  check(key, limit = config.rateLimit.general) {
    const now = Date.now();
    const windowStart = now - limit.window;
    
    if (!this.limits.has(key)) {
      this.limits.set(key, []);
    }
    
    const requests = this.limits.get(key);
    
    // Remove old requests outside window
    const validRequests = requests.filter(time => time > windowStart);
    this.limits.set(key, validRequests);
    
    if (validRequests.length >= limit.requests) {
      const oldestRequest = Math.min(...validRequests);
      const resetTime = oldestRequest + limit.window;
      throw new Error(`Rate limit exceeded. Reset in ${Math.ceil((resetTime - now) / 1000)}s`);
    }
    
    // Add current request
    validRequests.push(now);
    return true;
  }
}

export const rateLimiter = new RateLimiter();

// Enhanced HTTP client with caching and error handling
export async function httpGet(url, options = {}) {
  const cacheKey = `http:${url}:${JSON.stringify(options)}`;
  
  // Check cache first
  if (cache.has(cacheKey) && !options.skipCache) {
    return cache.get(cacheKey);
  }

  // Rate limiting
  rateLimiter.check(`http:${new URL(url).hostname}`);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.server.requestTimeout);

  try {
    const response = await fetch(url, {
      agent: httpAgent,
      signal: controller.signal,
      headers: {
        'User-Agent': 'NewsXP-AI MCP Server/1.0',
        ...options.headers
      },
      ...options
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    
    // Cache successful responses
    if (!options.skipCache) {
      cache.set(cacheKey, data, options.cacheTTL);
    }
    
    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timeout after ${config.server.requestTimeout}ms`);
    }
    throw error;
  }
}

// Input validation helpers
export const validators = {
  required: (value, name) => {
    if (value === undefined || value === null || value === '') {
      throw new Error(`${name} is required`);
    }
    return value;
  },

  string: (value, name, options = {}) => {
    if (typeof value !== 'string') {
      throw new Error(`${name} must be a string`);
    }
    if (options.minLength && value.length < options.minLength) {
      throw new Error(`${name} must be at least ${options.minLength} characters`);
    }
    if (options.maxLength && value.length > options.maxLength) {
      throw new Error(`${name} must be at most ${options.maxLength} characters`);
    }
    if (options.pattern && !options.pattern.test(value)) {
      throw new Error(`${name} format is invalid`);
    }
    return value;
  },

  integer: (value, name, options = {}) => {
    const num = parseInt(value);
    if (isNaN(num)) {
      throw new Error(`${name} must be a valid integer`);
    }
    if (options.min !== undefined && num < options.min) {
      throw new Error(`${name} must be at least ${options.min}`);
    }
    if (options.max !== undefined && num > options.max) {
      throw new Error(`${name} must be at most ${options.max}`);
    }
    return num;
  },

  date: (value, name) => {
    const date = new Date(value);
    if (isNaN(date.getTime())) {
      throw new Error(`${name} must be a valid date (YYYY-MM-DD)`);
    }
    return date;
  },

  enum: (value, name, allowedValues) => {
    if (!allowedValues.includes(value)) {
      throw new Error(`${name} must be one of: ${allowedValues.join(', ')}`);
    }
    return value;
  }
};

// Error response helper
export function createErrorResponse(message, details = null) {
  return {
    content: [{
      type: 'text',
      text: JSON.stringify({
        error: message,
        details: details,
        timestamp: new Date().toISOString()
      }, null, 2)
    }]
  };
}

// Success response helper
export function createSuccessResponse(data, metadata = {}) {
  return {
    content: [{
      type: 'text',
      text: JSON.stringify({
        success: true,
        data: data,
        metadata: {
          timestamp: new Date().toISOString(),
          ...metadata
        }
      }, null, 2)
    }]
  };
}

// Exponential backoff utility
export async function withRetry(fn, maxAttempts = 3, baseDelay = 1000) {
  let lastError;
  
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (attempt === maxAttempts) {
        throw error;
      }
      
      const delay = baseDelay * Math.pow(2, attempt - 1);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError;
}

// Environment validation
export function validateEnvironment() {
  const required = ['GITHUB_OWNER', 'GITHUB_TOKEN'];
  const missing = required.filter(key => !process.env[key]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
  
  return true;
}

// Utility to get date range
export function getDateRange(days) {
  const endDate = new Date();
  const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);
  return { startDate, endDate };
}

// Extract top items from frequency object
export function getTopItems(obj, topN = 10) {
  return Object.entries(obj)
    .sort(([,a], [,b]) => b - a)
    .slice(0, topN)
    .map(([key, count]) => ({ name: key, count }));
}

// Sources configuration helpers
export function getSourceCategories() {
  if (!config.sources?.categories) return [];
  return Object.keys(config.sources.categories);
}

export function getEnabledSources(category = null) {
  if (!config.sources?.sources) return {};
  
  let sources = config.sources.sources;
  
  // Filter by category if specified
  if (category && category !== 'all') {
    sources = Object.fromEntries(
      Object.entries(sources).filter(([, source]) => source.category === category)
    );
  }
  
  // Filter enabled sources
  return Object.fromEntries(
    Object.entries(sources).filter(([, source]) => source.enabled !== false)
  );
}

export function getSourcesByReliability(minReliability = 0.8) {
  if (!config.sources?.sources) return {};
  
  return Object.fromEntries(
    Object.entries(config.sources.sources).filter(([, source]) => 
      source.reliability >= minReliability && source.enabled !== false
    )
  );
}

export function validateSourceConfig(source) {
  const errors = [];
  
  if (!source.url) errors.push('URL is required');
  if (!source.type) errors.push('Type is required');
  if (!source.category) errors.push('Category is required');
  
  if (source.reliability !== undefined && (source.reliability < 0 || source.reliability > 1)) {
    errors.push('Reliability must be between 0 and 1');
  }
  
  if (source.priority !== undefined && (source.priority < 1 || source.priority > 10)) {
    errors.push('Priority must be between 1 and 10');
  }
  
  return errors;
}