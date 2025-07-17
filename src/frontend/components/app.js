import { DateUtils, ArticleUtils, DOMUtils } from '../utils/utils.js';
import { ArticleRenderer, ArticleHandler } from './articles.js';
import { PerformanceMonitor, Analytics, LazyLoader } from '../utils/performance.js';
import { CustomAudioPlayer } from './custom-audio-player.js';

// Main news application class
export class NewsApp {
    constructor() {
        this.newsData = null;
        this.isLoading = false;
        this.appVersion = '2024.1.0'; // Increment this when you want to force cache refresh
        this.currentRoute = this.detectRoute();
        try {
            this.performance = new PerformanceMonitor();
        } catch (error) {
            console.warn('PerformanceMonitor not available:', error);
            this.performance = {
                mark: () => {},
                report: () => {}
            };
        }
    }

    detectRoute() {
        const path = window.location.pathname;
        if (path.includes('/telugu')) {
            return 'telugu';
        }
        return 'default';
    }

    getApiEndpoint() {
        switch (this.currentRoute) {
            case 'telugu':
                return './api/telugu.json';
            default:
                return './api/latest.json';
        }
    }

    navigateToRoute(route) {
        const newPath = route === 'telugu' ? '/telugu' : '/';
        window.history.pushState({}, '', newPath);
        this.currentRoute = this.detectRoute();
        this.loadNews();
    }

    async initialize() {
        try {
            this.performance.mark('app_init_start');
            
            // Check for cached version and force refresh if needed
            this.checkVersionAndRefresh();
            
            // Initialize analytics
            try {
                if (typeof Analytics !== 'undefined') {
                    Analytics.init();
                } else {
                    console.warn('Analytics not available, skipping analytics initialization');
                }
            } catch (analyticsError) {
                console.warn('Analytics initialization failed:', analyticsError);
            }
            
            // No service worker registration - no caching
            
            // Show loading states
            this.showLoadingStates();
            
            // Initialize article handlers
            try {
                if (typeof ArticleHandler !== 'undefined') {
                    ArticleHandler.initializeTooltips();
                } else {
                    console.warn('ArticleHandler not available, skipping tooltip initialization');
                }
            } catch (handlerError) {
                console.warn('Article handler initialization failed:', handlerError);
            }
            
            // Initialize lazy loading
            try {
                if (typeof LazyLoader !== 'undefined') {
                    LazyLoader.init();
                } else {
                    console.warn('LazyLoader not available, skipping lazy loading initialization');
                }
            } catch (lazyError) {
                console.warn('Lazy loader initialization failed:', lazyError);
            }
            
            // Add cache-busting to logo
            this.addCacheBustingToAssets();
            
            // Load news data
            await this.loadNews();
            
            this.performance.mark('app_init_complete');
            this.performance.report();
            
            // Track page view
            if (typeof Analytics !== 'undefined') {
                Analytics.trackPageView(this.currentRoute === 'telugu' ? 'telugu' : 'home');
            }

            // Handle browser back/forward buttons
            window.addEventListener('popstate', () => {
                this.currentRoute = this.detectRoute();
                this.loadNews();
            });
            
        } catch (error) {
            console.error('Failed to initialize news app:', error);
            // Only track error if Analytics is available
            if (typeof Analytics !== 'undefined') {
                Analytics.trackError(error, { context: 'app_initialization' });
            }
            this.showErrorStates();
        }
    }



    showLoadingStates() {
        DOMUtils.showLoading('main-story');
        DOMUtils.showLoading('news-column-1');
        DOMUtils.showLoading('news-column-2');
        DOMUtils.showLoading('research-grid');
    }

    addCacheBustingToAssets() {
        // Add cache-busting timestamp to logo and other static assets
        const timestamp = Date.now();
        
        // Update logo src with cache-busting parameter
        const logoElement = document.getElementById('logo-icon');
        if (logoElement) {
            const originalSrc = logoElement.src.split('?')[0]; // Remove existing parameters
            logoElement.src = `${originalSrc}?t=${timestamp}`;
        }
        
        // Add cache-busting to favicon if needed
        const favicon = document.querySelector('link[rel="icon"]');
        if (favicon) {
            const originalHref = favicon.href.split('?')[0];
            favicon.href = `${originalHref}?t=${timestamp}`;
        }
        
        // Add cache-busting to CSS files
        const cssLinks = document.querySelectorAll('link[rel="stylesheet"]');
        cssLinks.forEach(link => {
            if (link.href && !link.href.includes('fonts.googleapis.com')) {
                const originalHref = link.href.split('?')[0];
                link.href = `${originalHref}?t=${timestamp}`;
            }
        });
        
        // Add cache-busting headers to all future fetch requests
        const originalFetch = window.fetch;
        window.fetch = function(url, options = {}) {
            // Only add cache-busting to same-origin requests
            if (typeof url === 'string' && (!url.startsWith('http') || url.startsWith(window.location.origin))) {
                options.headers = {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    ...options.headers
                };
                options.cache = 'no-store';
            }
            return originalFetch.call(this, url, options);
        };
    }

    checkVersionAndRefresh() {
        try {
            const storedVersion = localStorage.getItem('newsxp_app_version');
            const lastRefreshTime = localStorage.getItem('newsxp_last_refresh');
            const currentTime = Date.now();
            
            // Force refresh if:
            // 1. No stored version (first visit)
            // 2. Version mismatch (app updated)
            // 3. Last refresh was more than 1 hour ago (safety mechanism)
            const oneHour = 60 * 60 * 1000;
            const shouldForceRefresh = !storedVersion || 
                                     storedVersion !== this.appVersion || 
                                     (!lastRefreshTime || (currentTime - parseInt(lastRefreshTime)) > oneHour);
            
            if (shouldForceRefresh) {
                console.log('🔄 Forcing cache refresh for fresh content...');
                
                // Store current version and refresh time
                localStorage.setItem('newsxp_app_version', this.appVersion);
                localStorage.setItem('newsxp_last_refresh', currentTime.toString());
                
                // Only force refresh if this isn't the first load after a refresh
                // (prevent infinite refresh loop)
                const isRecentRefresh = lastRefreshTime && (currentTime - parseInt(lastRefreshTime)) < 5000;
                if (!isRecentRefresh) {
                    // Force hard refresh silently
                    window.location.reload(true); // Hard refresh
                    return true; // Indicate refresh is happening
                }
            }
            
            return false; // No refresh needed
        } catch (error) {
            console.warn('Version check failed:', error);
            return false;
        }
    }

    showErrorStates() {
        DOMUtils.showError('main-story', 'Unable to load main story');
        DOMUtils.showError('news-column-1', 'Unable to load news');
        DOMUtils.showError('news-column-2', 'Unable to load news');
        DOMUtils.showError('research-grid', 'Unable to load research papers');
    }



    async loadNews() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.performance.mark('news_load_start');
        
                try {
            // Simple fetch with cache-busting timestamp - no complex caching logic
            const timestamp = Date.now();
            const apiUrl = `${this.getApiEndpoint()}?t=${timestamp}&v=${this.appVersion}`;
            
            // Fetch news data with timeout and no-cache headers
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch(apiUrl, {
                signal: controller.signal,
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'If-Modified-Since': 'Thu, 01 Jan 1970 00:00:00 GMT'
                }
            });
            
            clearTimeout(timeoutId);
            
            // Handle 304 (Not Modified) as success - just means content hasn't changed
            if (!response.ok && response.status !== 304) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // For 304, try to get JSON anyway (some servers still return content)
            // If it fails, we'll catch it in the outer try-catch
            this.newsData = await response.json();
            this.performance.mark('news_fetch_complete');
            
            // Content freshness checked silently (no notifications)
            
            // Process and render news
            this.processAndRenderNews();
            this.performance.mark('news_render_complete');
            
        } catch (error) {
            console.error('Error loading news:', error);
            this.handleLoadError(error);
        } finally {
            this.isLoading = false;
        }
    }

    processAndRenderNews() {
        if (!this.newsData || !this.newsData.articles) {
            throw new Error('Invalid news data format');
        }

        // All articles are already filtered to top 25, no need for additional quality filtering
        const allArticles = this.newsData.articles;
        const { headlineArticles, researchArticles, regularArticles } = ArticleUtils.categorizeArticles(allArticles);

        // Update header with filtering info
        this.updateHeader(allArticles.length, this.newsData.filter_type);

        // **FIX**: Render content with proper headline distinction
        this.renderContent(headlineArticles, regularArticles, researchArticles);
    }

    updateHeader(totalArticles, filterType = 'keyword_based') {
        try {
            // Update date
            const generatedDate = new Date(this.newsData.generated_at);
            const formattedDate = DateUtils.formatHeaderDate(generatedDate);
            DOMUtils.setElementContent('current-date', formattedDate);
            
            // Update edition info with custom audio player
            const audioTimestamp = Date.now();
            const editionInfo = `
                <div class="edition-left">
                    <span class="edition-name">Ramana Siddanth Emani</span>
                    <div class="social-icons">
                        <a href="https://github.com/SiddanthEmani" target="_blank" rel="noopener noreferrer" class="social-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.30 3.297-1.30.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                            </svg>
                        </a>
                        <a href="https://linkedin.com/in/siddanth-emani" target="_blank" rel="noopener noreferrer" class="social-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                            </svg>
                        </a>
                    </div>
                </div>
                <div class="audio-player" id="custom-audio-container"></div>
                <div class="edition-right">
                    <span class="articles-count">${totalArticles} featured articles</span>
                    <br>
                    <span class="last-updated">Last updated: ${DateUtils.formatLastUpdated(generatedDate)}</span>
                </div>`;
                
            DOMUtils.setElementContent('edition-text', editionInfo);
            
            // Initialize custom audio player
            setTimeout(() => {
                const audioContainer = document.getElementById('custom-audio-container');
                if (audioContainer) {
                    new CustomAudioPlayer(`assets/audio/latest-podcast.wav?t=${audioTimestamp}`, audioContainer);
                }
            }, 100);
            
        } catch (error) {
            console.error('Error updating header:', error);
        }
    }



    renderContent(headlineArticles, regularArticles, researchArticles) {
        try {
            // Render designated headline as main story
            if (headlineArticles.length > 0) {
                console.log('Rendering headline as main story:', headlineArticles[0].title);
                try {
                    ArticleRenderer.renderMainStory(headlineArticles[0]);
                } catch (error) {
                    console.error('Error rendering main story:', error);
                    DOMUtils.showError('main-story', 'Unable to load main story');
                }
            } else {
                console.error('No headline article found in API response');
                // Only show error for main story, don't stop other content from loading
                DOMUtils.showError('main-story', 'No headline article available');
            }
            
            // Render regular articles in news grid - continue regardless of headline status
            try {
                ArticleRenderer.renderNewsGrid(regularArticles);
            } catch (error) {
                console.error('Error rendering news grid:', error);
                DOMUtils.showError('news-column-1', 'Unable to load news');
                DOMUtils.showError('news-column-2', 'Unable to load news');
            }
            
            // Render research papers separately - continue regardless of other sections
            try {
                if (researchArticles.length > 0) {
                    ArticleRenderer.renderResearchGrid(researchArticles);
                } else {
                    DOMUtils.showError('research-grid', 'No research papers available');
                }
            } catch (error) {
                console.error('Error rendering research papers:', error);
                DOMUtils.showError('research-grid', 'Unable to load research papers');
            }
            
            // Log the final article distribution for debugging
            console.log(`Article distribution: ${headlineArticles.length} headline, ${regularArticles.length} regular articles, ${researchArticles.length} research papers`);
            
        } catch (error) {
            console.error('Error in renderContent:', error);
            // Fallback: only show error states if everything fails
            this.showErrorStates();
        }
    }

    handleLoadError(error) {
        let errorMessage = 'Unable to load news';
        let errorCode = 'UNKNOWN_ERROR';
        
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out. Please check back later.';
            errorCode = 'TIMEOUT_ERROR';
        } else if (error.message.includes('HTTP error')) {
            errorMessage = 'News service temporarily unavailable';
            errorCode = 'HTTP_ERROR';
        } else if (!navigator.onLine) {
            errorMessage = 'No internet connection detected';
            errorCode = 'OFFLINE_ERROR';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Network connection issue';
            errorCode = 'NETWORK_ERROR';
        }

        // Track the error
        if (typeof Analytics !== 'undefined') {
            Analytics.trackError(error, { 
                context: 'news_loading',
                error_code: errorCode 
            });
        }

        // Show fallback content in main story
        const fallbackHTML = `
            <div class="category-tag error">Error</div>
            <h2 class="main-headline">Unable to Load News</h2>
            <div class="decorative-line"></div>
            <p class="main-description">
                ${errorMessage}. Please check back later.
            </p>
        `;
        
        DOMUtils.setElementContent('main-story', fallbackHTML);
        DOMUtils.showError('news-column-1', errorMessage);
        DOMUtils.showError('news-column-2', errorMessage);
        DOMUtils.showError('research-grid', errorMessage);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const app = new NewsApp();
        await app.initialize();
        
        // Make app available globally for debugging
        window.newsApp = app;
        
        console.log('✅ NewsXP AI app initialized successfully');
    } catch (error) {
        console.error('Failed to initialize news app:', error);
        
        // Show a basic error message to the user
        const errorMessage = 'Failed to load the news application. Please check back later.';
        document.body.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #333;">
                <h1>NewsXP AI</h1>
                <p style="color: #d32f2f;">${errorMessage}</p>
            </div>
        `;
    }
});
