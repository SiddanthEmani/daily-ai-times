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
                Analytics.trackPageView('home');
            }
            
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
        DOMUtils.showLoading('research-column-1');
        DOMUtils.showLoading('research-column-2');
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
        DOMUtils.showError('research-column-1', 'Unable to load research papers');
        DOMUtils.showError('research-column-2', 'Unable to load research papers');
    }



    async loadNews() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.performance.mark('news_load_start');
        
                try {
            // Simple fetch with cache-busting timestamp - no complex caching logic
            const timestamp = Date.now();
            const apiUrl = `./api/latest.json?t=${timestamp}&v=${this.appVersion}`;
            
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

        // Render collection summary if available
        this.renderCollectionSummary(this.newsData.collection_summary);

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
                    <a href="https://github.com/SiddanthEmani/newsxp-ai" target="_blank" rel="noopener noreferrer" class="how-it-works-link">
                        How was this news generated?
                    </a>
                    <span class="articles-count">${totalArticles} featured articles</span>
                </div>
                <div class="audio-player" id="custom-audio-container"></div>
                <div class="edition-right">
                    <span class="update-frequency">Updated every 4 hours</span>
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

    renderCollectionSummary(summaryData) {
        const summarySection = document.getElementById('collection-summary');
        const summaryContent = document.getElementById('summary-content');
        
        if (!summarySection || !summaryContent) return;
        
        if (summaryData && summaryData.summary) {
            // Show and populate the summary section
            summarySection.style.display = 'block';
            
            let summaryHtml = `<p class="summary-text">${summaryData.summary}</p>`;
            
            // Add key themes if available
            if (summaryData.key_themes && summaryData.key_themes.length > 0) {
                summaryHtml += `
                    <div class="summary-themes">
                        <div class="themes-title">Key Themes:</div>
                        <div class="theme-tags">
                            ${summaryData.key_themes.map(theme => `<span class="theme-tag">${theme}</span>`).join('')}
                        </div>
                    </div>
                `;
            }
            
            summaryContent.innerHTML = summaryHtml;
        } else {
            // Hide the summary section if no summary available
            summarySection.style.display = 'none';
        }
    }

    renderContent(headlineArticles, regularArticles, researchArticles) {
        try {
            // Render designated headline as main story
            if (headlineArticles.length > 0) {
                console.log('Rendering headline as main story:', headlineArticles[0].title);
                ArticleRenderer.renderMainStory(headlineArticles[0]);
            } else {
                console.error('No headline article found in API response');
                throw new Error('No headline article available');
            }
            
            // Render regular articles in news grid
            ArticleRenderer.renderNewsGrid(regularArticles);
            
            // Render research papers separately
            if (researchArticles.length > 0) {
                ArticleRenderer.renderResearchGrid(researchArticles);
            }
            
            // Log the final article distribution for debugging
            console.log(`Article distribution: ${headlineArticles.length} headline, ${regularArticles.length} regular articles, ${researchArticles.length} research papers`);
            
        } catch (error) {
            console.error('Error rendering content:', error);
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
        DOMUtils.showError('research-column-1', errorMessage);
        DOMUtils.showError('research-column-2', errorMessage);
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
