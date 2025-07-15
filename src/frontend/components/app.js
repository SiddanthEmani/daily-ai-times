import { DateUtils, ArticleUtils, DOMUtils } from '../utils/utils.js';
import { ArticleRenderer, ArticleHandler } from './articles.js';
import { PerformanceMonitor, CacheManager, Analytics, LazyLoader } from '../utils/performance.js';

// Main news application class
export class NewsApp {
    constructor() {
        this.newsData = null;
        this.isLoading = false;
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
            
            // DISABLED: Service worker registration to prevent caching
            // Service workers can cache API responses which prevents fresh content
            
            // Unregister any existing service workers to prevent caching
            try {
                if ('serviceWorker' in navigator) {
                    const registrations = await navigator.serviceWorker.getRegistrations();
                    for (let registration of registrations) {
                        await registration.unregister();
                        console.log('Unregistered existing service worker');
                    }
                }
            } catch (swError) {
                console.warn('Failed to unregister service workers:', swError);
            }
            
            // Preload critical resources
            await this.preloadCriticalResources();
            
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

    async preloadCriticalResources() {
        // DISABLED: Preloading API resources to prevent caching
        // Preloading can cause browsers to cache API responses
        // Only preload static assets, never dynamic news data
        
        console.log('Skipping API preloading to ensure fresh content');
        
        // Don't preload API data - fetch fresh each time
        // Service worker should only cache static assets, not dynamic news data
    }

    showLoadingStates() {
        DOMUtils.showLoading('main-story');
        DOMUtils.showLoading('news-column-1');
        DOMUtils.showLoading('news-column-2');
        DOMUtils.showLoading('research-grid');
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
            // Use relative path for API with cache-busting parameter and timestamp
            const timestamp = Date.now();
            const randomSuffix = Math.random().toString(36).substring(7);
            const apiUrl = `./api/latest.json?t=${timestamp}&r=${randomSuffix}`;  // Double cache busting
            
            // Fetch news data with timeout and aggressive no-cache headers
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch(apiUrl, {
                signal: controller.signal,
                cache: 'no-store', // Prevent browser caching
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'If-Modified-Since': 'Mon, 26 Jul 1997 05:00:00 GMT', // Force fresh request
                    'If-None-Match': '*' // Disable ETag caching
                }
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.newsData = await response.json();
            this.performance.mark('news_fetch_complete');
            
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
            
            // Update edition info
            const editionInfo = `
                <div class="edition-left">
                    <span class="edition-name">Ramana Siddanth Emani</span>
                    <div class="social-icons">
                        <a href="https://github.com/SiddanthEmani" target="_blank" rel="noopener noreferrer" class="social-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                            </svg>
                        </a>
                        <a href="https://linkedin.com/in/siddanth-emani" target="_blank" rel="noopener noreferrer" class="social-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                            </svg>
                        </a>
                    </div>
                </div>
                <div class="audio-player"><audio controls src="assets/audio/latest-podcast.wav" title="Latest News Podcast"></audio></div>
                <div class="edition-right">
                    <span class="articles-count">${totalArticles} featured articles</span>
                    <br>
                    <span class="last-updated">Last updated: ${DateUtils.formatLastUpdated(generatedDate)}</span>
                </div>`;
                
            DOMUtils.setElementContent('edition-text', editionInfo);
            
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
