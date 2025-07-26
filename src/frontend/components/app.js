import { DateUtils, ArticleUtils, DOMUtils } from '../utils/utils.js';
import { ArticleRenderer, ArticleHandler } from './articles.js';
import { PerformanceMonitor, Analytics, LazyLoader } from '../utils/performance.js';
import { CustomAudioPlayer } from './custom-audio-player.js';

// Main news application class
export class NewsApp {
    constructor() {
        this.newsData = null;
        this.categoryData = {};
        this.currentCategory = 'ai';
        this.isLoading = false;
        this.appVersion = '2024.2.0'; // Increment this when you want to force cache refresh
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
            
            // Initialize audio player
            this.initializeAudioPlayer();
            
            // Set up category navigation
            this.initializeCategoryNavigation();
            
            // Load news data for current category
            await this.loadCategoryNews(this.currentCategory);
            
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
        const categoryContent = document.getElementById('category-content');
        if (categoryContent) {
            categoryContent.innerHTML = '<div class="loading">Loading articles...</div>';
        }
    }

    initializeAudioPlayer() {
        try {
            // Update date
            const formattedDate = DateUtils.formatHeaderDate(new Date());
            DOMUtils.setElementContent('current-date', formattedDate);
            
            // Update edition info with custom audio player
            const audioTimestamp = Date.now();
            const editionInfo = `
                <div class="audio-player" id="custom-audio-container"></div>`;
                
            DOMUtils.setElementContent('edition-text', editionInfo);
            
            // Initialize custom audio player
            setTimeout(() => {
                const audioContainer = document.getElementById('custom-audio-container');
                if (audioContainer) {
                    new CustomAudioPlayer(`assets/audio/latest-podcast.wav?t=${audioTimestamp}`, audioContainer);
                }
            }, 100);
            
        } catch (error) {
            console.error('Error initializing audio player:', error);
        }
    }

    initializeCategoryNavigation() {
        // Set up category tab click handlers
        const categoryTabs = document.querySelectorAll('.category-tab');
        categoryTabs.forEach(tab => {
            tab.addEventListener('click', async (e) => {
                const category = e.target.getAttribute('data-category');
                if (category && category !== this.currentCategory) {
                    await this.switchToCategory(category);
                }
            });
        });
    }

    async switchToCategory(category) {
        // Update active tab
        document.querySelectorAll('.category-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-category="${category}"]`).classList.add('active');
        
        // Update current category
        this.currentCategory = category;
        
        // Show loading state
        this.showLoadingStates();
        
        // Load category content
        await this.loadCategoryNews(category);
        
        // Track category switch
        if (typeof Analytics !== 'undefined') {
            Analytics.trackEvent('category_switch', { category });
        }
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
            const storedVersion = localStorage.getItem('daily_ai_times_app_version');
            const lastRefreshTime = localStorage.getItem('daily_ai_times_last_refresh');
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
                localStorage.setItem('daily_ai_times_app_version', this.appVersion);
                localStorage.setItem('daily_ai_times_last_refresh', currentTime.toString());
                
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
        const categoryContent = document.getElementById('category-content');
        if (categoryContent) {
            categoryContent.innerHTML = `
                <div class="error-state">
                    <h3>Unable to load articles</h3>
                    <p>Please try again later or switch to a different category.</p>
                </div>
            `;
        }
    }



    async loadCategoryNews(category) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.performance.mark('category_load_start');
        
        try {
            // Check if we have cached data for this category
            if (this.categoryData[category]) {
                this.renderCategoryContent(this.categoryData[category], category);
                return;
            }
            
            // Fetch category-specific data
            const timestamp = Date.now();
            const apiUrl = `./api/categories/${category}.json?t=${timestamp}&v=${this.appVersion}`;
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000);
            
            const response = await fetch(apiUrl, {
                signal: controller.signal,
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok && response.status !== 304) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const categoryData = await response.json();
            this.categoryData[category] = categoryData;
            this.performance.mark('category_fetch_complete');
            
            // Render category content
            this.renderCategoryContent(categoryData, category);
            this.performance.mark('category_render_complete');
            
        } catch (error) {
            console.error(`Error loading ${category} category:`, error);
            this.handleCategoryLoadError(error, category);
        } finally {
            this.isLoading = false;
        }
    }

    renderCategoryContent(categoryData, category) {
        const categoryContent = document.getElementById('category-content');
        if (!categoryContent) return;
        
        const articles = categoryData.articles || [];
        if (articles.length === 0) {
            categoryContent.innerHTML = `
                <div class="no-articles">
                    <h3>No articles found for ${category.toUpperCase()}</h3>
                    <p>Check back later for new content in this category.</p>
                </div>
            `;
            return;
        }
        
        // Create two-column layout
        const leftColumnArticles = [];
        const rightColumnArticles = [];
        
        articles.forEach((article, index) => {
            if (index % 2 === 0) {
                leftColumnArticles.push(article);
            } else {
                rightColumnArticles.push(article);
            }
        });
        
        const leftColumnHtml = leftColumnArticles.map(article => this.createArticleHtml(article)).join('');
        const rightColumnHtml = rightColumnArticles.map(article => this.createArticleHtml(article)).join('');
        
        categoryContent.innerHTML = `
            <div class="category-header">
                <h2>${category.toUpperCase()} News</h2>
                <p>${articles.length} articles</p>
            </div>
            <div class="category-articles">
                <div class="category-column-left">
                    ${leftColumnHtml}
                </div>
                <div class="category-column-right">
                    ${rightColumnHtml}
                </div>
            </div>
        `;
        
        // Add click handlers to articles
        let articleIndex = 0;
        categoryContent.querySelectorAll('.category-article').forEach((articleEl) => {
            const currentIndex = articleIndex;
            articleEl.addEventListener('click', () => {
                this.openArticle(articles[currentIndex]);
            });
            articleIndex++;
        });
    }

    createArticleHtml(article) {
        const publishedDate = new Date(article.published_date).toLocaleDateString();
        const truncatedDescription = article.description.length > 100 
            ? article.description.substring(0, 100) + '...' 
            : article.description;
        
        return `
            <article class="category-article">
                <h3 class="category-article-title">${article.title}</h3>
                <div class="category-article-meta">
                    <span class="category-article-source">${article.source}</span>
                    <span>•</span>
                    <span class="category-article-date">${publishedDate}</span>
                </div>
                <p class="category-article-description">${truncatedDescription}</p>
            </article>
        `;
    }

    openArticle(article) {
        if (article.url) {
            window.open(article.url, '_blank', 'noopener,noreferrer');
        }
        
        // Track article click
        if (typeof Analytics !== 'undefined') {
            Analytics.trackEvent('article_click', {
                category: this.currentCategory,
                source: article.source,
                title: article.title.substring(0, 50)
            });
        }
    }

    handleCategoryLoadError(error, category) {
        let errorMessage = `Unable to load ${category} news`;
        let errorCode = 'UNKNOWN_ERROR';
        
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out. Please try again.';
            errorCode = 'TIMEOUT_ERROR';
        } else if (error.message.includes('HTTP error')) {
            errorMessage = 'News service temporarily unavailable';
            errorCode = 'HTTP_ERROR';
        } else if (!navigator.onLine) {
            errorMessage = 'No internet connection detected';
            errorCode = 'OFFLINE_ERROR';
        }

        // Track the error
        if (typeof Analytics !== 'undefined') {
            Analytics.trackError(error, { 
                context: 'category_loading',
                category: category,
                error_code: errorCode 
            });
        }

        const categoryContent = document.getElementById('category-content');
        if (categoryContent) {
            categoryContent.innerHTML = `
                <div class="error-state">
                    <h3>Unable to Load ${category.toUpperCase()} News</h3>
                    <p>${errorMessage}</p>
                    <button onclick="newsApp.loadCategoryNews('${category}')" class="retry-button">
                        Try Again
                    </button>
                </div>
            `;
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
                <div class="audio-player" id="custom-audio-container"></div>`;
                
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
        
        console.log('✅ Daily AI Times app initialized successfully');
    } catch (error) {
        console.error('Failed to initialize news app:', error);
        
        // Show a basic error message to the user
        const errorMessage = 'Failed to load the news application. Please check back later.';
        document.body.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #333;">
                <h1>Daily AI Times</h1>
                <p style="color: #d32f2f;">${errorMessage}</p>
            </div>
        `;
    }
});
