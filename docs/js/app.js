import { DateUtils, ArticleUtils, DOMUtils } from './utils.js';
import { ArticleRenderer, ArticleHandler } from './articles.js';
import { PerformanceMonitor, CacheManager, Analytics } from './performance.js';

// Main news application class
export class NewsApp {
    constructor() {
        this.newsData = null;
        this.isLoading = false;
        this.performance = new PerformanceMonitor();
    }

    async initialize() {
        try {
            this.performance.mark('app_init_start');
            
            // Register service worker for caching
            await CacheManager.register();
            
            // Show loading states
            this.showLoadingStates();
            
            // Initialize article handlers
            ArticleHandler.initializeTooltips();
            
            // Load news data
            await this.loadNews();
            
            this.performance.mark('app_init_complete');
            this.performance.report();
            
            // Track page view
            Analytics.trackPageView('home');
            
        } catch (error) {
            console.error('Failed to initialize news app:', error);
            this.showErrorStates();
        }
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
            // Fetch news data with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch('../data/news.json', {
                signal: controller.signal
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

        // Filter and categorize articles
        const qualityArticles = this.newsData.articles.filter(ArticleUtils.isQualityArticle);
        const { researchArticles, regularArticles } = ArticleUtils.categorizeArticles(qualityArticles);

        // Update header
        this.updateHeader(regularArticles.length + researchArticles.length);

        // Render content
        this.renderContent(regularArticles, researchArticles);
    }

    updateHeader(totalArticles) {
        try {
            // Update date
            const generatedDate = new Date(this.newsData.generated_at);
            const formattedDate = DateUtils.formatHeaderDate(generatedDate);
            DOMUtils.setElementContent('current-date', formattedDate);
            
            // Update edition info
            const editionInfo = `Ramana Siddanth Emani • ${totalArticles} articles featured`;
            DOMUtils.setElementContent('edition-info', editionInfo);
            
        } catch (error) {
            console.error('Error updating header:', error);
        }
    }

    renderContent(regularArticles, researchArticles) {
        try {
            // Render main story
            if (regularArticles.length > 0) {
                ArticleRenderer.renderMainStory(regularArticles[0]);
            }
            
            // Render news grid (remaining articles)
            const gridArticles = regularArticles.slice(1, 15);
            ArticleRenderer.renderNewsGrid(gridArticles);
            
            // Render research papers
            ArticleRenderer.renderResearchGrid(researchArticles);
            
        } catch (error) {
            console.error('Error rendering content:', error);
            this.showErrorStates();
        }
    }

    handleLoadError(error) {
        let errorMessage = 'Unable to load news';
        
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out. Please try again.';
        } else if (error.message.includes('HTTP error')) {
            errorMessage = 'News service temporarily unavailable';
        } else if (!navigator.onLine) {
            errorMessage = 'No internet connection detected';
        }

        // Show fallback content in main story
        const fallbackHTML = `
            <div class="category-tag">Error</div>
            <h2 class="main-headline">Unable to Load News</h2>
            <div class="decorative-line"></div>
            <p class="main-description">
                ${errorMessage}. Please try refreshing the page or check back later.
            </p>
        `;
        
        DOMUtils.setElementContent('main-story', fallbackHTML);
        DOMUtils.showError('news-column-1');
        DOMUtils.showError('news-column-2');
        DOMUtils.showError('research-grid');
    }

    // Public method to refresh news
    async refresh() {
        this.showLoadingStates();
        await this.loadNews();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new NewsApp();
    app.initialize();
    
    // Make app available globally for debugging
    window.newsApp = app;
});
