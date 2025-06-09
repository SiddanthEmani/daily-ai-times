// Performance and analytics utilities
export class PerformanceMonitor {
    constructor() {
        this.metrics = {};
        this.startTime = Date.now();
    }

    mark(name) {
        this.metrics[name] = Date.now() - this.startTime;
    }

    report() {
        console.log('Performance Metrics:', this.metrics);
        return this.metrics;
    }

    // Monitor Core Web Vitals
    observeWebVitals() {
        if ('web-vital' in window) {
            // This would integrate with a real web vitals library
            console.log('Web Vitals monitoring enabled');
        }
    }
}

// Service Worker registration for caching
export class CacheManager {
    static async register() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/sw.js');
                console.log('Service Worker registered:', registration);
                return registration;
            } catch (error) {
                console.log('Service Worker registration failed:', error);
            }
        }
    }

    static async updateCache() {
        if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
            navigator.serviceWorker.controller.postMessage({ type: 'SKIP_WAITING' });
        }
    }
}

// Image lazy loading
export class LazyLoader {
    static init() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        observer.unobserve(img);
                    }
                });
            });

            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        }
    }
}

// Analytics helper
export class Analytics {
    static trackEvent(eventName, properties = {}) {
        // This would integrate with your analytics service
        console.log('Event tracked:', eventName, properties);
        
        // Example: Google Analytics 4
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, properties);
        }
    }

    static trackPageView(page) {
        this.trackEvent('page_view', { page });
    }

    static trackArticleClick(articleTitle, articleUrl) {
        this.trackEvent('article_click', {
            article_title: articleTitle,
            article_url: articleUrl
        });
    }
}
