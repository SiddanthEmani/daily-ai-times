// Performance and analytics utilities - Enhanced Version
export class PerformanceMonitor {
    constructor() {
        this.metrics = {};
        this.startTime = performance.now();
        this.observers = new Map();
        this.initWebVitalsObserver();
    }

    mark(name) {
        this.metrics[name] = performance.now() - this.startTime;
        
        // Use Performance API for more accurate measurements
        if (performance.mark) {
            performance.mark(`daily_ai_times-${name}`);
        }
    }

    measure(name, startMark, endMark) {
        if (performance.measure && performance.mark) {
            try {
                performance.measure(`daily_ai_times-${name}`, `daily_ai_times-${startMark}`, `daily_ai_times-${endMark}`);
            } catch (e) {
                console.warn('Performance measurement failed:', e);
            }
        }
    }

    report() {
        const metrics = {
            ...this.metrics,
            navigationTiming: this.getNavigationTiming(),
            resourceTiming: this.getResourceTiming(),
            memoryUsage: this.getMemoryUsage()
        };
        
        console.group('📊 Performance Metrics');
        console.table(this.metrics);
        console.log('Navigation Timing:', metrics.navigationTiming);
        console.log('Memory Usage:', metrics.memoryUsage);
        console.groupEnd();
        
        return metrics;
    }

    getNavigationTiming() {
        if (!performance.timing) return null;
        
        const timing = performance.timing;
        return {
            dns: timing.domainLookupEnd - timing.domainLookupStart,
            tcp: timing.connectEnd - timing.connectStart,
            request: timing.responseStart - timing.requestStart,
            response: timing.responseEnd - timing.responseStart,
            domParsing: timing.domContentLoadedEventStart - timing.responseEnd,
            domReady: timing.domContentLoadedEventEnd - timing.navigationStart,
            pageLoad: timing.loadEventEnd - timing.navigationStart
        };
    }

    getResourceTiming() {
        if (!performance.getEntriesByType) return null;
        
        const resources = performance.getEntriesByType('resource');
        const apiRequests = resources.filter(r => r.name.includes('/api/'));
        
        return {
            totalResources: resources.length,
            apiRequests: apiRequests.length,
            averageApiTime: apiRequests.length > 0 
                ? apiRequests.reduce((sum, r) => sum + r.duration, 0) / apiRequests.length 
                : 0
        };
    }

    getMemoryUsage() {
        if (!performance.memory) return null;
        
        return {
            used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
            total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
            limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
        };
    }

    initWebVitalsObserver() {
        // Check if PerformanceObserver is supported
        if (!('PerformanceObserver' in window)) {
            console.log('PerformanceObserver not supported');
            return;
        }

        // Observe Long Tasks (performance bottlenecks)
        try {
            // Check if longtask is supported
            const supportedEntryTypes = PerformanceObserver.supportedEntryTypes || [];
            
            if (supportedEntryTypes.includes('longtask')) {
                const longTaskObserver = new PerformanceObserver((list) => {
                    list.getEntries().forEach((entry) => {
                        if (entry.duration > 50) {
                            console.warn(`Long task detected: ${entry.duration}ms`);
                            Analytics.trackEvent('long_task', {
                                duration: entry.duration,
                                startTime: entry.startTime
                            });
                        }
                    });
                });
                longTaskObserver.observe({ entryTypes: ['longtask'] });
                this.observers.set('longtask', longTaskObserver);
            }
            // Removed console.log for unsupported entry types to reduce noise
        } catch (e) {
            // Silently fail for unsupported performance observers
            console.debug('Long task observer initialization skipped:', e.message);
        }

        // Observe Layout Shifts
        try {
            const supportedEntryTypes = PerformanceObserver.supportedEntryTypes || [];
            
            if (supportedEntryTypes.includes('layout-shift')) {
                const clsObserver = new PerformanceObserver((list) => {
                    list.getEntries().forEach((entry) => {
                        if (entry.value > 0.1) {
                            console.warn(`Layout shift detected: ${entry.value}`);
                            Analytics.trackEvent('layout_shift', { value: entry.value });
                        }
                    });
                });
                clsObserver.observe({ entryTypes: ['layout-shift'] });
                this.observers.set('layout-shift', clsObserver);
            }
            // Removed console.log for unsupported entry types to reduce noise
        } catch (e) {
            // Silently fail for unsupported performance observers
            console.debug('Layout shift observer initialization skipped:', e.message);
        }
    }

    disconnect() {
        this.observers.forEach(observer => observer.disconnect());
        this.observers.clear();
    }
}



// Enhanced Lazy Loading with Intersection Observer
export class LazyLoader {
    static init() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        const src = img.dataset.src;
                        
                        if (src) {
                            // Create a new image to preload
                            const newImg = new Image();
                            newImg.onload = () => {
                                img.src = src;
                                img.classList.remove('lazy');
                                img.classList.add('loaded');
                            };
                            newImg.onerror = () => {
                                img.classList.add('error');
                                console.warn('Failed to load image:', src);
                            };
                            newImg.src = src;
                            
                            observer.unobserve(img);
                        }
                    }
                });
            }, {
                rootMargin: '50px', // Start loading 50px before the image is visible
                threshold: 0.1
            });

            // Observe all lazy images
            document.querySelectorAll('img[data-src]').forEach(img => {
                img.classList.add('lazy');
                imageObserver.observe(img);
            });
            
            // Content lazy loading for articles
            const contentObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                        Analytics.trackEvent('content_viewed', {
                            element: entry.target.tagName.toLowerCase(),
                            id: entry.target.id
                        });
                    }
                });
            }, { threshold: 0.5 });

            document.querySelectorAll('[data-lazy-content]').forEach(el => {
                contentObserver.observe(el);
            });

            return { imageObserver, contentObserver };
        } else {
            // Fallback for browsers without Intersection Observer
            document.querySelectorAll('img[data-src]').forEach(img => {
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                img.classList.add('loaded');
            });
            
            document.querySelectorAll('[data-lazy-content]').forEach(el => {
                el.classList.add('visible');
            });
            
            return null;
        }
    }
}

// Enhanced Analytics with better error handling and batching
export class Analytics {
    static eventQueue = [];
    static batchTimeout = null;
    static isInitialized = false;

    static init() {
        this.isInitialized = true;
        
        // Track page performance
        window.addEventListener('load', () => {
            setTimeout(() => {
                this.trackPagePerformance();
            }, 1000);
        });

        // Track errors
        window.addEventListener('error', (event) => {
            this.trackEvent('javascript_error', {
                message: event.message,
                filename: event.filename,
                line: event.lineno,
                column: event.colno
            });
        });

        // Track unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.trackEvent('promise_rejection', {
                reason: event.reason?.toString() || 'Unknown error'
            });
        });
    }

    static trackEvent(eventName, properties = {}) {
        if (!this.isInitialized) {
            this.init();
        }

        const event = {
            name: eventName,
            properties: {
                ...properties,
                timestamp: Date.now(),
                url: window.location.href,
                userAgent: navigator.userAgent.substring(0, 100) // Truncate for privacy
            }
        };

        // Add to queue for batching
        this.eventQueue.push(event);
        
        // Batch events to reduce requests
        if (this.batchTimeout) {
            clearTimeout(this.batchTimeout);
        }
        
        this.batchTimeout = setTimeout(() => {
            this.flushEvents();
        }, 1000);

        // Also log to console in development
        if (window.location.hostname === 'localhost' || window.location.hostname.includes('127.0.0.1')) {
            console.log('📊 Analytics Event:', eventName, properties);
        }

        // Send to Google Analytics if available
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, properties);
        } else if (window.location.hostname !== 'localhost' && !window.location.hostname.includes('127.0.0.1')) {
            // Only warn about missing gtag in production environments
            console.warn('Google Analytics (gtag) not available - check if GA script loaded correctly');
        }
    }

    static flushEvents() {
        if (this.eventQueue.length === 0) return;

        // Here you would send batched events to your analytics service
        // Only log in development mode to reduce console noise in production
        if (window.location.hostname === 'localhost' || window.location.hostname.includes('127.0.0.1')) {
            console.log('📊 Flushing analytics events:', this.eventQueue.length);
        }
        
        // Clear the queue
        this.eventQueue = [];
        this.batchTimeout = null;
    }

    static trackPageView(page) {
        this.trackEvent('page_view', { 
            page,
            referrer: document.referrer,
            loadTime: performance.timing ? 
                performance.timing.loadEventEnd - performance.timing.navigationStart : null
        });
    }

    static trackArticleClick(articleTitle, articleUrl) {
        this.trackEvent('article_click', {
            article_title: articleTitle.substring(0, 100), // Truncate for data limits
            article_url: articleUrl,
            click_time: Date.now()
        });
    }

    static trackPagePerformance() {
        if (!performance.timing) return;

        const timing = performance.timing;
        const loadTime = timing.loadEventEnd - timing.navigationStart;
        const domContentLoaded = timing.domContentLoadedEventEnd - timing.navigationStart;

        this.trackEvent('page_performance', {
            load_time: loadTime,
            dom_content_loaded: domContentLoaded,
            first_paint: performance.getEntriesByName('first-paint')[0]?.startTime || null,
            first_contentful_paint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || null
        });
    }

    static trackError(error, context = {}) {
        this.trackEvent('application_error', {
            message: error.message,
            stack: error.stack?.substring(0, 500), // Truncate stack trace
            context: JSON.stringify(context).substring(0, 200)
        });
    }
}
