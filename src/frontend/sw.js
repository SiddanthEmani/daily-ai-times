// Service Worker for NewsXP AI - Serverless Edition
const CACHE_NAME = 'newsxp-serverless-v1';
const DATA_CACHE = 'newsxp-data-v1';

// Static assets to cache - optimized for serverless
const STATIC_ASSETS = [
    './',
    './index.html',
    './styles/main.css',
    './components/app.js',
    './utils/utils.js',
    './components/articles.js',
    './utils/performance.js',
    './utils/dom-helpers.js',
    './utils/state-management.js',
    './assets/images/logo.png',
    './favicon.ico'
];

// API endpoints - never cache fresh news (serverless updates every 4h)
const NEVER_CACHE_ENDPOINTS = [
    './api/latest.json',
    './api/widget.json'
];

// API endpoints with longer cache tolerance
const CACHE_FRIENDLY_ENDPOINTS = [
    './api/archives.json' // Historical data changes less frequently
];

// Cache duration for different types of content (in milliseconds)
const CACHE_DURATIONS = {
    static: 24 * 60 * 60 * 1000,    // 24 hours for static assets
    api: 5 * 60 * 1000,             // 5 minutes for API data
    categories: 10 * 60 * 1000       // 10 minutes for category data
};

// Install event - cache static assets
self.addEventListener('install', event => {
    console.log('Service Worker installing...');
    event.waitUntil(
        Promise.all([
            caches.open(CACHE_NAME).then(cache => {
                console.log('Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            }),
            caches.open(DATA_CACHE).then(cache => {
                console.log('Data cache initialized');
                return cache;
            })
        ]).then(() => {
            console.log('Service Worker installation complete');
            return self.skipWaiting();
        })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    console.log('Service Worker activating...');
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME && cacheName !== DATA_CACHE) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            console.log('Service Worker activation complete');
            return self.clients.claim();
        })
    );
});

// Utility function to check if cached response is fresh
function isCacheResponseFresh(response, maxAge) {
    if (!response) return false;
    
    const cachedDate = response.headers.get('sw-cached-date');
    if (!cachedDate) return false;
    
    const cacheTime = new Date(cachedDate).getTime();
    const now = Date.now();
    
    return (now - cacheTime) < maxAge;
}

// Add timestamp to cached responses
function addCacheTimestamp(response) {
    const headers = new Headers(response.headers);
    headers.set('sw-cached-date', new Date().toISOString());
    
    return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: headers
    });
}

// Network-first strategy for API endpoints
async function networkFirstStrategy(request, cacheName, maxAge) {
    try {
        // Try network first
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            // Cache the fresh response
            const cache = await caches.open(cacheName);
            const responseToCache = addCacheTimestamp(networkResponse.clone());
            await cache.put(request, responseToCache);
            return networkResponse;
        }
        
        throw new Error(`Network response not ok: ${networkResponse.status}`);
    } catch (error) {
        console.log('Network failed, trying cache:', error.message);
        
        // Fallback to cache
        const cache = await caches.open(cacheName);
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            console.log('Serving from cache:', request.url);
            return cachedResponse;
        }
        
        // If no cache, return a fallback response for API endpoints
        if (request.url.includes('/api/')) {
            return new Response(JSON.stringify({
                error: 'Data temporarily unavailable',
                offline: true,
                timestamp: new Date().toISOString()
            }), {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        throw error;
    }
}

// Cache-first strategy for static assets
async function cacheFirstStrategy(request, cacheName, maxAge) {
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(request);
    
    // Check if cached response is fresh
    if (isCacheResponseFresh(cachedResponse, maxAge)) {
        return cachedResponse;
    }
    
    try {
        // Try to fetch fresh version
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            const responseToCache = addCacheTimestamp(networkResponse.clone());
            await cache.put(request, responseToCache);
            return networkResponse;
        }
        
        // If network fails but we have cache, use it
        if (cachedResponse) {
            return cachedResponse;
        }
        
        throw new Error(`Network response not ok: ${networkResponse.status}`);
    } catch (error) {
        // If everything fails and we have cache, use it
        if (cachedResponse) {
            console.log('Network failed, using stale cache:', request.url);
            return cachedResponse;
        }
        
        throw error;
    }
}

// Fetch event - intelligent caching strategies
self.addEventListener('fetch', event => {
    // Skip cross-origin requests
    if (!event.request.url.startsWith(self.location.origin)) {
        return;
    }
    
    // Skip non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }
    
    const url = new URL(event.request.url);
    const pathname = url.pathname;
    
    // Never cache fresh news endpoints - always fetch from network
    if (NEVER_CACHE_ENDPOINTS.some(endpoint => pathname.includes(endpoint.replace('./', '/')))) {
        event.respondWith(
            fetch(event.request, {
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache'
                }
            }).catch(() => {
                // Only provide fallback if completely offline
                return new Response(JSON.stringify({
                    error: 'Fresh news data temporarily unavailable',
                    offline: true,
                    timestamp: new Date().toISOString()
                }), {
                    status: 503,
                    headers: { 'Content-Type': 'application/json' }
                });
            })
        );
    } else if (API_ENDPOINTS.some(endpoint => pathname.includes(endpoint))) {
        // Other API endpoints: Network-first with short cache
        event.respondWith(
            networkFirstStrategy(event.request, DATA_CACHE, CACHE_DURATIONS.api)
        );
    } else if (pathname.includes('/api/categories/')) {
        // Category endpoints: Network-first with medium cache
        event.respondWith(
            networkFirstStrategy(event.request, DATA_CACHE, CACHE_DURATIONS.categories)
        );
    } else if (STATIC_ASSETS.includes(pathname) || pathname.match(/\.(css|js|png|jpg|jpeg|gif|svg|ico)$/)) {
        // Static assets: Cache-first with long cache
        event.respondWith(
            cacheFirstStrategy(event.request, CACHE_NAME, CACHE_DURATIONS.static)
        );
    } else {
        // Default: Try cache first, then network
        event.respondWith(
            caches.match(event.request).then(response => {
                return response || fetch(event.request).catch(() => {
        // Fallback for HTML pages when offline
                    if (event.request.headers.get('accept').includes('text/html')) {
                        return caches.match(`${BASE_PATH}/index.html`);
                    }
                    return new Response('Offline', { status: 503 });
                });
            })
        );
    }
});

// Background sync for failed requests (if supported)
self.addEventListener('sync', event => {
    if (event.tag === 'background-sync') {
        console.log('Background sync triggered');
        event.waitUntil(
            // Only sync non-news data to avoid serving stale news
            caches.open(DATA_CACHE).then(cache => {
                return fetch(`${BASE_PATH}/api/archives.json`).then(response => {
                    if (response.ok) {
                        return cache.put(`${BASE_PATH}/api/archives.json`, addCacheTimestamp(response));
                    }
                });
            }).catch(error => {
                console.log('Background sync failed:', error);
            })
        );
    }
});

// Handle messages from the main thread
self.addEventListener('message', event => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    } else if (event.data && event.data.type === 'CACHE_URLS') {
        // Preload specific URLs
        const urls = event.data.urls;
        event.waitUntil(
            caches.open(DATA_CACHE).then(cache => {
                return Promise.all(
                    urls.map(url => 
                        fetch(url).then(response => {
                            if (response.ok) {
                                return cache.put(url, addCacheTimestamp(response));
                            }
                        }).catch(error => {
                            console.log('Failed to preload:', url, error);
                        })
                    )
                );
            })
        );
    }
});
self.addEventListener('fetch', event => {
    // Skip cross-origin requests
    if (!event.request.url.startsWith(self.location.origin)) {
        return;
    }

    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Return cached version or fetch from network
                if (response) {
                    return response;
                }

                return fetch(event.request).then(response => {
                    // Don't cache non-successful responses
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response;
                    }

                    // Clone the response
                    const responseToCache = response.clone();

                    caches.open(CACHE_NAME)
                        .then(cache => {
                            cache.put(event.request, responseToCache);
                        });

                    return response;
                });
            })
            .catch(() => {
                // Return offline page for navigation requests
                if (event.request.mode === 'navigate') {
                    return caches.match(`${BASE_PATH}/index.html`);
                }
            })
    );
});

// Handle messages from main thread
self.addEventListener('message', event => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});
