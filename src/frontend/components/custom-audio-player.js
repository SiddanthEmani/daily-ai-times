export class CustomAudioPlayer {
    constructor(src, container) {
        this.src = src;
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        this.audio = new Audio(src);
        this.isPlaying = false;
        this.currentTime = 0;
        this.duration = 0;
        this.volume = 0.7;
        this.hasStartedPlaying = false;
        
        this.init();
    }
    
    init() {
        this.injectStyles();
        this.createPlayer();
        this.setupAudioEvents();
        this.setupControlEvents();
    }
    
    injectStyles() {
        if (document.querySelector('#custom-audio-player-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'custom-audio-player-styles';
        style.textContent = `
            .audio-player-container {
                background: var(--paper-bg, #faf8f3);
                border: 1px solid var(--border-gray, #ddd);
                border-radius: 12px;
                min-width: 380px;
                position: relative;
                overflow: hidden;
            }
            
            .audio-nav-tabs {
                display: flex;
                background: var(--paper-bg, #faf8f3);
                border-radius: 12px 12px 0 0;
                padding: 8px;
                gap: 4px;
                border-bottom: 1px solid var(--border-gray, #ddd);
            }
            
            .audio-nav-tab {
                flex: 1;
                padding: 12px 16px;
                text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 0.9rem;
                font-weight: 500;
                cursor: pointer;
                background: transparent;
                color: var(--secondary-gray, #666);
                transition: all 0.3s ease;
                border: none;
                border-radius: 8px;
                position: relative;
            }
            

            
            .audio-nav-tab:hover {
                background: rgba(44, 44, 44, 0.05);
                color: var(--primary-dark, #2c2c2c);
                transform: translateY(-1px);
            }
            
            .audio-nav-tab.active {
                background: var(--primary-dark, #2c2c2c);
                color: white;
                box-shadow: 0 2px 8px rgba(44, 44, 44, 0.2);
            }
            

            
            .custom-audio-player {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 16px 20px;
                font-family: var(--font-masthead, 'Times New Roman', serif);
                gap: 12px;
                min-height: 60px;
                position: relative;
                border-radius: 0 0 12px 12px;
                background: var(--paper-bg, #faf8f3);
            }
            

            
            .audio-label {
                font-weight: bold;
                font-size: 0.85rem;
                color: var(--primary-dark, #2c2c2c);
                letter-spacing: 2px;
                text-transform: uppercase;
                font-family: var(--font-masthead, 'Times New Roman', serif);
                cursor: pointer;
                transition: color 0.2s ease;
            }
            
            .audio-label:hover {
                color: var(--secondary-gray, #666);
            }
            
            
            
            .audio-btn {
                width: 44px;
                height: 44px;
                border: 1px solid var(--primary-dark, #2c2c2c);
                background: var(--paper-bg, #faf8f3);
                color: var(--primary-dark, #2c2c2c);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.15s ease;
                border-radius: 50%;
            }
            
            .audio-btn:hover {
                background: var(--border-gray, #ddd);
            }
            
            .audio-btn:active {
                background: var(--light-gray, #888);
                color: white;
                transform: translateY(1px);
            }
            
            .audio-progress {
                flex: 1;
                height: 12px;
                background: var(--border-gray, #ddd);
                border: 1px solid var(--secondary-gray, #666);
                border-radius: 6px;
                position: relative;
                cursor: pointer;
                margin: 0 12px;
                display: flex;
            }
            

            
            .audio-progress-fill {
                height: 100%;
                background: var(--primary-dark, #2c2c2c);
                width: 0%;
                transition: width 0.2s ease;
                border-radius: 4px;
            }
            
            .audio-progress-thumb {
                position: absolute;
                top: -10px;
                width: 16px;
                height: 28px;
                background: var(--paper-bg, #faf8f3);
                border: 1px solid var(--primary-dark, #2c2c2c);
                border-radius: 3px;
                transform: translateX(-50%);
                cursor: pointer;
                transition: all 0.2s ease;
                left: 0%;
            }
            
            .audio-progress-thumb:hover {
                transform: translateX(-50%) scale(1.05);
                background: var(--border-gray, #ddd);
            }
            
            .audio-time-display {
                display: flex;
                align-items: center;
                gap: 4px;
                font-size: 12px;
                font-weight: 600;
                color: var(--primary-dark, #2c2c2c);
                font-variant-numeric: tabular-nums;
                min-width: 85px;
                white-space: nowrap;
                background: var(--paper-bg, #faf8f3);
                padding: 6px 10px;
                border-radius: 4px;
                border: 1px solid var(--secondary-gray, #666);
                font-family: var(--font-masthead, 'Times New Roman', serif);
            }
            
            .audio-time {
                font-family: 'Courier New', monospace;
            }
            

            
            .audio-loading {
                color: var(--secondary-gray, #666);
                font-size: 12px;
                font-style: italic;
            }
            
            /* Sticky positioning for audio player */
            .sticky-audio-player {
                position: fixed !important;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 1000;
                box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
                max-width: calc(100vw - 40px);
            }
            
            /* Smooth transition when becoming sticky */
            .audio-player-container {
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            }
            
            .audio-player-container.sticky-transition {
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            }
            
            @media (max-width: 480px) {
                .audio-player-container {
                    min-width: 300px;
                }
                
                .custom-audio-player {
                    gap: 8px;
                    padding: 12px 16px;
                    min-height: 50px;
                }
                .audio-time-display { 
                    font-size: 10px; 
                    min-width: 65px;
                    padding: 3px 5px;
                }
                .audio-btn { 
                    width: 36px; 
                    height: 36px; 
                    font-size: 14px; 
                }
                .audio-label { 
                    font-size: 0.7rem; 
                    letter-spacing: 1px;
                }
                .audio-progress {
                    height: 12px;
                    margin: 0 8px;
                    cursor: pointer;
                }
                .audio-progress-thumb {
                    width: 16px;
                    height: 28px;
                    top: -10px;
                }
                .audio-nav-tabs {
                    padding: 6px;
                    gap: 3px;
                }
                
                .audio-nav-tab {
                    font-size: 0.8rem;
                    padding: 10px 12px;
                }
                
                .sticky-audio-player {
                    bottom: 60px;
                    max-width: calc(100vw - 20px);
                }
                
                .audio-player-container.sticky-transition {
                    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    createPlayer() {
        this.container.innerHTML = `
            <div class="audio-player-container">
                <div class="audio-nav-tabs">
                    <button class="audio-nav-tab active" data-target="headlines">Headlines</button>
                    <button class="audio-nav-tab" data-target="research">Research Papers</button>
                </div>
                <div class="custom-audio-player">
                    <button class="audio-btn" id="playBtn" aria-label="Play/Pause">▶</button>
                    <span class="audio-label">PLAY ME</span>
                    <div class="audio-progress" id="progress" style="display: none;">
                        <div class="audio-progress-fill" id="progressFill"></div>
                        <div class="audio-progress-thumb" id="progressThumb"></div>
                    </div>
                    <div class="audio-time-display" id="timeDisplay" style="display: none;">
                        <span class="audio-time" id="currentTime">0:00</span>
                        <span>/</span>
                        <span class="audio-time" id="duration">0:00</span>
                    </div>
                </div>
            </div>
        `;
        
        this.elements = {
            container: this.container.querySelector('.audio-player-container'),
            playBtn: this.container.querySelector('#playBtn'),
            audioLabel: this.container.querySelector('.audio-label'),
            progress: this.container.querySelector('#progress'),
            progressFill: this.container.querySelector('#progressFill'),
            progressThumb: this.container.querySelector('#progressThumb'),
            timeDisplay: this.container.querySelector('#timeDisplay'),
            currentTime: this.container.querySelector('#currentTime'),
            duration: this.container.querySelector('#duration'),
            navTabs: this.container.querySelectorAll('.audio-nav-tab')
        };
    }
    
    setupAudioEvents() {
        this.audio.volume = this.volume;
        
        this.audio.addEventListener('loadstart', () => {
            this.elements.duration.textContent = '...';
        });
        
        this.audio.addEventListener('loadedmetadata', () => {
            this.duration = this.audio.duration;
            this.elements.duration.textContent = this.formatTime(this.duration);
        });
        
        this.audio.addEventListener('timeupdate', () => {
            this.currentTime = this.audio.currentTime;
            this.updateProgress();
        });
        
        this.audio.addEventListener('ended', () => {
            this.isPlaying = false;
            this.elements.playBtn.textContent = '▶';
        });
        
        this.audio.addEventListener('error', () => {
            this.elements.duration.textContent = 'Error';
        });
    }
    
    setupControlEvents() {
        // Play/Pause
        this.elements.playBtn.addEventListener('click', () => this.togglePlay());
        
        // Audio label click to play
        this.elements.audioLabel.addEventListener('click', () => this.togglePlay());
        
        // Progress bar - click/tap support
        this.elements.progress.addEventListener('click', (e) => this.seek(e));
        this.elements.progress.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.seek(e.touches[0]);
        });
        
        // Progress thumb drag - mouse and touch support
        let isDragging = false;
        
        // Mouse events
        this.elements.progressThumb.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging) this.seek(e);
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        // Touch events
        this.elements.progressThumb.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDragging = true;
        });
        
        document.addEventListener('touchmove', (e) => {
            if (isDragging) {
                e.preventDefault();
                this.seek(e.touches[0]);
            }
        }, { passive: false });
        
        document.addEventListener('touchend', () => {
            isDragging = false;
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.target.closest('.custom-audio-player')) {
                if (e.code === 'Space') {
                    e.preventDefault();
                    this.togglePlay();
                }
            }
        });
        
        // Navigation tabs
        this.elements.navTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const target = tab.dataset.target;
                
                // Set active tab
                this.setActiveTab(target);
                
                // Scroll to target section
                this.scrollToSection(target);
            });
        });
        
        // Sticky audio player on scroll
        this.setupStickyBehavior();
        
        // Auto-switch active tab based on scroll position
        this.setupScrollActiveTab();
    }
    
    togglePlay() {
        if (this.isPlaying) {
            this.audio.pause();
            this.elements.playBtn.textContent = '▶';
        } else {
            // Show controls and hide label on first play
            if (!this.hasStartedPlaying) {
                this.elements.progress.style.display = 'flex';
                this.elements.timeDisplay.style.display = 'flex';
                this.elements.audioLabel.style.display = 'none';
                this.hasStartedPlaying = true;
            }
            this.audio.play();
            this.elements.playBtn.textContent = '■';
        }
        this.isPlaying = !this.isPlaying;
    }
    
    seek(e) {
        const rect = this.elements.progress.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        const time = percent * this.duration;
        this.audio.currentTime = Math.max(0, Math.min(time, this.duration));
    }
    

    
    updateProgress() {
        if (this.duration > 0) {
            const percent = (this.currentTime / this.duration) * 100;
            this.elements.progressFill.style.width = `${percent}%`;
            this.elements.progressThumb.style.left = `${percent}%`;
            this.elements.currentTime.textContent = this.formatTime(this.currentTime);
        }
    }
    

    
    formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    scrollToSection(target) {
        let sectionElement;
        if (target === 'headlines') {
            sectionElement = document.querySelector('.news-grid');
        } else if (target === 'research') {
            sectionElement = document.querySelector('.research-section');
        }
        
        if (sectionElement) {
            sectionElement.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }
    }
    
    setupStickyBehavior() {
        let isSticky = false;
        let isTransitioning = false;
        const originalRect = this.container.getBoundingClientRect();
        const originalTop = originalRect.top + window.scrollY;
        
        const handleScroll = () => {
            const currentScrollY = window.scrollY;
            const shouldBeSticky = currentScrollY > originalTop + 100; // Add some buffer
            
            if (shouldBeSticky && !isSticky && !isTransitioning) {
                isTransitioning = true;
                
                // Add transition class for smooth animation
                this.elements.container.classList.add('sticky-transition');
                
                // Small delay to ensure transition class is applied
                requestAnimationFrame(() => {
                    this.elements.container.classList.add('sticky-audio-player');
                    isSticky = true;
                    
                    // Clean up transition class after animation
                    setTimeout(() => {
                        this.elements.container.classList.remove('sticky-transition');
                        isTransitioning = false;
                    }, 400);
                });
                
            } else if (!shouldBeSticky && isSticky && !isTransitioning) {
                isTransitioning = true;
                
                // Add transition class for smooth animation
                this.elements.container.classList.add('sticky-transition');
                
                requestAnimationFrame(() => {
                    this.elements.container.classList.remove('sticky-audio-player');
                    isSticky = false;
                    
                    // Clean up transition class after animation
                    setTimeout(() => {
                        this.elements.container.classList.remove('sticky-transition');
                        isTransitioning = false;
                    }, 400);
                });
            }
        };
        
        window.addEventListener('scroll', handleScroll, { passive: true });
        
        // Store the function reference for cleanup
        this.scrollHandler = handleScroll;
    }
    
    setupScrollActiveTab() {
        // Get the sections we want to observe
        const headlinesSection = document.querySelector('.news-grid');
        const researchSection = document.querySelector('.research-section');
        
        if (!headlinesSection || !researchSection) return;
        
        // Create intersection observer
        const observerOptions = {
            root: null,
            rootMargin: '-20% 0px -60% 0px', // Trigger when section is 20% from top
            threshold: 0
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    let targetTab;
                    
                    if (entry.target.classList.contains('news-grid')) {
                        targetTab = 'headlines';
                    } else if (entry.target.classList.contains('research-section')) {
                        targetTab = 'research';
                    }
                    
                    if (targetTab) {
                        this.setActiveTab(targetTab);
                    }
                }
            });
        }, observerOptions);
        
        // Observe the sections
        observer.observe(headlinesSection);
        observer.observe(researchSection);
        
        // Store observer for cleanup
        this.sectionObserver = observer;
    }
    
    setActiveTab(target) {
        // Remove active class from all tabs
        this.elements.navTabs.forEach(tab => tab.classList.remove('active'));
        
        // Add active class to target tab
        const targetTab = Array.from(this.elements.navTabs).find(tab => tab.dataset.target === target);
        if (targetTab) {
            targetTab.classList.add('active');
        }
    }
    
    // Public API
    play() { this.audio.play(); this.isPlaying = true; this.elements.playBtn.textContent = '■'; }
    pause() { this.audio.pause(); this.isPlaying = false; this.elements.playBtn.textContent = '▶'; }
    setSource(src) { this.audio.src = src; }
    destroy() { 
        this.audio.pause(); 
        this.audio.src = ''; 
        if (this.scrollHandler) {
            window.removeEventListener('scroll', this.scrollHandler);
        }
        if (this.sectionObserver) {
            this.sectionObserver.disconnect();
        }
        this.container.innerHTML = ''; 
    }
} 