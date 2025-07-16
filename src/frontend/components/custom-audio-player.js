export class CustomAudioPlayer {
    constructor(src, container) {
        this.src = src;
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        this.audio = new Audio(src);
        this.isPlaying = false;
        this.currentTime = 0;
        this.duration = 0;
        this.volume = 0.7;
        
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
            .custom-audio-player {
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--paper-bg, #faf8f3);
                border: 1px solid var(--border-gray, #ddd);
                padding: 12px 16px;
                font-family: var(--font-masthead, 'Times New Roman', serif);
                gap: 16px;
                min-height: 48px;
                min-width: 320px;
                position: relative;
            }
            
            
            
            .audio-btn {
                width: 32px;
                height: 32px;
                border: 1px solid var(--primary-dark, #2c2c2c);
                background: var(--paper-bg, #faf8f3);
                color: var(--primary-dark, #2c2c2c);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.2s ease;
                border-radius: 50%;
            }
            
            .audio-btn:hover {
                background: var(--border-gray, #ddd);
            }
            
            .audio-btn:active {
                background: var(--light-gray, #888);
                color: white;
            }
            
            .audio-progress {
                flex: 1;
                height: 6px;
                background: var(--border-gray, #ddd);
                border: 1px solid var(--primary-dark, #2c2c2c);
                position: relative;
                cursor: pointer;
                margin: 0 8px;
            }
            
            .audio-progress-fill {
                height: 100%;
                background: var(--primary-dark, #2c2c2c);
                width: 0%;
                transition: width 0.2s ease;
            }
            
            .audio-progress-thumb {
                position: absolute;
                top: -6px;
                width: 18px;
                height: 18px;
                background: var(--paper-bg, #faf8f3);
                border: 2px solid var(--primary-dark, #2c2c2c);
                border-radius: 50%;
                transform: translateX(-50%);
                cursor: pointer;
                transition: all 0.2s ease;
                left: 0%;
            }
            
            .audio-progress-thumb:hover {
                transform: translateX(-50%) scale(1.1);
            }
            
            .audio-time-display {
                display: flex;
                align-items: center;
                gap: 4px;
                font-size: 11px;
                font-weight: 600;
                color: var(--primary-dark, #2c2c2c);
                font-variant-numeric: tabular-nums;
                min-width: 75px;
                white-space: nowrap;
            }
            
            .audio-time {
                font-family: 'Courier New', monospace;
            }
            

            
            .audio-loading {
                color: var(--primary-dark, #2c2c2c);
                font-size: 12px;
                font-style: italic;
            }
            
            @media (max-width: 480px) {
                .custom-audio-player {
                    min-width: 280px;
                    gap: 8px;
                    padding: 10px 12px;
                }
                .audio-time-display { 
                    font-size: 10px; 
                    min-width: 65px;
                }
                .audio-btn { width: 28px; height: 28px; font-size: 12px; }
            }
        `;
        document.head.appendChild(style);
    }
    
    createPlayer() {
        this.container.innerHTML = `
            <div class="custom-audio-player">
                <button class="audio-btn" id="playBtn" aria-label="Play/Pause">▶</button>
                <div class="audio-progress" id="progress">
                    <div class="audio-progress-fill" id="progressFill"></div>
                    <div class="audio-progress-thumb" id="progressThumb"></div>
                </div>
                <div class="audio-time-display">
                    <span class="audio-time" id="currentTime">0:00</span>
                    <span>/</span>
                    <span class="audio-time" id="duration">0:00</span>
                </div>
            </div>
        `;
        
        this.elements = {
            playBtn: this.container.querySelector('#playBtn'),
            progress: this.container.querySelector('#progress'),
            progressFill: this.container.querySelector('#progressFill'),
            progressThumb: this.container.querySelector('#progressThumb'),
            currentTime: this.container.querySelector('#currentTime'),
            duration: this.container.querySelector('#duration')
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
        
        // Progress bar
        this.elements.progress.addEventListener('click', (e) => this.seek(e));
        
        // Progress thumb drag
        let isDragging = false;
        this.elements.progressThumb.addEventListener('mousedown', () => isDragging = true);
        document.addEventListener('mousemove', (e) => {
            if (isDragging) this.seek(e);
        });
        document.addEventListener('mouseup', () => isDragging = false);
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.target.closest('.custom-audio-player')) {
                if (e.code === 'Space') {
                    e.preventDefault();
                    this.togglePlay();
                }
            }
        });
    }
    
    togglePlay() {
        if (this.isPlaying) {
            this.audio.pause();
            this.elements.playBtn.textContent = '▶';
        } else {
            this.audio.play();
            this.elements.playBtn.textContent = '⏸';
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
    
    // Public API
    play() { this.audio.play(); this.isPlaying = true; this.elements.playBtn.textContent = '⏸'; }
    pause() { this.audio.pause(); this.isPlaying = false; this.elements.playBtn.textContent = '▶'; }
    setSource(src) { this.audio.src = src; }
    destroy() { this.audio.pause(); this.audio.src = ''; this.container.innerHTML = ''; }
} 