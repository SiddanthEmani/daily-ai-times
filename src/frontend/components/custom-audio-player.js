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
        this.isDragging = false;
        this.isHovered = false;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        this.injectStyles();
        this.createPlayer();
        this.setupAudioEvents();
        this.setupControlEvents();
        this.setupHoverEffects();
        this.initAudioVisualizer();
    }
    
    injectStyles() {
        if (document.querySelector('#custom-audio-player-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'custom-audio-player-styles';
        style.textContent = `
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Crimson+Text:ital,wght@0,400;0,600;1,400;1,600&family=Special+Elite&family=Orbitron:wght@400;500;600;700&family=News+Cycle:wght@400;700&display=swap');
            
            .audio-player-container {
                background: #FFFAF2;
                width: 400px;
                position: fixed;
                bottom: 24px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 1000;
                box-shadow: 0 8px 32px rgba(0,0,0,0.08);
                border-radius: 16px;
                overflow: hidden;
                margin: 0 auto;
                border: 1px solid #E8E4D8;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .audio-player-container:hover {
                box-shadow: 0 12px 40px rgba(0,0,0,0.12);
                transform: translateX(-50%) translateY(-2px);
            }
            
            .custom-audio-player {
                display: flex;
                flex-direction: column;
                padding: 0;
                font-family: 'Crimson Text', serif;
                background: transparent;
                align-items: center;
                position: relative;
                height: 80px;
                transition: height 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .custom-audio-player.expanded {
                height: 200px;
            }
            
            .visualizer-section {
                width: 100%;
                height: 0;
                background: #F5F1E8;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
                border-bottom: 1px solid #E8E4D8;
                transition: height 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                opacity: 0;
            }
            
            .visualizer-section.expanded {
                height: 120px;
                opacity: 1;
            }
            
            .visualizer-container {
                width: 100%;
                height: 100%;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .visualizer-title {
                position: absolute;
                top: 12px;
                left: 50%;
                transform: translateX(-50%);
                font-family: 'Crimson Text', serif;
                font-size: 10px;
                color: #000000;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                z-index: 10;
                font-weight: 600;
                text-align: center;
                width: 100%;
            }
            
            .main-visualizer {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                display: flex;
                align-items: flex-end;
                gap: 1px;
                height: 60px;
                width: 280px;
            }
            
            .visualizer-bar {
                flex: 1;
                background: #000000;
                border-radius: 1px;
                transition: height 0.1s ease;
                position: relative;
                min-width: 2px;
            }
            
            .controls-section {
                width: 100%;
                height: 80px;
                background: #FFFAF2;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 12px 20px;
                gap: 8px;
            }
            
            .audio-header {
                display: flex;
                justify-content: flex-end;
                align-items: center;
                width: 100%;
                margin-bottom: 4px;
            }
            
            .audio-status {
                font-size: 8px;
                color: #000000;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                font-family: 'Crimson Text', serif;
            }
            
            .audio-controls {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 16px;
                margin: 4px 0;
            }
            
            .audio-btn {
                width: 32px;
                height: 32px;
                border: 1px solid #000000;
                background: #FFFAF2;
                color: #000000;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
                transition: all 0.2s ease;
                border-radius: 50%;
                position: relative;
            }
            
            .audio-btn:hover {
                background: #000000;
                color: #FFFAF2;
                transform: scale(1.05);
            }
            
            .audio-btn:active {
                transform: scale(0.95);
            }
            
            .audio-btn.play-btn {
                width: 40px;
                height: 40px;
                font-size: 14px;
                border-width: 2px;
                border-color: #000000;
                color: #000000;
            }
            
            .audio-btn.play-btn:hover {
                background: #000000;
                color: #FFFAF2;
            }
            
            .audio-progress-container {
                position: relative;
                width: 100%;
                margin: 6px 0;
            }
            
            .audio-progress {
                width: 100%;
                height: 3px;
                background: #E8E4D8;
                border-radius: 2px;
                position: relative;
                cursor: pointer;
                margin: 0;
                overflow: hidden;
            }
            
            .audio-progress-fill {
                height: 100%;
                background: #000000;
                width: 0%;
                transition: width 0.1s ease;
                border-radius: 2px;
                position: relative;
            }
            
            .audio-progress-thumb {
                position: absolute;
                top: -2px;
                width: 8px;
                height: 8px;
                background: #000000;
                border: 1px solid #FFFAF2;
                border-radius: 50%;
                transform: translateX(-50%);
                cursor: pointer;
                transition: all 0.2s ease;
                left: 0%;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            }
            
            .audio-progress-thumb:hover {
                transform: translateX(-50%) scale(1.2);
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            }
            
            .audio-time-display {
                display: flex;
                justify-content: space-between;
                font-size: 9px;
                color: #000000;
                font-family: 'Crimson Text', serif;
                margin: 0;
                font-weight: 500;
                letter-spacing: 0.3px;
            }
            
            .audio-time {
                font-family: 'Crimson Text', serif;
                position: relative;
            }
            
            .audio-time.current {
                color: #2C2C2C;
                font-weight: 600;
            }
            
            .audio-loading {
                color: #000000;
                font-size: 10px;
                font-style: italic;
                text-align: center;
                padding: 4px;
                font-family: 'Crimson Text', serif;
            }
            
            /* Sticky positioning for audio player */
            .sticky-audio-player {
                position: fixed !important;
                bottom: 24px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 1000;
                box-shadow: 0 12px 40px rgba(0,0,0,0.12);
                max-width: calc(100vw - 48px);
            }
            
            /* Initial ease-in animation when becoming sticky */
            .sticky-audio-player.entering {
                animation: easeInFromBottom 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
            }
            
            @keyframes easeInFromBottom {
                0% {
                    transform: translateX(-50%) translateY(100%);
                    opacity: 0;
                }
                100% {
                    transform: translateX(-50%) translateY(0);
                    opacity: 1;
                }
            }
            
            /* Responsive design */
            @media (max-width: 600px) {
                .audio-player-container {
                    width: 360px;
                    bottom: 20px;
                }
                
                .custom-audio-player {
                    height: 80px;
                }
                
                .custom-audio-player.expanded {
                    height: 180px;
                }
                
                .visualizer-section {
                    height: 0;
                }
                
                .visualizer-section.expanded {
                    height: 100px;
                }
                
                .main-visualizer {
                    width: 240px;
                    height: 50px;
                }
                
                .controls-section {
                    height: 80px;
                    padding: 10px 16px;
                }
                
                .audio-controls {
                    gap: 12px;
                }
                
                .audio-btn {
                    width: 28px;
                    height: 28px;
                    font-size: 10px;
                }
                
                .audio-btn.play-btn {
                    width: 36px;
                    height: 36px;
                    font-size: 12px;
                }
                
                .sticky-audio-player {
                    bottom: 20px;
                    max-width: calc(100vw - 40px);
                }
            }
            
            @media (max-width: 480px) {
                .audio-player-container {
                    width: 320px;
                    bottom: 16px;
                }
                
                .custom-audio-player {
                    height: 80px;
                }
                
                .custom-audio-player.expanded {
                    height: 160px;
                }
                
                .visualizer-section {
                    height: 0;
                }
                
                .visualizer-section.expanded {
                    height: 80px;
                }
                
                .main-visualizer {
                    width: 200px;
                    height: 40px;
                }
                
                .controls-section {
                    height: 80px;
                    padding: 8px 12px;
                }
                
                .audio-controls {
                    gap: 10px;
                }
                
                .audio-btn {
                    width: 24px;
                    height: 24px;
                    font-size: 8px;
                }
                
                .audio-btn.play-btn {
                    width: 32px;
                    height: 32px;
                    font-size: 10px;
                }
                
                .sticky-audio-player {
                    bottom: 16px;
                    max-width: calc(100vw - 32px);
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    createPlayer() {
        this.container.innerHTML = `
            <div class="audio-player-container">
                <div class="custom-audio-player">
                    <div class="visualizer-section">
                        <div class="visualizer-container">
                            <div class="visualizer-title">Today's Headlines</div>
                            <div class="main-visualizer" id="mainVisualizer">
                                ${this.generateVisualizerBars()}
                            </div>
                        </div>
                    </div>
                    <div class="controls-section">
                        <div class="audio-header">
                            <span class="audio-status" id="audioStatus">Ready</span>
                        </div>
                        <div class="audio-controls">
                            <button class="audio-btn" id="skipBackBtn" aria-label="Skip Backward 10 seconds">⏮</button>
                            <button class="audio-btn play-btn" id="playBtn" aria-label="Play/Pause">▶</button>
                            <button class="audio-btn" id="skipForwardBtn" aria-label="Skip Forward 10 seconds">⏭</button>
                        </div>
                        <div class="audio-progress-container">
                            <div class="audio-progress" id="progress" style="display: none;">
                                <div class="audio-progress-fill" id="progressFill"></div>
                                <div class="audio-progress-thumb" id="progressThumb"></div>
                            </div>
                        </div>
                        <div class="audio-time-display" id="timeDisplay" style="display: none;">
                            <span class="audio-time current" id="currentTime">0:00</span>
                            <span class="audio-time" id="duration">0:00</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.elements = {
            container: this.container.querySelector('.audio-player-container'),
            player: this.container.querySelector('.custom-audio-player'),
            visualizerSection: this.container.querySelector('.visualizer-section'),
            playBtn: this.container.querySelector('#playBtn'),
            skipBackBtn: this.container.querySelector('#skipBackBtn'),
            skipForwardBtn: this.container.querySelector('#skipForwardBtn'),
            audioStatus: this.container.querySelector('#audioStatus'),
            progress: this.container.querySelector('#progress'),
            progressFill: this.container.querySelector('#progressFill'),
            progressThumb: this.container.querySelector('#progressThumb'),
            timeDisplay: this.container.querySelector('#timeDisplay'),
            currentTime: this.container.querySelector('#currentTime'),
            duration: this.container.querySelector('#duration'),
            mainVisualizer: this.container.querySelector('#mainVisualizer')
        };
        
        this.visualizerBarElements = this.elements.mainVisualizer.querySelectorAll('.visualizer-bar');
    }
    
    generateVisualizerBars() {
        let bars = '';
        for (let i = 0; i < 40; i++) {
            bars += `<div class="visualizer-bar" data-index="${i}" style="height: 5px;"></div>`;
        }
        return bars;
    }
    
    initAudioVisualizer() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 128;
            this.bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(this.bufferLength);
            
            const source = this.audioContext.createMediaElementSource(this.audio);
            source.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);
            
            this.startVisualizer();
        } catch (error) {
            console.log('Audio visualizer not supported:', error);
            this.fallbackVisualizer();
        }
    }
    
    startVisualizer() {
        const animate = () => {
            if (!this.isPlaying || !this.elements.visualizerSection.classList.contains('expanded')) {
                this.animationId = requestAnimationFrame(animate);
                return;
            }
            
            this.analyser.getByteFrequencyData(this.dataArray);
            this.updateVisualizer();
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    updateVisualizer() {
        const maxHeight = 50;
        
        this.visualizerBarElements.forEach((bar, index) => {
            const value = this.dataArray[index] || 0;
            const height = (value / 255) * maxHeight + 3;
            bar.style.height = `${height}px`;
        });
    }
    
    fallbackVisualizer() {
        // Fallback animation when Web Audio API is not available
        let frame = 0;
        
        const animate = () => {
            if (!this.isPlaying || !this.elements.visualizerSection.classList.contains('expanded')) {
                this.animationId = requestAnimationFrame(animate);
                return;
            }
            
            this.visualizerBarElements.forEach((bar, index) => {
                const time = frame * 0.05;
                const height = Math.sin(time + index * 0.2) * 15 + 20;
                bar.style.height = `${Math.max(3, height)}px`;
            });
            
            frame++;
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    setupAudioEvents() {
        this.audio.volume = this.volume;
        
        this.audio.addEventListener('loadedmetadata', () => {
            this.duration = this.audio.duration;
            this.updateDuration();
            this.showProgress();
            this.elements.audioStatus.textContent = 'Ready';
        });
        
        this.audio.addEventListener('timeupdate', () => {
            this.currentTime = this.audio.currentTime;
            this.updateProgress();
            this.updateCurrentTime();
        });
        
        this.audio.addEventListener('play', () => {
            this.elements.audioStatus.textContent = 'Playing';
            this.elements.container.classList.add('playing');
            this.expandVisualizer();
            this.resumeAudioContext();
        });
        
        this.audio.addEventListener('pause', () => {
            this.elements.audioStatus.textContent = 'Paused';
            this.elements.container.classList.remove('playing');
            this.collapseVisualizer();
        });
        
        this.audio.addEventListener('ended', () => {
            this.isPlaying = false;
            this.updatePlayButton();
            this.elements.audioStatus.textContent = 'Ended';
            this.elements.container.classList.remove('playing');
            this.collapseVisualizer();
        });
        
        this.audio.addEventListener('error', (e) => {
            console.error('Audio error:', e);
            this.elements.audioStatus.textContent = 'Error';
        });
        
        this.audio.addEventListener('loadstart', () => {
            this.elements.audioStatus.textContent = 'Loading...';
        });
        
        this.audio.addEventListener('canplay', () => {
            this.elements.audioStatus.textContent = 'Ready';
        });
    }
    
    resumeAudioContext() {
        if (this.audioContext && this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }
    
    setupControlEvents() {
        this.elements.playBtn.addEventListener('click', () => {
            this.togglePlay();
        });
        
        this.elements.skipBackBtn.addEventListener('click', () => {
            this.skipBackward();
        });
        
        this.elements.skipForwardBtn.addEventListener('click', () => {
            this.skipForward();
        });
        
        this.elements.progress.addEventListener('click', (e) => {
            this.seekTo(e);
        });
        
        this.elements.progressThumb.addEventListener('mousedown', (e) => {
            this.startDragging(e);
        });
        
        document.addEventListener('mousemove', (e) => {
            this.drag(e);
        });
        
        document.addEventListener('mouseup', () => {
            this.stopDragging();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.code) {
                case 'Space':
                    e.preventDefault();
                    this.togglePlay();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.skipBackward();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.skipForward();
                    break;
            }
        });
    }
    
    setupHoverEffects() {
        this.elements.container.addEventListener('mouseenter', () => {
            this.isHovered = true;
            this.elements.container.style.transform = 'translateX(-50%) translateY(-2px)';
        });
        
        this.elements.container.addEventListener('mouseleave', () => {
            this.isHovered = false;
            this.elements.container.style.transform = 'translateX(-50%) translateY(0)';
        });
    }
    
    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }
    
    play() {
        this.audio.play();
        this.isPlaying = true;
        this.updatePlayButton();
        this.hasStartedPlaying = true;
    }
    
    pause() {
        this.audio.pause();
        this.isPlaying = false;
        this.updatePlayButton();
    }
    
    skipBackward() {
        this.audio.currentTime = Math.max(0, this.audio.currentTime - 10);
        // Don't change status when skipping
    }
    
    skipForward() {
        this.audio.currentTime = Math.min(this.audio.duration, this.audio.currentTime + 10);
        // Don't change status when skipping
    }
    
    updatePlayButton() {
        this.elements.playBtn.textContent = this.isPlaying ? '⏸' : '▶';
    }
    
    updateProgress() {
        if (this.duration > 0) {
            const progress = (this.currentTime / this.duration) * 100;
            this.elements.progressFill.style.width = `${progress}%`;
            this.elements.progressThumb.style.left = `${progress}%`;
        }
    }
    
    updateCurrentTime() {
        this.elements.currentTime.textContent = this.formatTime(this.currentTime);
    }
    
    updateDuration() {
        this.elements.duration.textContent = this.formatTime(this.duration);
    }
    
    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    showProgress() {
        this.elements.progress.style.display = 'block';
        this.elements.timeDisplay.style.display = 'flex';
    }
    
    seekTo(e) {
        const rect = this.elements.progress.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        this.audio.currentTime = percent * this.duration;
    }
    
    startDragging(e) {
        this.isDragging = true;
        e.preventDefault();
    }
    
    drag(e) {
        if (!this.isDragging) return;
        const rect = this.elements.progress.getBoundingClientRect();
        const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        this.audio.currentTime = percent * this.duration;
    }
    
    stopDragging() {
        this.isDragging = false;
    }
    
    makeSticky() {
        this.elements.container.classList.add('sticky-audio-player');
        this.elements.container.classList.add('entering');
        setTimeout(() => {
            this.elements.container.classList.remove('entering');
        }, 500);
    }
    
    expandVisualizer() {
        this.elements.player.classList.add('expanded');
        this.elements.visualizerSection.classList.add('expanded');
    }
    
    resetVisualizerBars() {
        this.visualizerBarElements.forEach((bar) => {
            bar.style.height = '5px';
        });
    }
    
    collapseVisualizer() {
        this.elements.player.classList.remove('expanded');
        this.elements.visualizerSection.classList.remove('expanded');
        // Reset bars after a short delay to allow for smooth transition
        setTimeout(() => {
            this.resetVisualizerBars();
        }, 200);
    }
    
    removeSticky() {
        this.elements.container.classList.remove('sticky-audio-player');
    }
    
    resetPosition() {
        if (this.audio) {
            this.audio.currentTime = 0;
        }
    }
} 