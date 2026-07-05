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
        // Deterministic bar heights — purely decorative, echoes the mockup's waveform.
        this.waveBars = Array.from({ length: 16 }, (_, i) => 7 + Math.abs(Math.sin(i * 12.9898 + 4.1)) * 19);

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
                font-family: var(--font-masthead, 'Times New Roman', serif);
                width: 100%;
            }

            .audio-loading {
                color: var(--secondary-gray, #666);
                font-size: 12px;
                font-style: italic;
            }
        `;
        document.head.appendChild(style);
    }

    createPlayer() {
        const bars = this.waveBars.map((h, i) => `
            <span class="wave-bar" style="height:${Math.min(h, 26)}px;animation-duration:${(0.7 + (i % 5) * 0.12)}s;animation-delay:${(i * 0.025)}s"></span>
        `).join('');
        this.container.innerHTML = `
            <div class="audio-briefing custom-audio-player">
                <div class="audio-title">
                    <span class="audio-live-dot"></span>
                    <span class="audio-badge">TODAY'S BRIEFING</span>
                </div>
                <div class="audio-row">
                    <button class="audio-play" id="playBtn" aria-label="Play/Pause">▶</button>
                    <div class="audio-wave" aria-hidden="true">${bars}</div>
                </div>
                <div class="audio-track" id="progress">
                    <div class="audio-progress" id="progressFill"></div>
                </div>
                <div class="audio-time">
                    <span id="currentTime">0:00</span>
                    <span id="duration">--:--</span>
                </div>
            </div>
        `;

        this.elements = {
            container: this.container.querySelector('.audio-briefing'),
            playBtn: this.container.querySelector('#playBtn'),
            progress: this.container.querySelector('#progress'),
            progressFill: this.container.querySelector('#progressFill'),
            currentTime: this.container.querySelector('#currentTime'),
            duration: this.container.querySelector('#duration'),
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
            this.elements.container.classList.remove('is-playing');
        });
        
        this.audio.addEventListener('error', () => {
            this.elements.duration.textContent = 'Error';
        });
    }
    
    setupControlEvents() {
        // Play/Pause
        this.elements.playBtn.addEventListener('click', () => this.togglePlay());

        // Progress track — click/tap to seek
        this.elements.progress.addEventListener('click', (e) => this.seek(e));
        this.elements.progress.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.seek(e.touches[0]);
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
    }

    togglePlay() {
        if (this.isPlaying) {
            this.audio.pause();
            this.elements.playBtn.textContent = '▶';
            this.elements.container.classList.remove('is-playing');
        } else {
            this.hasStartedPlaying = true;
            this.audio.play();
            this.elements.playBtn.textContent = '❚❚';
            this.elements.container.classList.add('is-playing');
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
    play() { this.audio.play(); this.isPlaying = true; this.elements.playBtn.textContent = '❚❚'; this.elements.container.classList.add('is-playing'); }
    pause() { this.audio.pause(); this.isPlaying = false; this.elements.playBtn.textContent = '▶'; this.elements.container.classList.remove('is-playing'); }
    setSource(src) { this.audio.src = src; }
    destroy() {
        this.audio.pause();
        this.audio.src = '';
        this.container.replaceChildren();
    }
}
