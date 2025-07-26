/**
 * Pipeline Statistics Modal Component
 * Handles loading and displaying comprehensive pipeline statistics including
 * collection, bulk agent processing, and consensus stages
 */

export class StatsModal {
    constructor() {
        this.modal = document.getElementById('stats-modal');
        this.modalBody = document.getElementById('stats-modal-body');
        this.statsButton = document.getElementById('stats-button');
        this.closeButton = document.getElementById('modal-close');
        this.closeBottomButton = document.getElementById('modal-close-bottom');
        this.isLoading = false;
        this.statsData = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Open modal
        this.statsButton.addEventListener('click', () => {
            this.openModal();
        });
        
        // Close modal
        this.closeButton.addEventListener('click', () => {
            this.closeModal();
        });

        // Close modal - bottom button
        this.closeBottomButton.addEventListener('click', () => {
            this.closeModal();
        });
        
        // Close on overlay click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });
        
        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal.style.display !== 'none') {
                this.closeModal();
            }
        });
    }
    
    async openModal() {
        this.modal.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
        
        if (!this.statsData) {
            await this.loadStats();
        }
        
        this.renderStats();
    }
    
    closeModal() {
        this.modal.style.display = 'none';
        document.body.style.overflow = ''; // Restore scrolling
    }
    
    async loadStats() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.showLoadingState();
        
        try {
            // Add cache-busting timestamp to ensure fresh stats
            const timestamp = Date.now();
            const response = await fetch(`./api/stats.json?t=${timestamp}`, {
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.statsData = await response.json();
            console.log('Pipeline stats loaded:', this.statsData);
            
        } catch (error) {
            console.error('Failed to load pipeline stats:', error);
            this.showErrorState(error.message);
        } finally {
            this.isLoading = false;
        }
    }
    
    showLoadingState() {
        this.modalBody.innerHTML = `
            <div class="stats-loading">
                <div class="loading-spinner"></div>
                <p>Loading pipeline statistics...</p>
            </div>
        `;
    }
    
    showErrorState(errorMessage) {
        this.modalBody.innerHTML = `
            <div class="stats-error">
                <h3>Unable to Load Statistics</h3>
                <p>Error: ${errorMessage}</p>
                <p>Statistics will be available after the next pipeline run.</p>
                <button class="retry-button" onclick="window.statsModal.loadStats()">Retry</button>
            </div>
        `;
    }
    
    renderStats() {
        if (!this.statsData) {
            this.showErrorState('No statistics data available');
            return;
        }
        
        const stats = this.statsData.collection_stats;
        const html = this.createStatsHTML(stats);
        this.modalBody.innerHTML = html;
    }
    
    createStatsHTML(stats) {
        // Fix success rate calculation
        const successRate = stats.total_sources > 0 ? 
            Math.round((stats.successful_sources / stats.total_sources) * 100) : 0;
        
        // Get all pipeline stage stats
        const bulkStats = this.statsData.bulk_stage_stats || {};
        const consensusStats = this.statsData.consensus_stage_stats || {};
        const deepIntelligenceStats = this.statsData.deep_intelligence_stage_stats || {};
        const finalConsensusStats = this.statsData.final_consensus_stage_stats || {};
        
        return `
            <div class="stats-dashboard">
                <!-- Pipeline Funnel -->
                <div class="pipeline-funnel">
                    <!-- Outside Metrics -->
                    <div class="outside-metrics">
                        <div class="outside-metric">
                            <span class="metric-number">${stats.processing_time}s</span>
                            <span class="metric-text">Duration</span>
                        </div>
                        <div class="outside-metric">
                            <span class="metric-number">${successRate}%</span>
                            <span class="metric-text">Success</span>
                        </div>
                        <div class="outside-metric">
                            <span class="metric-number">${stats.total_sources}</span>
                            <span class="metric-text">Sources</span>
                        </div>
                    </div>

                    <!-- Funnel Container -->
                    <div class="funnel-container">
                        <!-- Collection -->
                        <div class="funnel-stage collection">
                            <div class="stage-label">Collection</div>
                            <div class="stage-metric">${stats.total_articles}</div>
                        </div>

                        <!-- Bulk Processing -->
                        ${bulkStats.total_agents ? `
                            <div class="funnel-stage bulk">
                                <div class="stage-label">Bulk AI</div>
                                <div class="stage-metric">${bulkStats.total_accepted || 0}</div>
                            </div>
                        ` : ''}

                        <!-- Consensus -->
                        ${consensusStats.total_articles ? `
                            <div class="funnel-stage consensus">
                                <div class="stage-label">Consensus</div>
                                <div class="stage-metric">${consensusStats.consensus_accepted || 0}</div>
                            </div>
                        ` : ''}

                        <!-- Deep Intelligence -->
                        ${deepIntelligenceStats.enabled ? `
                            <div class="funnel-stage deep">
                                <div class="stage-label">Deep AI</div>
                                <div class="stage-metric">${deepIntelligenceStats.total_articles_processed || 0}</div>
                            </div>
                        ` : ''}

                        <!-- Final -->
                        ${finalConsensusStats.total_articles ? `
                            <div class="funnel-stage final">
                                <div class="stage-label">Final</div>
                                <div class="stage-metric">${finalConsensusStats.accepted_articles || 0}</div>
                            </div>
                        ` : ''}
                    </div>
                </div>

                ${Object.keys(stats.failure_details || {}).length > 0 ? `
                    <div class="issues-section">
                        <h3 class="section-title">Issues</h3>
                        <div class="issues-list">
                            ${this.createSimpleIssuesList(stats.failure_details)}
                        </div>
                    </div>
                ` : ''}

                <!-- Timestamp -->
                <div class="stats-timestamp">
                    Generated: ${new Date(this.statsData.generated_at).toLocaleString()}
                </div>
            </div>
        `;
    }
    
    createSimpleIssuesList(failureDetails) {
        return Object.entries(failureDetails).map(([source, error]) => `
            <div class="issue-item">
                <strong>${this.formatSourceName(source)}:</strong> ${error}
            </div>
        `).join('');
    }
    
    formatSourceName(source) {
        // Convert source IDs to readable names
        const nameMap = {
            'techcrunch_ai': 'TechCrunch AI',
            'nvidia_blog': 'NVIDIA Blog',
            'openai_blog': 'OpenAI News',
            'deepmind_research': 'DeepMind Research',
            'uk_dsit_ai': 'UK Government AI',
            'nist_ai_news': 'NIST News',
            'towards_data_science': 'Towards Data Science',
            'huggingface_papers_api': 'HuggingFace Papers'
        };
        
        return nameMap[source] || source.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    formatAlgorithm(algorithm) {
        // Convert algorithm names to readable format
        const algorithmMap = {
            'weighted_voting': 'Weighted',
            'semantic_clustering': 'Semantic',
            'simple_majority': 'Majority'
        };
        
        return algorithmMap[algorithm] || algorithm || 'N/A';
    }
    
    truncateText(text, maxLength) {
        return text.length > maxLength ? text.substring(0, maxLength - 3) + '...' : text;
    }
    
    // Force refresh stats (useful for testing)
    async refreshStats() {
        this.statsData = null;
        await this.loadStats();
        if (this.modal.style.display !== 'none') {
            this.renderStats();
        }
    }
}

// Initialize stats modal when DOM is loaded
let statsModal;
document.addEventListener('DOMContentLoaded', () => {
    statsModal = new StatsModal();
    window.statsModal = statsModal; // Make available globally for debugging
}); 