import { DateUtils, TextUtils, DOMUtils, SourceUtils } from '../utils/utils.js';
import { Analytics } from '../utils/performance.js';

// Article rendering and interaction handling
export class ArticleRenderer {
    static createArticleHTML(article, isMainStory = false) {
        const dateInfo = DateUtils.formatDate(article.published_date);
        const sourceInfo = SourceUtils.formatSource(article.source);
        
        if (isMainStory) {
            return `
                <h2 class="main-headline">${TextUtils.sanitizeText(article.title)}</h2>
                <div class="decorative-line"></div>
                <p class="main-description">${TextUtils.sanitizeText(article.description)}</p>
                <div class="source">
                    <span class="source-formatted">${sourceInfo.formatted}</span>
                    <span class="date-info" data-tooltip="${dateInfo.tooltip}">${dateInfo.relative}</span>
                </div>
            `;
        } else {
            return `
                <article class="article" data-url="${article.url}" onclick="ArticleHandler.handleClick(this, '${article.url}')">
                    <h3 class="headline">${TextUtils.sanitizeText(TextUtils.truncateText(article.title, 80))}</h3>
                    <p class="description">${TextUtils.sanitizeText(TextUtils.truncateText(article.description, 200))}</p>
                    <div class="source">
                        <span class="source-formatted">${sourceInfo.formatted}</span>
                        <span class="date-info" data-tooltip="${dateInfo.tooltip}">${dateInfo.relative}</span>
                    </div>
                </article>
            `;
        }
    }

    static createResearchArticleHTML(article) {
        const dateInfo = DateUtils.formatDate(article.published_date);
        const sourceInfo = SourceUtils.formatSource(article.source);
        
        return `
            <article class="article" data-url="${article.url}" onclick="ArticleHandler.handleClick(this, '${article.url}')">
                <h3 class="headline">${TextUtils.sanitizeText(TextUtils.truncateText(article.title, 80))}</h3>
                ${article.author ? `<p class="byline">By: ${TextUtils.sanitizeText(article.author)}</p>` : ''}
                <p class="description">${TextUtils.sanitizeText(TextUtils.truncateText(article.description, 200))}</p>
                <div class="source">
                    <span class="source-formatted">${sourceInfo.formatted}</span>
                    <span class="date-info" data-tooltip="${dateInfo.tooltip}">${dateInfo.relative}</span>
                </div>
            </article>
        `;
    }

    static renderMainStory(article) {
        const mainStoryElement = document.getElementById('main-story');
        if (mainStoryElement && article) {
            const html = this.createArticleHTML(article, true);
            mainStoryElement.innerHTML = html;
            mainStoryElement.setAttribute('data-url', article.url);
            mainStoryElement.setAttribute('onclick', `ArticleHandler.handleClick(this, '${article.url}')`);
        }
    }

    static renderNewsGrid(articles) {
        const column1 = document.getElementById('news-column-1');
        const column2 = document.getElementById('news-column-2');
        
        if (!column1 || !column2) return;

        // Clear existing content
        column1.innerHTML = '';
        column2.innerHTML = '';
        
        // Distribute articles between columns for better balance
        // Put more articles in each column to accommodate up to 24 articles
        articles.forEach((article, index) => {
            const articleHTML = this.createArticleHTML(article);
            
            // Alternate between columns for even distribution
            if (index % 2 === 0) {
                column1.innerHTML += articleHTML;
            } else {
                column2.innerHTML += articleHTML;
            }
        });
        
        // Show article count for debugging if needed
        console.log(`Rendered ${articles.length} articles in news grid`);
    }

    static renderResearchGrid(articles) {
        const column1 = document.getElementById('research-column-1');
        const column2 = document.getElementById('research-column-2');
        
        if (!column1 || !column2) return;

        // Clear existing content
        column1.innerHTML = '';
        column2.innerHTML = '';
        
        // Distribute research articles between columns for better balance
        articles.forEach((article, index) => {
            const researchHTML = this.createResearchArticleHTML(article);
            
            // Alternate between columns for even distribution
            if (index % 2 === 0) {
                column1.innerHTML += researchHTML;
            } else {
                column2.innerHTML += researchHTML;
            }
        });
        
        // Show research article count for debugging
        console.log(`Rendered ${articles.length} research articles in research grid`);
    }
}

// Article click handling and interactions
export class ArticleHandler {
    static handleClick(element, url) {
        if (!element || !url) return;

        // Add pressed effect
        DOMUtils.addPressedEffect(element);
        
        // Track article click
        const title = element.querySelector('.headline, .main-headline, .research-headline')?.textContent || 'Unknown';
        Analytics.trackArticleClick(title, url);
        
        // Open article in new tab
        window.open(url, '_blank', 'noopener,noreferrer');
    }

    static initializeTooltips() {
        // Initialize tooltips for date info
        document.addEventListener('mouseover', (e) => {
            if (e.target.matches('.date-info[data-tooltip]')) {
                this.showTooltip(e.target, e.target.dataset.tooltip);
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.matches('.date-info[data-tooltip]')) {
                this.hideTooltip();
            }
        });
    }

    static showTooltip(element, text) {
        // Simple tooltip implementation
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: var(--primary-dark);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            z-index: 1000;
            pointer-events: none;
            white-space: nowrap;
            visibility: hidden;
        `;
        
        document.body.appendChild(tooltip);
        
        // Get element and tooltip dimensions
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        // Calculate initial position (above the element)
        let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
        let top = rect.top - tooltipRect.height - 5;
        
        // Adjust horizontal position if tooltip goes off-screen
        if (left < 5) {
            left = 5;
        } else if (left + tooltipRect.width > viewportWidth - 5) {
            left = viewportWidth - tooltipRect.width - 5;
        }
        
        // If tooltip goes above viewport, position it below the element
        if (top < 5) {
            top = rect.bottom + 5;
        }
        
        // Apply final position and make visible
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
        tooltip.style.visibility = 'visible';
    }

    static hideTooltip() {
        const tooltip = document.querySelector('.tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }
}

// Make ArticleHandler available globally for onclick handlers
window.ArticleHandler = ArticleHandler;
