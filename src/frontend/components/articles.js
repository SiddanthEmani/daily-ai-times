import { DateUtils, TextUtils, DOMUtils } from '../utils/utils.js';
import { Analytics } from '../utils/performance.js';

// Article rendering and interaction handling
export class ArticleRenderer {
    static createArticleHTML(article, isMainStory = false) {
        const dateInfo = DateUtils.formatDate(article.published_date);
        
        if (isMainStory) {
            return `
                <h2 class="main-headline">${TextUtils.sanitizeText(article.title)}</h2>
                <div class="decorative-line"></div>
                <p class="main-description">${TextUtils.sanitizeText(article.description)}</p>
                <div class="source">
                    <span>Source: ${TextUtils.sanitizeText(article.source)}</span>
                    <span class="date-info" data-tooltip="${dateInfo.tooltip}">${dateInfo.relative}</span>
                </div>
            `;
        } else {
            return `
                <article class="article" data-url="${article.url}" onclick="ArticleHandler.handleClick(this, '${article.url}')">
                    <h3 class="headline">${TextUtils.sanitizeText(TextUtils.truncateText(article.title, 80))}</h3>
                    <p class="description">${TextUtils.sanitizeText(TextUtils.truncateText(article.description, 200))}</p>
                    <div class="source">
                        <span>Source: ${TextUtils.sanitizeText(article.source)}</span>
                        <span class="date-info" data-tooltip="${dateInfo.tooltip}">${dateInfo.relative}</span>
                    </div>
                </article>
            `;
        }
    }

    static createResearchArticleHTML(article) {
        const dateInfo = DateUtils.formatDate(article.published_date);
        
        return `
            <article class="research-article" data-url="${article.url}" onclick="ArticleHandler.handleClick(this, '${article.url}')">
                <h3 class="research-headline">${TextUtils.sanitizeText(TextUtils.truncateText(article.title, 120))}</h3>
                ${article.author ? `<p class="research-author">By: ${TextUtils.sanitizeText(article.author)}</p>` : ''}
                <p class="research-description">${TextUtils.sanitizeText(TextUtils.truncateText(article.description, 300))}</p>
                <div class="research-source">
                    <span>Source: ${TextUtils.sanitizeText(article.source)}</span>
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
        
        // Distribute articles between columns
        articles.forEach((article, index) => {
            const articleHTML = this.createArticleHTML(article);
            
            if (index % 2 === 0) {
                column1.innerHTML += articleHTML;
            } else {
                column2.innerHTML += articleHTML;
            }
        });
    }

    static renderResearchGrid(articles) {
        const researchGrid = document.getElementById('research-grid');
        if (!researchGrid) return;

        researchGrid.innerHTML = '';
        
        articles.forEach(article => {
            const researchHTML = this.createResearchArticleHTML(article);
            researchGrid.innerHTML += researchHTML;
        });
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
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
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
