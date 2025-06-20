// Date and formatting utilities
export class DateUtils {
    static formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        const formattedDate = date.toLocaleDateString('en-US', options);
        
        if (diffDays === 0) {
            return { relative: 'today', tooltip: formattedDate };
        } else if (diffDays === 1) {
            return { relative: '1 day ago', tooltip: formattedDate };
        } else {
            return { relative: `${diffDays} days ago`, tooltip: formattedDate };
        }
    }

    static formatHeaderDate(dateString) {
        const date = new Date(dateString);
        const options = { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        return date.toLocaleDateString('en-US', options);
    }

    static parseDate(dateString) {
        return new Date(dateString);
    }

    static compareDates(dateA, dateB) {
        return new Date(dateB) - new Date(dateA); // Newest first
    }

    static formatLastUpdated(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffMinutes = Math.floor(diffTime / (1000 * 60));
        const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffMinutes < 1) {
            return 'just now';
        } else if (diffMinutes < 60) {
            return `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`;
        } else if (diffHours < 24) {
            return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
        } else if (diffDays < 7) {
            return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
        } else {
            // For older dates, show the actual date
            const options = { 
                month: 'short', 
                day: 'numeric',
                year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
            };
            return date.toLocaleDateString('en-US', options);
        }
    }
}

// Text manipulation utilities
export class TextUtils {
    static truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text;
        return text.substring(0, maxLength).trim() + '...';
    }

    static sanitizeText(text) {
        if (!text) return '';
        return text.replace(/[<>]/g, '');
    }
}

// Article quality and filtering utilities
export class ArticleUtils {
    static isQualityArticle(article) {
        if (!article || !article.title || !article.description) return false;

        // Filter out GitHub events
        const githubEventTypes = ['WatchEvent', 'ForkEvent', 'PullRequestEvent', 'IssueCommentEvent', 'IssuesEvent', 'PushEvent', 'CreateEvent', 'DeleteEvent'];
        const isGitHubEvent = githubEventTypes.some(eventType => article.title.includes(eventType));
        
        // Filter out articles with very short descriptions or raw JSON
        const hasGoodDescription = article.description && 
                                 article.description.length > 100 && 
                                 !article.description.startsWith('{') && 
                                 !article.description.includes("'action':") &&
                                 !article.description.includes('"action"');
        
        // Filter out articles from certain sources that tend to be low quality
        const isFromGitHubEvents = article.source_id === 'github_openai' && isGitHubEvent;
        
        return !isGitHubEvent && hasGoodDescription && !isFromGitHubEvents;
    }

    static getArticlePriority(article) {
        const qualitySources = {
            'openai_blog': 100,
            'nvidia_blog': 95,
            'google_research_blog': 90,
            'towards_data_science': 85,
            'techcrunch_ai': 80,
            'pytorch_blog': 75,
            'huggingface_papers_api': 70
        };
        
        const sourceScore = qualitySources[article.source_id] || 50;
        const descriptionScore = Math.min(article.description.length / 10, 50);
        
        return sourceScore + descriptionScore + (article.score || 0);
    }

    static categorizeArticles(articles) {
        const researchArticles = articles
            .filter(article => article.category === 'Research')
            .sort((a, b) => {
                // Sort by published date (newest first), then by priority
                const dateA = new Date(a.published_date);
                const dateB = new Date(b.published_date);
                const dateDiff = dateB - dateA;
                if (dateDiff !== 0) return dateDiff;
                return this.getArticlePriority(b) - this.getArticlePriority(a);
            })
            .slice(0, 10);
        
        const regularArticles = articles
            .filter(article => article.category !== 'Research')
            .sort((a, b) => {
                // Sort by published date (newest first), then by priority
                const dateA = new Date(a.published_date);
                const dateB = new Date(b.published_date);
                const dateDiff = dateB - dateA;
                if (dateDiff !== 0) return dateDiff;
                return this.getArticlePriority(b) - this.getArticlePriority(a);
            })
            .slice(0, 15);

        return { researchArticles, regularArticles };
    }
}

// DOM manipulation utilities
export class DOMUtils {
    static createElement(tag, className = '', content = '') {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (content) element.innerHTML = content;
        return element;
    }

    static setElementContent(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
    }

    static addPressedEffect(element) {
        element.classList.add('pressed');
        setTimeout(() => {
            element.classList.remove('pressed');
        }, 200);
    }

    static showLoading(elementId) {
        this.setElementContent(elementId, '<div class="loading">Loading news...</div>');
    }

    static showError(elementId, message = 'Unable to load content') {
        this.setElementContent(elementId, `<div class="error-message">${message}</div>`);
    }
}
