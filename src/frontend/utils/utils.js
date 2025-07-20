// Date and formatting utilities
export class DateUtils {
    static formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        
        // Check if date is valid
        if (isNaN(date.getTime())) {
            console.warn('Invalid date string:', dateString);
            return { relative: 'unknown date', tooltip: 'Invalid date' };
        }
        
        // Calculate difference in milliseconds
        const diffTime = now.getTime() - date.getTime();
        
        // Calculate days difference more accurately
        const msPerDay = 1000 * 60 * 60 * 24;
        const diffDays = Math.floor(diffTime / msPerDay);
        
        // Format the full date for tooltip
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        const formattedDate = date.toLocaleDateString('en-US', options);
        
        // Handle future dates
        if (diffTime < 0) {
            const futureDays = Math.floor(Math.abs(diffTime) / msPerDay);
            if (futureDays === 0) {
                return { relative: 'NEW', tooltip: formattedDate };
            } else if (futureDays === 1) {
                return { relative: 'tomorrow', tooltip: formattedDate };
            } else {
                return { relative: `in ${futureDays} days`, tooltip: formattedDate };
            }
        }
        
        // Handle same day (today)
        if (diffDays === 0) {
            return { relative: 'NEW', tooltip: formattedDate };
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
        
        // Find the last complete sentence within maxLength
        const truncated = text.substring(0, maxLength);
        const lastSentenceEnd = Math.max(
            truncated.lastIndexOf('.'),
            truncated.lastIndexOf('!'),
            truncated.lastIndexOf('?')
        );
        
        // If we found a sentence ending, use it
        if (lastSentenceEnd > text.length * 0.3) { // At least 30% of original length
            return text.substring(0, lastSentenceEnd + 1).trim();
        }
        
        // Otherwise, find the last complete word
        const lastSpace = truncated.lastIndexOf(' ');
        if (lastSpace > text.length * 0.3) {
            return text.substring(0, lastSpace).trim() + '...';
        }
        
        // Fallback to original behavior
        return text.substring(0, maxLength).trim() + '...';
    }

    static sanitizeText(text) {
        if (!text) return '';
        return text.replace(/[<>]/g, '');
    }
}

// Source formatting and icon utilities
export class SourceUtils {
    // Replace icon with text prefix
    static SOURCE_PREFIX = 'Source : ';
    
    static sourceMapping = {
        'the_verge': 'The Verge',
        'techcrunch_ai': 'TechCrunch', 
        'google_research_blog': 'Google Research',
        'nvidia_blog': 'NVIDIA Blog',
        'nvidia_developer': 'NVIDIA Developer',
        'openai_blog': 'OpenAI',
        'aws_ml_blog': 'AWS ML Blog',
        'azure_ai_blog': 'Azure AI Blog',
        'towards_ai': 'Towards AI',
        'towards_data_science': 'Towards Data Science',
        'mit_ai_news': 'MIT AI News',
        'ieee_spectrum_ai': 'IEEE Spectrum',
        'acm_ai_news': 'ACM News',
        'guardian_ai': 'The Guardian',
        'nist_ai_news': 'NIST AI News',
        'analytics_india_magazine': 'Analytics India Magazine',
        'the_decoder': 'The Decoder',
        'elastic_blog': 'Elastic Blog',
        'dataconomy': 'Dataconomy',
        'siliconangle_ai': 'SiliconANGLE',
        'ai_news': 'AI News',
        'tech_xplore': 'Tech Xplore',
        'pytorch_blog': 'PyTorch Blog',
        'huggingface_papers_api': 'Hugging Face Papers'
    };

    static formatSource(sourceKey) {
        const sourceName = this.sourceMapping[sourceKey];
        if (sourceName) {
            return {
                name: sourceName,
                prefix: this.SOURCE_PREFIX,
                formatted: `${this.SOURCE_PREFIX}${sourceName}`
            };
        }
        
        // Fallback for unknown sources - convert snake_case to Title Case
        const fallbackName = sourceKey
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
        
        return {
            name: fallbackName,
            prefix: this.SOURCE_PREFIX,
            formatted: `${this.SOURCE_PREFIX}${fallbackName}`
        };
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
        // Use article_type metadata to properly categorize articles
        
        // Find the headline article (should be exactly 1)
        const headlineArticles = articles
            .filter(article => article.article_type === 'headline')
            .sort((a, b) => {
                // Sort by published date (newest first), then by priority
                const dateA = new Date(a.published_date);
                const dateB = new Date(b.published_date);
                const dateDiff = dateB - dateA;
                if (dateDiff !== 0) return dateDiff;
                return this.getArticlePriority(b) - this.getArticlePriority(a);
            });
        
        // Get research papers
        const researchArticles = articles
            .filter(article => article.article_type === 'research')
            .sort((a, b) => {
                // Sort by published date (newest first), then by priority
                const dateA = new Date(a.published_date);
                const dateB = new Date(b.published_date);
                const dateDiff = dateB - dateA;
                if (dateDiff !== 0) return dateDiff;
                return this.getArticlePriority(b) - this.getArticlePriority(a);
            });
        
        // Get regular articles
        const regularArticles = articles
            .filter(article => article.article_type === 'article')
            .sort((a, b) => {
                // Sort by published date (newest first), then by priority
                const dateA = new Date(a.published_date);
                const dateB = new Date(b.published_date);
                const dateDiff = dateB - dateA;
                if (dateDiff !== 0) return dateDiff;
                return this.getArticlePriority(b) - this.getArticlePriority(a);
            });
        
        return {
            headlineArticles,
            researchArticles,
            regularArticles
        };
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
