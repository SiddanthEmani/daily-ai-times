// Enhanced DOM utilities with template literals and better event handling
export class TemplateEngine {
    static html(strings, ...values) {
        // Simple tagged template for safer HTML
        const result = strings.reduce((acc, str, i) => {
            const value = values[i] ? this.escapeHtml(values[i]) : '';
            return acc + str + value;
        }, '');
        return result;
    }

    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static createElement(tag, attrs = {}, children = []) {
        const element = document.createElement(tag);
        
        // Set attributes
        Object.entries(attrs).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'onclick') {
                element.addEventListener('click', value);
            } else {
                element.setAttribute(key, value);
            }
        });

        // Add children
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else if (child instanceof HTMLElement) {
                element.appendChild(child);
            }
        });

        return element;
    }
}

export class ComponentRenderer {
    static renderArticle(article, isMainStory = false) {
        const dateInfo = DateUtils.formatDate(article.published_date);
        
        if (isMainStory) {
            return TemplateEngine.createElement('div', {}, [
                TemplateEngine.createElement('h2', { className: 'main-headline' }, [article.title]),
                TemplateEngine.createElement('div', { className: 'decorative-line' }),
                TemplateEngine.createElement('p', { className: 'main-description' }, [article.description]),
                TemplateEngine.createElement('div', { className: 'source' }, [
                    TemplateEngine.createElement('span', {}, [`Source: ${article.source}`]),
                    TemplateEngine.createElement('span', { 
                        className: 'date-info',
                        'data-tooltip': dateInfo.tooltip 
                    }, [dateInfo.relative])
                ])
            ]);
        }
        
        return TemplateEngine.createElement('article', {
            className: 'article',
            'data-url': article.url,
            onclick: () => ArticleHandler.handleClick(article.url)
        }, [
            TemplateEngine.createElement('h3', { className: 'headline' }, [
                TextUtils.truncateText(article.title, 80)
            ]),
            TemplateEngine.createElement('p', { className: 'description' }, [
                TextUtils.truncateText(article.description, 200)
            ]),
            TemplateEngine.createElement('div', { className: 'source' }, [
                TemplateEngine.createElement('span', {}, [`Source: ${article.source}`]),
                TemplateEngine.createElement('span', { 
                    className: 'date-info',
                    'data-tooltip': dateInfo.tooltip 
                }, [dateInfo.relative])
            ])
        ]);
    }
}
