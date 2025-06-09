// Modern event delegation system
export class EventManager {
    constructor(container = document) {
        this.container = container;
        this.handlers = new Map();
    }

    on(selector, event, handler) {
        const key = `${event}:${selector}`;
        if (!this.handlers.has(key)) {
            this.handlers.set(key, new Set());
            this.container.addEventListener(event, (e) => {
                if (e.target.matches(selector)) {
                    this.handlers.get(key).forEach(h => h(e));
                }
            });
        }
        this.handlers.get(key).add(handler);
    }

    off(selector, event, handler) {
        const key = `${event}:${selector}`;
        if (this.handlers.has(key)) {
            this.handlers.get(key).delete(handler);
        }
    }
}

// Simple reactive state management
export class StateManager {
    constructor(initialState = {}) {
        this.state = { ...initialState };
        this.listeners = new Map();
    }

    setState(updates) {
        const oldState = { ...this.state };
        this.state = { ...this.state, ...updates };
        
        Object.keys(updates).forEach(key => {
            if (this.listeners.has(key)) {
                this.listeners.get(key).forEach(callback => {
                    callback(this.state[key], oldState[key]);
                });
            }
        });
    }

    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, new Set());
        }
        this.listeners.get(key).add(callback);
        
        // Return unsubscribe function
        return () => this.listeners.get(key).delete(callback);
    }

    getState() {
        return { ...this.state };
    }
}
