"""
Centralized prompt management for the Daily AI Times pipeline.

This module contains all system prompts used across different agents and processing stages.
Separating prompts allows for easier maintenance, testing, and optimization.
"""

from typing import Dict, Any, List


class BulkIntelligencePrompts:
    """System prompts for AI news binary classification agents."""
    
    @staticmethod
    def get_base_prompt(article_count: int, model_name: str = "") -> str:
        """Get optimized system prompt for binary AI relevance classification.
        
        OPTIMIZED FOR BINARY AI CLASSIFICATION:
        - Simple true/false classification for AI relevance
        - Handles real-world scenarios and edge cases through intelligent prompting
        - No static title processing - relies on content analysis
        - Designed for high-volume processing (10,000+ articles across 500 sources)
        
        Args:
            article_count: Number of articles to be processed
            model_name: Name of the model (for model-specific adjustments)
            
        Returns:
            Complete system prompt for binary AI classification
        """
        # Model-specific JSON emphasis
        json_emphasis = ""
        if model_name == 'gemma2-9b-it':
            json_emphasis = "\n\nCRITICAL: You MUST return ONLY valid JSON. No explanatory text before or after. Start with { and end with }."
        
        # Binary classification prompt optimized for AI relevance detection
        base_prompt = f"""You are an AI Content Binary Classifier. Analyze EXACTLY {article_count} articles and classify each as AI-related or not.{json_emphasis}

TASK: For each article, determine if it is meaningfully related to Artificial Intelligence, Machine Learning, or AI applications.

INPUT FIELDS: Each article includes Title, Description, and Source. Use the Source to infer context (e.g., academic venues like arXiv, Nature, IEEE).

CLASSIFICATION CRITERIA - Mark as TRUE if the article discusses:
• AI/ML algorithms, models, or techniques (neural networks, deep learning, LLMs, etc.)
• AI company news, product launches, acquisitions, or business developments
• AI research breakthroughs, academic papers, or scientific developments  
• AI applications in any industry (healthcare, finance, entertainment, etc.)
• AI policy, regulation, ethics, or governance discussions
• AI tools, platforms, services, or infrastructure
• Machine learning implementations or data science applications
• Robotics with AI/ML components
• Computer vision, NLP, or other AI subfields
• AI impact on jobs, society, or economic trends

RESEARCH PAPER DETECTION (reduce false negatives):
• If the Source indicates an academic venue (e.g., arXiv, Nature, Science, ACM, IEEE, AAAI, NeurIPS, ICML, ICLR, bioRxiv), treat as a research paper signal.
• If such a signal is present AND Title/Description mention AI/ML, models, training, benchmarks, or terms like "preprint", "arXiv", or "DOI", then classify TRUE even if the description is brief.
• Prefer recall over precision specifically for academic sources about AI/ML to avoid missing AI-related research.

CLASSIFICATION CRITERIA - Mark as FALSE if the article:
• Only mentions AI tangentially or as background context
• Is primarily about non-AI technology (basic software, hardware without AI)
• Discusses general business/economic news without AI focus
• Covers entertainment, sports, politics without AI angles
• Reports on cybersecurity, data breaches without AI components
• Covers basic automation or traditional programming
• Discusses general tech trends without specific AI relevance

EDGE CASE HANDLING:
• If the Source indicates an academic venue and Title/Description suggest AI/ML, lean TRUE to avoid false negatives.
• If uncertain and no academic signals, lean towards FALSE to maintain precision
• Companies that use AI but article doesn't mention AI aspects → FALSE
• "Smart" products without explicit AI/ML mention → FALSE  
• Data analysis without ML/AI methods → FALSE
• General tech conferences unless AI is primary focus → FALSE

REQUIRED JSON FORMAT (respond with ONLY this structure):
{{
  "articles": [
    {{"is_ai_related": true}}"""
        
        # Add example for multiple articles
        if article_count > 1:
            base_prompt += """,
    {"is_ai_related": false}"""
        
        base_prompt += """
  ]
}

Remember: Return ONLY the JSON object above. Classify based on actual AI/ML content relevance, not peripheral mentions."""
        
        return base_prompt
    
    @staticmethod 
    def get_distribution_prompt() -> str:
        """Get optimized system prompt for token estimation during distribution.
        
        OPTIMIZED: Compressed version for accurate token calculations in binary AI classification.
        
        Returns:
            Simplified system prompt for distribution calculations
        """
        return """AI Content Binary Classifier. TRUE for AI/ML content, FALSE otherwise.

CRITERIA: AI algorithms/models, AI companies, AI research, AI applications, AI policy, ML implementations, robotics with AI, computer vision/NLP, AI economic impact.
RESEARCH SIGNAL: If Source indicates an academic venue (e.g., arXiv, Nature, Science, ACM, IEEE, AAAI, NeurIPS, ICML, ICLR, bioRxiv) and title/description indicate AI/ML, lean TRUE.

JSON: {"articles": [{"is_ai_related": true}]}"""


class DeepIntelligencePrompts:
    """System prompts for deep intelligence analysis agents."""
    
    @staticmethod
    def get_single_article_prompt(article: Dict[str, Any]) -> str:
        """Create comprehensive analysis prompt for single article deep intelligence processing.
        
        Args:
            article: Article dictionary containing title, content, source, etc.
            
        Returns:
            Complete analysis prompt for the specific article
        """
        # Extract existing scores if available
        consensus_score = article.get('consensus_multi_dimensional_score', {})
        initial_confidence = consensus_score.get('confidence_mean', 0.0)
        
        return f"""Deep AI news validation for pre-filtered article.

ARTICLE:
Title: {article.get('title', 'N/A')}
Source: {article.get('source', 'N/A')}
Content: {article.get('content', article.get('description', 'N/A'))[:1000]}

CONSENSUS: Overall={consensus_score.get('overall_score', 0.0):.2f}, Confidence={initial_confidence:.2f}

AI VALIDATION FOCUS:
1. AI RELEVANCE: Verify genuine AI/ML content (not just tech mentioning AI)
2. CATEGORY VALIDATION: Confirm proper AI application domain classification
3. CREDIBILITY: Source reputation for AI coverage, factual accuracy
4. IMPACT: Significance to AI ecosystem within the assigned category

CATEGORY VALIDATION (AI-focused domains):
- TECH: Core AI/ML developments, research, companies, infrastructure, AI tools
- ENTERTAINMENT: AI applications in content, gaming, media production, creative AI
- FINANCE: AI in financial services, trading algorithms, fintech, economic analysis
- HEALTH: Medical AI, diagnostics, healthcare applications, biotech, health AI
- POLITICS: AI policy, regulation, ethics, governance decisions, government AI

GUIDELINES:
- ACCEPT: Strong AI relevance with proper category fit (target 75% acceptance)
- CONDITIONAL: Moderate AI relevance, category validation needed
- REJECT: Weak AI connection or incorrect category classification

JSON Response:
{{
  "fact_check_confidence": 0.8,
  "bias_score": 0.3,
  "credibility_score": 0.8,
  "impact_potential": 0.7,
  "overall_score": 0.75,
  "confidence": 0.8,
  "recommendation": "ACCEPT",
  "key_insights": ["insight"],
  "risk_factors": ["risk"],
  "ai_relevance_score": 0.9,
  "category_validation": "confirmed",
  "category_probabilities": {{
    "tech": 0.8,
    "entertainment": 0.1,
    "finance": 0.05,
    "health": 0.025,
    "politics": 0.025
  }}
}}"""
    
    @staticmethod
    def get_batch_analysis_prompt(articles: List[Dict[str, Any]]) -> str:
        """Create optimized prompt for analyzing multiple articles in a single API call.
        
        Args:
            articles: List of article dictionaries to analyze
            specialization: Agent specialization area
            
        Returns:
            Optimized batch analysis prompt
        """
        # Create compact article summaries for token efficiency
        article_summaries = []
        for i, article in enumerate(articles):
            title = article.get('title', 'N/A')[:60]   # Shorter titles
            source = article.get('source', 'N/A')[:20]  # Truncate source names
            
            # Aggressive content truncation for batch processing
            content = article.get('description', article.get('content', 'N/A'))
            if len(content) > 150:
                content = content[:150] + "..."
            
            # Include key existing scores for context
            consensus_score = article.get('consensus_multi_dimensional_score', {})
            overall_score = consensus_score.get('overall_score', 0.0)
            
            # Compact format to save tokens
            article_summaries.append(f"{i+1}. {title} | {source} | {overall_score:.1f} | {content}")
        
        articles_text = "\n".join(article_summaries)
        
        return f"""Batch AI news validation for {len(articles)} pre-filtered articles.

FOCUS: AI relevance validation across categories

CRITERIA: 
- ACCEPT: Strong AI/ML content with proper category fit (target 75% acceptance)
- CONDITIONAL: Moderate AI relevance, needs category validation  
- REJECT: Weak AI connection or misclassified category

CATEGORIES: TECH (core AI), ENTERTAINMENT (creative AI), FINANCE (fintech AI), HEALTH (medical AI), POLITICS (AI policy)

ARTICLES:
{articles_text}

JSON Response:
{{
  "batch_analysis": [
    {{
      "article_id": 1,
      "overall_score": 0.75,
      "recommendation": "ACCEPT", 
      "confidence": 0.8,
      "ai_relevance_score": 0.9,
      "category_validation": "confirmed",
      "key_insights": ["AI relevance insight"],
      "risk_factors": ["category risk"]
    }},
    {{"article_id": 2, "overall_score": 0.6, "recommendation": "CONDITIONAL", "confidence": 0.7}}
  ]
}}"""


class CategorizationPrompts:
    """Minimal prompts for categorizing articles into 8 buckets."""

    CATEGORIES = ["U.S.", "World", "Industry", "Business", "Research", "Health", "Ethics", "Learn"]

    @staticmethod
    def get_batch_prompt(articles: List[Dict[str, Any]]) -> str:
        """Ultra-compact batch categorization prompt (title, source, description)."""
        lines: List[str] = []
        for i, a in enumerate(articles, start=1):
            title = (a.get('title') or 'N/A')[:60]
            source = (a.get('source') or a.get('url') or 'N/A')
            # Derive a short source label if a URL was provided
            try:
                import re
                from urllib.parse import urlparse
                if source and (source.startswith('http://') or source.startswith('https://')):
                    domain = urlparse(source).netloc
                    source = re.sub(r'^www\.', '', domain)[:20]
                else:
                    source = str(source)[:20]
            except Exception:
                source = str(source)[:20]
            desc = (a.get('description') or '')
            if len(desc) > 150:
                desc = desc[:150] + '...'
            lines.append(f"{i}. {title} | {source} | {desc}")

        summary = "\n".join(lines)
        cats = ", ".join(CategorizationPrompts.CATEGORIES)
        return (
            f"Categorize EXACTLY {len(articles)} items into one of: {cats}.\n"
            f"Return ONLY JSON with key 'batch': a list of objects: {{'i': index, 'c': category, 'p': probabilities, 'conf': number}}.\n"
            f"Rules: Use only Title, Source, Description. No explanations. JSON only. Categories must match list exactly.\n\n"
            f"ARTICLES:\n{summary}\n\n"
            f"JSON schema example: {{\"batch\":[{{\"i\":1,\"c\":\"U.S.\",\"p\":{{\"U.S.\":0.7,\"World\":0.3}},\"conf\":0.7}}]}}"
        )

    @staticmethod
    def get_distribution_prompt() -> str:
        return "Categorize news into: U.S., World, Industry, Business, Research, Health, Ethics, Learn. JSON only."

class PromptManager:
    """Centralized prompt management for the Daily AI Times pipeline."""
    
    @staticmethod
    def get_bulk_intelligence_prompt(article_count: int, model_name: str = "") -> str:
        """Get the bulk intelligence system prompt for AI news evaluation."""
        return BulkIntelligencePrompts.get_base_prompt(article_count, model_name)
    
    @staticmethod
    def get_bulk_intelligence_distribution_prompt() -> str:
        """Get simplified prompt for distribution calculations."""
        return BulkIntelligencePrompts.get_distribution_prompt()
    
    @staticmethod
    def get_deep_intelligence_prompt(article: Dict[str, Any]) -> str:
        """Get the deep intelligence system prompt for AI validation."""
        return DeepIntelligencePrompts.get_single_article_prompt(article)
    
    @staticmethod
    def get_deep_intelligence_distribution_prompt() -> str:
        """Get simplified deep intelligence prompt for distribution calculations."""
        return "AI news validation for pre-filtered articles. Validate AI relevance and category classification."
    
    # Note: Deep intelligence batch uses categorization prompt; no separate deep batch prompt.

    @staticmethod
    def get_categorization_batch_prompt(articles: List[Dict[str, Any]]) -> str:
        """Get batch categorization prompt for 8-category taxonomy."""
        return CategorizationPrompts.get_batch_prompt(articles)

    @staticmethod
    def get_categorization_distribution_prompt() -> str:
        """Compact distribution prompt for token estimation and load balancing."""
        return CategorizationPrompts.get_distribution_prompt()
