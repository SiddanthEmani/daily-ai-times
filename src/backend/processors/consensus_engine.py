#!/usr/bin/env python3
"""
Consensus Engine for NewsXP.ai Swarm Intelligence System

Implements consensus algorithms for combining multiple agent decisions.
Enhanced with Multi-Dimensional Scoring System (Phase 1.1):
- Processes multi-dimensional scores from bulk agents
- Provides rich metadata for deep intelligence agents
- Maintains backward compatibility with legacy binary decisions
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

# Import multi-dimensional scoring system
try:
    from ...shared.types.scoring import MultiDimensionalScore, ArticleScore, aggregate_multi_dimensional_scores
except ImportError:
    try:
        from shared.types.scoring import MultiDimensionalScore, ArticleScore, aggregate_multi_dimensional_scores
    except ImportError:
        # Fallback for development/testing
        MultiDimensionalScore = None
        ArticleScore = None
        aggregate_multi_dimensional_scores = None

logger = logging.getLogger(__name__)

class ConsensusEngine:
    """
    Consensus engine for combining multiple agent decisions.
    Implements various consensus algorithms including weighted voting,
    semantic clustering, and simple majority voting.
    """
    
    def __init__(self, consensus_config: Dict[str, Any], swarm_config: Dict[str, Any]):
        """Initialize consensus engine with configuration."""
        self.consensus_config = consensus_config
        self.swarm_config = swarm_config
        
        # Extract consensus algorithms configuration
        self.consensus_algorithms = self.consensus_config.get('algorithms', {})
        self.required_agreement = self.consensus_config.get('validation', {}).get('required_agreement', 0.7)
        self.min_confidence = self.consensus_config.get('validation', {}).get('quality_gates', {}).get('min_confidence', 0.8)
        
        # Extract swarm configuration for weight calculations
        self.bulk_swarm_config = self.swarm_config.get('agents', {}).get('bulk_intelligence_swarm', {})
        
        logger.info(f"Consensus engine initialized with required_agreement={self.required_agreement}, min_confidence={self.min_confidence}")
    
    def _calculate_consensus_weight(self, agent_name: str, confidence: float) -> float:
        """Calculate consensus weight for an agent's decision."""
        agent_config = self.bulk_swarm_config.get('agents', {}).get(agent_name, {})
        base_weight = agent_config.get('consensus_weight', 0.33)
        
        # Adjust weight based on confidence
        confidence_factor = max(0.1, confidence)  # Minimum 0.1 to avoid zero weight
        return base_weight * confidence_factor
    
    def _weighted_voting_consensus(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Implement weighted voting consensus algorithm."""
        if not agent_results:
            return []
        
        # Get all articles (assuming all agents processed same articles)
        first_agent_results = next(iter(agent_results.values()))
        articles = [result[0] for result in first_agent_results]
        
        consensus_results = []
        
        for article_idx, article in enumerate(articles):
            # Collect weighted votes for this article
            total_weight_accept = 0.0
            total_weight_reject = 0.0
            max_confidence = 0.0
            
            for agent_name, results in agent_results.items():
                if article_idx < len(results):
                    _, decision, confidence = results[article_idx]
                    weight = self._calculate_consensus_weight(agent_name, confidence)
                    
                    if decision:
                        total_weight_accept += weight
                    else:
                        total_weight_reject += weight
                    
                    max_confidence = max(max_confidence, confidence)
            
            # Make consensus decision
            total_weight = total_weight_accept + total_weight_reject
            if total_weight > 0:
                accept_ratio = total_weight_accept / total_weight
                consensus_decision = accept_ratio >= self.required_agreement
                consensus_confidence = max_confidence * min(1.0, total_weight / len(agent_results))
            else:
                # No valid votes, default to reject
                consensus_decision = False
                consensus_confidence = 0.0
            
            consensus_results.append((article, consensus_decision, consensus_confidence))
        
        return consensus_results
    
    def _semantic_clustering_consensus(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Implement semantic clustering consensus algorithm."""
        # For now, implement as majority voting with clustering logic
        if not agent_results:
            return []
        
        first_agent_results = next(iter(agent_results.values()))
        articles = [result[0] for result in first_agent_results]
        
        consensus_results = []
        
        for article_idx, article in enumerate(articles):
            # Collect decisions for this article
            decisions = []
            confidences = []
            
            for agent_name, results in agent_results.items():
                if article_idx < len(results):
                    _, decision, confidence = results[article_idx]
                    decisions.append(decision)
                    confidences.append(confidence)
            
            if decisions:
                # Simple majority voting with clustering concept
                accept_count = sum(decisions)
                total_count = len(decisions)
                
                # Check if we have sufficient clustering (similarity)
                similarity_threshold = self.consensus_algorithms.get('semantic_clustering', {}).get('similarity_threshold', 0.75)
                consensus_decision = (accept_count / total_count) >= similarity_threshold
                consensus_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            else:
                consensus_decision = False
                consensus_confidence = 0.0
            
            consensus_results.append((article, consensus_decision, consensus_confidence))
        
        return consensus_results
    
    def _simple_majority_consensus(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Simple majority voting consensus as fallback."""
        if not agent_results:
            return []
        
        first_agent_results = next(iter(agent_results.values()))
        articles = [result[0] for result in first_agent_results]
        
        consensus_results = []
        
        for article_idx, article in enumerate(articles):
            accept_votes = 0
            total_votes = 0
            confidences = []
            
            for agent_name, results in agent_results.items():
                if article_idx < len(results):
                    _, decision, confidence = results[article_idx]
                    if decision:
                        accept_votes += 1
                    total_votes += 1
                    confidences.append(confidence)
            
            if total_votes > 0:
                consensus_decision = accept_votes > (total_votes / 2)
                consensus_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            else:
                consensus_decision = False
                consensus_confidence = 0.0
            
            consensus_results.append((article, consensus_decision, consensus_confidence))
        
        return consensus_results
    
    def apply_consensus(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Apply the configured consensus algorithm."""
        if not agent_results:
            return []
        
        logger.info(f"Applying consensus algorithm to {len(agent_results)} agent results")
        
        # Detect if this is distributed processing (different articles) or overlapping processing (same articles)
        is_distributed = self._is_distributed_processing(agent_results)
        
        if is_distributed:
            logger.info("Detected distributed processing - using distributed consensus")
            return self._distributed_consensus(agent_results)
        
        # Check if we have multi-dimensional scores available
        has_multi_dimensional = False
        for agent_name, results in agent_results.items():
            if results and 'multi_dimensional_score' in results[0][0]:
                has_multi_dimensional = True
                break
        
        if has_multi_dimensional:
            logger.info("Using multi-dimensional consensus algorithm")
            return self._multi_dimensional_consensus(agent_results)
        
        # Fall back to legacy algorithms for overlapping processing
        # Check which consensus algorithms are enabled
        weighted_voting = self.consensus_algorithms.get('weighted_voting', {})
        semantic_clustering = self.consensus_algorithms.get('semantic_clustering', {})
        
        if weighted_voting.get('enabled', True):
            logger.info("Using weighted voting consensus")
            return self._weighted_voting_consensus(agent_results)
        elif semantic_clustering.get('enabled', True):
            logger.info("Using semantic clustering consensus")
            return self._semantic_clustering_consensus(agent_results)
        else:
            logger.info("Using simple majority consensus")
            return self._simple_majority_consensus(agent_results)
    
    def _is_distributed_processing(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> bool:
        """Detect if agent results represent distributed processing (different articles) or overlapping processing (same articles)."""
        if len(agent_results) < 2:
            return False
        
        # Get article IDs from each agent
        agent_article_ids = {}
        for agent_name, results in agent_results.items():
            article_ids = set()
            for article, _, _ in results:
                article_id = article.get('id', article.get('url', 'unknown'))
                article_ids.add(article_id)
            agent_article_ids[agent_name] = article_ids
        
        # Check for overlap between agents
        agent_names = list(agent_article_ids.keys())
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                overlap = agent_article_ids[agent_names[i]] & agent_article_ids[agent_names[j]]
                if len(overlap) > 0:
                    # Found overlap, this is not distributed processing
                    return False
        
        # No overlap found between any agents - this is distributed processing
        return True
    
    def filter_by_confidence(self, consensus_results: List[Tuple[Dict[str, Any], bool, float]], 
                           optimize_for_deep_intelligence: bool = False) -> List[Dict[str, Any]]:
        """Filter accepted articles by minimum confidence threshold with optional optimization."""
        if optimize_for_deep_intelligence:
            # Use optimized data structure for deep intelligence
            return self.prepare_for_deep_intelligence(consensus_results)
        
        # Legacy method - keep full article data
        accepted_articles = []
        
        for article, decision, confidence in consensus_results:
            if decision and confidence >= self.min_confidence:
                # Add consensus metadata
                article_copy = article.copy()
                article_copy['swarm_consensus'] = {
                    'decision': decision,
                    'confidence': confidence,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                accepted_articles.append(article_copy)
        
        logger.info(f"Confidence filter: {len([r for r in consensus_results if r[1]])} decisions -> {len(accepted_articles)} accepted articles")
        return accepted_articles
    
    def get_consensus_stats(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]], 
                          consensus_results: List[Tuple[Dict[str, Any], bool, float]]) -> Dict[str, Any]:
        """Generate consensus statistics for monitoring."""
        if not agent_results or not consensus_results:
            return {}
        
        # Calculate agent agreement statistics
        total_articles = len(consensus_results)
        total_accepted = sum(1 for _, decision, _ in consensus_results if decision)
        
        # Calculate per-agent statistics
        agent_stats = {}
        for agent_name, results in agent_results.items():
            agent_accepted = sum(1 for _, decision, _ in results if decision)
            agent_stats[agent_name] = {
                'total_processed': len(results),
                'accepted': agent_accepted,
                'acceptance_rate': agent_accepted / len(results) if results else 0.0
            }
        logger.info(f"Agent stats: {agent_stats}")
        return {
            'total_articles': total_articles,
            'consensus_accepted': total_accepted,
            'consensus_acceptance_rate': total_accepted / total_articles if total_articles > 0 else 0.0,
            'agents': agent_stats,
            'algorithm_used': self._get_active_algorithm(),
            'required_agreement': self.required_agreement,
            'min_confidence': self.min_confidence
        }
    
    def _get_active_algorithm(self) -> str:
        """Get the name of the currently active consensus algorithm."""
        weighted_voting = self.consensus_algorithms.get('weighted_voting', {})
        semantic_clustering = self.consensus_algorithms.get('semantic_clustering', {})
        
        if weighted_voting.get('enabled', True):
            return 'weighted_voting'
        elif semantic_clustering.get('enabled', True):
            return 'semantic_clustering'
        else:
            return 'simple_majority'
    
    def _extract_multi_dimensional_scores(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract multi-dimensional scores from agent results."""
        multi_dimensional_results = {}
        
        for agent_name, results in agent_results.items():
            md_scores = []
            for article, decision, confidence in results:
                # Check if article has multi-dimensional score metadata
                if 'multi_dimensional_score' in article:
                    md_scores.append(article['multi_dimensional_score'])
                else:
                    # Create a simplified multi-dimensional score from legacy data
                    md_score = {
                        'relevance_score': 0.7 if decision else 0.3,
                        'quality_score': 0.6 if decision else 0.3,
                        'novelty_score': 0.5 if decision else 0.3,
                        'impact_score': 0.5 if decision else 0.3,
                        'confidence_mean': confidence,
                        'confidence_std': 0.1,
                        'confidence_interval_low': max(0.0, confidence - 0.1),
                        'confidence_interval_high': min(1.0, confidence + 0.1),
                        'agent_name': agent_name,
                        'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                        'model_name': agent_name,
                        'specialization': 'general_filtering',
                        'overall_score': 0.7 if decision else 0.3,
                        'decision_threshold': 0.7,
                        'binary_decision': decision
                    }
                    md_scores.append(md_score)
            
            multi_dimensional_results[agent_name] = md_scores
        
        return multi_dimensional_results
    
    def _multi_dimensional_consensus(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Implement multi-dimensional consensus algorithm."""
        if not agent_results:
            return []
        
        # Extract multi-dimensional scores
        md_results = self._extract_multi_dimensional_scores(agent_results)
        
        # Get all articles (assuming all agents processed same articles)
        first_agent_results = next(iter(agent_results.values()))
        articles = [result[0] for result in first_agent_results]
        
        consensus_results = []
        
        for article_idx, article in enumerate(articles):
            # Collect multi-dimensional scores for this article
            article_md_scores = []
            agent_weights = {}
            
            for agent_name, md_scores in md_results.items():
                if article_idx < len(md_scores):
                    md_score = md_scores[article_idx]
                    article_md_scores.append(md_score)
                    
                    # Calculate weight based on confidence and agent configuration
                    confidence = md_score.get('confidence_mean', 0.5)
                    weight = self._calculate_consensus_weight(agent_name, confidence)
                    agent_weights[agent_name] = weight
            
            if article_md_scores:
                # Aggregate multi-dimensional scores if we have the aggregation function
                if aggregate_multi_dimensional_scores and MultiDimensionalScore:
                    try:
                        # Convert dict scores to MultiDimensionalScore objects
                        score_objects = []
                        for score_dict in article_md_scores:
                            score_obj = MultiDimensionalScore.from_dict(score_dict)
                            score_objects.append(score_obj)
                        
                        # Aggregate scores with weights
                        aggregated_score = aggregate_multi_dimensional_scores(score_objects, agent_weights)
                        
                        # Store aggregated multi-dimensional score in article
                        article['consensus_multi_dimensional_score'] = aggregated_score.to_dict()
                        
                        # Return in legacy format for backward compatibility
                        consensus_decision = aggregated_score.binary_decision
                        consensus_confidence = aggregated_score.confidence_mean
                        
                        consensus_results.append((article, consensus_decision, consensus_confidence))
                        
                    except Exception as e:
                        logger.warning(f"Multi-dimensional aggregation failed for article {article_idx}: {e}")
                        # Fall back to simple weighted voting
                        consensus_results.append(self._simple_weighted_vote_for_article(article, article_md_scores, agent_weights))
                else:
                    # Fall back to simple weighted voting
                    consensus_results.append(self._simple_weighted_vote_for_article(article, article_md_scores, agent_weights))
            else:
                # No scores available, default to reject
                consensus_results.append((article, False, 0.0))
        
        return consensus_results
    
    def _simple_weighted_vote_for_article(self, article: Dict[str, Any], md_scores: List[Dict[str, Any]], weights: Dict[str, float]) -> Tuple[Dict[str, Any], bool, float]:
        """Simple weighted voting for a single article using multi-dimensional scores."""
        total_weight_accept = 0.0
        total_weight_reject = 0.0
        max_confidence = 0.0
        
        for md_score in md_scores:
            agent_name = md_score.get('agent_name', 'unknown')
            weight = weights.get(agent_name, 0.33)
            decision = md_score.get('binary_decision', False)
            confidence = md_score.get('confidence_mean', 0.5)
            
            if decision:
                total_weight_accept += weight
            else:
                total_weight_reject += weight
            
            max_confidence = max(max_confidence, confidence)
        
        # Make consensus decision
        total_weight = total_weight_accept + total_weight_reject
        if total_weight > 0:
            accept_ratio = total_weight_accept / total_weight
            consensus_decision = accept_ratio >= self.required_agreement
            consensus_confidence = max_confidence * min(1.0, total_weight / len(md_scores))
        else:
            # No valid votes, default to reject
            consensus_decision = False
            consensus_confidence = 0.0
        
        return (article, consensus_decision, consensus_confidence)
    
    def prepare_for_deep_intelligence(self, consensus_results: List[Tuple[Dict[str, Any], bool, float]]) -> List[Dict[str, Any]]:
        """
        Prepare minimal, optimized data structure for deep intelligence agents.
        Reduces data size by 60-70% by removing redundant metadata and truncating content.
        **FIX**: Now preserves frontend-required fields (category, description, author).
        """
        optimized_articles = []
        
        for article, decision, confidence in consensus_results:
            if decision and confidence >= self.min_confidence:
                # Extract consensus scores (only what's used in deep intelligence prompt)
                consensus_score = article.get('consensus_multi_dimensional_score', {})
                
                # Get content and truncate early to save memory/tokens
                content = article.get('content', article.get('description', ''))
                if len(content) > 2000:
                    content = content[:2000] + "..."
                
                # Create minimal article structure - only essential data
                # **FIX**: Preserve frontend-required fields for proper categorization and display
                optimized_article = {
                    # Core identifiers
                    'article_id': article.get('article_id', article.get('id', '')),  # Fallback to 'id' field
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'published_date': article.get('published_date', ''),
                    
                    # **FIX**: Frontend-required fields for proper categorization and rendering
                    'category': article.get('category', 'Media'),  # Required for research vs regular article categorization
                    'description': content,  # Frontend expects 'description' field, not 'content'
                    'author': article.get('author', ''),  # Required for research article display
                    
                    # Also include 'content' for backward compatibility with deep intelligence processing
                    'content': content,
                    
                    # Minimal consensus scores (only 6 values used in prompt)
                    'consensus_multi_dimensional_score': {
                        'relevance_score': consensus_score.get('relevance_score', 0.0),
                        'quality_score': consensus_score.get('quality_score', 0.0),
                        'novelty_score': consensus_score.get('novelty_score', 0.0),
                        'impact_score': consensus_score.get('impact_score', 0.0),
                        'overall_score': consensus_score.get('overall_score', 0.0),
                        'confidence_mean': consensus_score.get('confidence_mean', 0.0)
                    }
                    # Removed: individual agent scores, processing timestamps, 
                    # confidence intervals, specializations, model names, etc.
                }
                
                optimized_articles.append(optimized_article)
        
        # Log optimization results
        if consensus_results:
            original_size = sum(len(str(article[0])) for article in consensus_results if article[1])
            optimized_size = sum(len(str(art)) for art in optimized_articles)
            reduction_pct = ((original_size - optimized_size) / original_size * 100) if original_size > 0 else 0
            
            logger.info(f"Data optimization: {len(optimized_articles)} articles, "
                       f"{reduction_pct:.1f}% size reduction ({original_size} -> {optimized_size} chars)")
        
        return optimized_articles
    
    def _distributed_consensus(self, agent_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """Handle consensus for distributed processing where each agent processes different articles."""
        if not agent_results:
            return []
        
        # In distributed mode, simply collect all results from all agents
        # since each agent processed different articles
        consensus_results = []
        
        for agent_name, results in agent_results.items():
            for article, decision, confidence in results:
                # Add agent metadata for tracking
                article_copy = article.copy()
                article_copy['processed_by_agent'] = agent_name
                article_copy['consensus_confidence'] = confidence
                article_copy['consensus_decision'] = decision
                
                # **FIX**: Preserve multi-dimensional scores from individual agents as consensus scores
                # This is critical for headline classification and downstream processing
                if 'multi_dimensional_score' in article:
                    # Copy the agent's multi-dimensional score as consensus score
                    md_score = article['multi_dimensional_score']
                    
                    # Convert agent score format to consensus score format
                    consensus_md_score = {
                        'relevance_score': md_score.get('relevance_score', 0.0),
                        'quality_score': md_score.get('quality_score', 0.0),
                        'novelty_score': md_score.get('novelty_score', 0.0),
                        'impact_score': md_score.get('impact_score', 0.0),
                        'overall_score': md_score.get('overall_score', 0.0),
                        'confidence_mean': md_score.get('confidence_mean', confidence),
                        'confidence_std': md_score.get('confidence_std', 0.1),
                        'binary_decision': md_score.get('binary_decision', decision),
                        'agent_name': agent_name,
                        'consensus_method': 'distributed_single_agent',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
                    article_copy['consensus_multi_dimensional_score'] = consensus_md_score
                    
                    # Log first few for debugging
                    if len(consensus_results) < 3:
                        logger.info(f"Distributed consensus preserved scores for {agent_name}: "
                                   f"R={consensus_md_score['relevance_score']:.2f}, "
                                   f"Q={consensus_md_score['quality_score']:.2f}, "
                                   f"I={consensus_md_score['impact_score']:.2f}, "
                                   f"O={consensus_md_score['overall_score']:.2f}")
                else:
                    # Fallback: create minimal consensus score from decision/confidence
                    logger.warning(f"Article from {agent_name} missing multi_dimensional_score, using fallback")
                    article_copy['consensus_multi_dimensional_score'] = {
                        'relevance_score': 0.7 if decision else 0.3,
                        'quality_score': 0.6 if decision else 0.3,
                        'novelty_score': 0.5 if decision else 0.3,
                        'impact_score': 0.5 if decision else 0.3,
                        'overall_score': 0.6 if decision else 0.3,
                        'confidence_mean': confidence,
                        'confidence_std': 0.1,
                        'binary_decision': decision,
                        'agent_name': agent_name,
                        'consensus_method': 'distributed_fallback',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                
                consensus_results.append((article_copy, decision, confidence))
        
        logger.info(f"Distributed consensus: collected {len(consensus_results)} articles from {len(agent_results)} agents with preserved multi-dimensional scores")
        return consensus_results
