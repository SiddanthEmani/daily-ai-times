#!/usr/bin/env python3
"""
Final Consensus Engine for Daily AI Times
Combines initial consensus results with deep intelligence analysis.
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class FinalConsensusEngine:
    """
    Advanced consensus engine that combines initial consensus filtering
    with deep intelligence analysis results to make final decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the final consensus engine."""
        self.config = config
        
        # Final consensus parameters
        self.deep_intelligence_weight = config.get('deep_intelligence_weight', 0.6)
        self.initial_consensus_weight = config.get('initial_consensus_weight', 0.4)
        self.min_deep_intelligence_confidence = config.get('min_deep_intelligence_confidence', 0.7)
        self.min_combined_score = config.get('min_combined_score', 0.6)
        
        # Consensus algorithms
        self.consensus_method = config.get('consensus_method', 'weighted_combination')
        self.require_unanimous_accept = config.get('require_unanimous_accept', False)
        self.veto_threshold = config.get('veto_threshold', 0.3)
        
        # Quality gates
        self.enable_quality_gates = config.get('enable_quality_gates', True)
        self.min_fact_check_confidence = config.get('min_fact_check_confidence', 0.4)  # Lowered default from 0.6 to 0.4
        self.max_bias_tolerance = config.get('max_bias_tolerance', 0.7)
        self.min_credibility_score = config.get('min_credibility_score', 0.5)
        
        logger.info(f"Initialized Final Consensus Engine: {self.consensus_method}")
    
    def apply_final_consensus(self, 
                            initial_consensus_results: List[Tuple[Dict[str, Any], bool, float]],
                            deep_intelligence_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Tuple[Dict[str, Any], bool, float]]:
        """
        Apply final consensus combining initial consensus with deep intelligence results.
        
        Args:
            initial_consensus_results: Results from initial consensus filtering
            deep_intelligence_results: Results from deep intelligence agents
            
        Returns:
            Final consensus decisions with enriched articles
        """
        logger.info(f"Applying final consensus to {len(initial_consensus_results)} articles")
        
        # Merge results by article
        merged_results = self._merge_consensus_results(initial_consensus_results, deep_intelligence_results)
        
        # Apply consensus algorithms
        final_decisions = []
        for article_data in merged_results:
            decision = self._make_final_decision(article_data)
            final_decisions.append(decision)
        
        accepted_count = sum(1 for _, accept, _ in final_decisions if accept)
        logger.info(f"Final consensus: {accepted_count}/{len(final_decisions)} articles accepted")
        
        return final_decisions
    
    def _merge_consensus_results(self, 
                                initial_results: List[Tuple[Dict[str, Any], bool, float]],
                                deep_intelligence_results: Dict[str, List[Tuple[Dict[str, Any], bool, float]]]) -> List[Dict[str, Any]]:
        """Merge initial consensus results with deep intelligence analysis."""
        
        # Create lookup for deep intelligence results by article title/url
        deep_intelligence_lookup = {}
        for agent_name, results in deep_intelligence_results.items():
            for article, accept, confidence in results:
                key = self._create_article_key(article)
                if key not in deep_intelligence_lookup:
                    deep_intelligence_lookup[key] = []
                deep_intelligence_lookup[key].append({
                    'agent': agent_name,
                    'article': article,
                    'accept': accept,
                    'confidence': confidence
                })
        
        # Merge results
        merged_results = []
        for initial_article, initial_accept, initial_confidence in initial_results:
            article_key = self._create_article_key(initial_article)
            
            # Find matching deep intelligence results
            deep_results = deep_intelligence_lookup.get(article_key, [])
            
            merged_data = {
                'initial_consensus': {
                    'article': initial_article,
                    'accept': initial_accept,
                    'confidence': initial_confidence
                },
                'deep_intelligence': deep_results,
                'article_key': article_key
            }
            
            merged_results.append(merged_data)
        
        return merged_results
    
    def _create_article_key(self, article: Dict[str, Any]) -> str:
        """Create a unique key for article matching."""
        title = article.get('title', '').strip()
        url = article.get('url', '').strip()
        source = article.get('source', '').strip()
        
        # Use URL if available, otherwise use title + source
        if url:
            return f"url:{url}"
        else:
            return f"title_source:{title}:{source}"
    
    def _make_final_decision(self, article_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """Make final decision for an article based on all available data."""
        
        initial_data = article_data['initial_consensus']
        deep_intelligence_data = article_data['deep_intelligence']
        
        # Base article from initial consensus
        base_article = initial_data['article']
        
        # If no deep intelligence data, use initial consensus
        if not deep_intelligence_data:
            logger.debug(f"No deep intelligence data for article: {base_article.get('title', 'N/A')}")
            return base_article, initial_data['accept'], initial_data['confidence']
        
        # Combine deep intelligence results
        combined_article = self._combine_deep_intelligence_results(base_article, deep_intelligence_data)
        
        # Apply consensus algorithm
        if self.consensus_method == 'weighted_combination':
            decision = self._weighted_combination_consensus(initial_data, deep_intelligence_data, combined_article)
        elif self.consensus_method == 'unanimous_accept':
            decision = self._unanimous_accept_consensus(initial_data, deep_intelligence_data, combined_article)
        elif self.consensus_method == 'veto_based':
            decision = self._veto_based_consensus(initial_data, deep_intelligence_data, combined_article)
        else:
            decision = self._weighted_combination_consensus(initial_data, deep_intelligence_data, combined_article)
        
        # Apply quality gates
        if self.enable_quality_gates:
            decision = self._apply_quality_gates(decision)
        
        return decision
    
    def _combine_deep_intelligence_results(self, base_article: Dict[str, Any], 
                                         deep_intelligence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple deep intelligence agents."""
        
        combined_article = base_article.copy()
        
        # Collect all deep intelligence analyses
        all_analyses = []
        all_scores = []
        all_confidences = []
        all_recommendations = []
        
        for result in deep_intelligence_data:
            article = result['article']
            all_analyses.append(article.get('deep_intelligence_analysis', {}))
            all_scores.append(article.get('deep_intelligence_score', 0.5))
            all_confidences.append(article.get('deep_intelligence_confidence', 0.5))
            all_recommendations.append(article.get('deep_intelligence_recommendation', 'CONDITIONAL'))
        
        # Combine analyses
        combined_analysis = self._merge_deep_intelligence_analyses(all_analyses)
        
        # Calculate combined metrics
        combined_score = statistics.mean(all_scores) if all_scores else 0.5
        combined_confidence = statistics.mean(all_confidences) if all_confidences else 0.5
        
        # Determine combined recommendation
        accept_count = sum(1 for rec in all_recommendations if rec == 'ACCEPT')
        reject_count = sum(1 for rec in all_recommendations if rec == 'REJECT')
        
        if accept_count > reject_count:
            combined_recommendation = 'ACCEPT'
        elif reject_count > accept_count:
            combined_recommendation = 'REJECT'
        else:
            combined_recommendation = 'CONDITIONAL'
        
        # Add combined deep intelligence data
        combined_article['combined_deep_intelligence'] = {
            'analysis': combined_analysis,
            'score': combined_score,
            'confidence': combined_confidence,
            'recommendation': combined_recommendation,
            'agents_count': len(deep_intelligence_data),
            'consensus_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return combined_article
    
    def _merge_deep_intelligence_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple deep intelligence analyses into a combined analysis."""
        
        if not analyses:
            return {}
        
        # Combine fact verification
        all_verified_claims = []
        all_unverified_claims = []
        fact_confidences = []
        
        # Combine bias detection
        all_bias_indicators = []
        bias_scores = []
        
        # Combine credibility assessment
        all_credibility_factors = []
        credibility_scores = []
        
        # Combine impact analysis
        all_impact_areas = []
        impact_potentials = []
        
        # Combine synthesis
        all_insights = []
        all_risks = []
        all_suggestions = []
        
        for analysis in analyses:
            if not analysis:
                continue
            
            # Fact verification
            fact_verification = analysis.get('fact_verification', {})
            all_verified_claims.extend(fact_verification.get('verified_claims', []))
            all_unverified_claims.extend(fact_verification.get('unverified_claims', []))
            fact_confidences.append(fact_verification.get('fact_check_confidence', 0.5))
            
            # Bias detection
            bias_detection = analysis.get('bias_detection', {})
            all_bias_indicators.extend(bias_detection.get('bias_indicators', []))
            bias_scores.append(bias_detection.get('bias_detection_score', 0.5))
            
            # Credibility assessment
            credibility_assessment = analysis.get('credibility_assessment', {})
            all_credibility_factors.extend(credibility_assessment.get('credibility_factors', []))
            credibility_scores.append(credibility_assessment.get('credibility_score', 0.5))
            
            # Impact analysis
            impact_analysis = analysis.get('impact_analysis', {})
            all_impact_areas.extend(impact_analysis.get('impact_areas', []))
            impact_potentials.append(impact_analysis.get('impact_potential', 0.5))
            
            # Synthesis
            synthesis = analysis.get('synthesis', {})
            all_insights.extend(synthesis.get('key_insights', []))
            all_risks.extend(synthesis.get('risk_factors', []))
            all_suggestions.extend(synthesis.get('enhancement_suggestions', []))
        
        # Create combined analysis
        combined_analysis = {
            'fact_verification': {
                'verified_claims': list(set(all_verified_claims)),
                'unverified_claims': list(set(all_unverified_claims)),
                'fact_check_confidence': statistics.mean(fact_confidences) if fact_confidences else 0.5
            },
            'bias_detection': {
                'bias_indicators': list(set(all_bias_indicators)),
                'bias_detection_score': statistics.mean(bias_scores) if bias_scores else 0.5
            },
            'credibility_assessment': {
                'credibility_factors': list(set(all_credibility_factors)),
                'credibility_score': statistics.mean(credibility_scores) if credibility_scores else 0.5
            },
            'impact_analysis': {
                'impact_areas': list(set(all_impact_areas)),
                'impact_potential': statistics.mean(impact_potentials) if impact_potentials else 0.5
            },
            'synthesis': {
                'key_insights': list(set(all_insights)),
                'risk_factors': list(set(all_risks)),
                'enhancement_suggestions': list(set(all_suggestions))
            }
        }
        
        return combined_analysis
    
    def _weighted_combination_consensus(self, initial_data: Dict[str, Any], 
                                      deep_intelligence_data: List[Dict[str, Any]], 
                                      combined_article: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """Apply weighted combination consensus algorithm."""
        
        # Get scores
        initial_score = initial_data['confidence']
        deep_intelligence_score = combined_article['combined_deep_intelligence']['score']
        deep_intelligence_confidence = combined_article['combined_deep_intelligence']['confidence']
        
        # Calculate weighted score
        weighted_score = (self.initial_consensus_weight * initial_score + 
                         self.deep_intelligence_weight * deep_intelligence_score)
        
        # Calculate combined confidence
        combined_confidence = (self.initial_consensus_weight * initial_data['confidence'] +
                             self.deep_intelligence_weight * deep_intelligence_confidence)
        
        # Make decision
        accept = (weighted_score >= self.min_combined_score and 
                 deep_intelligence_confidence >= self.min_deep_intelligence_confidence)
        
        # Add final consensus metadata
        combined_article['final_consensus'] = {
            'algorithm': 'weighted_combination',
            'weighted_score': weighted_score,
            'combined_confidence': combined_confidence,
            'decision': 'ACCEPT' if accept else 'REJECT',
            'weights': {
                'initial_consensus': self.initial_consensus_weight,
                'deep_intelligence': self.deep_intelligence_weight
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return combined_article, accept, combined_confidence
    
    def _unanimous_accept_consensus(self, initial_data: Dict[str, Any], 
                                  deep_intelligence_data: List[Dict[str, Any]], 
                                  combined_article: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """Apply unanimous accept consensus algorithm."""
        
        # Check if initial consensus accepts
        initial_accept = initial_data['accept']
        
        # Check if all deep intelligence agents accept
        all_deep_accept = all(result['accept'] for result in deep_intelligence_data)
        
        # Final decision requires unanimous acceptance
        accept = initial_accept and all_deep_accept
        
        # Calculate confidence as minimum of all confidences
        all_confidences = [initial_data['confidence']] + [result['confidence'] for result in deep_intelligence_data]
        combined_confidence = min(all_confidences) if all_confidences else 0.0
        
        # Add final consensus metadata
        combined_article['final_consensus'] = {
            'algorithm': 'unanimous_accept',
            'initial_accept': initial_accept,
            'deep_intelligence_accept': all_deep_accept,
            'decision': 'ACCEPT' if accept else 'REJECT',
            'combined_confidence': combined_confidence,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return combined_article, accept, combined_confidence
    
    def _veto_based_consensus(self, initial_data: Dict[str, Any], 
                            deep_intelligence_data: List[Dict[str, Any]], 
                            combined_article: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """Apply veto-based consensus algorithm."""
        
        # Start with initial consensus
        accept = initial_data['accept']
        
        # Check for veto from deep intelligence agents
        veto_scores = []
        for result in deep_intelligence_data:
            if not result['accept'] and result['confidence'] >= self.veto_threshold:
                veto_scores.append(result['confidence'])
        
        # Apply veto if any agent strongly rejects
        if veto_scores:
            accept = False
            veto_applied = True
            strongest_veto = max(veto_scores)
        else:
            veto_applied = False
            strongest_veto = 0.0
        
        # Calculate confidence
        all_confidences = [initial_data['confidence']] + [result['confidence'] for result in deep_intelligence_data]
        combined_confidence = statistics.mean(all_confidences) if all_confidences else 0.0
        
        # Add final consensus metadata
        combined_article['final_consensus'] = {
            'algorithm': 'veto_based',
            'initial_accept': initial_data['accept'],
            'veto_applied': veto_applied,
            'strongest_veto': strongest_veto,
            'decision': 'ACCEPT' if accept else 'REJECT',
            'combined_confidence': combined_confidence,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return combined_article, accept, combined_confidence
    
    def _apply_quality_gates(self, decision: Tuple[Dict[str, Any], bool, float]) -> Tuple[Dict[str, Any], bool, float]:
        """Apply quality gates to filter out low-quality articles."""
        
        article, accept, confidence = decision
        
        if not accept:
            return decision  # Already rejected
        
        # Check quality gates
        quality_checks = []
        
        # Fact check confidence gate
        deep_intelligence = article.get('combined_deep_intelligence', {})
        analysis = deep_intelligence.get('analysis', {})
        fact_verification = analysis.get('fact_verification', {})
        fact_confidence = fact_verification.get('fact_check_confidence', 1.0)
        
        if fact_confidence < self.min_fact_check_confidence:
            quality_checks.append(f"Fact check confidence too low: {fact_confidence:.2f}")
        
        # Bias detection gate - FIXED: Higher bias score means MORE bias, which is bad
        bias_detection = analysis.get('bias_detection', {})
        bias_score = bias_detection.get('bias_detection_score', 0.0)  # Default to low bias (good)
        
        if bias_score > self.max_bias_tolerance:  # Fixed: > instead of <
            quality_checks.append(f"Bias level too high: {bias_score:.2f}")
        
        # Credibility gate
        credibility_assessment = analysis.get('credibility_assessment', {})
        credibility_score = credibility_assessment.get('credibility_score', 1.0)
        
        if credibility_score < self.min_credibility_score:
            quality_checks.append(f"Credibility score too low: {credibility_score:.2f}")
        
        # Apply quality gates
        if quality_checks:
            accept = False
            article['quality_gate_failures'] = quality_checks
            logger.debug(f"Article failed quality gates: {quality_checks}")
        
        return article, accept, confidence
    
    def get_final_consensus_stats(self, final_decisions: List[Tuple[Dict[str, Any], bool, float]]) -> Dict[str, Any]:
        """Generate statistics for final consensus decisions."""
        
        if not final_decisions:
            return {}
        
        total_articles = len(final_decisions)
        accepted_articles = sum(1 for _, accept, _ in final_decisions if accept)
        rejected_articles = total_articles - accepted_articles
        
        # Calculate confidence statistics
        confidences = [confidence for _, _, confidence in final_decisions]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Count articles with deep intelligence analysis
        deep_intelligence_count = sum(1 for article, _, _ in final_decisions 
                                    if 'combined_deep_intelligence' in article)
        
        # Count quality gate failures
        quality_gate_failures = sum(1 for article, _, _ in final_decisions 
                                  if 'quality_gate_failures' in article)
        
        return {
            'total_articles': total_articles,
            'accepted_articles': accepted_articles,
            'rejected_articles': rejected_articles,
            'acceptance_rate': (accepted_articles / total_articles * 100) if total_articles > 0 else 0,
            'average_confidence': avg_confidence,
            'deep_intelligence_coverage': {
                'articles_analyzed': deep_intelligence_count,
                'coverage_percentage': (deep_intelligence_count / total_articles * 100) if total_articles > 0 else 0
            },
            'quality_gates': {
                'enabled': self.enable_quality_gates,
                'failures': quality_gate_failures,
                'failure_rate': (quality_gate_failures / total_articles * 100) if total_articles > 0 else 0
            },
            'consensus_method': self.consensus_method,
            'algorithm_weights': {
                'initial_consensus': self.initial_consensus_weight,
                'deep_intelligence': self.deep_intelligence_weight
            }
        }
