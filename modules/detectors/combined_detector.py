# modules/detectors/combined_detector.py
"""
Combined cannibalization detection using multiple methods
This is the recommended approach for comprehensive analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
import asyncio

from .competition_detector import CompetitionDetector
from .similarity_detector import SimilarityDetector
from ..analyzers.ml_scoring import MLScoringEngine
from ..analyzers.ai_analyzer import AIAnalyzer
from ..analyzers.serp_analyzer import SERPAnalyzer
from utils.url_normalizer import URLNormalizer

logger = logging.getLogger(__name__)

class CombinedDetector:
    """Comprehensive cannibalization detection using all available methods"""
    
    def __init__(self, internal_data: pd.DataFrame, gsc_data: pd.DataFrame,
                 embeddings_data: Optional[pd.DataFrame] = None,
                 ai_analyzer: Optional[AIAnalyzer] = None,
                 config: Optional[Dict] = None):
        """
        Initialize combined detector with all data sources
        
        Args:
            internal_data: SEO crawler data
            gsc_data: Google Search Console data
            embeddings_data: Optional embeddings data
            ai_analyzer: Optional AI analyzer
            config: Configuration settings
        """
        self.internal_data = internal_data
        self.gsc_data = gsc_data
        self.embeddings_data = embeddings_data
        self.ai_analyzer = ai_analyzer
        self.config = config or {}
        
        # Initialize sub-detectors
        self.competition_detector = CompetitionDetector(gsc_data)
        self.similarity_detector = SimilarityDetector(
            internal_data, 
            embeddings_data,
            config.get('similarity_settings', {})
        )
        
        # Initialize ML scoring engine
        self.ml_scorer = MLScoringEngine()
        
        # SERP analyzer (if configured)
        self.serp_analyzer = None
        if config.get('use_serp') and hasattr(APIManager, 'get_serper_api_key'):
            from utils.api_manager import APIManager
            serper_key = APIManager.get_serper_api_key()
            if serper_key:
                self.serp_analyzer = SERPAnalyzer(serper_key)
    
    async def detect_all(self) -> pd.DataFrame:
        """
        Run comprehensive cannibalization detection
        
        Returns:
            DataFrame with all detected cannibalization issues
        """
        logger.info("Starting combined cannibalization detection")
        
        all_detections = []
        
        # 1. Competition-based detection (Primary)
        logger.info("Phase 1: Competition-based detection")
        competition_results = await self._run_competition_detection()
        if not competition_results.empty:
            all_detections.append(competition_results)
        
        # 2. Similarity-based detection (Secondary)
        logger.info("Phase 2: Similarity-based detection")
        similarity_results = await self._run_similarity_detection()
        if not similarity_results.empty:
            all_detections.append(similarity_results)
        
        # 3. Combine and deduplicate results
        logger.info("Phase 3: Combining results")
        combined_results = self._combine_results(all_detections)
        
        if combined_results.empty:
            logger.warning("No cannibalization issues detected")
            return combined_results
        
        # 4. ML-based risk scoring
        logger.info("Phase 4: ML risk scoring")
        combined_results = self._apply_ml_scoring(combined_results)
        
        # 5. AI intent analysis (if available)
        if self.ai_analyzer:
            logger.info("Phase 5: AI intent analysis")
            combined_results = await self._enhance_with_ai_analysis(combined_results)
        
        # 6. Final prioritization
        logger.info("Phase 6: Final prioritization")
        combined_results = self._prioritize_results(combined_results)
        
        logger.info(f"Combined detection complete: {len(combined_results)} issues found")
        
        return combined_results
    
    async def _run_competition_detection(self) -> pd.DataFrame:
        """Run competition-based detection"""
        try:
            keyword_settings = self.config.get('keyword_settings', {})
            
            results = await self.competition_detector.detect_all_competition(
                min_shared_queries=keyword_settings.get('min_shared_queries', 2),
                min_impressions=keyword_settings.get('min_impressions', 5),
                use_serp=self.config.get('use_serp', False) and self.serp_analyzer is not None
            )
            
            # Add method marker
            results['detection_method'] = 'competition'
            
            return results
            
        except Exception as e:
            logger.error(f"Competition detection error: {e}")
            return pd.DataFrame()
    
    async def _run_similarity_detection(self) -> pd.DataFrame:
        """Run similarity-based detection"""
        try:
            # Check if we should use content analysis
            use_content = self.config.get('analyze_content', False)
            
            if use_content and self.ai_analyzer:
                results = await self.similarity_detector.detect_with_content_analysis(
                    self.ai_analyzer
                )
            else:
                results = self.similarity_detector.detect_all_similarity()
            
            # Add method marker
            results['detection_method'] = 'similarity'
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity detection error: {e}")
            return pd.DataFrame()
    
    def _combine_results(self, detection_results: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine and deduplicate results from different detection methods"""
        if not detection_results:
            return pd.DataFrame()
        
        # Concatenate all results
        combined = pd.concat(detection_results, ignore_index=True)
        
        if combined.empty:
            return combined
        
        # Normalize URLs for deduplication
        combined['url1_norm'] = combined['url1'].apply(URLNormalizer.normalize_for_matching)
        combined['url2_norm'] = combined['url2'].apply(URLNormalizer.normalize_for_matching)
        
        # Create canonical pair identifier (sorted URLs)
        combined['pair_id'] = combined.apply(
            lambda row: tuple(sorted([row['url1_norm'], row['url2_norm']])),
            axis=1
        )
        
        # Group by pair and combine data
        grouped = combined.groupby('pair_id')
        
        deduplicated = []
        for pair_id, group in grouped:
            # Combine data from different detection methods
            combined_row = self._merge_detection_data(group)
            deduplicated.append(combined_row)
        
        result_df = pd.DataFrame(deduplicated)
        
        # Clean up temporary columns
        if 'url1_norm' in result_df.columns:
            result_df = result_df.drop(['url1_norm', 'url2_norm', 'pair_id'], axis=1)
        
        return result_df
    
    def _merge_detection_data(self, group: pd.DataFrame) -> Dict:
        """Merge data from multiple detection methods for the same URL pair"""
        # Start with the first row
        merged = group.iloc[0].to_dict()
        
        # Track which detection methods found this pair
        detection_methods = group['detection_method'].unique().tolist()
        merged['detection_methods'] = detection_methods
        merged['detection_count'] = len(detection_methods)
        
        # Combine scores from different methods
        if 'competition' in detection_methods and 'similarity' in detection_methods:
            # Both methods detected - this is high confidence
            comp_row = group[group['detection_method'] == 'competition'].iloc[0]
            sim_row = group[group['detection_method'] == 'similarity'].iloc[0]
            
            # Combine scores
            merged['competition_score'] = comp_row.get('competition_score', 0)
            merged['similarity_score'] = sim_row.get('competition_score', 0)  # Note: similarity uses 'competition_score' as primary
            
            # Combined confidence score
            merged['confidence_score'] = 0.9  # High confidence when both methods agree
            
            # Preserve method-specific data
            if 'shared_queries' in comp_row:
                merged['shared_queries'] = comp_row['shared_queries']
                merged['shared_queries_count'] = comp_row['shared_queries_count']
            
            if 'title_similarity' in sim_row:
                merged['title_similarity'] = sim_row['title_similarity']
                merged['semantic_similarity'] = sim_row.get('semantic_similarity', 0)
        else:
            # Single method detection
            merged['confidence_score'] = 0.6  # Medium confidence
        
        return merged
    
    def _apply_ml_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ML-based risk scoring to combined results"""
        risk_scores = []
        
        for idx, row in df.iterrows():
            # Prepare features for ML scoring
            features = {
                'title_similarity': row.get('title_similarity', 0),
                'h1_similarity': row.get('h1_similarity', 0),
                'semantic_similarity': row.get('semantic_similarity', 0),
                'keyword_overlap': row.get('shared_queries_count', 0) / 100,  # Normalize
                'intent_match': row.get('intent_match', 0.5),
                'serp_overlap': row.get('serp_competition_rate', 0),
                'traffic_difference': 0,  # Will be calculated
                'position_variance': row.get('avg_position_spread', 10) / 20,  # Normalize
                'ctr_ratio': 1.0  # Default
            }
            
            # Calculate ML risk score
            risk_score, contributions = self.ml_scorer.calculate_adaptive_score(features)
            
            risk_scores.append({
                'ml_risk_score': risk_score,
                'risk_contributions': contributions
            })
        
        # Add ML scores to dataframe
        risk_df = pd.DataFrame(risk_scores)
        df = pd.concat([df, risk_df], axis=1)
        
        # Update risk categories based on ML score
        thresholds = self.config.get('thresholds', {})
        df['risk_category'] = pd.cut(
            df['ml_risk_score'],
            bins=[0, 
                  thresholds.get('medium', 0.3),
                  thresholds.get('high', 0.5),
                  thresholds.get('critical', 0.7),
                  1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df
    
    async def _enhance_with_ai_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance results with AI-powered analysis"""
        # Add intent classification if not present
        if 'intent1' not in df.columns:
            # Get unique URLs
            all_urls = set(df['url1'].unique()) | set(df['url2'].unique())
            
            # Get internal data for these URLs
            url_data = self.internal_data[self.internal_data['url'].isin(all_urls)].copy()
            
            # Analyze intent
            if not url_data.empty:
                url_data = await self.ai_analyzer.analyze_intent_batch(url_data, batch_size=10)
                
                # Map intents back to pairs
                intent_map = dict(zip(url_data['url'], url_data['ai_intent']))
                
                df['intent1'] = df['url1'].map(intent_map).fillna('Unknown')
                df['intent2'] = df['url2'].map(intent_map).fillna('Unknown')
                df['intent_match'] = (df['intent1'] == df['intent2']).astype(float)
        
        return df
    
    def _prioritize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final prioritization of results"""
        # Calculate priority score
        df['priority_score'] = (
            df.get('ml_risk_score', df.get('competition_score', 0)) * 0.3 +
            df.get('confidence_score', 0.5) * 0.2 +
            df.get('shared_queries_count', 0) / 100 * 0.2 +
            df.get('traffic_opportunity', 0) / 1000 * 0.2 +
            df.get('detection_count', 1) / 2 * 0.1
        )
        
        # Sort by priority
        df = df.sort_values('priority_score', ascending=False)
        
        # Add rank
        df['priority_rank'] = range(1, len(df) + 1)
        
        # Add actionable insights
        df['recommended_action'] = df.apply(self._determine_action, axis=1)
        
        return df
    
    def _determine_action(self, row: pd.Series) -> str:
        """Determine recommended action based on detection data"""
        risk_score = row.get('ml_risk_score', 0)
        similarity = row.get('overall_similarity', row.get('similarity_score', 0))
        shared_queries = row.get('shared_queries_count', 0)
        
        if risk_score > 0.8 or (similarity > 0.9 and shared_queries > 10):
            return "Consolidate immediately"
        elif risk_score > 0.6 or (similarity > 0.7 and shared_queries > 5):
            return "Merge or differentiate"
        elif risk_score > 0.4 or (similarity > 0.5 and shared_queries > 2):
            return "Optimize and differentiate"
        else:
            return "Monitor and optimize"
    
    def get_detection_summary(self) -> Dict:
        """Get summary of detection results"""
        return {
            'detection_methods': {
                'competition': self.competition_detector is not None,
                'similarity': self.similarity_detector is not None,
                'ml_scoring': self.ml_scorer is not None,
                'ai_analysis': self.ai_analyzer is not None,
                'serp_analysis': self.serp_analyzer is not None
            },
            'configuration': {
                'min_shared_queries': self.config.get('keyword_settings', {}).get('min_shared_queries', 2),
                'min_similarity': self.config.get('similarity_settings', {}).get('min_similarity', 0.20),
                'use_serp': self.config.get('use_serp', False),
                'analyze_content': self.config.get('analyze_content', False)
            }
        }