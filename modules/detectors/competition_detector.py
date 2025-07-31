# modules/detectors/competition_detector.py
"""
Competition-based cannibalization detection
Focuses on actual SERP and keyword competition
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..analyzers.keyword_analyzer import KeywordAnalyzer
from ..analyzers.serp_analyzer import SERPAnalyzer

logger = logging.getLogger(__name__)

class CompetitionDetector:
    """Detect cannibalization through competition signals"""
    
    def __init__(self, gsc_data: pd.DataFrame, serp_analyzer: Optional[SERPAnalyzer] = None):
        """
        Initialize competition detector
        
        Args:
            gsc_data: Google Search Console data
            serp_analyzer: Optional SERP analyzer for real-time competition data
        """
        self.gsc_data = gsc_data
        self.keyword_analyzer = KeywordAnalyzer(gsc_data)
        self.serp_analyzer = serp_analyzer
        
    async def detect_all_competition(self, 
                                   min_shared_queries: int = 2,
                                   min_impressions: int = 10,
                                   use_serp: bool = True) -> pd.DataFrame:
        """
        Detect all types of competition-based cannibalization
        
        Returns:
            DataFrame with competition-based cannibalization issues
        """
        logger.info("Starting competition-based detection")
        
        # 1. Keyword-based competition (primary method)
        keyword_competition = self.keyword_analyzer.detect_keyword_cannibalization(
            min_impressions=min_impressions,
            min_shared_queries=min_shared_queries
        )
        
        if keyword_competition.empty:
            logger.warning("No keyword competition found")
            return pd.DataFrame()
        
        # Add source column
        keyword_competition['detection_source'] = 'keyword_competition'
        
        # 2. SERP-based competition (if available)
        if use_serp and self.serp_analyzer:
            serp_competition = await self._detect_serp_competition(keyword_competition)
            if not serp_competition.empty:
                # Merge SERP data with keyword data
                keyword_competition = self._merge_serp_data(keyword_competition, serp_competition)
        
        # 3. Calculate combined competition score
        keyword_competition = self._calculate_competition_scores(keyword_competition)
        
        # 4. Add performance impact analysis
        keyword_competition = self._analyze_performance_impact(keyword_competition)
        
        return keyword_competition
    
    async def _detect_serp_competition(self, url_pairs: pd.DataFrame) -> pd.DataFrame:
        """Detect SERP-level competition for URL pairs"""
        if self.serp_analyzer is None:
            return pd.DataFrame()
        
        logger.info(f"Checking SERP competition for {len(url_pairs)} URL pairs")
        
        serp_results = []
        
        # Get top keywords for SERP analysis
        for _, pair in url_pairs.iterrows():
            # Get shared queries for this pair
            shared_queries = pair['shared_queries'][:10]  # Top 10 queries
            
            if shared_queries:
                # Analyze SERP competition
                serp_data = await self.serp_analyzer.analyze_url_pair_serp(
                    pair['url1'],
                    pair['url2'],
                    shared_queries
                )
                
                serp_results.append({
                    'url1': pair['url1'],
                    'url2': pair['url2'],
                    'serp_competition_rate': serp_data['metrics']['competition_rate'],
                    'serp_competing_keywords': len(serp_data['competing_keywords']),
                    'avg_position_difference': serp_data['metrics']['avg_position_diff'],
                    'serp_data': serp_data
                })
        
        return pd.DataFrame(serp_results)
    
    def _merge_serp_data(self, keyword_df: pd.DataFrame, 
                        serp_df: pd.DataFrame) -> pd.DataFrame:
        """Merge SERP competition data with keyword data"""
        if serp_df.empty:
            return keyword_df
        
        # Merge on URL pairs
        merged = keyword_df.merge(
            serp_df,
            on=['url1', 'url2'],
            how='left'
        )
        
        # Fill missing SERP data with defaults
        merged['serp_competition_rate'] = merged['serp_competition_rate'].fillna(0)
        merged['serp_competing_keywords'] = merged['serp_competing_keywords'].fillna(0)
        
        return merged
    
    def _calculate_competition_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive competition scores"""
        # Keyword competition components
        df['keyword_competition_score'] = df['keyword_cannibalization_score']
        
        # SERP competition score (if available)
        if 'serp_competition_rate' in df.columns:
            df['serp_competition_score'] = (
                df['serp_competition_rate'] * 0.6 +
                (df['serp_competing_keywords'] / df['shared_queries_count']).clip(0, 1) * 0.4
            )
        else:
            df['serp_competition_score'] = 0
        
        # Combined competition score
        df['competition_score'] = (
            df['keyword_competition_score'] * 0.7 +  # Primary weight on keyword data
            df['serp_competition_score'] * 0.3       # Secondary weight on SERP data
        )
        
        # Competition severity
        df['competition_severity'] = pd.cut(
            df['competition_score'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df
    
    def _analyze_performance_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze the performance impact of competition"""
        # Get overall performance for each URL
        url_performance = self.gsc_data.groupby('url').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        
        # Add performance data for each URL in the pair
        df = df.merge(
            url_performance.rename(columns={'url': 'url1'}).add_suffix('_url1'),
            on='url1',
            how='left'
        )
        df = df.merge(
            url_performance.rename(columns={'url': 'url2'}).add_suffix('_url2'),
            on='url2',
            how='left'
        )
        
        # Calculate performance metrics
        df['total_clicks_both_urls'] = df['clicks_url1'] + df['clicks_url2']
        df['total_impressions_both_urls'] = df['impressions_url1'] + df['impressions_url2']
        
        # Click distribution (how evenly clicks are split)
        df['click_distribution'] = df.apply(
            lambda row: min(row['clicks_url1'], row['clicks_url2']) / 
                       max(row['clicks_url1'], row['clicks_url2'], 1),
            axis=1
        )
        
        # Traffic opportunity (potential gain from consolidation)
        df['traffic_opportunity'] = df.apply(
            lambda row: self._estimate_traffic_opportunity(row),
            axis=1
        )
        
        return df
    
    def _estimate_traffic_opportunity(self, row: pd.Series) -> int:
        """Estimate potential traffic gain from fixing cannibalization"""
        # Conservative estimate: consolidated page could capture 70% of combined traffic
        # Plus 20% boost from improved rankings
        current_clicks = row['total_shared_clicks']
        potential_clicks = current_clicks * 0.7 * 1.2
        
        return max(0, int(potential_clicks - row['clicks_url1']))  # Assume url1 is kept
    
    def get_competition_summary(self) -> Dict:
        """Get summary of competition analysis"""
        impact = self.keyword_analyzer.calculate_cannibalization_impact()
        
        summary = {
            'total_competing_pairs': 0,  # Will be set by caller
            'keyword_metrics': {
                'cannibalized_queries': impact['cannibalized_queries'],
                'cannibalization_rate': f"{impact['cannibalization_rate']:.1%}",
                'affected_clicks': impact['cannibalized_clicks'],
                'potential_click_gain': impact['estimated_click_gain'],
                'potential_gain_percentage': f"{impact['estimated_gain_percentage']:.1%}"
            },
            'recommendations': self._generate_competition_recommendations(impact)
        }
        
        return summary
    
    def _generate_competition_recommendations(self, impact: Dict) -> List[str]:
        """Generate recommendations based on competition analysis"""
        recommendations = []
        
        if impact['cannibalization_rate'] > 0.3:
            recommendations.append(
                "Critical: Over 30% of queries show cannibalization. "
                "Implement comprehensive content consolidation strategy."
            )
        elif impact['cannibalization_rate'] > 0.15:
            recommendations.append(
                "High: 15-30% query cannibalization detected. "
                "Focus on consolidating high-traffic keyword groups."
            )
        else:
            recommendations.append(
                "Moderate: Some keyword cannibalization detected. "
                "Review and optimize affected content clusters."
            )
        
        if impact['estimated_gain_percentage'] > 0.1:
            recommendations.append(
                f"Significant opportunity: Fixing cannibalization could increase "
                f"traffic by {impact['estimated_gain_percentage']:.0%}"
            )
        
        return recommendations