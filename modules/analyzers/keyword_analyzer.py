# modules/analyzers/keyword_analyzer.py
"""
Keyword-based cannibalization detection - the most critical component
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """Analyze keyword-level competition between URLs"""
    
    def __init__(self, gsc_data: pd.DataFrame):
        """
        Initialize with GSC performance data
        
        Args:
            gsc_data: DataFrame with columns: url, query, clicks, impressions, position
        """
        self.gsc_data = gsc_data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and enrich GSC data"""
        # Ensure numeric types
        numeric_cols = ['clicks', 'impressions', 'position']
        for col in numeric_cols:
            if col in self.gsc_data.columns:
                self.gsc_data[col] = pd.to_numeric(self.gsc_data[col], errors='coerce')
        
        # Calculate additional metrics
        self.gsc_data['ctr'] = self.gsc_data.apply(
            lambda x: x['clicks'] / x['impressions'] if x['impressions'] > 0 else 0, 
            axis=1
        )
        
        # Add search volume tier
        self.gsc_data['volume_tier'] = pd.cut(
            self.gsc_data['impressions'],
            bins=[0, 10, 100, 1000, 10000, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # Calculate query value (combination of volume and CTR)
        self.gsc_data['query_value'] = (
            self.gsc_data['clicks'] * 1.0 + 
            self.gsc_data['impressions'] * 0.1
        )
    
    def detect_keyword_cannibalization(self, 
                                     min_impressions: int = 10,
                                     min_shared_queries: int = 2,
                                     position_threshold: float = 20.0) -> pd.DataFrame:
        """
        Detect URLs competing for the same keywords
        
        Args:
            min_impressions: Minimum impressions for a query to be considered
            min_shared_queries: Minimum shared queries for cannibalization
            position_threshold: Maximum position to consider (e.g., top 20)
            
        Returns:
            DataFrame with cannibalization pairs and metrics
        """
        logger.info("Starting keyword-based cannibalization detection")
        
        # Filter queries by impressions and position
        relevant_data = self.gsc_data[
            (self.gsc_data['impressions'] >= min_impressions) &
            (self.gsc_data['position'] <= position_threshold)
        ].copy()
        
        # Find queries with multiple URLs
        query_url_counts = relevant_data.groupby('query')['url'].nunique()
        multi_url_queries = query_url_counts[query_url_counts > 1].index.tolist()
        
        logger.info(f"Found {len(multi_url_queries)} queries with multiple ranking URLs")
        
        # Analyze each multi-URL query
        cannibalization_data = []
        
        for query in multi_url_queries:
            query_data = relevant_data[relevant_data['query'] == query]
            urls = query_data['url'].unique()
            
            # Create pairs
            for i, url1 in enumerate(urls):
                for url2 in urls[i+1:]:
                    pair_data = self._analyze_url_pair_for_query(
                        url1, url2, query, query_data
                    )
                    cannibalization_data.append(pair_data)
        
        # Aggregate by URL pairs
        aggregated_data = self._aggregate_cannibalization_data(cannibalization_data)
        
        # Calculate risk scores
        result_df = self._calculate_keyword_risk_scores(aggregated_data)
        
        # Filter by minimum shared queries
        result_df = result_df[result_df['shared_queries_count'] >= min_shared_queries]
        
        logger.info(f"Found {len(result_df)} URL pairs with keyword cannibalization")
        
        return result_df.sort_values('keyword_cannibalization_score', ascending=False)
    
    def _analyze_url_pair_for_query(self, url1: str, url2: str, 
                                   query: str, query_data: pd.DataFrame) -> Dict:
        """Analyze competition between two URLs for a specific query"""
        url1_data = query_data[query_data['url'] == url1].iloc[0]
        url2_data = query_data[query_data['url'] == url2].iloc[0]
        
        # Calculate competition metrics
        position_spread = abs(url1_data['position'] - url2_data['position'])
        total_clicks = url1_data['clicks'] + url2_data['clicks']
        total_impressions = url1_data['impressions'] + url2_data['impressions']
        
        # Determine competition type
        if position_spread <= 3:
            competition_type = 'direct'  # Very close positions
        elif position_spread <= 10:
            competition_type = 'moderate'
        else:
            competition_type = 'indirect'
        
        return {
            'url1': url1,
            'url2': url2,
            'query': query,
            'url1_position': url1_data['position'],
            'url2_position': url2_data['position'],
            'url1_clicks': url1_data['clicks'],
            'url2_clicks': url2_data['clicks'],
            'total_clicks': total_clicks,
            'total_impressions': total_impressions,
            'position_spread': position_spread,
            'competition_type': competition_type,
            'query_value': url1_data['query_value']
        }
    
    def _aggregate_cannibalization_data(self, 
                                      cannibalization_data: List[Dict]) -> List[Dict]:
        """Aggregate cannibalization data by URL pairs"""
        pair_data = defaultdict(lambda: {
            'queries': [],
            'total_clicks': 0,
            'total_impressions': 0,
            'weighted_position_spread': 0,
            'competition_types': Counter(),
            'query_values': []
        })
        
        for item in cannibalization_data:
            # Create consistent pair key
            pair_key = tuple(sorted([item['url1'], item['url2']]))
            
            # Aggregate data
            pair_data[pair_key]['queries'].append(item['query'])
            pair_data[pair_key]['total_clicks'] += item['total_clicks']
            pair_data[pair_key]['total_impressions'] += item['total_impressions']
            pair_data[pair_key]['weighted_position_spread'] += (
                item['position_spread'] * item['total_impressions']
            )
            pair_data[pair_key]['competition_types'][item['competition_type']] += 1
            pair_data[pair_key]['query_values'].append(item['query_value'])
        
        # Convert to list of dicts
        aggregated = []
        for (url1, url2), data in pair_data.items():
            avg_position_spread = (
                data['weighted_position_spread'] / data['total_impressions']
                if data['total_impressions'] > 0 else 0
            )
            
            aggregated.append({
                'url1': url1,
                'url2': url2,
                'shared_queries': data['queries'],
                'shared_queries_count': len(data['queries']),
                'total_shared_clicks': data['total_clicks'],
                'total_shared_impressions': data['total_impressions'],
                'avg_position_spread': avg_position_spread,
                'direct_competition_count': data['competition_types']['direct'],
                'moderate_competition_count': data['competition_types']['moderate'],
                'indirect_competition_count': data['competition_types']['indirect'],
                'total_query_value': sum(data['query_values'])
            })
        
        return aggregated
    
    def _calculate_keyword_risk_scores(self, 
                                     aggregated_data: List[Dict]) -> pd.DataFrame:
        """Calculate keyword cannibalization risk scores"""
        df = pd.DataFrame(aggregated_data)
        
        if df.empty:
            return df
        
        # Normalize metrics for scoring
        max_queries = df['shared_queries_count'].max()
        max_impressions = df['total_shared_impressions'].max()
        max_clicks = df['total_shared_clicks'].max()
        max_value = df['total_query_value'].max()
        
        # Calculate component scores
        df['query_overlap_score'] = df['shared_queries_count'] / max_queries if max_queries > 0 else 0
        df['traffic_impact_score'] = df['total_shared_impressions'] / max_impressions if max_impressions > 0 else 0
        df['click_loss_score'] = df['total_shared_clicks'] / max_clicks if max_clicks > 0 else 0
        df['value_score'] = df['total_query_value'] / max_value if max_value > 0 else 0
        
        # Position spread score (inverse - closer positions = higher score)
        df['position_competition_score'] = 1 / (1 + df['avg_position_spread'] / 10)
        
        # Direct competition weight
        df['competition_severity'] = (
            df['direct_competition_count'] * 1.0 +
            df['moderate_competition_count'] * 0.5 +
            df['indirect_competition_count'] * 0.2
        ) / df['shared_queries_count']
        
        # Calculate final keyword cannibalization score
        df['keyword_cannibalization_score'] = (
            df['query_overlap_score'] * 0.20 +
            df['traffic_impact_score'] * 0.25 +
            df['click_loss_score'] * 0.20 +
            df['value_score'] * 0.15 +
            df['position_competition_score'] * 0.10 +
            df['competition_severity'] * 0.10
        )
        
        # Add risk category
        df['keyword_risk_category'] = pd.cut(
            df['keyword_cannibalization_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return df
    
    def get_query_level_analysis(self, url1: str, url2: str) -> pd.DataFrame:
        """Get detailed query-level analysis for a URL pair"""
        # Find all queries where both URLs appear
        url1_queries = set(self.gsc_data[self.gsc_data['url'] == url1]['query'])
        url2_queries = set(self.gsc_data[self.gsc_data['url'] == url2]['query'])
        shared_queries = url1_queries & url2_queries
        
        if not shared_queries:
            return pd.DataFrame()
        
        # Get data for shared queries
        shared_data = self.gsc_data[
            self.gsc_data['query'].isin(shared_queries) &
            self.gsc_data['url'].isin([url1, url2])
        ].copy()
        
        # Pivot to compare URLs side by side
        pivot_data = shared_data.pivot_table(
            index='query',
            columns='url',
            values=['clicks', 'impressions', 'position', 'ctr'],
            aggfunc='first'
        )
        
        # Flatten column names
        pivot_data.columns = [f'{col[0]}_{col[1]}' for col in pivot_data.columns]
        pivot_data = pivot_data.reset_index()
        
        # Calculate competition metrics
        pivot_data['position_difference'] = abs(
            pivot_data[f'position_{url1}'] - pivot_data[f'position_{url2}']
        )
        pivot_data['clicks_total'] = (
            pivot_data[f'clicks_{url1}'] + pivot_data[f'clicks_{url2}']
        )
        pivot_data['impressions_total'] = (
            pivot_data[f'impressions_{url1}'] + pivot_data[f'impressions_{url2}']
        )
        
        # Identify winner for each query
        pivot_data['winning_url'] = pivot_data.apply(
            lambda row: url1 if row[f'position_{url1}'] < row[f'position_{url2}'] else url2,
            axis=1
        )
        
        return pivot_data.sort_values('impressions_total', ascending=False)
    
    def find_keyword_clusters(self, min_cluster_size: int = 3) -> Dict[str, List[str]]:
        """Find clusters of related keywords that multiple URLs compete for"""
        # Create keyword co-occurrence matrix
        keyword_urls = defaultdict(set)
        for _, row in self.gsc_data.iterrows():
            keyword_urls[row['query']].add(row['url'])
        
        # Find keywords that share multiple URLs
        keyword_clusters = defaultdict(set)
        
        for query1, urls1 in keyword_urls.items():
            if len(urls1) < 2:  # Skip keywords with single URL
                continue
                
            for query2, urls2 in keyword_urls.items():
                if query1 >= query2:  # Avoid duplicates
                    continue
                    
                # Check URL overlap
                common_urls = urls1 & urls2
                if len(common_urls) >= 2:  # At least 2 URLs compete for both keywords
                    # Add to cluster
                    cluster_key = tuple(sorted(common_urls))
                    keyword_clusters[cluster_key].add(query1)
                    keyword_clusters[cluster_key].add(query2)
        
        # Filter clusters by size
        significant_clusters = {
            urls: list(keywords)
            for urls, keywords in keyword_clusters.items()
            if len(keywords) >= min_cluster_size
        }
        
        return significant_clusters
    
    def calculate_cannibalization_impact(self) -> Dict[str, float]:
        """Calculate the overall impact of cannibalization on the site"""
        # Find all cannibalized queries
        query_url_counts = self.gsc_data.groupby('query')['url'].nunique()
        cannibalized_queries = query_url_counts[query_url_counts > 1].index
        
        # Calculate metrics
        cannibalized_data = self.gsc_data[self.gsc_data['query'].isin(cannibalized_queries)]
        all_data = self.gsc_data
        
        impact_metrics = {
            'total_queries': len(all_data['query'].unique()),
            'cannibalized_queries': len(cannibalized_queries),
            'cannibalization_rate': len(cannibalized_queries) / len(query_url_counts),
            'cannibalized_clicks': cannibalized_data['clicks'].sum(),
            'total_clicks': all_data['clicks'].sum(),
            'click_loss_rate': cannibalized_data['clicks'].sum() / all_data['clicks'].sum(),
            'cannibalized_impressions': cannibalized_data['impressions'].sum(),
            'total_impressions': all_data['impressions'].sum(),
            'impression_loss_rate': cannibalized_data['impressions'].sum() / all_data['impressions'].sum(),
            'avg_position_cannibalized': cannibalized_data['position'].mean(),
            'avg_position_overall': all_data['position'].mean()
        }
        
        # Estimate potential traffic gain from fixing cannibalization
        # Assume best-performing URL could capture 70% of combined traffic
        potential_gain = 0
        for query in cannibalized_queries:
            query_data = self.gsc_data[self.gsc_data['query'] == query]
            total_clicks = query_data['clicks'].sum()
            best_clicks = query_data['clicks'].max()
            potential_gain += (total_clicks * 0.7) - best_clicks
        
        impact_metrics['estimated_click_gain'] = max(0, potential_gain)
        impact_metrics['estimated_gain_percentage'] = (
            potential_gain / all_data['clicks'].sum() if all_data['clicks'].sum() > 0 else 0
        )
        
        return impact_metrics