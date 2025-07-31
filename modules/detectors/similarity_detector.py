# modules/detectors/similarity_detector.py
"""
Similarity-based cannibalization detection
Traditional approach based on content similarity
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from ..analyzers.similarity_analyzer import SimilarityAnalyzer
from ..analyzers.content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)

class SimilarityDetector:
    """Detect cannibalization through content similarity"""
    
    def __init__(self, internal_data: pd.DataFrame, 
                 embeddings_data: Optional[pd.DataFrame] = None,
                 config: Optional[Dict] = None):
        """
        Initialize similarity detector
        
        Args:
            internal_data: Crawler data with metadata
            embeddings_data: Optional embeddings data
            config: Configuration settings
        """
        self.internal_data = internal_data
        self.embeddings_data = embeddings_data
        self.config = config or {}
        
        # Normalize URLs in embeddings if provided
        if self.embeddings_data is not None:
            self._normalize_embeddings_urls()
        
    def _normalize_embeddings_urls(self):
        """Normalize URLs in embeddings data for better matching"""
        # Import here to avoid circular imports
        from utils.url_normalizer import URLNormalizer
        
        # Find URL column
        url_columns = ['URL', 'url', 'Address', 'address']
        url_col = None
        
        for col in url_columns:
            if col in self.embeddings_data.columns:
                url_col = col
                break
        
        if url_col:
            # Add normalized URL column
            self.embeddings_data['url_normalized'] = self.embeddings_data[url_col].apply(
                URLNormalizer.normalize_for_matching
            )
            logger.info(f"Normalized {len(self.embeddings_data)} URLs in embeddings data")
    
    def detect_all_similarity(self, use_content_analysis: bool = False) -> pd.DataFrame:
        """
        Detect all similarity-based cannibalization
        
        Args:
            use_content_analysis: Whether to fetch and analyze page content
            
        Returns:
            DataFrame with similarity-based cannibalization issues
        """
        logger.info("Starting similarity-based detection")
        
        # Initialize similarity analyzer
        similarity_analyzer = SimilarityAnalyzer(
            embeddings_data=self.embeddings_data,
            use_content_embeddings=use_content_analysis
        )
        
        # 1. Calculate similarities
        min_similarity = self.config.get('min_similarity', 0.20)
        similarity_results = similarity_analyzer.calculate_all_similarities(
            self.internal_data,
            min_similarity
        )
        
        if similarity_results.empty:
            logger.warning("No similarity matches found")
            return pd.DataFrame()
        
        # Add detection source
        similarity_results['detection_source'] = 'content_similarity'
        
        # 2. Apply custom weights if provided
        if 'weights' in self.config:
            similarity_results = self._recalculate_with_weights(
                similarity_results, 
                self.config['weights']
            )
        
        # 3. Calculate risk scores
        similarity_results = self._calculate_similarity_risk_scores(similarity_results)
        
        # 4. Add metadata
        similarity_results = self._enrich_with_metadata(similarity_results)
        
        return similarity_results
    
    def _recalculate_with_weights(self, df: pd.DataFrame, weights: Dict) -> pd.DataFrame:
        """Recalculate overall similarity with custom weights"""
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            norm_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            norm_weights = weights
        
        # Recalculate overall similarity
        df['overall_similarity'] = 0
        
        if 'title_similarity' in df.columns:
            df['overall_similarity'] += df['title_similarity'] * norm_weights.get('title', 0.25)
        
        if 'h1_similarity' in df.columns:
            df['overall_similarity'] += df['h1_similarity'] * norm_weights.get('h1', 0.15)
        
        if 'semantic_similarity' in df.columns:
            df['overall_similarity'] += df['semantic_similarity'] * norm_weights.get('content', 0.35)
        elif 'content_similarity' in df.columns:
            df['overall_similarity'] += df['content_similarity'] * norm_weights.get('content', 0.35)
        
        if 'meta_description_similarity' in df.columns:
            df['overall_similarity'] += df['meta_description_similarity'] * norm_weights.get('meta', 0.15)
        
        return df
    
    def _calculate_similarity_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk scores based on similarity"""
        # Component scores
        df['title_risk'] = df['title_similarity'].apply(
            lambda x: 1.0 if x > 0.9 else (0.7 if x > 0.7 else (0.3 if x > 0.5 else 0))
        )
        
        df['content_risk'] = df.get('semantic_similarity', df.get('content_similarity', 0)).apply(
            lambda x: 1.0 if x > 0.8 else (0.6 if x > 0.6 else (0.3 if x > 0.4 else 0))
        )
        
        # Intent matching bonus
        if 'intent1' in df.columns and 'intent2' in df.columns:
            df['intent_match'] = (df['intent1'] == df['intent2']).astype(float)
        else:
            df['intent_match'] = 0.5  # Neutral if no intent data
        
        # Calculate similarity risk score
        df['similarity_risk_score'] = (
            df['overall_similarity'] * 0.5 +
            df['title_risk'] * 0.2 +
            df['content_risk'] * 0.2 +
            df['intent_match'] * 0.1
        )
        
        # Add risk category
        df['risk_category'] = pd.cut(
            df['similarity_risk_score'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Use similarity score as primary score for compatibility
        df['competition_score'] = df['similarity_risk_score']
        
        return df
    
    def _enrich_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata from internal data"""
        # Get metadata for each URL
        url_metadata = self.internal_data.set_index('url')[
            ['title', 'h1', 'meta_description']
        ].to_dict('index')
        
        # Add metadata
        for idx, row in df.iterrows():
            url1_meta = url_metadata.get(row['url1'], {})
            url2_meta = url_metadata.get(row['url2'], {})
            
            df.at[idx, 'title1_full'] = url1_meta.get('title', '')
            df.at[idx, 'title2_full'] = url2_meta.get('title', '')
            df.at[idx, 'h1_1_full'] = url1_meta.get('h1', '')
            df.at[idx, 'h1_2_full'] = url2_meta.get('h1', '')
        
        return df
    
    async def detect_with_content_analysis(self, ai_analyzer=None) -> pd.DataFrame:
        """
        Enhanced detection with actual content analysis
        
        Args:
            ai_analyzer: Optional AI analyzer for advanced analysis
            
        Returns:
            DataFrame with content-based cannibalization issues
        """
        logger.info("Starting content-based similarity detection")
        
        # Initialize content analyzer
        content_analyzer = ContentAnalyzer(
            ai_analyzer=ai_analyzer,
            extraction_method='smart'
        )
        
        # Analyze content
        logger.info("Fetching and analyzing page content...")
        internal_with_content = await content_analyzer.analyze_content_similarity(
            self.internal_data,
            fetch_content=True
        )
        
        # Update internal data
        self.internal_data = internal_with_content
        
        # Run similarity detection with content embeddings
        return self.detect_all_similarity(use_content_analysis=True)
    
    def get_similarity_stats(self, results: pd.DataFrame) -> Dict:
        """Get statistics about similarity detection"""
        if results.empty:
            return {
                'total_pairs': 0,
                'avg_similarity': 0,
                'high_similarity_pairs': 0,
                'identical_titles': 0
            }
        
        stats = {
            'total_pairs': len(results),
            'avg_similarity': results['overall_similarity'].mean(),
            'high_similarity_pairs': len(results[results['overall_similarity'] > 0.7]),
            'identical_titles': len(results[results['title_similarity'] > 0.95]),
            'similarity_distribution': {
                '0-20%': len(results[results['overall_similarity'] < 0.2]),
                '20-40%': len(results[(results['overall_similarity'] >= 0.2) & 
                                    (results['overall_similarity'] < 0.4)]),
                '40-60%': len(results[(results['overall_similarity'] >= 0.4) & 
                                    (results['overall_similarity'] < 0.6)]),
                '60-80%': len(results[(results['overall_similarity'] >= 0.6) & 
                                    (results['overall_similarity'] < 0.8)]),
                '80-100%': len(results[results['overall_similarity'] >= 0.8])
            }
        }
        
        return stats
