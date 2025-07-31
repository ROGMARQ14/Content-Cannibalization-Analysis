# modules/analyzers/similarity_analyzer.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    """Calculate similarities between URLs using various methods"""
    
    def __init__(self, embeddings_data: Optional[pd.DataFrame] = None, 
                 use_content_embeddings: bool = False):
        """
        Initialize similarity analyzer
        
        Args:
            embeddings_data: DataFrame with URL and embedding columns (e.g., from Screaming Frog)
            use_content_embeddings: Whether to use content embeddings from content analysis
        """
        self.embeddings_data = embeddings_data
        self.use_content_embeddings = use_content_embeddings
        self.tfidf_vectorizer = None  # Initialize on first use
        
        logger.info(f"SimilarityAnalyzer initialized with embeddings_data: {embeddings_data is not None}, "
                   f"use_content_embeddings: {use_content_embeddings}")
        
        # Debug embeddings data
        if embeddings_data is not None:
            logger.info(f"Embeddings data shape: {embeddings_data.shape}")
            logger.info(f"Embeddings columns: {embeddings_data.columns.tolist()}")
            if len(embeddings_data) > 0:
                logger.info(f"Sample embedding URL: {embeddings_data.iloc[0].get('URL', embeddings_data.iloc[0].get('url', 'No URL column'))}")
        
    def calculate_all_similarities(self, 
                                 urls_df: pd.DataFrame, 
                                 min_similarity: float = 0.3) -> pd.DataFrame:
        """Calculate all pairwise similarities between URLs"""
        logger.info(f"Calculating similarities for {len(urls_df)} URLs")
        logger.info(f"Columns in urls_df: {urls_df.columns.tolist()}")
        logger.info(f"Minimum similarity threshold: {min_similarity}")
        
        # Debug: Show sample URLs
        if len(urls_df) > 0:
            logger.info(f"Sample URLs from internal data:")
            for i in range(min(3, len(urls_df))):
                logger.info(f"  - {urls_df.iloc[i]['url']}")
        
        similarity_pairs = []
        total_comparisons = 0
        above_threshold = 0
        
        # Calculate pairwise similarities
        for i in range(len(urls_df)):
            for j in range(i + 1, len(urls_df)):
                total_comparisons += 1
                url1_data = urls_df.iloc[i]
                url2_data = urls_df.iloc[j]
                
                # Log first pair for debugging
                if i == 0 and j == 1:
                    logger.info(f"Sample URL1 data: {url1_data.to_dict()}")
                    logger.info(f"Sample URL2 data: {url2_data.to_dict()}")
                
                # Calculate individual similarities
                similarities = self._calculate_pair_similarities(url1_data, url2_data)
                
                # Log first similarity calculation
                if i == 0 and j == 1:
                    logger.info(f"Sample similarities: {similarities}")
                
                # Debug: Log some similarity scores
                if total_comparisons <= 5:
                    logger.info(f"Pair {total_comparisons}: {url1_data['url'][:50]} vs {url2_data['url'][:50]}")
                    logger.info(f"  Overall similarity: {similarities['overall_similarity']:.3f}")
                
                # Check if above minimum threshold
                if similarities['overall_similarity'] >= min_similarity:
                    above_threshold += 1
                    pair_data = {
                        'url1': url1_data['url'],
                        'url2': url2_data['url'],
                        'title1': url1_data.get('title', ''),
                        'title2': url2_data.get('title', ''),
                        'intent1': url1_data.get('ai_intent', 'Unknown'),
                        'intent2': url2_data.get('ai_intent', 'Unknown'),
                        **similarities
                    }
                    similarity_pairs.append(pair_data)
        
        logger.info(f"Total comparisons made: {total_comparisons}")
        logger.info(f"Pairs above threshold ({min_similarity}): {above_threshold}")
        logger.info(f"Found {len(similarity_pairs)} pairs above {min_similarity} similarity threshold")
        
        # Debug: Show distribution of similarities
        if total_comparisons > 0:
            all_sims = []
            for i in range(len(urls_df)):
                for j in range(i + 1, len(urls_df)):
                    url1_data = urls_df.iloc[i]
                    url2_data = urls_df.iloc[j]
                    sims = self._calculate_pair_similarities(url1_data, url2_data)
                    all_sims.append(sims['overall_similarity'])
            
            if all_sims:
                logger.info(f"Similarity distribution:")
                logger.info(f"  Min: {min(all_sims):.3f}")
                logger.info(f"  Max: {max(all_sims):.3f}")
                logger.info(f"  Mean: {np.mean(all_sims):.3f}")
                logger.info(f"  Median: {np.median(all_sims):.3f}")
                
                # Show how many fall into different ranges
                ranges = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
                for low, high in ranges:
                    count = sum(1 for s in all_sims if low <= s < high)
                    logger.info(f"  {low:.1f}-{high:.1f}: {count} pairs")
        
        return pd.DataFrame(similarity_pairs)
