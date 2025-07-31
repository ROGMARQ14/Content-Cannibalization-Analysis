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
    
    def _calculate_pair_similarities(self, url1_data: pd.Series, url2_data: pd.Series) -> Dict[str, float]:
        """Calculate all similarity metrics for a URL pair"""
        
        # Title similarity
        title_sim = self._calculate_text_similarity(
            str(url1_data.get('title', '')), 
            str(url2_data.get('title', ''))
        )
        
        # H1 similarity
        h1_sim = self._calculate_text_similarity(
            str(url1_data.get('h1', '')), 
            str(url2_data.get('h1', ''))
        )
        
        # Meta description similarity
        meta_sim = self._calculate_text_similarity(
            str(url1_data.get('meta_description', '')), 
            str(url2_data.get('meta_description', ''))
        )
        
        # Debug log for text similarities
        logger.debug(f"Text similarities - Title: {title_sim:.3f}, H1: {h1_sim:.3f}, Meta: {meta_sim:.3f}")
        
        # Semantic/Content similarity
        if self.use_content_embeddings and 'content_embedding' in url1_data.index:
            # Use content embeddings from our content analysis
            semantic_sim = self._calculate_embedding_similarity(
                url1_data.get('content_embedding'),
                url2_data.get('content_embedding')
            )
            similarity_type = 'content_similarity'
            logger.debug(f"Using content embeddings, similarity: {semantic_sim:.3f}")
        elif self.embeddings_data is not None:
            # Use Screaming Frog embeddings
            semantic_sim = self._calculate_sf_embedding_similarity(
                url1_data['url'],
                url2_data['url']
            )
            similarity_type = 'semantic_similarity'
            logger.debug(f"Using SF embeddings, similarity: {semantic_sim:.3f}")
        else:
            # Fallback to text-based semantic similarity
            semantic_sim = self._calculate_semantic_similarity_fallback(url1_data, url2_data)
            similarity_type = 'semantic_similarity'
            logger.debug(f"Using fallback semantic similarity: {semantic_sim:.3f}")
        
        # Calculate overall similarity (weighted average)
        # Note: These weights will be overridden by user settings
        overall_sim = (
            title_sim * 0.30 +
            h1_sim * 0.20 +
            semantic_sim * 0.35 +
            meta_sim * 0.15
        )
        
        return {
            'title_similarity': title_sim,
            'h1_similarity': h1_sim,
            'meta_description_similarity': meta_sim,
            similarity_type: semantic_sim,
            'overall_similarity': overall_sim
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        # Clean and normalize text
        text1 = self._normalize_text(str(text1))
        text2 = self._normalize_text(str(text2))
        
        # Quick exact match check
        if text1 == text2:
            return 1.0
        
        # Use TF-IDF for similarity
        try:
            # Create new vectorizer for each comparison to avoid state issues
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Fit and transform both texts
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # Fallback to simple word overlap
            return self._calculate_word_overlap(text1, text2)
    
    def _calculate_embedding_similarity(self, embedding1, embedding2) -> float:
        """Calculate similarity between two embedding vectors"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Convert to numpy arrays if needed
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            embedding1 = embedding1.reshape(1, -1)
            embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def _calculate_sf_embedding_similarity(self, url1: str, url2: str) -> float:
        """Calculate similarity using Screaming Frog embeddings"""
        if self.embeddings_data is None:
            return 0.0
        
        try:
            # Find embeddings for both URLs
            url1_embedding = self._get_sf_embedding(url1)
            url2_embedding = self._get_sf_embedding(url2)
            
            if url1_embedding is None or url2_embedding is None:
                # Log which URL is missing
                if url1_embedding is None:
                    logger.debug(f"No embedding for URL1: {url1}")
                if url2_embedding is None:
                    logger.debug(f"No embedding for URL2: {url2}")
                return 0.0
            
            return self._calculate_embedding_similarity(url1_embedding, url2_embedding)
            
        except Exception as e:
            logger.error(f"Error with SF embeddings: {e}")
            return 0.0
    
    def _get_sf_embedding(self, url: str) -> Optional[np.ndarray]:
        """Get Screaming Frog embedding for a URL"""
        try:
            # ALWAYS normalize the input URL first
            normalized_input_url = url.rstrip('/').lower()
            logger.debug(f"Looking for embedding for URL: {url} (normalized: {normalized_input_url})")
            
            # Try different URL column names
            url_columns = ['URL', 'url', 'Address', 'address', 'Url']
            
            # Find which column contains URLs in embeddings data
            url_col_found = None
            for col in url_columns:
                if col in self.embeddings_data.columns:
                    url_col_found = col
                    logger.debug(f"Found URL column: {col}")
                    break
            
            if not url_col_found:
                logger.error(f"No URL column found in embeddings data. Columns: {self.embeddings_data.columns.tolist()}")
                return None
            
            # NORMALIZE ALL URLs in embeddings data for comparison
            # Create a temporary normalized column if it doesn't exist
            if '_normalized_url' not in self.embeddings_data.columns:
                self.embeddings_data['_normalized_url'] = self.embeddings_data[url_col_found].str.rstrip('/').str.lower()
            
            # Try to find match using normalized URLs
            mask = self.embeddings_data['_normalized_url'] == normalized_input_url
            
            if not mask.any():
                # Try without protocol as fallback
                url_without_protocol = normalized_input_url.replace('https://', '').replace('http://', '')
                mask = self.embeddings_data['_normalized_url'].str.replace('https://', '').str.replace('http://', '') == url_without_protocol
                
                if not mask.any():
                    logger.warning(f"No embedding found for URL: {url}")
                    # Debug: Show some sample normalized URLs from embeddings
                    if len(self.embeddings_data) > 0:
                        logger.debug("Sample normalized URLs in embeddings data:")
                        for i in range(min(3, len(self.embeddings_data))):
                            logger.debug(f"  - Original: {self.embeddings_data.iloc[i][url_col_found]}")
                            logger.debug(f"  - Normalized: {self.embeddings_data.iloc[i]['_normalized_url']}")
                    return None
            
            # Get the row
            row = self.embeddings_data[mask].iloc[0]
            
            # Get embedding columns (all numeric columns except URL columns)
            embedding_cols = [c for c in self.embeddings_data.columns 
                            if c not in url_columns + ['_normalized_url'] and 
                            pd.api.types.is_numeric_dtype(self.embeddings_data[c])]
            
            if not embedding_cols:
                logger.error("No numeric columns found for embeddings")
                return None
            
            logger.debug(f"Found {len(embedding_cols)} embedding columns")
            return row[embedding_cols].values.astype(float)
            
        except Exception as e:
            logger.error(f"Error getting SF embedding: {e}")
            return None
    
    def _calculate_semantic_similarity_fallback(self, url1_data: pd.Series, url2_data: pd.Series) -> float:
        """Calculate semantic similarity without embeddings (fallback method)"""
        # Combine all text fields
        text1 = ' '.join([
            str(url1_data.get('title', '')),
            str(url1_data.get('h1', '')),
            str(url1_data.get('meta_description', ''))
        ])
        
        text2 = ' '.join([
            str(url2_data.get('title', '')),
            str(url2_data.get('h1', '')),
            str(url2_data.get('meta_description', ''))
        ])
        
        return self._calculate_text_similarity(text1, text2)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Simple word overlap calculation (Jaccard similarity)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_similarity_stats(self, similarity_df: pd.DataFrame) -> Dict:
        """Get statistics about the similarity analysis"""
        if similarity_df.empty:
            return {
                'avg_title_similarity': 0,
                'avg_h1_similarity': 0,
                'avg_semantic_similarity': 0,
                'high_similarity_count': 0
            }
        
        stats = {
            'avg_title_similarity': similarity_df['title_similarity'].mean(),
            'avg_h1_similarity': similarity_df['h1_similarity'].mean(),
            'avg_semantic_similarity': similarity_df.get('semantic_similarity', 
                                                       similarity_df.get('content_similarity', 0)).mean(),
            'high_similarity_count': len(similarity_df[similarity_df['overall_similarity'] > 0.7])
        }
        
        return stats
