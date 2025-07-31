# modules/analyzers/similarity_analyzer.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import Dict, List, Optional, Tuple
import re
from utils.url_normalizer import URLNormalizer

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
        self.tfidf_vectorizer = None
        
        # Prepare embeddings data if provided
        if self.embeddings_data is not None:
            self._prepare_embeddings_data()
        
        logger.info(f"SimilarityAnalyzer initialized with embeddings: {embeddings_data is not None}, "
                   f"use_content: {use_content_embeddings}")
    
    def _prepare_embeddings_data(self):
        """Prepare embeddings data with normalized URLs"""
        # Find URL column
        url_columns = ['URL', 'url', 'Address', 'address', 'Url']
        url_col = None
        
        for col in url_columns:
            if col in self.embeddings_data.columns:
                url_col = col
                break
        
        if not url_col:
            logger.error(f"No URL column found in embeddings. Columns: {self.embeddings_data.columns.tolist()}")
            return
        
        # Normalize URLs for matching
        self.embeddings_data['url_normalized'] = self.embeddings_data[url_col].apply(
            URLNormalizer.normalize_for_matching
        )
        
        # Create lookup dictionary for faster access
        self.embeddings_lookup = {}
        
        for idx, row in self.embeddings_data.iterrows():
            normalized_url = row['url_normalized']
            
            # Get embedding columns (numeric columns)
            embedding_cols = [c for c in self.embeddings_data.columns 
                            if c not in url_columns + ['url_normalized'] and 
                            pd.api.types.is_numeric_dtype(self.embeddings_data[c])]
            
            if embedding_cols:
                self.embeddings_lookup[normalized_url] = row[embedding_cols].values.astype(float)
        
        logger.info(f"Prepared {len(self.embeddings_lookup)} embeddings for matching")
    
    def calculate_all_similarities(self, 
                                 urls_df: pd.DataFrame, 
                                 min_similarity: float = 0.3) -> pd.DataFrame:
        """Calculate all pairwise similarities between URLs"""
        logger.info(f"Calculating similarities for {len(urls_df)} URLs")
        logger.info(f"Minimum similarity threshold: {min_similarity}")
        
        # Ensure URLs are normalized in the input data
        if 'url_normalized' not in urls_df.columns:
            urls_df['url_normalized'] = urls_df['url'].apply(URLNormalizer.normalize_for_matching)
        
        similarity_pairs = []
        total_comparisons = 0
        above_threshold = 0
        
        # Calculate pairwise similarities
        for i in range(len(urls_df)):
            for j in range(i + 1, len(urls_df)):
                total_comparisons += 1
                url1_data = urls_df.iloc[i]
                url2_data = urls_df.iloc[j]
                
                # Calculate similarities
                similarities = self._calculate_pair_similarities(url1_data, url2_data)
                
                # Check threshold
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
        
        logger.info(f"Total comparisons: {total_comparisons}, Above threshold: {above_threshold}")
        
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
        
        # Semantic/Content similarity
        if self.use_content_embeddings and 'content_embedding' in url1_data.index:
            # Use content embeddings from content analysis
            semantic_sim = self._calculate_embedding_similarity(
                url1_data.get('content_embedding'),
                url2_data.get('content_embedding')
            )
            similarity_type = 'content_similarity'
        elif self.embeddings_lookup:
            # Use Screaming Frog embeddings with improved matching
            semantic_sim = self._calculate_embeddings_similarity_improved(
                url1_data.get('url_normalized', url1_data['url']),
                url2_data.get('url_normalized', url2_data['url'])
            )
            similarity_type = 'semantic_similarity'
        else:
            # Fallback to text-based semantic similarity
            semantic_sim = self._calculate_semantic_similarity_fallback(url1_data, url2_data)
            similarity_type = 'semantic_similarity'
        
        # Calculate overall similarity (weighted average)
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
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # Fallback to word overlap
            return self._calculate_word_overlap(text1, text2)
    
    def _calculate_embeddings_similarity_improved(self, url1_norm: str, url2_norm: str) -> float:
        """Calculate similarity using embeddings with improved URL matching"""
        # Try direct lookup first
        embedding1 = self.embeddings_lookup.get(url1_norm)
        embedding2 = self.embeddings_lookup.get(url2_norm)
        
        # If direct lookup fails, try variations
        if embedding1 is None:
            embedding1 = self._find_embedding_fuzzy(url1_norm)
        if embedding2 is None:
            embedding2 = self._find_embedding_fuzzy(url2_norm)
        
        if embedding1 is None or embedding2 is None:
            logger.debug(f"No embeddings found for: {url1_norm} or {url2_norm}")
            return 0.0
        
        return self._calculate_embedding_similarity(embedding1, embedding2)
    
    def _find_embedding_fuzzy(self, normalized_url: str) -> Optional[np.ndarray]:
        """Try to find embedding with fuzzy matching"""
        # Try exact match first
        if normalized_url in self.embeddings_lookup:
            return self.embeddings_lookup[normalized_url]
        
        # Try without domain variations
        url_parts = normalized_url.split('/')
        if len(url_parts) > 1:
            # Try path only
            path_only = '/'.join(url_parts[1:])
            for stored_url, embedding in self.embeddings_lookup.items():
                if stored_url.endswith(path_only):
                    logger.debug(f"Fuzzy matched {normalized_url} to {stored_url}")
                    return embedding
        
        # Try with common variations
        variations = [
            normalized_url.rstrip('/'),
            normalized_url + '/',
            normalized_url.replace('http://', 'https://'),
            normalized_url.replace('https://', 'http://')
        ]
        
        for variant in variations:
            if variant in self.embeddings_lookup:
                return self.embeddings_lookup[variant]
        
        return None
    
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
            
            # Ensure same shape
            if embedding1.shape != embedding2.shape:
                logger.warning(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
                return 0.0
            
            # Calculate cosine similarity
            embedding1 = embedding1.reshape(1, -1)
            embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(max(0, similarity))  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return 0.0
    
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