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
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def calculate_all_similarities(self, 
                                 urls_df: pd.DataFrame, 
                                 min_similarity: float = 0.3) -> pd.DataFrame:
        """Calculate all pairwise similarities between URLs"""
        logger.info(f"Calculating similarities for {len(urls_df)} URLs")
        
        similarity_pairs = []
        
        # Calculate pairwise similarities
        for i in range(len(urls_df)):
            for j in range(i + 1, len(urls_df)):
                url1_data = urls_df.iloc[i]
                url2_data = urls_df.iloc[j]
                
                # Calculate individual similarities
                similarities = self._calculate_pair_similarities(url1_data, url2_data)
                
                # Only keep if above minimum threshold
                if similarities['overall_similarity'] >= min_similarity:
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
        
        logger.info(f"Found {len(similarity_pairs)} pairs above {min_similarity} similarity threshold")
        
        return pd.DataFrame(similarity_pairs)
    
    def _calculate_pair_similarities(self, url1_data: pd.Series, url2_data: pd.Series) -> Dict[str, float]:
        """Calculate all similarity metrics for a URL pair"""
        
        # Title similarity
        title_sim = self._calculate_text_similarity(
            url1_data.get('title', ''), 
            url2_data.get('title', '')
        )
        
        # H1 similarity
        h1_sim = self._calculate_text_similarity(
            url1_data.get('h1', ''), 
            url2_data.get('h1', '')
        )
        
        # Meta description similarity
        meta_sim = self._calculate_text_similarity(
            url1_data.get('meta_description', ''), 
            url2_data.get('meta_description', '')
        )
        
        # Semantic/Content similarity
        if self.use_content_embeddings and 'content_embedding' in url1_data.index:
            # Use content embeddings from our content analysis
            semantic_sim = self._calculate_embedding_similarity(
                url1_data.get('content_embedding'),
                url2_data.get('content_embedding')
            )
            similarity_type = 'content_similarity'
        elif self.embeddings_data is not None:
            # Use Screaming Frog embeddings
            semantic_sim = self._calculate_sf_embedding_similarity(
                url1_data['url'],
                url2_data['url']
            )
            similarity_type = 'semantic_similarity'
        else:
            # Fallback to text-based semantic similarity
            semantic_sim = self._calculate_semantic_similarity_fallback(url1_data, url2_data)
            similarity_type = 'semantic_similarity'
        
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
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        # Quick exact match check
        if text1 == text2:
            return 1.0
        
        # Use TF-IDF for similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
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
                return 0.0
            
            return self._calculate_embedding_similarity(url1_embedding, url2_embedding)
            
        except Exception as e:
            logger.error(f"Error with SF embeddings: {e}")
            return 0.0
    
    def _get_sf_embedding(self, url: str) -> Optional[np.ndarray]:
        """Get Screaming Frog embedding for a URL"""
        try:
            # Try different URL column names
            url_columns = ['URL', 'url', 'Address', 'address']
            
            for col in url_columns:
                if col in self.embeddings_data.columns:
                    # Find the row for this URL
                    mask = self.embeddings_data[col] == url
                    if not mask.any():
                        # Try with normalized URL
                        normalized_url = url.rstrip('/').lower()
                        mask = self.embeddings_data[col].str.rstrip('/').str.lower() == normalized_url
                    
                    if mask.any():
                        row = self.embeddings_data[mask].iloc[0]
                        
                        # Get embedding columns (all numeric columns except URL)
                        embedding_cols = [c for c in self.embeddings_data.columns 
                                        if c not in url_columns and 
                                        pd.api.types.is_numeric_dtype(self.embeddings_data[c])]
                        
                        if embedding_cols:
                            return row[embedding_cols].values.astype(float)
            
            logger.warning(f"No embedding found for URL: {url}")
            return None
            
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