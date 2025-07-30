# modules/analyzers/content_analyzer.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup, Comment
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import logging
from readability.readability import Document as Readability
import trafilatura
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Analyze full page content for deeper cannibalization detection"""
    
    # Common patterns for non-content areas
    EXCLUDE_PATTERNS = {
        'classes': [
            'header', 'footer', 'nav', 'navigation', 'menu', 'sidebar', 
            'widget', 'banner', 'ad', 'advertisement', 'popup', 'modal',
            'cookie', 'breadcrumb', 'pagination', 'comments', 'related',
            'social', 'share', 'newsletter', 'subscribe', 'copyright',
            'disclaimer', 'privacy', 'terms', 'conditions'
        ],
        'ids': [
            'header', 'footer', 'nav', 'navigation', 'menu', 'sidebar',
            'comments', 'related-posts', 'cookie-banner', 'popup'
        ],
        'tags': [
            'header', 'footer', 'nav', 'aside', 'script', 'style',
            'noscript', 'iframe'
        ],
        'attributes': {
            'role': ['navigation', 'banner', 'complementary', 'contentinfo'],
            'aria-label': ['navigation', 'breadcrumb', 'footer']
        }
    }
    
    # Patterns that indicate main content
    CONTENT_PATTERNS = {
        'classes': [
            'content', 'main-content', 'article-content', 'post-content',
            'entry-content', 'article-body', 'post-body', 'story-body',
            'text-content', 'body-content', 'page-content', 'main',
            'article', 'post', 'entry', 'story', 'blog-post'
        ],
        'ids': [
            'content', 'main-content', 'article', 'main', 'post',
            'article-body', 'story-body'
        ],
        'tags': [
            'main', 'article'
        ],
        'attributes': {
            'role': ['main', 'article'],
            'itemprop': ['articleBody', 'text']
        }
    }
    
    def __init__(self, ai_analyzer=None, extraction_method='smart'):
        """
        Initialize content analyzer
        
        Args:
            ai_analyzer: AI analyzer instance for advanced analysis
            extraction_method: 'smart', 'trafilatura', 'readability', or 'custom'
        """
        self.ai_analyzer = ai_analyzer
        self.extraction_method = extraction_method
        self.sentence_model = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    async def analyze_content_similarity(self, 
                                       urls_df: pd.DataFrame,
                                       fetch_content: bool = True,
                                       content_column: Optional[str] = None,
                                       extraction_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Analyze content similarity between URLs with smart content extraction
        """
        logger.info(f"Analyzing content for {len(urls_df)} URLs using {self.extraction_method} extraction")
        
        # Allow custom extraction configuration
        if extraction_config:
            self.update_extraction_config(extraction_config)
        
        # Step 1: Get content for each URL
        if fetch_content and 'Address' in urls_df.columns:
            urls_df = await self._fetch_page_content(urls_df)
        elif content_column and content_column in urls_df.columns:
            urls_df['content'] = urls_df[content_column]
        else:
            logger.warning("No content available for analysis")
            return urls_df
        
        # Step 2: Extract content features
        urls_df = self._extract_content_features(urls_df)
        
        # Step 3: Calculate content embeddings
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings for content
        content_texts = urls_df['content_clean'].fillna('').tolist()
        content_embeddings = self.sentence_model.encode(content_texts, show_progress_bar=True)
        urls_df['content_embedding'] = list(content_embeddings)
        
        return urls_df
    
    async def _fetch_page_content(self, urls_df: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
        """Fetch actual page content from URLs"""
        urls = urls_df['Address'].tolist()
        contents = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                batch_contents = await asyncio.gather(
                    *[self._fetch_single_page(session, url) for url in batch],
                    return_exceptions=True
                )
                contents.extend(batch_contents)
        
        urls_df['content'] = contents
        urls_df['content_extraction_method'] = [
            self.extraction_method if content else 'failed' 
            for content in contents
        ]
        
        return urls_df
    
    async def _fetch_single_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch content from a single URL with smart extraction"""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Use selected extraction method
                    if self.extraction_method == 'trafilatura':
                        return self._extract_with_trafilatura(html, url)
                    elif self.extraction_method == 'readability':
                        return self._extract_with_readability(html, url)
                    elif self.extraction_method == 'custom':
                        return self._extract_with_custom_rules(html)
                    else:  # 'smart' - tries multiple methods
                        return self._extract_with_smart_method(html, url)
                else:
                    return ""
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
    
    def _extract_with_smart_method(self, html: str, url: str) -> str:
        """Smart extraction that tries multiple methods"""
        # Try trafilatura first (usually best for articles/blogs)
        content = self._extract_with_trafilatura(html, url)
        
        if len(content) < 100:  # Too short, try readability
            content = self._extract_with_readability(html, url)
        
        if len(content) < 100:  # Still too short, use custom rules
            content = self._extract_with_custom_rules(html)
        
        return content
    
    def _extract_with_trafilatura(self, html: str, url: str) -> str:
        """Extract content using trafilatura (excellent for articles)"""
        try:
            # Trafilatura is excellent at extracting main content
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
                url=url
            )
            return extracted or ""
        except Exception as e:
            logger.error(f"Trafilatura extraction error: {e}")
            return ""
    
    def _extract_with_readability(self, html: str, url: str) -> str:
        """Extract content using readability (good for various content types)"""
        try:
            doc = Readability(html, url=url)
            summary = doc.summary()
            
            # Parse the summary to get text
            soup = BeautifulSoup(summary, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.error(f"Readability extraction error: {e}")
            return ""
    
    def _extract_with_custom_rules(self, html: str) -> str:
        """Extract content using custom rules based on patterns"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Step 1: Remove all excluded elements
        self._remove_excluded_elements(soup)
        
        # Step 2: Find main content area
        main_content = self._find_main_content(soup)
        
        if main_content:
            # Extract text from main content
            text = self._extract_text_from_element(main_content)
        else:
            # Fallback: Get all paragraphs not in excluded areas
            text = self._fallback_extraction(soup)
        
        return text[:10000]  # Limit to 10k chars
    
    def _remove_excluded_elements(self, soup: BeautifulSoup):
        """Remove all excluded elements from soup"""
        # Remove by tag
        for tag in self.EXCLUDE_PATTERNS['tags']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove by class
        for class_pattern in self.EXCLUDE_PATTERNS['classes']:
            for element in soup.find_all(class_=re.compile(class_pattern, re.I)):
                element.decompose()
        
        # Remove by id
        for id_pattern in self.EXCLUDE_PATTERNS['ids']:
            for element in soup.find_all(id=re.compile(id_pattern, re.I)):
                element.decompose()
        
        # Remove by attributes
        for attr, values in self.EXCLUDE_PATTERNS['attributes'].items():
            for value in values:
                for element in soup.find_all(attrs={attr: re.compile(value, re.I)}):
                    element.decompose()
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content area of the page"""
        # Try to find by semantic HTML5 tags first
        main = soup.find('main')
        if main:
            return main
        
        article = soup.find('article')
        if article:
            return article
        
        # Try to find by class patterns
        for class_pattern in self.CONTENT_PATTERNS['classes']:
            content = soup.find(class_=re.compile(class_pattern, re.I))
            if content:
                return content
        
        # Try to find by id patterns
        for id_pattern in self.CONTENT_PATTERNS['ids']:
            content = soup.find(id=re.compile(id_pattern, re.I))
            if content:
                return content
        
        # Try to find by attributes
        for attr, values in self.CONTENT_PATTERNS['attributes'].items():
            for value in values:
                content = soup.find(attrs={attr: value})
                if content:
                    return content
        
        # Try to find the element with the most text
        return self._find_element_with_most_text(soup)
    
    def _find_element_with_most_text(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the element containing the most text content"""
        max_length = 0
        best_element = None
        
        # Check common container elements
        for element in soup.find_all(['div', 'section', 'main', 'article']):
            text_length = len(element.get_text(strip=True))
            # Must have substantial text and more than just a few paragraphs
            if text_length > max_length and text_length > 200:
                paragraphs = element.find_all('p')
                if len(paragraphs) > 2:  # At least 3 paragraphs
                    max_length = text_length
                    best_element = element
        
        return best_element
    
    def _extract_text_from_element(self, element: BeautifulSoup) -> str:
        """Extract clean text from an element"""
        # Get all text-bearing elements
        text_elements = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        
        text_parts = []
        for elem in text_elements:
            text = elem.get_text(separator=' ', strip=True)
            if text and len(text) > 20:  # Skip very short text
                text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _fallback_extraction(self, soup: BeautifulSoup) -> str:
        """Fallback extraction when main content can't be identified"""
        # Get all paragraphs
        paragraphs = soup.find_all('p')
        
        # Filter paragraphs by length and content
        valid_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Must be substantial and not look like navigation/footer text
            if (len(text) > 50 and 
                not any(pattern in text.lower() for pattern in 
                       ['copyright', 'all rights reserved', 'privacy policy', 
                        'terms of service', 'cookie policy'])):
                valid_paragraphs.append(text)
        
        return ' '.join(valid_paragraphs)
    
    def update_extraction_config(self, config: Dict):
        """Update extraction configuration"""
        if 'exclude_classes' in config:
            self.EXCLUDE_PATTERNS['classes'].extend(config['exclude_classes'])
        
        if 'exclude_ids' in config:
            self.EXCLUDE_PATTERNS['ids'].extend(config['exclude_ids'])
        
        if 'content_classes' in config:
            self.CONTENT_PATTERNS['classes'].extend(config['content_classes'])
        
        if 'content_ids' in config:
            self.CONTENT_PATTERNS['ids'].extend(config['content_ids'])
    
    def _extract_content_features(self, urls_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from content"""
        features = []
        
        for _, row in urls_df.iterrows():
            content = row.get('content', '')
            
            if content:
                # Clean content
                clean_content = self._clean_text(content)
                
                # Extract features
                feature_dict = {
                    'content_clean': clean_content,
                    'word_count': len(clean_content.split()),
                    'unique_words': len(set(clean_content.lower().split())),
                    'top_keywords': self._extract_keywords(clean_content, n=20),
                    'content_topics': self._extract_topics(clean_content)
                }
            else:
                feature_dict = {
                    'content_clean': '',
                    'word_count': 0,
                    'unique_words': 0,
                    'top_keywords': [],
                    'content_topics': []
                }
            
            features.append(feature_dict)
        
        # Add features to dataframe
        for key in features[0].keys():
            urls_df[key] = [f[key] for f in features]
        
        return urls_df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_keywords(self, text: str, n: int = 20) -> List[str]:
        """Extract top keywords from text"""
        words = text.lower().split()
        # Remove stopwords
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Get word frequency
        word_freq = Counter(words)
        
        # Return top n keywords
        return [word for word, _ in word_freq.most_common(n)]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics using simple TF-IDF approach"""
        # This is simplified - could use more sophisticated topic modeling
        keywords = self._extract_keywords(text, n=10)
        
        # Group related keywords (simplified approach)
        topics = []
        for i in range(0, len(keywords), 3):
            topic = ' '.join(keywords[i:i+3])
            if topic:
                topics.append(topic)
        
        return topics[:5]  # Return top 5 topics
    
    def calculate_content_similarity_matrix(self, urls_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pairwise content similarity"""
        if 'content_embedding' not in urls_df.columns:
            logger.error("Content embeddings not found. Run analyze_content_similarity first.")
            return pd.DataFrame()
        
        embeddings = np.vstack(urls_df['content_embedding'].values)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create pairwise similarity dataframe
        similarity_pairs = []
        
        for i in range(len(urls_df)):
            for j in range(i + 1, len(urls_df)):
                similarity_pairs.append({
                    'url1': urls_df.iloc[i]['Address'],
                    'url2': urls_df.iloc[j]['Address'],
                    'content_similarity': similarity_matrix[i, j],
                    'keyword_overlap': self._calculate_keyword_overlap(
                        urls_df.iloc[i]['top_keywords'],
                        urls_df.iloc[j]['top_keywords']
                    ),
                    'topic_overlap': self._calculate_topic_overlap(
                        urls_df.iloc[i]['content_topics'],
                        urls_df.iloc[j]['content_topics']
                    )
                })
        
        return pd.DataFrame(similarity_pairs)
    
    def _calculate_keyword_overlap(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate Jaccard similarity between keyword sets"""
        if not keywords1 or not keywords2:
            return 0.0
        
        set1 = set(keywords1)
        set2 = set(keywords2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_topic_overlap(self, topics1: List[str], topics2: List[str]) -> float:
        """Calculate topic overlap"""
        if not topics1 or not topics2:
            return 0.0
        
        # Simple approach - could be enhanced with semantic similarity
        common_topics = set(topics1) & set(topics2)
        all_topics = set(topics1) | set(topics2)
        
        return len(common_topics) / len(all_topics) if all_topics else 0.0
    
    async def get_ai_content_analysis(self, url_pair: Dict) -> Dict:
        """Get AI-powered content analysis for a URL pair"""
        if not self.ai_analyzer:
            return {}
        
        prompt = f"""
        Analyze the content overlap between these two pages:
        
        URL 1: {url_pair['url1']}
        Keywords: {url_pair.get('keywords1', [])}
        Topics: {url_pair.get('topics1', [])}
        
        URL 2: {url_pair['url2']}
        Keywords: {url_pair.get('keywords2', [])}
        Topics: {url_pair.get('topics2', [])}
        
        Content Similarity: {url_pair.get('content_similarity', 0):.2%}
        Keyword Overlap: {url_pair.get('keyword_overlap', 0):.2%}
        
        Provide a JSON response with:
        {{
            "content_overlap_type": "duplicate|near_duplicate|topically_similar|complementary|distinct",
            "cannibalization_risk": "high|medium|low",
            "content_differentiation_needed": true/false,
            "differentiation_suggestions": ["suggestion1", "suggestion2"],
            "merge_recommendation": true/false
        }}
        """
        
        try:
            analysis = await self.ai_analyzer.analyze_json(prompt)
            return analysis
        except Exception as e:
            logger.error(f"AI content analysis error: {e}")
            return {}
