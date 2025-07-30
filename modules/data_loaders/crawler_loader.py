# modules/data_loaders/crawler_loader.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import re
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

class CrawlerDataLoader:
    """Load and process data from various SEO crawlers"""
    
    # Known crawler export patterns
    CRAWLER_PATTERNS = {
        'screaming_frog': {
            'url_columns': ['Address'],
            'title_columns': ['Title 1', 'Title'],
            'h1_columns': ['H1-1', 'H1 1', 'H1'],
            'meta_columns': ['Meta Description 1', 'Meta Description'],
            'content_columns': ['Content', 'Body Text'],
            'status_columns': ['Status Code', 'Response'],
            'canonical_columns': ['Canonical Link Element 1', 'Canonical']
        },
        'sitebulb': {
            'url_columns': ['URL', 'Crawled URL'],
            'title_columns': ['Title', 'Page Title'],
            'h1_columns': ['H1', 'First H1'],
            'meta_columns': ['Meta Description', 'Description'],
            'content_columns': ['Content', 'Page Content'],
            'status_columns': ['HTTP Status Code', 'Status'],
            'canonical_columns': ['Canonical URL', 'Canonical']
        },
        'deepcrawl': {
            'url_columns': ['URL', 'Found at URL'],
            'title_columns': ['Page Title', 'Title'],
            'h1_columns': ['H1', 'Primary H1'],
            'meta_columns': ['Meta Description', 'Description'],
            'content_columns': ['Body Content', 'Page Text'],
            'status_columns': ['HTTP Status', 'Status Code'],
            'canonical_columns': ['Canonical Tag', 'Canonical URL']
        },
        'generic': {
            'url_columns': ['url', 'address', 'page', 'link'],
            'title_columns': ['title', 'page title', 'seo title'],
            'h1_columns': ['h1', 'heading', 'main heading'],
            'meta_columns': ['description', 'meta description'],
            'content_columns': ['content', 'text', 'body'],
            'status_columns': ['status', 'status code', 'response'],
            'canonical_columns': ['canonical', 'canonical url']
        }
    }
    
    def __init__(self):
        self.detected_crawler = None
        self.column_mapping = {}
        
    def load_data(self, file_path_or_buffer, detect_crawler: bool = True) -> pd.DataFrame:
        """
        Load crawler data with automatic detection and validation
        
        Args:
            file_path_or_buffer: Path to file or file buffer
            detect_crawler: Whether to auto-detect the crawler type
            
        Returns:
            Standardized DataFrame with crawler data
        """
        try:
            # Read the file
            df = pd.read_csv(file_path_or_buffer)
            logger.info(f"Loaded {len(df)} rows from crawler export")
            
            # Detect crawler type
            if detect_crawler:
                self.detected_crawler = self._detect_crawler_type(df)
                logger.info(f"Detected crawler type: {self.detected_crawler}")
            
            # Map columns to standard names
            self.column_mapping = self._map_columns(df, self.detected_crawler)
            
            # Validate required columns
            self._validate_required_columns(df, self.column_mapping)
            
            # Standardize the dataframe
            df_standardized = self._standardize_dataframe(df, self.column_mapping)
            
            # Clean and process data
            df_cleaned = self._clean_crawler_data(df_standardized)
            
            # Add derived fields
            df_enhanced = self._enhance_data(df_cleaned)
            
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Error loading crawler data: {e}")
            raise
    
    def _detect_crawler_type(self, df: pd.DataFrame) -> str:
        """Detect which crawler was used based on column patterns"""
        columns_lower = [col.lower() for col in df.columns]
        
        # Check each crawler's patterns
        best_match = 'generic'
        best_score = 0
        
        for crawler, patterns in self.CRAWLER_PATTERNS.items():
            if crawler == 'generic':
                continue
                
            score = 0
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if pattern.lower() in columns_lower:
                        score += 1
                        break
            
            if score > best_score:
                best_score = score
                best_match = crawler
        
        return best_match
    
    def _map_columns(self, df: pd.DataFrame, crawler_type: str) -> Dict[str, str]:
        """Map crawler columns to standard names"""
        mapping = {}
        columns_lower = {col.lower(): col for col in df.columns}
        
        patterns = self.CRAWLER_PATTERNS.get(crawler_type, self.CRAWLER_PATTERNS['generic'])
        
        # Map each column type
        for column_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern.lower() in columns_lower:
                    actual_column = columns_lower[pattern.lower()]
                    standard_name = column_type.replace('_columns', '')
                    mapping[standard_name] = actual_column
                    break
        
        return mapping
    
    def _validate_required_columns(self, df: pd.DataFrame, mapping: Dict[str, str]):
        """Validate that required columns are present"""
        required = ['url', 'title', 'h1', 'meta']
        missing = []
        
        for req in required:
            if req not in mapping:
                missing.append(req)
        
        if missing:
            available_columns = list(df.columns)[:10]  # Show first 10 columns
            error_msg = (
                f"Missing required columns: {missing}\n"
                f"Available columns: {available_columns}...\n"
                f"Detected crawler: {self.detected_crawler}"
            )
            raise ValueError(error_msg)
    
    def _standardize_dataframe(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Create standardized dataframe with consistent column names"""
        df_standard = pd.DataFrame()
        
        # Map columns
        for standard_name, actual_column in mapping.items():
            if actual_column in df.columns:
                df_standard[standard_name] = df[actual_column]
        
        # Add any additional columns that might be useful
        extra_columns = ['Indexability', 'Word Count', 'Unique Inlinks', 'Response Time']
        for col in extra_columns:
            if col in df.columns:
                df_standard[col.lower().replace(' ', '_')] = df[col]
        
        return df_standard
    
    def _clean_crawler_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize crawler data"""
        # Create a copy
        df_clean = df.copy()
        
        # Clean URLs
        if 'url' in df_clean.columns:
            df_clean['url'] = df_clean['url'].apply(self._clean_url)
            # Remove any rows with invalid URLs
            df_clean = df_clean[df_clean['url'].notna()]
        
        # Clean text fields
        text_fields = ['title', 'h1', 'meta', 'content']
        for field in text_fields:
            if field in df_clean.columns:
                df_clean[field] = df_clean[field].fillna('').astype(str).apply(self._clean_text)
        
        # Filter by status code if available
        if 'status' in df_clean.columns:
            # Keep only 200 status codes
            df_clean['status'] = pd.to_numeric(df_clean['status'], errors='coerce')
            df_clean = df_clean[df_clean['status'] == 200]
            logger.info(f"Filtered to {len(df_clean)} URLs with 200 status")
        
        # Handle canonical URLs
        if 'canonical' in df_clean.columns:
            df_clean['canonical'] = df_clean['canonical'].fillna(df_clean['url'])
            df_clean['is_canonical'] = df_clean['url'] == df_clean['canonical']
        
        return df_clean
    
    def _enhance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields and enhancements"""
        df_enhanced = df.copy()
        
        # Add URL components
        if 'url' in df_enhanced.columns:
            df_enhanced['domain'] = df_enhanced['url'].apply(self._extract_domain)
            df_enhanced['path'] = df_enhanced['url'].apply(self._extract_path)
            df_enhanced['url_depth'] = df_enhanced['path'].apply(lambda x: x.count('/') - 1)
        
        # Add text length metrics
        if 'title' in df_enhanced.columns:
            df_enhanced['title_length'] = df_enhanced['title'].str.len()
            df_enhanced['title_words'] = df_enhanced['title'].str.split().str.len()
        
        if 'meta' in df_enhanced.columns:
            df_enhanced['meta_length'] = df_enhanced['meta'].str.len()
        
        # Add content metrics if available
        if 'content' in df_enhanced.columns and df_enhanced['content'].notna().any():
            df_enhanced['content_length'] = df_enhanced['content'].str.len()
            df_enhanced['has_content'] = df_enhanced['content_length'] > 100
        
        # Identify potential page types
        df_enhanced['page_type'] = df_enhanced.apply(self._identify_page_type, axis=1)
        
        return df_enhanced
    
    def _clean_url(self, url: str) -> Optional[str]:
        """Clean and validate URL"""
        if pd.isna(url) or not url:
            return None
        
        url = str(url).strip()
        
        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Remove trailing slash
        url = url.rstrip('/')
        
        # Validate URL
        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                return url
        except:
            pass
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text fields"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common SEO plugin artifacts
        text = re.sub(r'\s*\|\s*$', '', text)  # Remove trailing pipes
        text = re.sub(r'^\s*\|\s*', '', text)  # Remove leading pipes
        
        return text.strip()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ''
    
    def _extract_path(self, url: str) -> str:
        """Extract path from URL"""
        try:
            parsed = urlparse(url)
            return parsed.path
        except:
            return '/'
    
    def _identify_page_type(self, row: pd.Series) -> str:
        """Identify the type of page based on URL and content"""
        url = str(row.get('url', '')).lower()
        path = str(row.get('path', '')).lower()
        title = str(row.get('title', '')).lower()
        
        # Homepage
        if path == '/' or path == '':
            return 'homepage'
        
        # Category/listing pages
        if any(indicator in path for indicator in ['/category/', '/categories/', '/tag/', '/tags/']):
            return 'category'
        
        # Product pages
        if any(indicator in path for indicator in ['/product/', '/products/', '/item/', '/p/']):
            return 'product'
        
        # Blog posts
        if any(indicator in path for indicator in ['/blog/', '/post/', '/article/', '/news/']):
            return 'blog'
        
        # About/contact pages
        if any(indicator in path for indicator in ['/about', '/contact', '/team', '/careers']):
            return 'informational'
        
        # Check title patterns
        if any(indicator in title for indicator in ['category', 'archive', 'tag']):
            return 'category'
        
        if any(indicator in title for indicator in ['product', 'buy', 'shop']):
            return 'product'
        
        # Default
        return 'other'
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics about the loaded data"""
        summary = {
            'total_urls': len(df),
            'detected_crawler': self.detected_crawler,
            'columns_mapped': len(self.column_mapping),
            'page_types': df['page_type'].value_counts().to_dict() if 'page_type' in df.columns else {},
            'avg_title_length': df['title_length'].mean() if 'title_length' in df.columns else 0,
            'has_content': 'content' in df.columns and df['content'].notna().any(),
            'canonical_urls': df['is_canonical'].sum() if 'is_canonical' in df.columns else 0
        }
        
        return summary