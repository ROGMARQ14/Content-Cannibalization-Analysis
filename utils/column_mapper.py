# utils/column_mapper.py
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SmartColumnMapper:
    """Intelligently map column names across different export formats"""
    
    # Define column mappings for different tools and variations
    COLUMN_MAPPINGS = {
        'url': {
            'patterns': ['address', 'url', 'urls', 'page', 'pages', 'landing page', 
                        'landing_page', 'destination', 'location', 'link', 'permalink'],
            'regex': r'(url|address|page|link)',
            'priority': 1
        },
        'title': {
            'patterns': ['title', 'title 1', 'title_1', 'page title', 'meta title', 
                        'seo title', 'h1', 'heading 1', 'primary title'],
            'regex': r'title|heading',
            'priority': 2
        },
        'h1': {
            'patterns': ['h1', 'h1-1', 'h1_1', 'heading 1', 'heading', 'main heading',
                        'primary heading', 'first heading'],
            'regex': r'h1|heading',
            'priority': 3
        },
        'meta_description': {
            'patterns': ['meta description', 'meta_description', 'description', 
                        'meta desc', 'meta description 1', 'seo description'],
            'regex': r'(meta.*desc|description)',
            'priority': 4
        },
        'query': {
            'patterns': ['query', 'queries', 'keyword', 'keywords', 'search term',
                        'search_term', 'term', 'search query'],
            'regex': r'(quer|keyword|term)',
            'priority': 5
        },
        'clicks': {
            'patterns': ['clicks', 'click', 'click count', 'total clicks'],
            'regex': r'click',
            'priority': 6
        },
        'impressions': {
            'patterns': ['impressions', 'impression', 'views', 'impr'],
            'regex': r'(impression|impr|view)',
            'priority': 7
        },
        'position': {
            'patterns': ['position', 'pos', 'rank', 'ranking', 'avg position',
                        'average position', 'avg_position'],
            'regex': r'(position|pos|rank)',
            'priority': 8
        },
        'ctr': {
            'patterns': ['ctr', 'click through rate', 'click_through_rate', 
                        'clickthrough rate', 'ct rate'],
            'regex': r'ctr|click.*rate',
            'priority': 9
        }
    }
    
    def __init__(self):
        self.mapping_cache = {}
    
    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect and map columns to standardized names
        
        Returns:
            Dict mapping standardized names to actual column names in the dataframe
        """
        detected_mappings = {}
        used_columns = set()
        
        # Get all column names in lowercase for comparison
        df_columns_lower = {col.lower().strip(): col for col in df.columns}
        
        # Sort by priority to handle most important columns first
        sorted_mappings = sorted(self.COLUMN_MAPPINGS.items(), 
                               key=lambda x: x[1]['priority'])
        
        for standard_name, mapping_info in sorted_mappings:
            # First try exact matches
            for pattern in mapping_info['patterns']:
                if pattern in df_columns_lower and df_columns_lower[pattern] not in used_columns:
                    actual_col = df_columns_lower[pattern]
                    detected_mappings[standard_name] = actual_col
                    used_columns.add(actual_col)
                    logger.info(f"Mapped '{actual_col}' to '{standard_name}' (exact match)")
                    break
            
            # If no exact match, try regex
            if standard_name not in detected_mappings:
                regex_pattern = re.compile(mapping_info['regex'], re.IGNORECASE)
                for col_lower, col_actual in df_columns_lower.items():
                    if col_actual not in used_columns and regex_pattern.search(col_lower):
                        detected_mappings[standard_name] = col_actual
                        used_columns.add(col_actual)
                        logger.info(f"Mapped '{col_actual}' to '{standard_name}' (regex match)")
                        break
        
        # Log any unmapped columns that might be important
        unmapped = set(df.columns) - used_columns
        if unmapped:
            logger.warning(f"Unmapped columns: {unmapped}")
        
        return detected_mappings
    
    def validate_required_columns(self, 
                                df: pd.DataFrame, 
                                required_columns: List[str],
                                file_type: str = "data") -> Tuple[bool, List[str]]:
        """
        Validate that required columns are present
        
        Args:
            df: DataFrame to validate
            required_columns: List of required standardized column names
            file_type: Type of file for better error messages
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        detected = self.auto_detect_columns(df)
        missing = []
        
        for required in required_columns:
            if required not in detected:
                missing.append(required)
        
        if missing:
            logger.error(f"Missing required columns in {file_type}: {missing}")
            return False, missing
        
        return True, []
    
    def standardize_dataframe(self, df: pd.DataFrame, mappings: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create a standardized dataframe with consistent column names
        
        Args:
            df: Original dataframe
            mappings: Optional custom mappings to use
            
        Returns:
            DataFrame with standardized column names
        """
        if mappings is None:
            mappings = self.auto_detect_columns(df)
        
        # Create a copy to avoid modifying original
        standardized_df = df.copy()
        
        # Rename columns based on mappings
        rename_dict = {}
        for standard_name, actual_name in mappings.items():
            if actual_name in standardized_df.columns:
                rename_dict[actual_name] = standard_name
        
        standardized_df = standardized_df.rename(columns=rename_dict)
        
        return standardized_df
    
    def get_column_suggestions(self, df: pd.DataFrame) -> str:
        """
        Get suggestions for column mapping
        
        Returns:
            String with suggestions for user
        """
        detected = self.auto_detect_columns(df)
        
        suggestions = ["Detected column mappings:"]
        for standard, actual in detected.items():
            suggestions.append(f"  • {standard} → {actual}")
        
        # Check for potentially important unmapped columns
        unmapped = set(df.columns) - set(detected.values())
        if unmapped:
            suggestions.append("\nUnmapped columns that might be useful:")
            for col in unmapped:
                suggestions.append(f"  • {col}")
        
        return "\n".join(suggestions)

# Enhanced data loader with smart column mapping
class FlexibleDataLoader:
    """Data loader that handles various column formats"""
    
    def __init__(self):
        self.column_mapper = SmartColumnMapper()
    
    def load_internal_data(self, file) -> pd.DataFrame:
        """Load internal SEO data with flexible column detection"""
        try:
            # Read the file
            df = pd.read_csv(file)
            
            # Required columns for internal data
            required = ['url', 'title', 'h1', 'meta_description']
            
            # Validate columns
            is_valid, missing = self.column_mapper.validate_required_columns(
                df, required, "internal SEO data"
            )
            
            if not is_valid:
                # Try to provide helpful error message
                suggestions = self.column_mapper.get_column_suggestions(df)
                raise ValueError(
                    f"Missing required columns: {missing}\n\n"
                    f"Your file has columns: {list(df.columns)}\n\n"
                    f"{suggestions}\n\n"
                    f"Please ensure your file contains URL, Title, H1, and Meta Description columns."
                )
            
            # Standardize the dataframe
            standardized_df = self.column_mapper.standardize_dataframe(df)
            
            # Additional data cleaning
            standardized_df = self._clean_internal_data(standardized_df)
            
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error loading internal data: {e}")
            raise
    
    def load_gsc_data(self, file) -> pd.DataFrame:
        """Load GSC data with flexible column detection"""
        try:
            # Read the file
            df = pd.read_csv(file)
            
            # Required columns for GSC data
            required = ['url', 'query', 'clicks', 'impressions', 'position']
            
            # Check if we need alternate column names (some GSC exports use different names)
            detected = self.column_mapper.auto_detect_columns(df)
            
            # Handle special case where 'url' might be 'page' in GSC
            if 'url' not in detected and 'page' in self.column_mapper.auto_detect_columns(df):
                detected['url'] = detected.get('page')
            
            # Validate columns
            is_valid, missing = self.column_mapper.validate_required_columns(
                df, required, "GSC performance data"
            )
            
            if not is_valid:
                suggestions = self.column_mapper.get_column_suggestions(df)
                raise ValueError(
                    f"Missing required columns: {missing}\n\n"
                    f"Your file has columns: {list(df.columns)}\n\n"
                    f"{suggestions}\n\n"
                    f"Please ensure your GSC export contains Landing Page/URL, Query, "
                    f"Clicks, Impressions, and Position columns."
                )
            
            # Standardize the dataframe
            standardized_df = self.column_mapper.standardize_dataframe(df)
            
            # Additional GSC-specific cleaning
            standardized_df = self._clean_gsc_data(standardized_df)
            
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error loading GSC data: {e}")
            raise
    
    def _clean_internal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean internal SEO data"""
        # Remove any rows with missing URLs
        df = df.dropna(subset=['url'])
        
        # Clean URLs (remove trailing slashes, normalize)
        df['url'] = df['url'].str.strip().str.rstrip('/')
        
        # Fill missing values with empty strings
        text_columns = ['title', 'h1', 'meta_description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        return df
    
    def _clean_gsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean GSC performance data"""
        # Remove any rows with missing URLs or queries
        df = df.dropna(subset=['url', 'query'])
        
        # Clean URLs
        df['url'] = df['url'].str.strip().str.rstrip('/')
        
        # Ensure numeric columns are proper type
        numeric_columns = ['clicks', 'impressions', 'position']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate CTR if not present
        if 'ctr' not in df.columns and 'clicks' in df.columns and 'impressions' in df.columns:
            df['ctr'] = df.apply(
                lambda row: row['clicks'] / row['impressions'] if row['impressions'] > 0 else 0,
                axis=1
            )
        
        return df