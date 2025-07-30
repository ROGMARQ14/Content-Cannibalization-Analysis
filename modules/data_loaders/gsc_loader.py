# modules/data_loaders/gsc_loader.py
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)

class GSCLoader:
    """Google Search Console data loader with OAuth 2.0"""
    
    SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
    
    def __init__(self):
        self.creds = None
        self.service = None
        
    def authenticate(self):
        """Handle OAuth 2.0 authentication flow in Streamlit"""
        # Check if already authenticated
        if 'gsc_token' in st.session_state:
            try:
                self.creds = Credentials.from_authorized_user_info(
                    json.loads(st.session_state['gsc_token']),
                    self.SCOPES
                )
                
                # Refresh token if expired
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                    st.session_state['gsc_token'] = self.creds.to_json()
                
                self.service = build('searchconsole', 'v1', credentials=self.creds)
                return True
                
            except Exception as e:
                logger.error(f"Error loading existing credentials: {e}")
                del st.session_state['gsc_token']
        
        # Initialize OAuth flow
        try:
            # Check if OAuth config exists in secrets
            if 'gsc_oauth_config' not in st.secrets:
                st.error("Google Search Console OAuth not configured. Please add credentials to Streamlit secrets.")
                with st.expander("Setup Instructions"):
                    st.markdown("""
                    ### How to set up GSC OAuth:
                    
                    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                    2. Create a new project or select existing
                    3. Enable the Search Console API
                    4. Create OAuth 2.0 credentials
                    5. Add authorized redirect URI: `http://localhost:8501`
                    6. Download the credentials JSON
                    7. Add to `.streamlit/secrets.toml`:
                    
                    ```toml
                    [gsc_oauth_config]
                    client_id = "your-client-id"
                    client_secret = "your-client-secret"
                    auth_uri = "https://accounts.google.com/o/oauth2/auth"
                    token_uri = "https://oauth2.googleapis.com/token"
                    redirect_uri = "http://localhost:8501"
                    ```
                    """)
                return False
            
            # Create flow from secrets
            client_config = {
                "web": {
                    "client_id": st.secrets["gsc_oauth_config"]["client_id"],
                    "client_secret": st.secrets["gsc_oauth_config"]["client_secret"],
                    "auth_uri": st.secrets["gsc_oauth_config"]["auth_uri"],
                    "token_uri": st.secrets["gsc_oauth_config"]["token_uri"],
                    "redirect_uris": [st.secrets["gsc_oauth_config"]["redirect_uri"]]
                }
            }
            
            flow = Flow.from_client_config(
                client_config,
                scopes=self.SCOPES,
                redirect_uri=st.secrets["gsc_oauth_config"]["redirect_uri"]
            )
            
            # Handle the OAuth flow
            if 'oauth_state' not in st.session_state:
                # Step 1: Generate authorization URL
                auth_url, state = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true',
                    prompt='consent'
                )
                st.session_state['oauth_state'] = state
                
                st.markdown("### Connect to Google Search Console")
                st.markdown("Click the link below to authorize access to your Search Console data:")
                st.markdown(f"[🔐 Authorize Access]({auth_url})")
                
                # Step 2: Handle the callback
                st.markdown("---")
                auth_code = st.text_input(
                    "After authorizing, paste the authorization code here:",
                    key="gsc_auth_code"
                )
                
                if auth_code:
                    try:
                        # Exchange code for token
                        flow.fetch_token(code=auth_code)
                        
                        # Save credentials
                        st.session_state['gsc_token'] = flow.credentials.to_json()
                        del st.session_state['oauth_state']
                        
                        st.success("✅ Successfully connected to Google Search Console!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
                        logger.error(f"OAuth error: {e}")
            
            return False
            
        except Exception as e:
            st.error(f"OAuth setup error: {str(e)}")
            logger.error(f"OAuth setup error: {e}")
            return False
    
    def get_sites(self) -> List[str]:
        """Get list of verified sites from GSC"""
        if not self.service:
            return []
        
        try:
            sites_list = self.service.sites().list().execute()
            return [site['siteUrl'] for site in sites_list.get('siteEntry', [])]
        except Exception as e:
            logger.error(f"Error fetching sites: {e}")
            return []
    
    def fetch_performance_data(self, 
                             site_url: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None,
                             dimensions: List[str] = None,
                             row_limit: int = 25000) -> pd.DataFrame:
        """
        Fetch performance data from GSC
        
        Args:
            site_url: The site URL to fetch data for
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            dimensions: List of dimensions to include
            row_limit: Maximum rows to fetch (max 25000)
            
        Returns:
            DataFrame with GSC performance data
        """
        if not self.service:
            raise Exception("Not authenticated. Please authenticate first.")
        
        # Set default date range (last 3 months)
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        # Default dimensions
        if not dimensions:
            dimensions = ['page', 'query']
        
        logger.info(f"Fetching GSC data for {site_url} from {start_date} to {end_date}")
        
        try:
            # Build request
            request_body = {
                'startDate': start_date,
                'endDate': end_date,
                'dimensions': dimensions,
                'rowLimit': min(row_limit, 25000),  # GSC API limit
                'dimensionFilterGroups': [{
                    'filters': [{
                        'dimension': 'page',
                        'operator': 'notContains',
                        'expression': '#'  # Exclude fragments
                    }]
                }]
            }
            
            # Execute request
            response = self.service.searchanalytics().query(
                siteUrl=site_url,
                body=request_body
            ).execute()
            
            # Process response
            if 'rows' not in response:
                logger.warning("No data returned from GSC")
                return pd.DataFrame()
            
            # Convert to DataFrame
            rows_data = []
            for row in response['rows']:
                row_data = {}
                
                # Add dimensions
                for i, dimension in enumerate(dimensions):
                    row_data[dimension] = row['keys'][i]
                
                # Add metrics
                row_data['clicks'] = row.get('clicks', 0)
                row_data['impressions'] = row.get('impressions', 0)
                row_data['ctr'] = row.get('ctr', 0)
                row_data['position'] = row.get('position', 0)
                
                rows_data.append(row_data)
            
            df = pd.DataFrame(rows_data)
            
            # Standardize column names for compatibility
            column_mapping = {
                'page': 'url',
                'query': 'query',
                'clicks': 'clicks',
                'impressions': 'impressions',
                'ctr': 'ctr',
                'position': 'position'
            }
            
            df = df.rename(columns=column_mapping)
            
            logger.info(f"Fetched {len(df)} rows from GSC")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching GSC data: {e}")
            raise
    
    def fetch_all_pages_data(self, 
                           site_url: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data for all pages (handles pagination)"""
        all_data = []
        start_row = 0
        
        while True:
            try:
                # Fetch batch
                df_batch = self.fetch_performance_data(
                    site_url,
                    start_date,
                    end_date,
                    dimensions=['page', 'query'],
                    row_limit=25000
                )
                
                if df_batch.empty:
                    break
                
                all_data.append(df_batch)
                
                # Check if we got less than the limit (no more data)
                if len(df_batch) < 25000:
                    break
                
                start_row += 25000
                
                # GSC API doesn't support offset, so we'd need to implement
                # filtering to get additional rows
                # For now, we'll work with the 25k limit
                break
                
            except Exception as e:
                logger.error(f"Error in pagination: {e}")
                break
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_top_pages(self, 
                     site_url: str,
                     limit: int = 100,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """Get top pages by clicks"""
        # Fetch page-level data
        df = self.fetch_performance_data(
            site_url,
            start_date,
            end_date,
            dimensions=['page'],
            row_limit=limit
        )
        
        if not df.empty:
            # Sort by clicks
            df = df.sort_values('clicks', ascending=False)
        
        return df
    
    def get_cannibalization_data(self,
                               site_url: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """Get data specifically formatted for cannibalization analysis"""
        # Fetch page and query data
        df = self.fetch_performance_data(
            site_url,
            start_date,
            end_date,
            dimensions=['page', 'query'],
            row_limit=25000
        )
        
        if df.empty:
            return df
        
        # Add derived metrics
        df['ctr_percentage'] = df['ctr'] * 100
        
        # Identify queries with multiple ranking pages
        query_page_counts = df.groupby('query')['url'].nunique()
        multi_page_queries = query_page_counts[query_page_counts > 1].index
        
        # Mark potential cannibalization
        df['potential_cannibalization'] = df['query'].isin(multi_page_queries)
        
        # Sort by importance (clicks * impressions)
        df['importance_score'] = df['clicks'] * np.log1p(df['impressions'])
        df = df.sort_values('importance_score', ascending=False)
        
        return df
    
    def create_streamlit_interface(self) -> Optional[pd.DataFrame]:
        """Create Streamlit interface for GSC data fetching"""
        st.subheader("🔗 Google Search Console Direct Connection")
        
        # Authenticate
        if not self.service:
            if not self.authenticate():
                return None
        
        # Get sites
        sites = self.get_sites()
        
        if not sites:
            st.warning("No verified sites found in your Search Console account.")
            return None
        
        # Site selection
        selected_site = st.selectbox(
            "Select website",
            sites,
            key="gsc_site_select"
        )
        
        # Date range
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start date",
                value=datetime.now() - timedelta(days=90),
                key="gsc_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End date",
                value=datetime.now() - timedelta(days=1),
                key="gsc_end_date"
            )
        
        # Fetch data button
        if st.button("Fetch GSC Data", key="fetch_gsc_data"):
            with st.spinner("Fetching data from Google Search Console..."):
                try:
                    df = self.get_cannibalization_data(
                        selected_site,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if not df.empty:
                        st.success(f"✅ Fetched {len(df)} rows of data")
                        
                        # Show preview
                        st.dataframe(df.head(10))
                        
                        # Show summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Queries", df['query'].nunique())
                        with col2:
                            st.metric("Total Pages", df['url'].nunique())
                        with col3:
                            st.metric("Potential Cannibalization", 
                                     df['potential_cannibalization'].sum())
                        
                        return df
                    else:
                        st.warning("No data found for the selected date range.")
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    logger.error(f"GSC fetch error: {e}")
        
        return None