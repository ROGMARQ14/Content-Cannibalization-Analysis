# modules/analyzers/serp_analyzer.py
import asyncio
import aiohttp
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import logging
from urllib.parse import urlparse
import streamlit as st

logger = logging.getLogger(__name__)

class SERPAnalyzer:
    """Analyze actual SERP overlap using Serper API"""
    
    def __init__(self, serper_api_key: str):
        self.api_key = serper_api_key
        self.base_url = "https://google.serper.dev/search"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_serp_overlap(self, 
                               keywords: List[str], 
                               domain: str,
                               location: str = "United States",
                               language: str = "en",
                               batch_size: int = 10) -> Dict:
        """Check which URLs from the domain appear for each keyword"""
        logger.info(f"Checking SERP overlap for {len(keywords)} keywords on domain: {domain}")
        
        # Normalize domain
        domain = self._normalize_domain(domain)
        
        # Process keywords in batches
        all_results = {}
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            
            # Create tasks for batch
            tasks = []
            for keyword in batch:
                task = self._search_keyword(keyword, location, language)
                tasks.append(task)
            
            # Execute batch
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for keyword, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error searching for '{keyword}': {result}")
                        continue
                    
                    overlap_data = self._analyze_keyword_overlap(result, domain, keyword)
                    if overlap_data:
                        all_results[keyword] = overlap_data
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
            
            # Update progress
            progress = min((i + len(batch)) / len(keywords), 1.0)
            if hasattr(st, 'progress'):
                st.progress(progress, text=f"Analyzing SERPs: {i + len(batch)}/{len(keywords)} keywords")
        
        # Calculate overall overlap statistics
        overlap_summary = self._calculate_overlap_summary(all_results)
        
        return {
            'keyword_overlaps': all_results,
            'summary': overlap_summary,
            'domain': domain
        }
    
    async def _search_keyword(self, keyword: str, location: str, language: str) -> Dict:
        """Search for a single keyword using Serper API"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': keyword,
            'location': location,
            'gl': self._get_country_code(location),
            'hl': language,
            'num': 20,  # Get top 20 results
            'serp_type': 'search'
        }
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(self.base_url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    raise Exception(f"Serper API error: {resp.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error searching for keyword '{keyword}': {e}")
            raise
    
    def _analyze_keyword_overlap(self, serp_result: Dict, domain: str, keyword: str) -> Optional[Dict]:
        """Analyze SERP results for domain overlap"""
        try:
            organic_results = serp_result.get('organic', [])
            
            # Find all URLs from the domain in the SERP
            domain_urls = []
            for i, item in enumerate(organic_results):
                url = item.get('link', '')
                if self._is_same_domain(url, domain):
                    domain_urls.append({
                        'url': url,
                        'position': i + 1,
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', '')
                    })
            
            # Only return if there's overlap (2+ URLs)
            if len(domain_urls) >= 2:
                return {
                    'overlapping_urls': domain_urls,
                    'overlap_count': len(domain_urls),
                    'overlap_score': self._calculate_overlap_score(domain_urls),
                    'positions': [u['position'] for u in domain_urls],
                    'keyword': keyword
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing overlap for keyword '{keyword}': {e}")
            return None
    
    def _calculate_overlap_score(self, urls: List[Dict]) -> float:
        """Calculate overlap severity based on positions"""
        if len(urls) < 2:
            return 0.0
        
        positions = [u['position'] for u in urls]
        
        # Factor 1: Position proximity (URLs close together are worse)
        position_range = max(positions) - min(positions)
        proximity_score = 1 / (position_range + 1)
        
        # Factor 2: Top position weight (overlap in top 10 is worse)
        avg_position = sum(positions) / len(positions)
        if avg_position <= 10:
            position_weight = 1.0
        elif avg_position <= 20:
            position_weight = 0.7
        else:
            position_weight = 0.4
        
        # Factor 3: Number of overlapping URLs
        count_factor = min(len(urls) / 3, 1.0)  # Max out at 3 URLs
        
        # Combined score
        overlap_score = (proximity_score * 0.4 + position_weight * 0.4 + count_factor * 0.2)
        
        return round(overlap_score, 3)
    
    def _calculate_overlap_summary(self, keyword_overlaps: Dict) -> Dict:
        """Calculate summary statistics for SERP overlaps"""
        if not keyword_overlaps:
            return {
                'total_keywords_analyzed': 0,
                'keywords_with_overlap': 0,
                'average_overlap_score': 0,
                'critical_overlaps': 0,
                'top_10_overlaps': 0
            }
        
        overlap_scores = [data['overlap_score'] for data in keyword_overlaps.values()]
        critical_overlaps = sum(1 for score in overlap_scores if score > 0.7)
        
        # Count overlaps in top 10
        top_10_overlaps = 0
        for data in keyword_overlaps.values():
            if any(pos <= 10 for pos in data['positions']):
                top_10_overlaps += 1
        
        return {
            'total_keywords_analyzed': len(keyword_overlaps),
            'keywords_with_overlap': len(keyword_overlaps),
            'average_overlap_score': round(sum(overlap_scores) / len(overlap_scores), 3),
            'critical_overlaps': critical_overlaps,
            'top_10_overlaps': top_10_overlaps,
            'max_overlap_score': round(max(overlap_scores), 3) if overlap_scores else 0
        }
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain for comparison"""
        # Remove protocol if present
        if domain.startswith(('http://', 'https://')):
            domain = urlparse(domain).netloc
        
        # Remove www. for consistency
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove trailing slash
        domain = domain.rstrip('/')
        
        return domain.lower()
    
    def _is_same_domain(self, url: str, domain: str) -> bool:
        """Check if URL belongs to the specified domain"""
        try:
            parsed = urlparse(url)
            url_domain = parsed.netloc.lower()
            
            # Remove www. for comparison
            if url_domain.startswith('www.'):
                url_domain = url_domain[4:]
            
            return url_domain == domain or url_domain.endswith(f'.{domain}')
            
        except Exception:
            return False
    
    def _get_country_code(self, location: str) -> str:
        """Get country code from location string"""
        # Map common locations to country codes
        country_codes = {
            'united states': 'us',
            'united kingdom': 'uk',
            'canada': 'ca',
            'australia': 'au',
            'germany': 'de',
            'france': 'fr',
            'spain': 'es',
            'italy': 'it',
            'netherlands': 'nl',
            'india': 'in',
            'japan': 'jp',
            'brazil': 'br'
        }
        
        location_lower = location.lower()
        for country, code in country_codes.items():
            if country in location_lower:
                return code
        
        # Default to US
        return 'us'
    
    async def analyze_url_pair_serp(self, 
                                   url1: str, 
                                   url2: str,
                                   keywords: List[str],
                                   location: str = "United States") -> Dict:
        """Analyze SERP competition between two specific URLs"""
        logger.info(f"Analyzing SERP competition between URLs for {len(keywords)} keywords")
        
        competition_data = {
            'url1': url1,
            'url2': url2,
            'competing_keywords': [],
            'url1_only': [],
            'url2_only': [],
            'neither': []
        }
        
        for keyword in keywords:
            try:
                result = await self._search_keyword(keyword, location, 'en')
                organic_results = result.get('organic', [])
                
                # Check which URLs appear
                url1_found = False
                url2_found = False
                url1_position = None
                url2_position = None
                
                for i, item in enumerate(organic_results):
                    link = item.get('link', '')
                    if link == url1 or link.rstrip('/') == url1.rstrip('/'):
                        url1_found = True
                        url1_position = i + 1
                    if link == url2 or link.rstrip('/') == url2.rstrip('/'):
                        url2_found = True
                        url2_position = i + 1
                
                # Categorize keyword
                if url1_found and url2_found:
                    competition_data['competing_keywords'].append({
                        'keyword': keyword,
                        'url1_position': url1_position,
                        'url2_position': url2_position,
                        'position_diff': abs(url1_position - url2_position)
                    })
                elif url1_found:
                    competition_data['url1_only'].append({
                        'keyword': keyword,
                        'position': url1_position
                    })
                elif url2_found:
                    competition_data['url2_only'].append({
                        'keyword': keyword,
                        'position': url2_position
                    })
                else:
                    competition_data['neither'].append(keyword)
                    
            except Exception as e:
                logger.error(f"Error analyzing keyword '{keyword}': {e}")
        
        # Calculate competition metrics
        total_keywords = len(keywords)
        competing_keywords = len(competition_data['competing_keywords'])
        
        competition_data['metrics'] = {
            'competition_rate': competing_keywords / total_keywords if total_keywords > 0 else 0,
            'total_competing': competing_keywords,
            'avg_position_diff': sum(k['position_diff'] for k in competition_data['competing_keywords']) / competing_keywords if competing_keywords > 0 else 0
        }
        
        return competition_data