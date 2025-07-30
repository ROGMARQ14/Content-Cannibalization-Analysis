# modules/analyzers/ai_analyzer.py
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

# Import AI providers after basic imports
try:
    from ..ai_providers.base_provider import BaseAIProvider
    from ..ai_providers.openai_provider import OpenAIProvider
    from ..ai_providers.anthropic_provider import AnthropicProvider
    from ..ai_providers.gemini_provider import GeminiProvider
except ImportError as e:
    logger.error(f"Error importing AI providers: {e}")
    raise

class AIAnalyzer:
    """AI-powered content analysis and recommendations"""
    
    def __init__(self, provider: str, model: str, api_key: str):
        """Initialize with specific AI provider"""
        self.provider_name = provider
        self.model_name = model
        self.ai_provider = self._create_provider(provider, model, api_key)
        self.batch_delay = 0.5  # Default delay between batches
        
    def _create_provider(self, provider: str, model: str, api_key: str) -> BaseAIProvider:
        """Create the appropriate AI provider instance"""
        if provider == 'openai':
            return OpenAIProvider(api_key, model)
        elif provider == 'anthropic':
            return AnthropicProvider(api_key, model)
        elif provider == 'gemini':
            return GeminiProvider(api_key, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def analyze_intent_batch(self, content_data: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
        """Analyze content intent using AI in batches"""
        logger.info(f"Analyzing intent for {len(content_data)} URLs using {self.provider_name}/{self.model_name}")
        
        # Prepare prompts
        prompts = []
        for _, row in content_data.iterrows():
            title = row.get('title', '')
            h1 = row.get('h1', '')
            meta_desc = row.get('meta_description', '')
            prompts.append(self.ai_provider.create_intent_prompt(title, h1, meta_desc))
        
        # Process in batches to avoid rate limits and resource constraints
        intents = []
        total_urls = len(prompts)
        total_batches = (total_urls + batch_size - 1) // batch_size  # Calculate total number of batches
        
        # Create a single progress bar container
        if hasattr(st, 'progress'):
            progress_bar = st.progress(0, text=f"Analyzing intent: 0/{total_batches} batches")
        
        for batch_num, i in enumerate(range(0, total_urls, batch_size), 1):
            batch = prompts[i:i + batch_size]
            
            try:
                # Add delay between batches to prevent overwhelming the system
                if i > 0:
                    await asyncio.sleep(self.batch_delay)
                
                batch_results = await self.ai_provider.batch_analyze(batch)
                intents.extend(batch_results)
                
                # Update single progress bar
                progress = batch_num / total_batches
                if hasattr(st, 'progress'):
                    progress_bar.progress(progress, text=f"Analyzing intent: {batch_num}/{total_batches} batches")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                # Add placeholder intents for failed batch
                intents.extend(['Unknown'] * len(batch))
        
        # Clean and standardize intents
        content_data['ai_intent'] = [self._standardize_intent(intent) for intent in intents]
        
        logger.info("Intent analysis completed")
        return content_data
    
    def _standardize_intent(self, intent: str) -> str:
        """Standardize intent classifications"""
        intent_lower = intent.lower().strip()
        
        if 'information' in intent_lower:
            return 'Informational'
        elif 'commercial' in intent_lower:
            return 'Commercial'
        elif 'transactional' in intent_lower:
            return 'Transactional'
        elif 'navigational' in intent_lower:
            return 'Navigational'
        else:
            # Default to informational if unclear
            return 'Informational'
    
    async def generate_recommendations(self, 
                                     cannibalization_pairs: List[Dict],
                                     performance_data: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Generate AI-powered recommendations for cannibalization issues"""
        logger.info(f"Generating recommendations for {len(cannibalization_pairs)} pairs")
        
        recommendations = []
        batch_size = 5  # Process 5 pairs at a time
        total_batches = (len(cannibalization_pairs) + batch_size - 1) // batch_size
        
        # Create a single progress bar
        if hasattr(st, 'progress'):
            progress_bar = st.progress(0, text=f"Generating recommendations: 0/{total_batches} batches")
        
        for batch_num, i in enumerate(range(0, len(cannibalization_pairs), batch_size), 1):
            batch = cannibalization_pairs[i:i + batch_size]
            
            # Prepare prompts for batch
            prompts = []
            for pair in batch:
                # Enhance pair data with performance metrics if available
                if performance_data is not None:
                    pair = self._enhance_with_performance(pair, performance_data)
                
                prompts.append(self.ai_provider.create_recommendation_prompt(pair))
            
            # Get recommendations
            try:
                batch_recommendations = await self.ai_provider.batch_analyze_json(prompts)
                
                # Add metadata to recommendations
                for j, rec in enumerate(batch_recommendations):
                    rec['pair_index'] = i + j
                    rec['generated_at'] = datetime.now().isoformat()
                    rec['ai_model'] = f"{self.provider_name}/{self.model_name}"
                    recommendations.append(rec)
                    
            except Exception as e:
                logger.error(f"Error generating recommendations for batch {batch_num}: {e}")
                # Add placeholder recommendation for failed items
                for j in range(len(batch)):
                    recommendations.append({
                        'pair_index': i + j,
                        'error': str(e),
                        'severity': 'unknown',
                        'recommended_action': 'manual_review',
                        'generated_at': datetime.now().isoformat()
                    })
            
            # Update progress
            progress = batch_num / total_batches
            if hasattr(st, 'progress'):
                progress_bar.progress(progress, text=f"Generating recommendations: {batch_num}/{total_batches} batches")
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _enhance_with_performance(self, pair: Dict, performance_data: pd.DataFrame) -> Dict:
        """Enhance cannibalization pair with performance metrics"""
        try:
            # Get performance metrics for URL 1
            url1_data = performance_data[performance_data['Landing page'] == pair['url1']]
            if not url1_data.empty:
                pair['clicks1'] = int(url1_data['Clicks'].sum())
                pair['impressions1'] = int(url1_data['Impressions'].sum())
                pair['avg_position1'] = float(url1_data['Position'].mean())
                pair['ctr1'] = float(url1_data['CTR'].mean()) if 'CTR' in url1_data else 0
            
            # Get performance metrics for URL 2
            url2_data = performance_data[performance_data['Landing page'] == pair['url2']]
            if not url2_data.empty:
                pair['clicks2'] = int(url2_data['Clicks'].sum())
                pair['impressions2'] = int(url2_data['Impressions'].sum())
                pair['avg_position2'] = float(url2_data['Position'].mean())
                pair['ctr2'] = float(url2_data['CTR'].mean()) if 'CTR' in url2_data else 0
                
        except Exception as e:
            logger.warning(f"Could not enhance pair with performance data: {e}")
        
        return pair
    
    async def analyze_content_clusters(self, 
                                     url_clusters: List[List[str]], 
                                     content_data: pd.DataFrame) -> List[Dict]:
        """Analyze content clusters for gaps and opportunities"""
        logger.info(f"Analyzing {len(url_clusters)} content clusters")
        
        cluster_analyses = []
        
        for i, cluster_urls in enumerate(url_clusters):
            # Get content data for cluster
            cluster_content = []
            for url in cluster_urls:
                row = content_data[content_data['Address'] == url]
                if not row.empty:
                    cluster_content.append({
                        'url': url,
                        'title': row.iloc[0].get('Title 1', ''),
                        'keywords': self._extract_keywords(row.iloc[0])
                    })
            
            if cluster_content:
                try:
                    prompt = self.ai_provider.create_content_analysis_prompt(cluster_content)
                    analysis = await self.ai_provider.analyze_json(prompt)
                    analysis['cluster_id'] = i
                    analysis['cluster_size'] = len(cluster_urls)
                    cluster_analyses.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Error analyzing cluster {i}: {e}")
                    cluster_analyses.append({
                        'cluster_id': i,
                        'error': str(e),
                        'cluster_size': len(cluster_urls)
                    })
            
            # Update progress
            progress = (i + 1) / len(url_clusters)
            if hasattr(st, 'progress'):
                st.progress(progress, text=f"Analyzing clusters: {i + 1}/{len(url_clusters)}")
        
        return cluster_analyses
    
    def _extract_keywords(self, row: pd.Series) -> List[str]:
        """Extract keywords from content row"""
        keywords = []
        
        # Extract from title and H1
        title = str(row.get('Title 1', ''))
        h1 = str(row.get('H1-1', ''))
        
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        
        # Combine and clean text
        text = f"{title} {h1}".lower()
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Filter out very common words (simple stopword removal)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Return top 10 unique keywords
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]
    
    async def generate_executive_summary(self, 
                                       analysis_results: Dict,
                                       recommendations: List[Dict]) -> str:
        """Generate an executive summary of the analysis"""
        prompt = f"""
        Create an executive summary for this content cannibalization analysis:
        
        Analysis Overview:
        - Total URLs analyzed: {analysis_results.get('total_urls', 0)}
        - Cannibalization pairs found: {analysis_results.get('total_pairs', 0)}
        - High-risk pairs: {analysis_results.get('high_risk_count', 0)}
        - Medium-risk pairs: {analysis_results.get('medium_risk_count', 0)}
        - Low-risk pairs: {analysis_results.get('low_risk_count', 0)}
        
        Top Issues:
        {self._format_top_issues(recommendations[:5])}
        
        Create a concise executive summary (200-300 words) that:
        1. Highlights the most critical findings
        2. Quantifies the potential SEO impact
        3. Provides 3-5 key action items
        4. Estimates the effort required
        
        Format in clear, professional language suitable for stakeholders.
        """
        
        try:
            summary = await self.ai_provider.analyze(prompt)
            return summary
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return self._generate_fallback_summary(analysis_results, recommendations)
    
    def _format_top_issues(self, recommendations: List[Dict]) -> str:
        """Format top issues for the summary prompt"""
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(f"{i}. {rec.get('primary_issue', 'Issue')} - "
                           f"Severity: {rec.get('severity', 'unknown')}, "
                           f"Action: {rec.get('recommended_action', 'review')}")
        return '\n'.join(formatted)
    
    def _generate_fallback_summary(self, analysis_results: Dict, recommendations: List[Dict]) -> str:
        """Generate a basic summary if AI fails"""
        high_risk = analysis_results.get('high_risk_count', 0)
        total_pairs = analysis_results.get('total_pairs', 0)
        
        return f"""
        ## Executive Summary
        
        The content cannibalization analysis identified {total_pairs} instances of potential 
        content overlap, with {high_risk} requiring immediate attention.
        
        ### Key Findings:
        - {high_risk} high-risk cannibalization issues that could be impacting rankings
        - Most common issue: Multiple pages targeting identical keywords
        - Estimated traffic impact: Significant ranking dilution detected
        
        ### Recommended Actions:
        1. Consolidate {high_risk} high-priority page pairs
        2. Implement content differentiation strategies
        3. Update internal linking structure
        4. Monitor performance post-implementation
        
        ### Next Steps:
        Review the detailed recommendations for each cannibalization pair and prioritize 
        based on traffic potential and current performance metrics.
        """
