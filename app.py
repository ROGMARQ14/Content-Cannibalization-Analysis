# app.py - Enhanced Content Cannibalization Analyzer
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utilities
from utils.api_manager import APIManager
from utils.column_mapper import SmartColumnMapper, FlexibleDataLoader
from utils.export_handler import ExportHandler
from utils.url_normalizer import URLNormalizer

# Import analyzers
from modules.analyzers.ai_analyzer import AIAnalyzer
from modules.analyzers.serp_analyzer import SERPAnalyzer
from modules.analyzers.ml_scoring import MLScoringEngine
from modules.analyzers.similarity_analyzer import SimilarityAnalyzer
from modules.analyzers.content_analyzer import ContentAnalyzer
from modules.analyzers.keyword_analyzer import KeywordAnalyzer

# Import detectors
from modules.detectors.competition_detector import CompetitionDetector
from modules.detectors.similarity_detector import SimilarityDetector
from modules.detectors.combined_detector import CombinedDetector

# Import data loaders
from modules.data_loaders.gsc_loader import GSCLoader

# Import reporting
from modules.reporting.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Enhanced Content Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .critical-risk {
        color: #d32f2f;
        font-weight: bold;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffa500;
        font-weight: bold;
    }
    .low-risk {
        color: #00cc00;
        font-weight: bold;
    }
    .detection-info {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'serp_data' not in st.session_state:
        st.session_state.serp_data = None
    if 'internal_data' not in st.session_state:
        st.session_state.internal_data = None
    if 'gsc_data' not in st.session_state:
        st.session_state.gsc_data = None
    if 'detection_method' not in st.session_state:
        st.session_state.detection_method = 'combined'
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.title("🔍 Enhanced Content Cannibalization Analyzer")
    st.markdown("**Competition-First** detection for identifying and resolving content cannibalization at scale")
    
    # Quick stats if analysis complete
    if st.session_state.analysis_complete:
        show_quick_stats()
    
    # API validation
    api_valid, api_issues = APIManager.validate_apis()
    
    if api_issues:
        with st.expander("⚠️ Configuration Status", expanded=not api_valid):
            for issue in api_issues:
                if "No AI provider" in issue:
                    st.error(issue)
                else:
                    st.warning(issue)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Detection Method Selection
        st.subheader("🎯 Detection Strategy")
        detection_method = st.selectbox(
            "Detection Method",
            ["Combined (Recommended)", "Competition-First", "Similarity-First", "Custom"],
            help="""
            **Combined**: Uses all detection methods for comprehensive analysis
            **Competition-First**: Starts with keyword/SERP competition
            **Similarity-First**: Traditional content similarity approach
            **Custom**: Configure your own detection pipeline
            """,
            key="detection_method_select"
        )
        st.session_state.detection_method = detection_method.split()[0].lower()
        
        # Debug Mode
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode",
            value=False,
            help="Show detailed analysis information"
        )
        
        st.divider()
        
        # AI Provider Selection
        available_providers = APIManager.get_available_providers()
        
        if available_providers:
            selected_provider = st.selectbox(
                "AI Provider",
                options=list(available_providers.keys()),
                format_func=lambda x: available_providers[x]['display_name']
            )
            
            model_options = available_providers[selected_provider]['models']
            selected_model = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                format_func=lambda x: f"{model_options[x]['name']} - {model_options[x]['best_for']}"
            )
            
            st.session_state.provider = selected_provider
            st.session_state.model = selected_model
        
        st.divider()
        
        # Analysis Options
        st.subheader("🔍 Analysis Options")
        
        # Keyword Analysis Settings
        with st.expander("Keyword Competition", expanded=True):
            min_shared_queries = st.slider(
                "Min Shared Keywords",
                1, 10, 2,
                help="Minimum keywords both URLs must rank for"
            )
            
            min_keyword_impressions = st.slider(
                "Min Keyword Impressions",
                0, 100, 5,
                help="Minimum impressions for a keyword to be considered"
            )
            
            keyword_position_threshold = st.slider(
                "Max Position to Consider",
                10, 50, 30,
                help="Ignore keywords ranking below this position"
            )
        
        # Similarity Settings
        with st.expander("Content Similarity"):
            min_similarity = st.slider(
                "Min Similarity Score",
                0.0, 1.0, 0.20, 0.05,
                help="Lower = more sensitive detection"
            )
            
            similarity_weights = {
                'title': st.slider("Title Weight", 0.0, 1.0, 0.25, 0.05),
                'h1': st.slider("H1 Weight", 0.0, 1.0, 0.15, 0.05),
                'content': st.slider("Content Weight", 0.0, 1.0, 0.35, 0.05),
                'keywords': st.slider("Keyword Overlap Weight", 0.0, 1.0, 0.25, 0.05)
            }
        
        # SERP Analysis
        use_serp = st.checkbox(
            "Enable SERP Analysis",
            value=APIManager.has_serper_api(),
            disabled=not APIManager.has_serper_api(),
            help="Real-time search result analysis"
        )
        
        if use_serp:
            serp_location = st.selectbox(
                "SERP Location",
                ["United States", "United Kingdom", "Canada", "Australia", 
                 "Germany", "France", "Spain", "India"]
            )
        
        st.divider()
        
        # Risk Thresholds
        st.subheader("⚠️ Risk Thresholds")
        high_threshold = st.slider("Critical Risk", 0.0, 1.0, 0.7, 0.05)
        medium_threshold = st.slider("High Risk", 0.0, 1.0, 0.5, 0.05)
        low_threshold = st.slider("Medium Risk", 0.0, 1.0, 0.3, 0.05)
        
        st.divider()
        
        # Performance Settings
        st.subheader("⚡ Performance")
        
        filter_by_performance = st.checkbox(
            "Filter Low-Traffic Pages",
            value=False,
            help="⚠️ May exclude cannibalized pages with suppressed metrics"
        )
        
        if filter_by_performance:
            min_clicks = st.number_input("Min Clicks", 0, 100, 5)
            min_impressions = st.number_input("Min Impressions", 0, 1000, 50)
    
    # Main content tabs
    tabs = st.tabs(["📊 Analysis", "🎯 Detection Results", "🤖 AI Insights", "📈 Reports", "❓ Help"])
    
    with tabs[0]:
        run_analysis_tab(
            selected_provider if 'selected_provider' in locals() else None,
            selected_model if 'selected_model' in locals() else None,
            {
                'detection_method': st.session_state.detection_method,
                'keyword_settings': {
                    'min_shared_queries': min_shared_queries if 'min_shared_queries' in locals() else 2,
                    'min_impressions': min_keyword_impressions if 'min_keyword_impressions' in locals() else 5,
                    'position_threshold': keyword_position_threshold if 'keyword_position_threshold' in locals() else 30
                },
                'similarity_settings': {
                    'min_similarity': min_similarity if 'min_similarity' in locals() else 0.20,
                    'weights': similarity_weights if 'similarity_weights' in locals() else {}
                },
                'use_serp': use_serp if 'use_serp' in locals() else False,
                'serp_location': serp_location if 'serp_location' in locals() else "United States",
                'thresholds': {
                    'critical': high_threshold if 'high_threshold' in locals() else 0.7,
                    'high': medium_threshold if 'medium_threshold' in locals() else 0.5,
                    'medium': low_threshold if 'low_threshold' in locals() else 0.3
                },
                'performance_filter': {
                    'enabled': filter_by_performance if 'filter_by_performance' in locals() else False,
                    'min_clicks': min_clicks if 'min_clicks' in locals() else 5,
                    'min_impressions': min_impressions if 'min_impressions' in locals() else 50
                }
            }
        )
    
    with tabs[1]:
        show_detection_results()
    
    with tabs[2]:
        show_ai_insights_tab()
    
    with tabs[3]:
        generate_reports_tab()
    
    with tabs[4]:
        show_help_tab()

def show_quick_stats():
    """Show quick statistics banner"""
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        cols = st.columns(5)
        with cols[0]:
            st.metric("URLs Analyzed", results.get('total_urls', 0))
        with cols[1]:
            st.metric("Issues Found", results.get('total_issues', 0))
        with cols[2]:
            st.metric("Critical Risk", results.get('critical_risk_count', 0))
        with cols[3]:
            st.metric("Est. Traffic Loss", results.get('traffic_impact', 'N/A'))
        with cols[4]:
            st.metric("Detection Method", st.session_state.detection_method.title())

def run_analysis_tab(provider, model, config):
    """Main analysis tab"""
    st.header("📊 Content Analysis")
    
    # File upload section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📄 SEO Crawler Data")
        internal_file = st.file_uploader(
            "Upload crawler export (CSV)",
            type=['csv'],
            help="Screaming Frog, Sitebulb, or similar",
            key="internal_file_upload"
        )
        
        if internal_file:
            try:
                preview_df = pd.read_csv(io.StringIO(internal_file.getvalue().decode('utf-8')), nrows=5)
                with st.expander("Preview & Column Detection"):
                    st.dataframe(preview_df)
                    mapper = SmartColumnMapper()
                    st.code(mapper.get_column_suggestions(preview_df))
            except Exception as e:
                st.error(f"Error previewing file: {e}")
    
    with col2:
        st.subheader("📊 GSC Performance Data")
        gsc_file = st.file_uploader(
            "Upload GSC export (CSV)",
            type=['csv'],
            help="Query & URL performance data",
            key="gsc_file_upload"
        )
    
    with col3:
        st.subheader("🧮 Embeddings (Optional)")
        embeddings_file = st.file_uploader(
            "Upload embeddings (CSV)",
            type=['csv'],
            help="Screaming Frog embeddings export",
            key="embeddings_file_upload"
        )
    
    # Analysis info box
    if internal_file and gsc_file:
        st.markdown("""
        <div class="detection-info">
        <h4>🎯 Analysis Pipeline</h4>
        <p><strong>Step 1:</strong> Keyword Competition Analysis (GSC data)</p>
        <p><strong>Step 2:</strong> Content Similarity Detection (Crawler data)</p>
        <p><strong>Step 3:</strong> Combined Risk Scoring (ML-based)</p>
        <p><strong>Step 4:</strong> AI Recommendations (Provider: {provider})</p>
        </div>
        """.format(provider=provider or "Not selected"), unsafe_allow_html=True)
    
    # Run analysis button
    if st.button("🚀 Run Cannibalization Analysis", type="primary", use_container_width=True):
        if internal_file and gsc_file:
            asyncio.run(run_comprehensive_analysis(
                internal_file, gsc_file, embeddings_file,
                provider, model, config
            ))
        else:
            st.error("Please upload both SEO crawler data and GSC performance data.")

async def run_comprehensive_analysis(internal_file, gsc_file, embeddings_file, 
                                   provider, model, config):
    """Run the comprehensive cannibalization analysis"""
    try:
        progress_bar = st.progress(0, text="Initializing analysis...")
        
        # Load data
        progress_bar.progress(10, text="Loading data files...")
        
        loader = FlexibleDataLoader()
        internal_data = loader.load_internal_data(internal_file)
        gsc_data = loader.load_gsc_data(gsc_file)
        
        # Store in session state for debugging
        st.session_state.internal_data = internal_data
        st.session_state.gsc_data = gsc_data
        
        # Normalize URLs for matching
        progress_bar.progress(20, text="Normalizing URLs...")
        internal_data['url_normalized'] = internal_data['url'].apply(URLNormalizer.normalize_for_matching)
        gsc_data['url_normalized'] = gsc_data['url'].apply(URLNormalizer.normalize_for_matching)
        
        # Performance filtering (if enabled)
        if config['performance_filter']['enabled']:
            progress_bar.progress(25, text="Applying performance filters...")
            
            performance_urls = gsc_data.groupby('url').agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            
            qualified_urls = performance_urls[
                (performance_urls['clicks'] >= config['performance_filter']['min_clicks']) |
                (performance_urls['impressions'] >= config['performance_filter']['min_impressions'])
            ]['url'].tolist()
            
            # Show filtering stats
            original_count = len(internal_data)
            internal_data = internal_data[internal_data['url'].isin(qualified_urls)]
            if st.session_state.debug_mode:
                st.info(f"Performance filter: {original_count} → {len(internal_data)} URLs")
        
        # Initialize analyzers
        progress_bar.progress(30, text="Initializing analyzers...")
        
        # AI Analyzer
        ai_analyzer = None
        if provider and model:
            api_key = APIManager.get_api_key(provider)
            ai_analyzer = AIAnalyzer(provider, model, api_key)
        
        # Initialize detectors based on method
        progress_bar.progress(40, text="Running detection pipeline...")
        
        if config['detection_method'] == 'combined':
            detector = CombinedDetector(
                internal_data=internal_data,
                gsc_data=gsc_data,
                embeddings_data=pd.read_csv(embeddings_file) if embeddings_file else None,
                ai_analyzer=ai_analyzer,
                config=config
            )
        elif config['detection_method'] == 'competition':
            detector = CompetitionDetector(gsc_data)
        else:  # similarity
            detector = SimilarityDetector(
                internal_data=internal_data,
                embeddings_data=pd.read_csv(embeddings_file) if embeddings_file else None,
                config=config['similarity_settings']
            )
        
        # Run detection
        progress_bar.progress(60, text="Detecting cannibalization issues...")
        
        if config['detection_method'] == 'combined':
            results = await detector.detect_all()
        elif config['detection_method'] == 'competition':
            results = await detector.detect_all_competition(
                min_shared_queries=config['keyword_settings']['min_shared_queries'],
                min_impressions=config['keyword_settings']['min_impressions'],
                use_serp=config['use_serp']
            )
        else:
            results = detector.detect_all_similarity()
        
        # Generate AI recommendations
        if ai_analyzer and not results.empty:
            progress_bar.progress(80, text="Generating AI recommendations...")
            
            # Convert results to format for AI
            cannibalization_pairs = results.head(50).to_dict('records')  # Top 50 for AI
            recommendations = await ai_analyzer.generate_recommendations(
                cannibalization_pairs, gsc_data
            )
            st.session_state.recommendations = recommendations
        
        # Calculate final metrics
        progress_bar.progress(90, text="Calculating impact metrics...")
        
        analysis_results = {
            'total_urls': len(internal_data),
            'total_issues': len(results),
            'critical_risk_count': len(results[results['risk_category'] == 'Critical']) if 'risk_category' in results else 0,
            'high_risk_count': len(results[results['risk_category'] == 'High']) if 'risk_category' in results else 0,
            'medium_risk_count': len(results[results['risk_category'] == 'Medium']) if 'risk_category' in results else 0,
            'low_risk_count': len(results[results['risk_category'] == 'Low']) if 'risk_category' in results else 0,
            'detection_method': config['detection_method'],
            'cannibalization_data': results,
            'traffic_impact': calculate_traffic_impact(results, gsc_data)
        }
        
        st.session_state.analysis_results = analysis_results
        st.session_state.analysis_complete = True
        
        progress_bar.progress(100, text="Analysis complete!")
        st.success(f"✅ Found {len(results)} potential cannibalization issues!")
        
        # Show immediate insights
        if len(results) > 0:
            st.balloons()
            
            # Quick insights
            col1, col2, col3 = st.columns(3)
            with col1:
                top_issue = results.iloc[0]
                st.info(f"**Top Issue:** Competition score {top_issue.get('competition_score', 0):.2%}")
            with col2:
                keyword_issues = results[results['detection_source'] == 'keyword_competition'] if 'detection_source' in results else results
                st.info(f"**Keyword Issues:** {len(keyword_issues)}")
            with col3:
                avg_queries = results['shared_queries_count'].mean() if 'shared_queries_count' in results else 0
                st.info(f"**Avg Shared Keywords:** {avg_queries:.1f}")
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        st.error(f"❌ Analysis failed: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)

def calculate_traffic_impact(results, gsc_data):
    """Calculate estimated traffic impact"""
    if results.empty:
        return "No impact"
    
    # Sum potential traffic loss
    if 'traffic_opportunity' in results.columns:
        total_opportunity = results['traffic_opportunity'].sum()
        current_clicks = gsc_data['clicks'].sum()
        
        if current_clicks > 0:
            impact_pct = (total_opportunity / current_clicks) * 100
            return f"+{impact_pct:.0f}% potential"
        else:
            return f"+{total_opportunity} clicks"
    else:
        return "Calculate after analysis"

def show_detection_results():
    """Show detailed detection results"""
    if not st.session_state.analysis_complete:
        st.info("👆 Run analysis first to see detection results")
        return
    
    results = st.session_state.analysis_results
    df = results.get('cannibalization_data', pd.DataFrame())
    
    if df.empty:
        st.warning("No cannibalization issues detected. Try adjusting detection settings.")
        return
    
    st.header("🎯 Detection Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Issues", len(df))
    with col2:
        if 'detection_source' in df.columns:
            keyword_issues = len(df[df['detection_source'] == 'keyword_competition'])
            st.metric("Keyword Competition", keyword_issues)
    with col3:
        if 'shared_queries_count' in df.columns:
            avg_queries = df['shared_queries_count'].mean()
            st.metric("Avg Shared Keywords", f"{avg_queries:.1f}")
    with col4:
        if 'competition_score' in df.columns:
            avg_score = df['competition_score'].mean()
            st.metric("Avg Competition Score", f"{avg_score:.2%}")
    
    # Detection method breakdown
    if 'detection_source' in df.columns:
        st.subheader("Detection Method Breakdown")
        
        method_counts = df['detection_source'].value_counts()
        fig = px.pie(values=method_counts.values, names=method_counts.index,
                    title="Issues by Detection Method")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk distribution
    if 'risk_category' in df.columns:
        st.subheader("Risk Distribution")
        
        risk_counts = df['risk_category'].value_counts()
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                    color=risk_counts.index,
                    color_discrete_map={
                        'Critical': '#d32f2f',
                        'High': '#ff4b4b',
                        'Medium': '#ffa500',
                        'Low': '#00cc00'
                    },
                    title="Issues by Risk Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top issues table
    st.subheader("🔥 Top Cannibalization Issues")
    
    # Prepare display columns
    display_cols = ['url1', 'url2']
    
    if 'competition_score' in df.columns:
        display_cols.append('competition_score')
    if 'shared_queries_count' in df.columns:
        display_cols.append('shared_queries_count')
    if 'traffic_opportunity' in df.columns:
        display_cols.append('traffic_opportunity')
    if 'risk_category' in df.columns:
        display_cols.append('risk_category')
    
    # Sort by score
    sort_col = 'competition_score' if 'competition_score' in df.columns else 'shared_queries_count'
    top_issues = df.nlargest(20, sort_col)[display_cols]
    
    # Format for display
    if 'competition_score' in top_issues.columns:
        top_issues['competition_score'] = top_issues['competition_score'].apply(lambda x: f"{x:.1%}")
    
    # Show with custom styling
    st.dataframe(
        top_issues,
        column_config={
            "url1": st.column_config.TextColumn("URL 1", width="medium"),
            "url2": st.column_config.TextColumn("URL 2", width="medium"),
            "competition_score": st.column_config.TextColumn("Competition", width="small"),
            "shared_queries_count": st.column_config.NumberColumn("Shared Keywords", width="small"),
            "traffic_opportunity": st.column_config.NumberColumn("Traffic Opp.", width="small"),
            "risk_category": st.column_config.TextColumn("Risk", width="small")
        },
        hide_index=True
    )
    
    # Detailed view
    st.subheader("📋 Detailed Analysis")
    
    selected_pair = st.selectbox(
        "Select URL pair for detailed analysis",
        options=range(len(df)),
        format_func=lambda x: f"{df.iloc[x]['url1'][:50]}... vs {df.iloc[x]['url2'][:50]}..."
    )
    
    if selected_pair is not None:
        pair = df.iloc[selected_pair]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**URL 1:**")
            st.code(pair['url1'])
            
            if 'clicks_url1' in pair:
                st.metric("Total Clicks", pair['clicks_url1'])
            if 'impressions_url1' in pair:
                st.metric("Total Impressions", pair['impressions_url1'])
        
        with col2:
            st.markdown("**URL 2:**")
            st.code(pair['url2'])
            
            if 'clicks_url2' in pair:
                st.metric("Total Clicks", pair['clicks_url2'])
            if 'impressions_url2' in pair:
                st.metric("Total Impressions", pair['impressions_url2'])
        
        # Shared queries
        if 'shared_queries' in pair and isinstance(pair['shared_queries'], list):
            st.markdown("**Shared Keywords:**")
            
            # Show top queries
            queries_to_show = pair['shared_queries'][:20]
            query_chips = " ".join([f"`{q}`" for q in queries_to_show])
            st.markdown(query_chips)
            
            if len(pair['shared_queries']) > 20:
                st.caption(f"...and {len(pair['shared_queries']) - 20} more keywords")
        
        # Competition details
        if 'competition_type' in df.columns:
            st.markdown(f"**Competition Analysis:**")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Competition Score", f"{pair.get('competition_score', 0):.1%}")
            with metrics_col2:
                st.metric("Avg Position Spread", f"{pair.get('avg_position_spread', 0):.1f}")
            with metrics_col3:
                st.metric("Traffic Opportunity", f"+{pair.get('traffic_opportunity', 0)} clicks")

def show_ai_insights_tab():
    """Display AI-generated insights and recommendations"""
    if not st.session_state.analysis_complete:
        st.info("👆 Run analysis first to see AI insights")
        return
    
    st.header("🤖 AI-Powered Insights")
    
    recommendations = st.session_state.recommendations
    
    if not recommendations:
        st.warning("No AI recommendations available. Make sure an AI provider is configured.")
        return
    
    # Group recommendations by severity
    critical_recs = [r for r in recommendations if r.get('severity') == 'critical']
    high_recs = [r for r in recommendations if r.get('severity') == 'high']
    medium_recs = [r for r in recommendations if r.get('severity') == 'medium']
    
    # Critical issues
    if critical_recs:
        st.subheader("🚨 Critical Issues Requiring Immediate Action")
        
        for i, rec in enumerate(critical_recs[:5], 1):
            with st.expander(f"Critical Issue {i}: {rec.get('primary_issue', 'Issue')}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Recommended Action:** {rec.get('recommended_action', 'Review')}")
                    
                    if 'implementation_steps' in rec:
                        st.markdown("**Implementation Steps:**")
                        for step in rec['implementation_steps']:
                            st.markdown(f"- {step}")
                
                with col2:
                    st.metric("Priority Score", f"{rec.get('priority_score', 0)}/10")
                    st.markdown(f"**Impact:** {rec.get('expected_impact', 'TBD')}")
    
    # High severity recommendations
    if high_recs:
        st.subheader("⚠️ High Priority Issues")
        
        for i, rec in enumerate(high_recs[:10], 1):
            with st.expander(f"High Priority {i}: {rec.get('primary_issue', 'Issue')}"):
                st.markdown(f"**Action:** {rec.get('recommended_action', 'Review')}")
                st.markdown(f"**Expected Impact:** {rec.get('expected_impact', 'TBD')}")
    
    # Executive summary
    st.divider()
    
    if st.button("Generate Executive Summary", key="gen_exec_summary"):
        with st.spinner("Generating executive summary..."):
            if st.session_state.provider and st.session_state.model:
                api_key = APIManager.get_api_key(st.session_state.provider)
                ai_analyzer = AIAnalyzer(st.session_state.provider, st.session_state.model, api_key)
                
                summary = asyncio.run(ai_analyzer.generate_executive_summary(
                    st.session_state.analysis_results,
                    recommendations
                ))
                
                st.markdown("### 📋 Executive Summary")
                st.markdown(summary)

def generate_reports_tab():
    """Generate and download reports"""
    if not st.session_state.analysis_complete:
        st.info("👆 Run analysis first to generate reports")
        return
    
    st.header("📈 Generate Reports")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Action Plan", "Technical Report"]
        )
    
    with col2:
        report_format = st.selectbox(
            "Format",
            ["Excel", "CSV", "JSON"]
        )
    
    # Report options
    st.subheader("Report Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_all_pairs = st.checkbox("Include all URL pairs", value=False)
        max_pairs = st.number_input("Max pairs to include", 10, 1000, 50) if not include_all_pairs else None
    
    with col2:
        include_ai_insights = st.checkbox("Include AI recommendations", value=True)
        include_technical_details = st.checkbox("Include technical details", value=False)
    
    with col3:
        include_charts = st.checkbox("Include visualizations", value=True)
        include_action_items = st.checkbox("Include action items", value=True)
    
    # Generate report
    if st.button("📥 Generate Report", type="primary"):
        with st.spinner(f"Generating {report_type} in {report_format} format..."):
            
            # Prepare data
            results = st.session_state.analysis_results
            recommendations = st.session_state.recommendations or []
            
            # Create report generator
            generator = ReportGenerator(
                results,
                recommendations,
                st.session_state.serp_data
            )
            
            # Generate report
            report = generator.generate_report(report_type, report_format.lower())
            
            # Download button
            st.download_button(
                label=f"📥 Download {report_type}",
                data=report['content'],
                file_name=report['filename'],
                mime=report['mime_type']
            )
            
            st.success(f"✅ {report_type} generated successfully!")

def show_help_tab():
    """Display help and documentation"""
    st.header("❓ Help & Documentation")
    
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Prepare Your Data:**
           - **GSC Export**: Performance report with URLs and queries
           - **Crawler Export**: Screaming Frog or similar with metadata
           - **Embeddings** (Optional): For enhanced similarity detection
        
        2. **Choose Detection Method:**
           - **Combined** (Recommended): Uses all detection methods
           - **Competition-First**: Focus on keyword/SERP competition
           - **Similarity-First**: Traditional content similarity
        
        3. **Configure Settings:**
           - Lower thresholds = more sensitive detection
           - Disable performance filtering to catch all issues
           - Enable debug mode to see detailed analysis
        
        4. **Run Analysis:**
           - Upload your files
           - Click "Run Cannibalization Analysis"
           - Review results in the Detection Results tab
        
        5. **Take Action:**
           - Review AI recommendations
           - Generate reports for stakeholders
           - Implement fixes based on priority
        """)
    
    with st.expander("📊 Understanding Detection Methods"):
        st.markdown("""
        ### Detection Methods Explained
        
        **1. Keyword Competition Detection** 🎯
        - Identifies URLs ranking for the same keywords
        - Uses GSC data to find actual competition
        - Most accurate method for active cannibalization
        
        **2. Content Similarity Detection** 📄
        - Compares titles, H1s, meta descriptions
        - Uses embeddings for semantic similarity
        - Identifies potential future cannibalization
        
        **3. SERP Competition Analysis** 🔍
        - Real-time search result analysis
        - Confirms actual SERP competition
        - Requires Serper API key
        
        **4. Combined Detection** 🏆
        - Uses all methods for comprehensive analysis
        - ML-based risk scoring
        - Provides confidence scores
        """)
    
    with st.expander("⚙️ Configuration Tips"):
        st.markdown("""
        ### Optimal Settings
        
        **For Maximum Detection:**
        - Min Shared Keywords: 1-2
        - Min Similarity: 0.15-0.20
        - Disable performance filtering
        - Use Combined detection method
        
        **For Focused Analysis:**
        - Min Shared Keywords: 3-5
        - Min Similarity: 0.30-0.40
        - Enable performance filtering
        - Use Competition-First method
        
        **For Large Sites (1000+ URLs):**
        - Increase thresholds slightly
        - Enable performance filtering
        - Process in batches if needed
        """)
    
    with st.expander("🐛 Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        **"No issues detected"**
        - Lower the minimum similarity threshold
        - Reduce minimum shared keywords to 1
        - Disable performance filtering
        - Check URL normalization in debug mode
        
        **"URL matching errors"**
        - Ensure consistent URL format across files
        - Check for trailing slashes
        - Verify protocol (http vs https)
        
        **"Analysis timeout"**
        - Reduce number of URLs
        - Enable performance filtering
        - Process in smaller batches
        
        **"Embeddings not working"**
        - Verify URL column name in embeddings file
        - Check that URLs match exactly
        - Enable debug mode to see matching attempts
        """)

if __name__ == "__main__":
    main()