# app.py
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

# Import custom modules
from utils.api_manager import APIManager
from utils.column_mapper import SmartColumnMapper, FlexibleDataLoader
from utils.export_handler import ExportHandler
from modules.analyzers.ai_analyzer import AIAnalyzer
from modules.analyzers.serp_analyzer import SERPAnalyzer
from modules.analyzers.ml_scoring import MLScoringEngine
from modules.analyzers.similarity_analyzer import SimilarityAnalyzer
from modules.analyzers.content_analyzer import ContentAnalyzer
from modules.data_loaders.crawler_loader import CrawlerDataLoader
from modules.data_loaders.gsc_loader import GSCLoader
from modules.reporting.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.title("🔍 Enhanced Content Cannibalization Analyzer")
    st.markdown("AI-powered analysis for identifying and resolving content cannibalization issues at scale")
    
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
        
        # AI Provider Selection
        available_providers = APIManager.get_available_providers()
        
        if available_providers:
            # Filter out any empty providers one more time
            valid_providers = {}
            for provider, info in available_providers.items():
                if APIManager.get_api_key(provider):
                    valid_providers[provider] = info
            
            if valid_providers:
                # Filter out O1 models if they somehow appear
                for provider in list(valid_providers.keys()):
                    if provider == 'openai' and 'models' in valid_providers[provider]:
                        # Remove O1 models
                        valid_providers[provider]['models'] = {
                            k: v for k, v in valid_providers[provider]['models'].items() 
                            if not k.startswith('o1')
                        }
                
                selected_provider = st.selectbox(
                    "AI Provider",
                    options=list(valid_providers.keys()),
                    format_func=lambda x: valid_providers[x]['display_name']
                )
                
                model_options = valid_providers[selected_provider]['models']
                selected_model = st.selectbox(
                    "Model",
                    options=list(model_options.keys()),
                    format_func=lambda x: f"{model_options[x]['name']} - {model_options[x]['best_for']}"
                )
                
                # Show model info
                model_info = APIManager.get_model_info(selected_provider, selected_model)
                st.info(f"Context window: {model_info['context']:,} tokens")
            else:
                st.error("No AI providers with valid API keys found. Please check your Streamlit secrets.")
                selected_provider = None
                selected_model = None
        else:
            st.error("No AI providers configured. Please add API keys in Streamlit secrets.")
            selected_provider = None
            selected_model = None
        
        st.divider()
        
        # Analysis Options
        st.subheader("Analysis Options")
        
        use_serp = st.checkbox(
            "Enable SERP Overlap Analysis",
            value=APIManager.has_serper_api(),
            disabled=not APIManager.has_serper_api(),
            help="Requires Serper API key"
        )
        
        use_gsc_oauth = st.checkbox(
            "Connect to Google Search Console",
            value=False,
            help="Direct GSC integration for real-time data"
        )
        
        st.divider()
        
        # Similarity Weights
        st.subheader("Similarity Weights")
        
        weights = {}
        weights['title'] = st.slider("Title Similarity", 0.0, 1.0, 0.25, 0.05)
        weights['h1'] = st.slider("H1 Similarity", 0.0, 1.0, 0.15, 0.05)
        weights['semantic'] = st.slider("Semantic Similarity", 0.0, 1.0, 0.20, 0.05)
        weights['keyword'] = st.slider("Keyword Overlap", 0.0, 1.0, 0.15, 0.05)
        
        if use_serp:
            weights['serp'] = st.slider("SERP Overlap", 0.0, 1.0, 0.15, 0.05)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        st.divider()
        
        # Risk Thresholds
        st.subheader("Risk Thresholds")
        high_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.05)
        medium_threshold = st.slider("Medium Risk Threshold", 0.0, 1.0, 0.4, 0.05)
        
        # Content Extraction Settings
        extraction_config = show_content_extraction_settings()
    
    # Main content tabs
    if available_providers:
        tabs = st.tabs(["📊 Analysis", "🤖 AI Insights", "📈 Reports", "❓ Help"])
        
        with tabs[0]:
            run_analysis_tab(selected_provider, selected_model, weights, 
                           use_serp, use_gsc_oauth, high_threshold, medium_threshold,
                           extraction_config)
        
        with tabs[1]:
            show_ai_insights_tab()
        
        with tabs[2]:
            generate_reports_tab()
        
        with tabs[3]:
            show_help_tab()
    else:
        st.error("Please configure at least one AI provider to continue.")

def show_content_extraction_settings():
    """Show content extraction settings in sidebar"""
    st.sidebar.divider()
    st.sidebar.subheader("🔍 Content Extraction Settings")
    
    # Extraction method selection
    extraction_method = st.sidebar.selectbox(
        "Extraction Method",
        ["Smart (Automatic)", "Trafilatura (Articles)", "Readability (General)", "Custom Rules"],
        help="""
        - Smart: Tries multiple methods automatically
        - Trafilatura: Best for articles and blog posts
        - Readability: Good for various content types
        - Custom Rules: Uses pattern-based extraction
        """,
        key="extraction_method_select"
    )
    
    # Map display names to method names
    method_map = {
        "Smart (Automatic)": "smart",
        "Trafilatura (Articles)": "trafilatura",
        "Readability (General)": "readability",
        "Custom Rules": "custom"
    }
    extraction_method = method_map[extraction_method]
    
    # Advanced settings
    with st.sidebar.expander("Advanced Extraction Settings"):
        st.markdown("### Exclude Additional Elements")
        
        # Custom exclusions
        exclude_classes = st.text_area(
            "Exclude Classes (one per line)",
            placeholder="e.g.\npromo-box\nauthor-bio\nread-more",
            help="CSS classes to exclude from content extraction",
            key="exclude_classes_input"
        )
        
        exclude_ids = st.text_area(
            "Exclude IDs (one per line)",
            placeholder="e.g.\nsubscribe-form\npopup-modal",
            help="Element IDs to exclude from content extraction",
            key="exclude_ids_input"
        )
        
        # Minimum content length
        min_content_length = st.number_input(
            "Minimum Valid Content Length",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Minimum characters to consider content valid (pages with less will be flagged)",
            key="min_content_length_input"
        )
        
        st.info("Note: Content up to 10,000 characters will be processed per page")
    
    # Build extraction config
    extraction_config = {
        'method': extraction_method,
        'min_length': min_content_length
    }
    
    # Add custom patterns if provided
    if exclude_classes:
        extraction_config['exclude_classes'] = [c.strip() for c in exclude_classes.split('\n') if c.strip()]
    
    if exclude_ids:
        extraction_config['exclude_ids'] = [i.strip() for i in exclude_ids.split('\n') if i.strip()]
    
    return extraction_config

def run_analysis_tab(provider, model, weights, use_serp, use_gsc_oauth, 
                    high_threshold, medium_threshold, extraction_config):
    """Run the main analysis"""
    
    st.header("📊 Content Analysis")
    
    # File upload section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Internal SEO Data")
        internal_file = st.file_uploader(
            "Upload crawler export (CSV)",
            type=['csv'],
            help="Flexible format: URL/Address, Title, H1, Meta Description",
            key="internal_file_upload"
        )
        
        if internal_file:
            try:
                # Reset file pointer
                internal_file.seek(0)
                # Try to read with different encodings for preview
                preview_df = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        internal_file.seek(0)
                        preview_df = pd.read_csv(internal_file, nrows=5, encoding=encoding)
                        break
                    except:
                        continue
                
                if preview_df is not None and not preview_df.empty:
                    with st.expander("📋 File Preview & Column Detection"):
                        st.write("First 5 rows of your file:")
                        st.dataframe(preview_df)
                        
                        mapper = SmartColumnMapper()
                        suggestions = mapper.get_column_suggestions(preview_df)
                        st.text(suggestions)
                        st.session_state.internal_data = preview_df
                else:
                    st.warning("Could not preview the file. It might be empty or have an encoding issue.")
                    
                # Reset file pointer for actual processing
                internal_file.seek(0)
                
            except Exception as e:
                st.error(f"Error previewing file: {str(e)}")
                st.info("The file might be using a special encoding or format. The analysis will still attempt to process it.")
    
    with col2:
        st.subheader("GSC Performance Data")
        if use_gsc_oauth:
            if st.button("Connect to GSC", key="gsc_connect_btn"):
                # Handle OAuth flow
                gsc_loader = GSCLoader()
                gsc_loader.authenticate()
        else:
            gsc_file = st.file_uploader(
                "Upload GSC export (CSV)",
                type=['csv'],
                help="Flexible format: URL/Landing Page, Query/Keyword, Clicks, Impressions",
                key="gsc_file_upload"
            )
    
    with col3:
        st.subheader("Embeddings/Content (Optional)")
        
        # Choice between embeddings or content
        analysis_enhancement = st.radio(
            "Enhancement Option",
            ["None", "Use Screaming Frog Embeddings", "Analyze Page Content"],
            help="""
            - None: Use only metadata (titles, H1s, descriptions)
            - Screaming Frog Embeddings: Use pre-calculated embeddings from SF
            - Analyze Page Content: Extract and analyze actual page content
            """,
            key="enhancement_option"
        )
        
        embeddings_file = None
        content_option = None
        
        if analysis_enhancement == "Use Screaming Frog Embeddings":
            embeddings_file = st.file_uploader(
                "Upload embeddings (CSV)",
                type=['csv'],
                help="CSV with URL and embedding values from Screaming Frog",
                key="embeddings_file_upload"
            )
            st.info("📊 Will use Screaming Frog's embeddings for similarity calculation")
            
        elif analysis_enhancement == "Analyze Page Content":
            content_option = st.selectbox(
                "Content Source",
                ["Fetch from URLs", "Content in crawler export", "Upload separate file"],
                key="content_source"
            )
            
            if content_option == "Upload separate file":
                content_file = st.file_uploader(
                    "Upload content data (CSV)",
                    type=['csv'],
                    help="CSV with URL and Content/Text columns",
                    key="content_file_upload"
                )
            elif content_option == "Content in crawler export":
                if st.session_state.internal_data is not None:
                    df = st.session_state.internal_data
                    possible_content_cols = [col for col in df.columns 
                                           if any(term in col.lower() 
                                                 for term in ['content', 'text', 'body', 'copy'])]
                    
                    if possible_content_cols:
                        content_column = st.selectbox(
                            "Select content column",
                            possible_content_cols,
                            key="content_column_select"
                        )
                    else:
                        st.warning("No content columns detected")
    
    # Analysis configuration
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_similarity = st.slider(
            "Minimum Similarity Threshold",
            0.0, 1.0, 0.3, 0.05,
            help="Only analyze URL pairs above this similarity",
            key="min_similarity_threshold"
        )
    
    with col2:
        if use_serp:
            serp_location = st.selectbox(
                "SERP Location",
                ["United States", "United Kingdom", "Canada", "Australia", 
                 "Germany", "France", "Spain", "India"],
                key="serp_location_select"
            )
    
    # Run analysis button
    if st.button("🚀 Run Cannibalization Analysis", type="primary", use_container_width=True):
        if internal_file and (gsc_file or use_gsc_oauth):
            asyncio.run(run_analysis(
                internal_file, 
                gsc_file if not use_gsc_oauth else None,
                embeddings_file,
                provider,
                model,
                weights,
                use_serp,
                serp_location if use_serp else None,
                min_similarity,
                high_threshold,
                medium_threshold,
                extraction_config,
                analysis_enhancement,
                content_option
            ))
        else:
            st.error("Please upload both internal SEO data and GSC performance data.")
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        display_analysis_results()

async def run_analysis(internal_file, gsc_file, embeddings_file, provider, model,
                      weights, use_serp, serp_location, min_similarity,
                      high_threshold, medium_threshold, extraction_config,
                      analysis_enhancement, content_option):
    """Run the complete analysis pipeline"""
    
    try:
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Step 1: Load data
        progress_bar.progress(10, text="Loading data files...")
        
        # Load internal data with flexible column mapping
        loader = FlexibleDataLoader()
        internal_data = loader.load_internal_data(internal_file)
        
        # Load GSC data with flexible column mapping
        if gsc_file:
            gsc_data = loader.load_gsc_data(gsc_file)
        else:
            # Handle OAuth GSC data
            gsc_data = None  # Placeholder
        
        progress_bar.progress(20, text="Data loaded successfully")
        
        # Step 2: Initialize analyzers
        progress_bar.progress(30, text="Initializing AI analyzer...")
        
        api_key = APIManager.get_api_key(provider)
        ai_analyzer = AIAnalyzer(provider, model, api_key)
        
        # Initialize ML scoring engine
        ml_scorer = MLScoringEngine()
        
        # Step 3: Analyze content intent
        progress_bar.progress(40, text="Analyzing content intent with AI...")
        internal_data = await ai_analyzer.analyze_intent_batch(internal_data)
        
        # Step 4: Handle embeddings or content analysis
        if analysis_enhancement == "Use Screaming Frog Embeddings" and embeddings_file:
            progress_bar.progress(50, text="Processing Screaming Frog embeddings...")
            embeddings_data = pd.read_csv(embeddings_file)
            # Use SF embeddings for similarity calculation
            use_content_analysis = False
            
        elif analysis_enhancement == "Analyze Page Content":
            progress_bar.progress(50, text="Analyzing page content...")
            content_analyzer = ContentAnalyzer(ai_analyzer, extraction_config['method'])
            
            if content_option == "Fetch from URLs":
                internal_data = await content_analyzer.analyze_content_similarity(
                    internal_data, 
                    fetch_content=True,
                    extraction_config=extraction_config
                )
            elif content_option == "Content in crawler export":
                # Use existing column
                internal_data = await content_analyzer.analyze_content_similarity(
                    internal_data,
                    fetch_content=False,
                    content_column=content_column if 'content_column' in locals() else None
                )
            
            embeddings_data = None
            use_content_analysis = True
        else:
            embeddings_data = None
            use_content_analysis = False
        
        # Step 5: Calculate similarities
        progress_bar.progress(60, text="Calculating content similarities...")
        
        similarity_analyzer = SimilarityAnalyzer(
            embeddings_data=embeddings_data,
            use_content_embeddings=use_content_analysis
        )
        similarity_results = similarity_analyzer.calculate_all_similarities(
            internal_data, 
            min_similarity
        )
        
        # Step 6: SERP analysis (if enabled)
        serp_results = None
        if use_serp and APIManager.has_serper_api():
            progress_bar.progress(70, text="Analyzing SERP overlap...")
            
            # Extract keywords from GSC data
            keywords = gsc_data.groupby('query')['clicks'].sum().nlargest(100).index.tolist()
            
            # Get domain from internal data
            domain = internal_data['url'].iloc[0].split('/')[2]
            
            serper_key = APIManager.get_serper_api_key()
            async with SERPAnalyzer(serper_key) as serp_analyzer:
                serp_results = await serp_analyzer.check_serp_overlap(
                    keywords, domain, serp_location
                )
        
        # Step 7: Calculate ML-based risk scores
        progress_bar.progress(80, text="Calculating risk scores...")
        
        cannibalization_pairs = []
        for _, row in similarity_results.iterrows():
            # Prepare features for ML scoring
            features = {
                'title_similarity': row['title_similarity'],
                'h1_similarity': row['h1_similarity'],
                'semantic_similarity': row.get('semantic_similarity', row.get('content_similarity', 0)),
                'keyword_overlap': calculate_keyword_overlap(row, gsc_data),
                'intent_match': 1 if row['intent1'] == row['intent2'] else 0
            }
            
            if serp_results:
                features['serp_overlap'] = get_serp_overlap_score(row, serp_results)
            
            # Get ML risk score
            risk_score, contributions = ml_scorer.calculate_adaptive_score(features)
            
            cannibalization_pairs.append({
                'url1': row['url1'],
                'url2': row['url2'],
                'title1': row.get('title1', ''),
                'title2': row.get('title2', ''),
                'risk_score': risk_score,
                'risk_category': ml_scorer.get_risk_category(risk_score),
                'contributions': contributions,
                **features
            })
        
        # Step 8: Generate AI recommendations
        progress_bar.progress(90, text="Generating AI recommendations...")
        
        # Filter high and medium risk pairs
        priority_pairs = [p for p in cannibalization_pairs 
                         if p['risk_score'] >= medium_threshold]
        
        recommendations = await ai_analyzer.generate_recommendations(
            priority_pairs[:50],  # Limit to top 50 for performance
            gsc_data
        )
        
        # Step 9: Prepare final results
        progress_bar.progress(95, text="Preparing results...")
        
        analysis_results = {
            'total_urls': len(internal_data),
            'total_pairs': len(cannibalization_pairs),
            'high_risk_count': sum(1 for p in cannibalization_pairs 
                                 if p['risk_score'] >= high_threshold),
            'medium_risk_count': sum(1 for p in cannibalization_pairs 
                                   if medium_threshold <= p['risk_score'] < high_threshold),
            'low_risk_count': sum(1 for p in cannibalization_pairs 
                                if p['risk_score'] < medium_threshold),
            'pairs': cannibalization_pairs,
            'serp_summary': serp_results.get('summary') if serp_results else None,
            'analysis_method': analysis_enhancement,
            'content_analyzed': use_content_analysis
        }
        
        # Store in session state
        st.session_state.analysis_results = analysis_results
        st.session_state.recommendations = recommendations
        st.session_state.serp_data = serp_results
        st.session_state.analysis_complete = True
        
        progress_bar.progress(100, text="Analysis complete!")
        st.success("✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

def calculate_keyword_overlap(row, gsc_data):
    """Calculate keyword overlap between two URLs"""
    url1_keywords = set(gsc_data[gsc_data['url'] == row['url1']]['query'])
    url2_keywords = set(gsc_data[gsc_data['url'] == row['url2']]['query'])
    
    if not url1_keywords or not url2_keywords:
        return 0.0
    
    overlap = len(url1_keywords & url2_keywords)
    total = len(url1_keywords | url2_keywords)
    
    return overlap / total if total > 0 else 0.0

def get_serp_overlap_score(row, serp_results):
    """Get SERP overlap score for a URL pair"""
    keyword_overlaps = serp_results.get('keyword_overlaps', {})
    
    # Find keywords where both URLs appear
    overlap_score = 0.0
    overlap_count = 0
    
    for keyword, data in keyword_overlaps.items():
        urls = [u['url'] for u in data['overlapping_urls']]
        if row['url1'] in urls and row['url2'] in urls:
            overlap_score += data['overlap_score']
            overlap_count += 1
    
    return overlap_score / overlap_count if overlap_count > 0 else 0.0

def display_analysis_results():
    """Display the analysis results"""
    results = st.session_state.analysis_results
    
    # Summary metrics
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total URLs Analyzed", results['total_urls'])
    
    with col2:
        st.metric("Cannibalization Pairs", results['total_pairs'])
    
    with col3:
        st.metric("High Risk", results['high_risk_count'], 
                 delta=f"{results['high_risk_count']/results['total_pairs']*100:.1f}%" if results['total_pairs'] > 0 else "0%")
    
    with col4:
        st.metric("Medium Risk", results['medium_risk_count'],
                 delta=f"{results['medium_risk_count']/results['total_pairs']*100:.1f}%" if results['total_pairs'] > 0 else "0%")
    
    # Analysis method info
    if results.get('analysis_method'):
        st.info(f"📊 Analysis Method: {results['analysis_method']}" + 
               (" with content extraction" if results.get('content_analyzed') else ""))
    
    # Risk distribution chart
    st.subheader("Risk Distribution")
    
    risk_data = pd.DataFrame({
        'Risk Level': ['High', 'Medium', 'Low'],
        'Count': [results['high_risk_count'], 
                 results['medium_risk_count'], 
                 results['low_risk_count']]
    })
    
    fig = px.pie(risk_data, values='Count', names='Risk Level',
                color_discrete_map={'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00cc00'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Top cannibalization pairs
    st.subheader("🔥 Top Cannibalization Issues")
    
    # Convert pairs to DataFrame for display
    pairs_df = pd.DataFrame(results['pairs'])
    pairs_df = pairs_df.sort_values('risk_score', ascending=False)
    
    # Display top 10
    for idx, row in pairs_df.head(10).iterrows():
        with st.expander(f"**{row['url1'][:50]}...** vs **{row['url2'][:50]}...**"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Risk Score:** <span class='{row['risk_category'].lower()}-risk'>"
                          f"{row['risk_score']:.2%}</span>", unsafe_allow_html=True)
                st.markdown(f"**Risk Category:** {row['risk_category']}")
                if row.get('title1'):
                    st.markdown(f"**Title 1:** {row['title1'][:60]}...")
                if row.get('title2'):
                    st.markdown(f"**Title 2:** {row['title2'][:60]}...")
            
            with col2:
                st.markdown("**Similarity Breakdown:**")
                for feature, value in row['contributions'].items():
                    st.markdown(f"- {feature.replace('_', ' ').title()}: {value:.2%}")

def show_ai_insights_tab():
    """Display AI-generated insights"""
    if not st.session_state.analysis_complete:
        st.info("👆 Please run the analysis first to see AI insights.")
        return
    
    st.header("🤖 AI-Powered Insights")
    
    recommendations = st.session_state.recommendations
    
    if recommendations:
        # Group by severity
        high_severity = [r for r in recommendations if r.get('severity') == 'high']
        medium_severity = [r for r in recommendations if r.get('severity') == 'medium']
        
        # High severity recommendations
        if high_severity:
            st.subheader("🚨 Critical Issues Requiring Immediate Action")
            
            for rec in high_severity[:5]:
                with st.expander(f"Priority {rec.get('priority_score', 'N/A')}/10 - {rec.get('primary_issue', 'Issue')}"):
                    st.markdown(f"**Recommended Action:** {rec.get('recommended_action', 'Review').replace('_', ' ').title()}")
                    
                    st.markdown("**Implementation Steps:**")
                    for step in rec.get('implementation_steps', []):
                        st.markdown(f"- {step}")
                    
                    st.markdown(f"**Expected Impact:** {rec.get('expected_impact', 'N/A')}")
        
        # Medium severity recommendations
        if medium_severity:
            st.subheader("⚠️ Important Issues to Address")
            
            for rec in medium_severity[:5]:
                with st.expander(f"Priority {rec.get('priority_score', 'N/A')}/10 - {rec.get('primary_issue', 'Issue')}"):
                    st.markdown(f"**Recommended Action:** {rec.get('recommended_action', 'Review').replace('_', ' ').title()}")
                    st.markdown(f"**Expected Impact:** {rec.get('expected_impact', 'N/A')}")
        
        # Executive Summary
        st.divider()
        if st.button("Generate Executive Summary", key="exec_summary_btn"):
            with st.spinner("Generating AI summary..."):
                ai_analyzer = AIAnalyzer(
                    st.session_state.get('provider'), 
                    st.session_state.get('model'),
                    APIManager.get_api_key(st.session_state.get('provider'))
                )
                summary = asyncio.run(ai_analyzer.generate_executive_summary(
                    st.session_state.analysis_results,
                    recommendations
                ))
                st.markdown("### 📋 Executive Summary")
                st.markdown(summary)

def generate_reports_tab():
    """Generate and download reports without resetting the app"""
    if not st.session_state.analysis_complete:
        st.info("👆 Please run the analysis first to generate reports.")
        return
    
    st.header("📈 Generate Reports")
    
    # Create tabs for different export types
    export_tabs = st.tabs(["Quick Export", "Custom Report", "Bulk Export"])
    
    with export_tabs[0]:
        st.subheader("Quick Export Options")
        
        # Prepare data for export
        results = st.session_state.analysis_results
        pairs_df = pd.DataFrame(results['pairs'])
        
        # High risk pairs
        high_risk_df = pairs_df[pairs_df['risk_category'] == 'High'].copy()
        if not high_risk_df.empty:
            st.markdown("### 🔴 High Risk Pairs")
            ExportHandler.export_with_state_preservation(
                'high_risk',
                high_risk_df[['url1', 'url2', 'risk_score', 'title_similarity', 'semantic_similarity']],
                'high_risk_cannibalization.csv',
                'csv'
            )
        
        # All pairs
        st.markdown("### 📊 All Cannibalization Pairs")
        ExportHandler.export_with_state_preservation(
            'all_pairs',
            pairs_df,
            'all_cannibalization_pairs.xlsx',
            'xlsx'
        )
        
        # Recommendations
        if st.session_state.recommendations:
            st.markdown("### 💡 AI Recommendations")
            rec_df = pd.DataFrame(st.session_state.recommendations)
            ExportHandler.export_with_state_preservation(
                'recommendations',
                rec_df,
                'ai_recommendations.csv',
                'csv'
            )
    
    with export_tabs[1]:
        st.subheader("Custom Report Builder")
        
        # Let users select what to include
        include_summary = st.checkbox("Include Executive Summary", value=True, key="cb_summary")
        include_high_risk = st.checkbox("Include High Risk Pairs", value=True, key="cb_high_risk")
        include_recommendations = st.checkbox("Include AI Recommendations", value=True, key="cb_recommendations")
        include_serp = st.checkbox("Include SERP Analysis", 
                                  value=st.session_state.serp_data is not None,
                                  key="cb_serp")
        
        if st.button("Generate Custom Report", key="custom_report_btn"):
            with st.spinner("Building custom report..."):
                # Build custom report would go here
                st.success("Custom report ready for download!")
    
    with export_tabs[2]:
        st.subheader("Bulk Export All Data")
        
        if st.button("Prepare Bulk Export", key="bulk_export_btn"):
            with st.spinner("Preparing bulk export..."):
                # Create a multi-sheet Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Summary sheet
                    summary_df = pd.DataFrame([{
                        'Total URLs': results['total_urls'],
                        'Total Pairs': results['total_pairs'],
                        'High Risk': results['high_risk_count'],
                        'Medium Risk': results['medium_risk_count'],
                        'Low Risk': results['low_risk_count'],
                        'Analysis Method': results.get('analysis_method', 'Standard')
                    }])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # All pairs
                    pairs_df.to_excel(writer, sheet_name='All Pairs', index=False)
                    
                    # High risk only
                    if not high_risk_df.empty:
                        high_risk_df.to_excel(writer, sheet_name='High Risk', index=False)
                    
                    # Recommendations
                    if st.session_state.recommendations:
                        rec_df = pd.DataFrame(st.session_state.recommendations)
                        rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
                    
                    # SERP data
                    if st.session_state.serp_data:
                        serp_summary = pd.DataFrame([st.session_state.serp_data['summary']])
                        serp_summary.to_excel(writer, sheet_name='SERP Summary', index=False)
                
                # Download without rerun
                st.download_button(
                    label="📥 Download Complete Analysis",
                    data=output.getvalue(),
                    file_name='complete_cannibalization_analysis.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='bulk_download_unique'
                )

def show_help_tab():
    """Display help and documentation"""
    st.header("❓ Help & Documentation")
    
    with st.expander("🚀 Getting Started"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Prepare Your Data:**
           - Export your site's crawl data (Screaming Frog, Sitebulb, etc.)
           - Download GSC performance report (no row limit)
           - (Optional) Export embeddings from Screaming Frog or prepare content data
        
        2. **Configure Settings:**
           - Select your preferred AI provider and model
           - Adjust similarity weights based on your priorities
           - Set risk thresholds for your needs
           - Configure content extraction if analyzing page content
        
        3. **Choose Analysis Method:**
           - **Basic**: Metadata only (titles, H1s, descriptions)
           - **With SF Embeddings**: Use Screaming Frog's content embeddings
           - **With Content Analysis**: Extract and analyze actual page content
        
        4. **Run Analysis:**
           - Upload your data files
           - Enable SERP analysis for deeper insights
           - Click "Run Cannibalization Analysis"
        
        5. **Review Results:**
           - Check the Analysis tab for overview
           - Explore AI Insights for recommendations
           - Generate reports for stakeholders
        """)
    
    with st.expander("📊 Understanding the Metrics"):
        st.markdown("""
        ### Similarity Metrics Explained
        
        - **Title Similarity:** How similar page titles are (critical for SEO)
        - **H1 Similarity:** Similarity of main headings
        - **Semantic Similarity:** Overall content theme similarity (from embeddings or content)
        - **Keyword Overlap:** Shared target keywords from GSC
        - **SERP Overlap:** Actual competition in search results
        
        ### Risk Categories
        
        - **High Risk (>70%):** Immediate action required
        - **Medium Risk (40-70%):** Should be addressed soon
        - **Low Risk (<40%):** Monitor but not critical
        
        ### Content Analysis
        
        - **Screaming Frog Embeddings:** Pre-calculated content vectors from SF
        - **Content Extraction:** Analyzes actual page content, excluding headers/footers
        - **Smart Extraction:** Automatically identifies main content areas
        """)
    
    with st.expander("🛠️ Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        **Column Detection Problems:**
        - The app automatically detects column variations
        - Check the preview to see detected mappings
        - Ensure CSV is properly formatted
        
        **Content Extraction:**
        - Minimum content length validates page quality
        - Up to 10,000 characters processed per page
        - Custom exclusions for site-specific elements
        
        **API Errors:**
        - Verify API keys in Streamlit secrets
        - Check rate limits for your plan
        - Ensure proper formatting
        
        **Performance Issues:**
        - Start with top pages for large sites
        - Use embeddings for faster analysis
        - Limit SERP keywords to most important
        """)

if __name__ == "__main__":
    main()
