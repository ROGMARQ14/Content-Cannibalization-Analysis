# utils/export_handler.py
import streamlit as st
import pandas as pd
import json
import io
from typing import Dict, Any
import base64

class ExportHandler:
    """Handle exports without resetting the app state"""
    
    @staticmethod
    def create_download_link(data: Any, filename: str, file_format: str) -> str:
        """Create a download link that doesn't trigger app rerun"""
        
        if file_format == 'csv':
            if isinstance(data, pd.DataFrame):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                mime = 'text/csv'
            else:
                raise ValueError("CSV format requires a DataFrame")
                
        elif file_format == 'json':
            if isinstance(data, (dict, list)):
                json_str = json.dumps(data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                mime = 'application/json'
            else:
                raise ValueError("JSON format requires a dict or list")
                
        elif file_format == 'xlsx':
            if isinstance(data, pd.DataFrame):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data.to_excel(writer, sheet_name='Analysis', index=False)
                b64 = base64.b64encode(output.getvalue()).decode()
                mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            else:
                raise ValueError("Excel format requires a DataFrame")
        
        href = f'<a href="data:{mime};base64,{b64}" download="{filename}">📥 Download {filename}</a>'
        return href
    
    @staticmethod
    def export_with_state_preservation(key: str, data: Any, filename: str, file_format: str):
        """Export data while preserving app state"""
        
        # Generate unique key for this export
        export_key = f"export_{key}_{filename}"
        
        # Store export state
        if export_key not in st.session_state:
            st.session_state[export_key] = False
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"Ready to export: {filename}")
        
        with col2:
            if file_format == 'csv':
                data_to_download = data.to_csv(index=False) if isinstance(data, pd.DataFrame) else str(data)
                mime_type = 'text/csv'
            elif file_format == 'json':
                data_to_download = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
                mime_type = 'application/json'
            elif file_format == 'xlsx':
                output = io.BytesIO()
                if isinstance(data, pd.DataFrame):
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        data.to_excel(writer, sheet_name='Analysis', index=False)
                data_to_download = output.getvalue()
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            
            # Use download button with unique key
            st.download_button(
                label=f"Download {file_format.upper()}",
                data=data_to_download,
                file_name=filename,
                mime=mime_type,
                key=export_key  # Unique key prevents rerun
            )

# Enhanced app.py export section
def generate_reports_tab_enhanced():
    """Enhanced report generation that doesn't reset the app"""
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
        include_summary = st.checkbox("Include Executive Summary", value=True)
        include_high_risk = st.checkbox("Include High Risk Pairs", value=True)
        include_recommendations = st.checkbox("Include AI Recommendations", value=True)
        include_serp = st.checkbox("Include SERP Analysis", value=st.session_state.serp_data is not None)
        
        if st.button("Generate Custom Report", key="custom_report_btn"):
            with st.spinner("Building custom report..."):
                custom_report = build_custom_report(
                    include_summary,
                    include_high_risk,
                    include_recommendations,
                    include_serp
                )
                
                # Export without triggering rerun
                ExportHandler.export_with_state_preservation(
                    'custom_report',
                    custom_report,
                    'custom_cannibalization_report.xlsx',
                    'xlsx'
                )
    
    with export_tabs[2]:
        st.subheader("Bulk Export All Data")
        
        if st.button("Export Everything", key="bulk_export_btn"):
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
                        'Low Risk': results['low_risk_count']
                    }])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # All pairs
                    pairs_df.to_excel(writer, sheet_name='All Pairs', index=False)
                    
                    # High risk only
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
                    key='bulk_download'  # Unique key
                )