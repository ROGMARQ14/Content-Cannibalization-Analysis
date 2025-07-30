# modules/reporting/report_generator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import io
from datetime import datetime
import logging
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, PieChart, Reference

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate various report formats for cannibalization analysis"""
    
    def __init__(self, analysis_results: Dict, recommendations: List[Dict], 
                 serp_data: Optional[Dict] = None):
        self.analysis_results = analysis_results
        self.recommendations = recommendations
        self.serp_data = serp_data
        self.timestamp = datetime.now()
        
    def generate_report(self, report_type: str, format_type: str) -> Dict:
        """
        Generate a report in the specified format
        
        Args:
            report_type: Type of report (executive_summary, detailed_analysis, etc.)
            format_type: Output format (excel, csv, json, pdf)
            
        Returns:
            Dict with 'content', 'filename', and 'mime_type'
        """
        # Generate report content based on type
        if report_type.lower() == "executive summary":
            content = self._generate_executive_summary()
        elif report_type.lower() == "detailed analysis":
            content = self._generate_detailed_analysis()
        elif report_type.lower() == "action plan":
            content = self._generate_action_plan()
        elif report_type.lower() == "technical report":
            content = self._generate_technical_report()
        else:
            content = self._generate_detailed_analysis()
        
        # Convert to requested format
        if format_type.lower() == "excel":
            return self._format_as_excel(content, report_type)
        elif format_type.lower() == "csv":
            return self._format_as_csv(content, report_type)
        elif format_type.lower() == "json":
            return self._format_as_json(content, report_type)
        else:
            # Default to Excel
            return self._format_as_excel(content, report_type)
    
    def _generate_executive_summary(self) -> Dict:
        """Generate executive summary report data"""
        pairs_df = pd.DataFrame(self.analysis_results['pairs'])
        
        # Calculate key metrics
        total_traffic_at_risk = self._calculate_traffic_at_risk(pairs_df)
        top_opportunities = self._identify_top_opportunities(pairs_df)
        
        summary_data = {
            'overview': {
                'analysis_date': self.timestamp.strftime('%Y-%m-%d'),
                'total_urls_analyzed': self.analysis_results['total_urls'],
                'cannibalization_pairs_found': self.analysis_results['total_pairs'],
                'high_risk_issues': self.analysis_results['high_risk_count'],
                'medium_risk_issues': self.analysis_results['medium_risk_count'],
                'estimated_traffic_impact': total_traffic_at_risk
            },
            'key_findings': self._generate_key_findings(),
            'top_priorities': top_opportunities[:5],
            'recommended_actions': self._generate_executive_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        
        return summary_data
    
    def _generate_detailed_analysis(self) -> Dict:
        """Generate detailed analysis report data"""
        pairs_df = pd.DataFrame(self.analysis_results['pairs'])
        
        detailed_data = {
            'summary': self._generate_summary_section(),
            'risk_analysis': self._generate_risk_analysis(pairs_df),
            'cannibalization_pairs': self._format_pairs_for_report(pairs_df),
            'recommendations': self._format_recommendations(),
            'serp_analysis': self._format_serp_analysis() if self.serp_data else None,
            'technical_details': self._generate_technical_details(pairs_df)
        }
        
        return detailed_data
    
    def _generate_action_plan(self) -> Dict:
        """Generate actionable plan report data"""
        pairs_df = pd.DataFrame(self.analysis_results['pairs'])
        high_risk_pairs = pairs_df[pairs_df['risk_category'] == 'High']
        
        action_plan = {
            'immediate_actions': self._generate_immediate_actions(high_risk_pairs),
            'consolidation_plan': self._generate_consolidation_plan(),
            'content_optimization': self._generate_optimization_plan(),
            'timeline': self._generate_implementation_timeline(),
            'success_metrics': self._generate_success_metrics()
        }
        
        return action_plan
    
    def _generate_technical_report(self) -> Dict:
        """Generate technical SEO report data"""
        pairs_df = pd.DataFrame(self.analysis_results['pairs'])
        
        technical_data = {
            'similarity_analysis': self._generate_similarity_analysis(pairs_df),
            'intent_analysis': self._generate_intent_analysis(pairs_df),
            'url_patterns': self._analyze_url_patterns(pairs_df),
            'content_gaps': self._identify_content_gaps(),
            'technical_recommendations': self._generate_technical_recommendations()
        }
        
        return technical_data
    
    def _format_as_excel(self, content: Dict, report_type: str) -> Dict:
        """Format report as Excel file with multiple sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Create workbook and remove default sheet
            workbook = writer.book
            
            # Summary sheet
            self._create_summary_sheet(writer, content, report_type)
            
            # Detailed data sheets based on report type
            if report_type.lower() == "executive summary":
                self._create_executive_sheets(writer, content)
            elif report_type.lower() == "detailed analysis":
                self._create_detailed_sheets(writer, content)
            elif report_type.lower() == "action plan":
                self._create_action_sheets(writer, content)
            elif report_type.lower() == "technical report":
                self._create_technical_sheets(writer, content)
            
            # Apply styling
            for sheet in workbook.worksheets:
                self._style_excel_sheet(sheet)
        
        output.seek(0)
        
        return {
            'content': output.getvalue(),
            'filename': f'{report_type.lower().replace(" ", "_")}_{self.timestamp.strftime("%Y%m%d")}.xlsx',
            'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
    
    def _format_as_csv(self, content: Dict, report_type: str) -> Dict:
        """Format report as CSV (simplified version)"""
        # For CSV, we'll focus on the main data
        pairs_df = pd.DataFrame(self.analysis_results['pairs'])
        
        # Add recommendations if available
        if self.recommendations:
            rec_df = pd.DataFrame(self.recommendations)
            # Merge recommendations with pairs
            if 'pair_index' in rec_df.columns:
                pairs_df = pairs_df.merge(
                    rec_df[['pair_index', 'recommended_action', 'priority_score']],
                    left_index=True,
                    right_on='pair_index',
                    how='left'
                )
        
        csv_content = pairs_df.to_csv(index=False)
        
        return {
            'content': csv_content,
            'filename': f'{report_type.lower().replace(" ", "_")}_{self.timestamp.strftime("%Y%m%d")}.csv',
            'mime_type': 'text/csv'
        }
    
    def _format_as_json(self, content: Dict, report_type: str) -> Dict:
        """Format report as JSON"""
        # Ensure all data is JSON serializable
        json_content = self._make_json_serializable(content)
        
        return {
            'content': json.dumps(json_content, indent=2),
            'filename': f'{report_type.lower().replace(" ", "_")}_{self.timestamp.strftime("%Y%m%d")}.json',
            'mime_type': 'application/json'
        }
    
    def _create_summary_sheet(self, writer, content: Dict, report_type: str):
        """Create summary sheet for Excel report"""
        summary_data = []
        
        # Add report metadata
        summary_data.append(['Report Type', report_type])
        summary_data.append(['Generated Date', self.timestamp.strftime('%Y-%m-%d %H:%M')])
        summary_data.append([''])
        
        # Add key metrics
        summary_data.append(['Key Metrics', ''])
        summary_data.append(['Total URLs Analyzed', self.analysis_results['total_urls']])
        summary_data.append(['Cannibalization Pairs Found', self.analysis_results['total_pairs']])
        summary_data.append(['High Risk Issues', self.analysis_results['high_risk_count']])
        summary_data.append(['Medium Risk Issues', self.analysis_results['medium_risk_count']])
        summary_data.append(['Low Risk Issues', self.analysis_results['low_risk_count']])
        
        # Create DataFrame and write to Excel
        df_summary = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    def _create_detailed_sheets(self, writer, content: Dict):
        """Create sheets for detailed analysis report"""
        # Cannibalization pairs sheet
        if 'cannibalization_pairs' in content:
            pairs_df = pd.DataFrame(content['cannibalization_pairs'])
            pairs_df.to_excel(writer, sheet_name='Cannibalization Pairs', index=False)
        
        # Recommendations sheet
        if self.recommendations:
            rec_df = pd.DataFrame(self.recommendations)
            rec_df.to_excel(writer, sheet_name='AI Recommendations', index=False)
        
        # Risk analysis sheet
        if 'risk_analysis' in content:
            risk_df = pd.DataFrame(content['risk_analysis'])
            risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
    
    def _style_excel_sheet(self, sheet):
        """Apply styling to Excel sheet"""
        # Header styling
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")
        
        # Apply header styling
        for cell in sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # Auto-adjust column widths
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Helper methods for generating report sections
    def _calculate_traffic_at_risk(self, pairs_df: pd.DataFrame) -> str:
        """Calculate estimated traffic at risk from cannibalization"""
        high_risk_count = len(pairs_df[pairs_df['risk_category'] == 'High'])
        
        # This is a simplified estimation
        if high_risk_count > 20:
            return "Significant (20+ high-risk issues)"
        elif high_risk_count > 10:
            return "Moderate (10-20 high-risk issues)"
        elif high_risk_count > 5:
            return "Low-Moderate (5-10 high-risk issues)"
        else:
            return "Low (<5 high-risk issues)"
    
    def _identify_top_opportunities(self, pairs_df: pd.DataFrame) -> List[Dict]:
        """Identify top optimization opportunities"""
        # Sort by risk score and get top opportunities
        top_pairs = pairs_df.nlargest(10, 'risk_score')
        
        opportunities = []
        for _, row in top_pairs.iterrows():
            opp = {
                'url1': row['url1'],
                'url2': row['url2'],
                'risk_score': f"{row['risk_score']:.2%}",
                'primary_issue': self._describe_primary_issue(row),
                'recommended_action': self._get_recommended_action(row)
            }
            opportunities.append(opp)
        
        return opportunities
    
    def _describe_primary_issue(self, row: pd.Series) -> str:
        """Describe the primary cannibalization issue"""
        if row['title_similarity'] > 0.9:
            return "Nearly identical titles"
        elif row.get('semantic_similarity', 0) > 0.8:
            return "Highly similar content"
        elif row.get('serp_overlap', 0) > 0.7:
            return "Competing in search results"
        else:
            return "Multiple similarity signals"
    
    def _get_recommended_action(self, row: pd.Series) -> str:
        """Get recommended action for a cannibalization pair"""
        # Find matching recommendation
        for rec in self.recommendations:
            if rec.get('pair_index') == row.name:
                return rec.get('recommended_action', 'Review and optimize')
        
        # Default recommendation based on similarity
        if row['risk_score'] > 0.8:
            return "Consolidate content"
        elif row['risk_score'] > 0.6:
            return "Differentiate content"
        else:
            return "Monitor and optimize"
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings for executive summary"""
        findings = []
        
        # Finding 1: Scale of issue
        total_pairs = self.analysis_results['total_pairs']
        high_risk = self.analysis_results['high_risk_count']
        
        if high_risk > 20:
            findings.append(f"Critical: {high_risk} high-risk cannibalization issues detected requiring immediate attention")
        elif high_risk > 10:
            findings.append(f"Significant: {high_risk} high-risk cannibalization issues impacting SEO performance")
        else:
            findings.append(f"Moderate: {high_risk} high-risk issues identified for optimization")
        
        # Finding 2: Most common issue type
        pairs_df = pd.DataFrame(self.analysis_results['pairs'])
        if not pairs_df.empty:
            # Identify most common issue
            title_issues = len(pairs_df[pairs_df['title_similarity'] > 0.8])
            content_issues = len(pairs_df[pairs_df.get('semantic_similarity', 0) > 0.8])
            
            if title_issues > content_issues:
                findings.append(f"{title_issues} pages have highly similar titles causing ranking confusion")
            else:
                findings.append(f"{content_issues} pages have overlapping content themes")
        
        # Finding 3: SERP impact if available
        if self.serp_data and self.serp_data.get('summary'):
            serp_summary = self.serp_data['summary']
            if serp_summary.get('top_10_overlaps', 0) > 0:
                findings.append(f"{serp_summary['top_10_overlaps']} keyword queries show multiple pages competing in top 10 results")
        
        return findings
    
    def _generate_executive_recommendations(self) -> List[str]:
        """Generate high-level recommendations for executives"""
        recommendations = []
        
        high_risk = self.analysis_results['high_risk_count']
        
        if high_risk > 20:
            recommendations.extend([
                "Implement immediate content consolidation strategy for highest-risk pages",
                "Establish content governance to prevent future cannibalization",
                "Allocate resources for comprehensive content audit and optimization"
            ])
        elif high_risk > 10:
            recommendations.extend([
                "Prioritize consolidation of top 10 high-risk page pairs",
                "Review and update content strategy to ensure clear differentiation",
                "Implement regular cannibalization monitoring"
            ])
        else:
            recommendations.extend([
                "Address high-risk cannibalization issues in next content update cycle",
                "Enhance content planning process to avoid overlap",
                "Schedule quarterly cannibalization reviews"
            ])
        
        return recommendations
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj