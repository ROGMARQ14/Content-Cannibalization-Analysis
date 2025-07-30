# 🔍 Enhanced Content Cannibalization Analyzer

An AI-powered tool for identifying and resolving content cannibalization issues at scale. This advanced analyzer helps SEO professionals detect when multiple pages compete for the same keywords, provides actionable recommendations, and generates comprehensive reports.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Key Features

### 🤖 AI-Powered Analysis
- **Multi-Provider Support**: Choose between OpenAI, Anthropic, or Google Gemini
- **Intent Classification**: Automatically identifies content intent (Informational, Commercial, Transactional, Navigational)
- **Smart Recommendations**: AI generates specific, actionable fixes for each issue
- **Executive Summaries**: Professional summaries for stakeholders

### 📊 Advanced Detection Methods
- **Flexible Similarity Analysis**:
  - Metadata comparison (titles, H1s, descriptions)
  - Screaming Frog embeddings support
  - Full content extraction and analysis
- **ML-Based Risk Scoring**: Dynamic scoring that adapts to your content patterns
- **SERP Overlap Detection**: Identifies actual search result competition via Serper API
- **Smart Column Detection**: Automatically handles different export formats

### 📈 Comprehensive Reporting
- **Multiple Report Types**:
  - Executive Summary
  - Detailed Analysis
  - Action Plan
  - Technical Report
- **Export Formats**: Excel (styled), CSV, JSON
- **No App Reset**: Export reports without losing your analysis

### 🔧 Flexible Data Sources
- **SEO Crawler Support**: Screaming Frog, Sitebulb, DeepCrawl, and more
- **Google Search Console**: Direct OAuth integration or CSV upload
- **Content Analysis**: Extract and analyze actual page content (excluding headers/footers)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Streamlit account (for secrets management)
- At least one AI provider API key (OpenAI, Anthropic, or Google Gemini)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ROGMARQ14/Enhanced-Content-Cannibalization-Analyzer.git
cd Enhanced-Content-Cannibalization-Analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**

Create `.streamlit/secrets.toml` in your project root:

```toml
# AI Providers (add at least one)
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GEMINI_API_KEY = "AI..."

# Optional: SERP Analysis
SERPER_API_KEY = "your-serper-key"

# Optional: Google Search Console OAuth
[gsc_oauth_config]
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
redirect_uri = "http://localhost:8501"
```

4. **Run the application**
```bash
streamlit run app.py
```

## 📋 Usage Guide

### 1. Prepare Your Data

#### Internal SEO Data (Required)
Export from your SEO crawler with these columns:
- URL/Address
- Title/Title 1
- H1/H1-1
- Meta Description

#### GSC Performance Data (Required)
- **Option 1**: Export from Google Search Console (no row limit)
- **Option 2**: Connect directly via OAuth

#### Content/Embeddings (Optional)
- **Option 1**: Use Screaming Frog embeddings export
- **Option 2**: Let the tool fetch and analyze page content
- **Option 3**: Provide pre-extracted content

### 2. Configure Analysis

1. **Select AI Provider**: Choose your preferred AI model
2. **Adjust Weights**: Customize similarity scoring weights
3. **Set Thresholds**: Define risk levels (High >70%, Medium 40-70%)
4. **Content Settings**: Configure extraction if analyzing page content

### 3. Run Analysis

1. Upload your data files
2. Select enhancement options (embeddings or content analysis)
3. Enable SERP analysis (if Serper API configured)
4. Click "Run Cannibalization Analysis"

### 4. Review Results

- **Analysis Tab**: View risk distribution and top issues
- **AI Insights**: Get AI-powered recommendations
- **Reports**: Generate and export comprehensive reports

## 🛠️ Advanced Configuration

### Content Extraction Settings

The tool can extract actual page content while excluding:
- Headers and navigation
- Footers and copyright
- Sidebars and widgets
- Advertisements and popups

Configure extraction method:
- **Smart**: Automatically tries multiple methods
- **Trafilatura**: Best for articles and blogs
- **Readability**: General-purpose extraction
- **Custom Rules**: Pattern-based extraction

### Custom Column Mapping

The tool automatically detects column variations:
- "URL" → "Address", "Landing Page"
- "Query" → "Keyword", "Search Term"
- "Title" → "Title 1", "Page Title"

## 📊 Understanding the Metrics

### Similarity Scores
- **Title Similarity**: How similar page titles are (critical for SEO)
- **H1 Similarity**: Main heading overlap
- **Semantic Similarity**: Overall content theme similarity
- **Keyword Overlap**: Shared GSC keywords
- **SERP Overlap**: Actual search result competition

### Risk Categories
- **High Risk (>70%)**: Immediate action required
- **Medium Risk (40-70%)**: Should be addressed soon
- **Low Risk (<40%)**: Monitor but not critical

### ML Risk Scoring
The system uses machine learning to:
- Adapt weights based on patterns
- Consider multiple factors simultaneously
- Provide more accurate risk assessment

## 🔧 Troubleshooting

### Common Issues

**"Missing required columns"**
- The tool shows which columns it detected
- Check the column mapping preview
- Ensure CSV is UTF-8 encoded

**"No AI providers configured"**
- Add at least one API key to secrets.toml
- Restart Streamlit after adding keys

**"Content extraction failed"**
- Check if URLs are accessible
- Adjust minimum content length threshold
- Try different extraction methods

**Performance Issues**
- Start with top pages for large sites
- Use embeddings for faster analysis
- Limit SERP keywords to most important

### API Rate Limits
- OpenAI: Varies by plan
- Anthropic: Varies by plan
- Gemini: 60 requests/minute
- Serper: 2,500 searches/month (free tier)

## 📈 Best Practices

1. **Data Quality**
   - Ensure URLs are properly formatted
   - Remove development/staging URLs
   - Focus on indexable pages (200 status)

2. **Analysis Strategy**
   - Start with high-traffic pages
   - Run quarterly cannibalization audits
   - Track improvements after fixes

3. **Content Fixes**
   - Consolidate truly duplicate content
   - Differentiate similar pages with unique angles
   - Update internal linking to support primary pages

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI providers: [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google](https://ai.google.dev/)
- SERP data: [Serper](https://serper.dev/)
- Content extraction: [Trafilatura](https://github.com/adbar/trafilatura)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

---

**Note**: This tool requires at least one AI provider API key to function. Ensure you have the necessary API credits for your usage volume.