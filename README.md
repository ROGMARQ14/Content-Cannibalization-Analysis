# 🔍 Content Cannibalization Analyzer v2.0

A **competition-first**, AI-powered tool for identifying and resolving content cannibalization issues at scale. This advanced analyzer helps SEO professionals detect when multiple pages compete for the same keywords, provides actionable recommendations, and generates comprehensive reports.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🆕 What's New in v2.0

### 🎯 Competition-First Detection
- **Keyword Competition Analysis**: Detects pages competing for the same search queries using GSC data
- **5-10x More Issues Detected**: Catches cannibalization that similarity-only methods miss
- **Multi-Method Detection**: Combined, Competition-First, or Similarity-First approaches

### 🔧 Enhanced URL Matching
- **Smart URL Normalization**: Handles protocol differences, parameters, and variations
- **Fuzzy Matching**: Better embeddings matching for Screaming Frog data
- **Debug Mode**: See exactly why URLs match or don't match

### 📊 Improved Detection Sensitivity
- **Lower Default Thresholds**: 20% similarity (was 30%) for better detection
- **Performance Filtering OFF by Default**: Catches cannibalized pages with suppressed metrics
- **Configurable Risk Levels**: Adjust thresholds to your needs

## 🌟 Key Features

### 🎯 Multiple Detection Methods
- **Combined (Recommended)**: Uses all detection methods for comprehensive analysis
- **Competition-First**: Focuses on keyword and SERP competition
- **Similarity-First**: Traditional content similarity approach
- **Custom Configuration**: Fine-tune detection parameters

### 🤖 AI-Powered Analysis
- **Multi-Provider Support**: OpenAI, Anthropic, or Google Gemini
- **Intent Classification**: Automatically identifies content intent
- **Smart Recommendations**: AI generates specific, actionable fixes
- **Confidence Scoring**: Know which issues are most likely true positives

### 📊 Advanced Detection Capabilities
- **Keyword Competition Detection**: Find URLs competing for the same queries
- **Content Similarity Analysis**: Compare titles, H1s, meta descriptions, and content
- **SERP Overlap Detection**: Identify actual search result competition
- **ML Risk Scoring**: Dynamic scoring that adapts to patterns
- **Embeddings Support**: Use Screaming Frog embeddings for semantic analysis

### 📈 Comprehensive Reporting
- **Detection Insights**: See which method found each issue
- **Priority Ranking**: Focus on high-impact cannibalization first
- **Traffic Impact Estimates**: Understand potential gains
- **Export Formats**: Excel, CSV, JSON

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- At least one AI provider API key (OpenAI, Anthropic, or Google Gemini)
- Google Search Console data export
- SEO crawler export (Screaming Frog, Sitebulb, etc.)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/enhanced-content-cannibalization-analyzer.git
cd enhanced-content-cannibalization-analyzer
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
```

4. **Run the application**
```bash
streamlit run app.py
```

## 📋 Usage Guide

### 1. Prepare Your Data

#### GSC Performance Data (Required)
Export from Google Search Console with:
- Landing Page (URL)
- Query
- Clicks
- Impressions
- Position

#### SEO Crawler Data (Required)
Export from Screaming Frog or similar with:
- URL/Address
- Title/Title 1
- H1/H1-1
- Meta Description

#### Embeddings Data (Optional but Recommended)
- Screaming Frog embeddings export for enhanced similarity detection

### 2. Configure Detection

1. **Choose Detection Method**:
   - **Combined**: Best for comprehensive analysis
   - **Competition-First**: Best for finding active cannibalization
   - **Similarity-First**: Best for finding potential issues

2. **Adjust Settings**:
   - **Min Shared Keywords**: 1-2 for maximum detection
   - **Min Similarity**: 0.15-0.20 for balanced results
   - **Performance Filter**: Keep OFF to catch all issues

3. **Enable Debug Mode**: See detailed analysis information

### 3. Run Analysis

1. Upload your data files
2. Click "Run Cannibalization Analysis"
3. Review results in the Detection Results tab

### 4. Review Results

- **Detection Results**: See all issues with competition scores
- **AI Insights**: Get actionable recommendations
- **Reports**: Export comprehensive analysis

## 🎯 Detection Methods Explained

### 1. Keyword Competition Detection 🏆
**How it works**: Analyzes GSC data to find URLs ranking for the same keywords

**Detects**:
- Pages splitting traffic for the same queries
- Ranking fluctuations due to Google confusion
- Actual SERP competition

**Best for**: Finding active cannibalization impacting traffic

### 2. Content Similarity Detection 📄
**How it works**: Compares page elements and content

**Detects**:
- Similar titles and headings
- Overlapping meta descriptions
- Content theme similarities

**Best for**: Finding potential future cannibalization

### 3. Combined Detection 🔄
**How it works**: Uses both methods plus ML scoring

**Detects**:
- All types of cannibalization
- Provides confidence scores
- Reduces false positives

**Best for**: Comprehensive analysis

## 📊 Understanding the Metrics

### Competition Score
- **0-30%**: Low risk - Monitor
- **30-50%**: Medium risk - Optimize
- **50-70%**: High risk - Take action
- **70-100%**: Critical - Immediate action needed

### Detection Sources
- **keyword_competition**: Found through GSC query analysis
- **content_similarity**: Found through content comparison
- **both**: Detected by multiple methods (high confidence)

### Key Metrics
- **Shared Queries Count**: Number of keywords both URLs rank for
- **Traffic Opportunity**: Estimated traffic gain from fixing
- **Confidence Score**: Likelihood of true cannibalization

## 🛠️ Troubleshooting

### "No issues detected"
1. **Lower thresholds**:
   - Set Min Shared Keywords to 1
   - Set Min Similarity to 0.15
2. **Disable performance filtering**
3. **Check URL format consistency** between files
4. **Enable debug mode** to see filtering details

### "URL matching errors"
1. The new URL normalizer handles most cases automatically
2. Check debug mode output for normalization details
3. Ensure URLs are complete (include protocol)

### "Too many results"
1. Increase Min Shared Keywords to 3-5
2. Increase Min Similarity to 0.30
3. Enable performance filtering
4. Focus on Competition-First method

## 📈 Best Practices

### For Maximum Detection
- Detection Method: Combined
- Min Shared Keywords: 1-2
- Min Similarity: 0.15-0.20
- Performance Filter: OFF
- Debug Mode: ON

### For Large Sites (1000+ URLs)
- Use Competition-First method
- Enable performance filtering
- Process high-traffic sections first
- Increase thresholds slightly

### For Actionable Results
- Focus on High/Critical risk issues
- Review AI recommendations
- Check traffic opportunity scores
- Validate with actual SERP analysis

## 🔧 Advanced Configuration

### Custom Detection Pipeline
Configure your own detection approach:
- Adjust individual method weights
- Set custom risk thresholds
- Enable/disable specific analyzers

### API Rate Limit Management
- Batch processing for large sites
- Configurable delays between API calls
- Progress tracking for long analyses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI providers: [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google](https://ai.google.dev/)
- SERP data: [Serper](https://serper.dev/)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Review the troubleshooting section
- Enable debug mode for detailed diagnostics

---

**v2.0 Release Notes**: Complete rewrite with competition-first detection, improved URL matching, and 5-10x better detection rates. Catches many more real cannibalization issues that impact SEO performance.
