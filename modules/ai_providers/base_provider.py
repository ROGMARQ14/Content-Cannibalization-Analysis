# modules/ai_providers/base_provider.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class BaseAIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider_name = self.__class__.__name__
        
    @abstractmethod
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Send a single prompt to the AI model"""
        pass
    
    @abstractmethod
    async def analyze_json(self, prompt: str, **kwargs) -> Dict:
        """Send a prompt and expect JSON response"""
        pass
    
    async def batch_analyze(self, prompts: List[str], max_concurrent: int = 1, **kwargs) -> List[str]:
        """Analyze multiple prompts in sequential batches to conserve memory"""
        results = []
        
        # Process sequentially to avoid memory issues
        for i, prompt in enumerate(prompts):
            try:
                result = await self.analyze(prompt, **kwargs)
                results.append(result)
                
                # Add small delay between requests to avoid rate limits
                if i < len(prompts) - 1:
                    await asyncio.sleep(0.3)
                    
            except Exception as e:
                logger.error(f"Error in prompt {i}: {e}")
                results.append("Unable to analyze")
        
        return results
    
    async def batch_analyze_json(self, prompts: List[str], max_concurrent: int = 1, **kwargs) -> List[Dict]:
        """Analyze multiple prompts expecting JSON responses - sequential processing"""
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = await self.analyze_json(prompt, **kwargs)
                results.append(result)
                
                # Add small delay between requests
                if i < len(prompts) - 1:
                    await asyncio.sleep(0.3)
                    
            except Exception as e:
                logger.error(f"Error in prompt {i}: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def create_intent_prompt(self, title: str, h1: str, meta_description: str) -> str:
        """Create a prompt for intent classification"""
        return f"""
        Analyze the search intent of this webpage based on its metadata:
        
        Title: {title}
        H1: {h1}
        Meta Description: {meta_description}
        
        Classify the primary intent as one of:
        - Informational: Seeking information, how-to guides, explanations
        - Commercial: Researching products/services, comparisons, reviews
        - Transactional: Ready to take action, buy, sign up, download
        - Navigational: Looking for a specific page or brand
        
        Respond with ONLY the intent category name.
        """
    
    def create_recommendation_prompt(self, cannibalization_data: Dict) -> str:
        """Create a prompt for generating recommendations"""
        return f"""
        Analyze this content cannibalization issue and provide specific recommendations:
        
        URL 1: {cannibalization_data['url1']}
        - Title: {cannibalization_data['title1']}
        - Intent: {cannibalization_data['intent1']}
        - Top Keywords: {', '.join(cannibalization_data['keywords1'][:5])}
        - Avg Position: {cannibalization_data['avg_position1']}
        - Clicks: {cannibalization_data['clicks1']}
        
        URL 2: {cannibalization_data['url2']}
        - Title: {cannibalization_data['title2']}
        - Intent: {cannibalization_data['intent2']}
        - Top Keywords: {', '.join(cannibalization_data['keywords2'][:5])}
        - Avg Position: {cannibalization_data['avg_position2']}
        - Clicks: {cannibalization_data['clicks2']}
        
        Similarity Scores:
        - Title Similarity: {cannibalization_data['title_similarity']:.2%}
        - Semantic Similarity: {cannibalization_data['semantic_similarity']:.2%}
        - Keyword Overlap: {cannibalization_data['keyword_overlap']:.2%}
        
        Provide a JSON response with:
        {{
            "severity": "high|medium|low",
            "primary_issue": "brief description of the main problem",
            "recommended_action": "consolidate|differentiate|optimize|monitor",
            "implementation_steps": [
                "Step 1...",
                "Step 2...",
                "Step 3..."
            ],
            "expected_impact": "description of expected SEO impact",
            "priority_score": 1-10
        }}
        """
    
    def create_content_analysis_prompt(self, content_cluster: List[Dict]) -> str:
        """Create a prompt for content gap analysis"""
        return f"""
        Analyze this cluster of related content to identify gaps and opportunities:
        
        Content Cluster:
        {self._format_content_cluster(content_cluster)}
        
        Identify:
        1. Missing subtopics that should be covered
        2. Content consolidation opportunities
        3. Differentiation strategies for similar content
        4. New content ideas that complement existing pages
        
        Provide a JSON response with:
        {{
            "content_gaps": [
                {{
                    "topic": "missing topic",
                    "priority": "high|medium|low",
                    "rationale": "why this is important"
                }}
            ],
            "consolidation_opportunities": [
                {{
                    "urls": ["url1", "url2"],
                    "strategy": "how to consolidate",
                    "target_url": "which URL to keep"
                }}
            ],
            "differentiation_strategies": [
                {{
                    "url": "page url",
                    "current_focus": "what it covers now",
                    "recommended_focus": "what it should focus on",
                    "changes_needed": ["change 1", "change 2"]
                }}
            ],
            "new_content_ideas": [
                {{
                    "title": "proposed title",
                    "intent": "content intent",
                    "target_keywords": ["keyword1", "keyword2"],
                    "rationale": "why this content is needed"
                }}
            ]
        }}
        """
    
    def _format_content_cluster(self, cluster: List[Dict]) -> str:
        """Format content cluster for prompt"""
        formatted = []
        for item in cluster:
            formatted.append(f"""
            - URL: {item['url']}
              Title: {item['title']}
              Keywords: {', '.join(item['keywords'][:5])}
            """)
        return '\n'.join(formatted)
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for the provider"""
        pass
    
    def validate_response(self, response: str, expected_format: str = "text") -> bool:
        """Validate AI response format"""
        if expected_format == "json":
            try:
                import json
                json.loads(response)
                return True
            except:
                return False
        return bool(response and response.strip())
