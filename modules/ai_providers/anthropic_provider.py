# modules/ai_providers/anthropic_provider.py
from anthropic import AsyncAnthropic
import json
from typing import Dict, Any
import logging
from .base_provider import BaseAIProvider

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseAIProvider):
    """Anthropic API provider implementation"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = AsyncAnthropic(api_key=api_key)
        
        # Model-specific settings
        self.model_settings = {
            'claude-3-5-sonnet-20241022': {'temperature': 0.3, 'max_tokens': 4000},
            'claude-3-opus-20240229': {'temperature': 0.3, 'max_tokens': 4000},
            'claude-3-haiku-20240307': {'temperature': 0.3, 'max_tokens': 4000}
        }
    
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Send a single prompt to Anthropic"""
        try:
            settings = self.model_settings.get(self.model, {'temperature': 0.3, 'max_tokens': 4000})
            
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', settings['temperature']),
                max_tokens=kwargs.get('max_tokens', settings['max_tokens'])
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def analyze_json(self, prompt: str, **kwargs) -> Dict:
        """Send a prompt and expect JSON response"""
        try:
            settings = self.model_settings.get(self.model, {'temperature': 0.3, 'max_tokens': 4000})
            
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no additional text or markdown."
            
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": json_prompt}],
                temperature=kwargs.get('temperature', settings['temperature']),
                max_tokens=kwargs.get('max_tokens', settings['max_tokens'])
            )
            
            content = response.content[0].text
            
            # Clean up response if needed
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise Exception(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for Anthropic models"""
        # Anthropic uses a similar tokenization to OpenAI
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4