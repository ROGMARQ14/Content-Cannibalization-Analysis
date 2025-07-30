# modules/ai_providers/openai_provider.py
from openai import AsyncOpenAI
import json
from typing import Dict, Any
import logging
from .base_provider import BaseAIProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Model-specific settings
        self.model_settings = {
            'gpt-4o': {'temperature': 0.3, 'max_tokens': 4000},
            'gpt-4o-mini': {'temperature': 0.3, 'max_tokens': 4000},
            'o1-preview': {'temperature': 0.1, 'max_tokens': 8000},
            'o1-mini': {'temperature': 0.1, 'max_tokens': 4000}
        }
    
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Send a single prompt to OpenAI"""
        try:
            settings = self.model_settings.get(self.model, {'temperature': 0.3, 'max_tokens': 4000})
            
            # O1 models don't support temperature
            if self.model.startswith('o1'):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings['max_tokens']
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', settings['temperature']),
                    max_tokens=kwargs.get('max_tokens', settings['max_tokens'])
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def analyze_json(self, prompt: str, **kwargs) -> Dict:
        """Send a prompt and expect JSON response"""
        try:
            settings = self.model_settings.get(self.model, {'temperature': 0.3, 'max_tokens': 4000})
            
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no additional text or markdown."
            
            # O1 models don't support temperature or response_format
            if self.model.startswith('o1'):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": json_prompt}],
                    max_tokens=settings['max_tokens']
                )
            else:
                # Use response_format for compatible models
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', settings['temperature']),
                    max_tokens=kwargs.get('max_tokens', settings['max_tokens']),
                    response_format={"type": "json_object"}
                )
            
            content = response.choices[0].message.content
            
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
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken"""
        try:
            import tiktoken
            
            # Get the appropriate encoding for the model
            if self.model.startswith('gpt-4'):
                encoding = tiktoken.encoding_for_model('gpt-4')
            else:
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            
            return len(encoding.encode(text))
        except Exception:
            # Fallback to character-based estimation
            return len(text) // 4