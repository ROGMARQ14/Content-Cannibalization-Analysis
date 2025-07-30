# modules/ai_providers/gemini_provider.py
import google.generativeai as genai
import json
from typing import Dict, Any
import logging
import asyncio
from .base_provider import BaseAIProvider

logger = logging.getLogger(__name__)

class GeminiProvider(BaseAIProvider):
    """Google Gemini API provider implementation"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        
        # Model-specific settings
        self.model_settings = {
            'gemini-1.5-pro': {
                'temperature': 0.3,
                'max_output_tokens': 8192,
                'candidate_count': 1
            },
            'gemini-1.5-flash': {
                'temperature': 0.3,
                'max_output_tokens': 8192,
                'candidate_count': 1
            },
            'gemini-2.0-flash-thinking-exp': {
                'temperature': 0.1,
                'max_output_tokens': 8192,
                'candidate_count': 1
            }
        }
        
        # Initialize the model
        self.model_instance = genai.GenerativeModel(model)
    
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Send a single prompt to Gemini"""
        try:
            settings = self.model_settings.get(self.model, {
                'temperature': 0.3,
                'max_output_tokens': 8192
            })
            
            # Create generation config
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', settings['temperature']),
                max_output_tokens=kwargs.get('max_tokens', settings['max_output_tokens']),
                candidate_count=settings.get('candidate_count', 1)
            )
            
            # Run in executor since Gemini SDK is not async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def analyze_json(self, prompt: str, **kwargs) -> Dict:
        """Send a prompt and expect JSON response"""
        try:
            settings = self.model_settings.get(self.model, {
                'temperature': 0.3,
                'max_output_tokens': 8192
            })
            
            # Add JSON instruction to prompt
            json_prompt = f"""{prompt}

            Return your response as valid JSON only. Do not include any markdown formatting, 
            code blocks, or additional text. The response should be parseable JSON."""
            
            # Create generation config
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', settings['temperature']),
                max_output_tokens=kwargs.get('max_tokens', settings['max_output_tokens']),
                candidate_count=settings.get('candidate_count', 1),
                response_mime_type="application/json"  # Gemini supports JSON mode
            )
            
            # Run in executor since Gemini SDK is not async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model_instance.generate_content(
                    json_prompt,
                    generation_config=generation_config
                )
            )
            
            content = response.text
            
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
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for Gemini models"""
        try:
            # Use Gemini's count_tokens method
            return self.model_instance.count_tokens(text).total_tokens
        except Exception:
            # Fallback to character-based estimation
            # Gemini tends to use slightly fewer tokens than GPT models
            return len(text) // 4