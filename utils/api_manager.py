# utils/api_manager.py
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class APIManager:
    """Manages API keys from Streamlit secrets and provides provider information"""
    
    SUPPORTED_PROVIDERS = {
        'openai': {
            'models': {
                'gpt-4o': {'name': 'GPT-4o', 'context': 128000, 'best_for': 'Best overall performance'},
                'gpt-4o-mini': {'name': 'GPT-4o Mini', 'context': 128000, 'best_for': 'Fast, cost-effective analysis'},
                'gpt-4-turbo': {'name': 'GPT-4 Turbo', 'context': 128000, 'best_for': 'High quality analysis'},
                'gpt-3.5-turbo': {'name': 'GPT-3.5 Turbo', 'context': 16385, 'best_for': 'Fast, affordable option'}
            },
            'key_name': 'OPENAI_API_KEY',
            'display_name': 'OpenAI'
        },
        'anthropic': {
            'models': {
                'claude-3-5-sonnet-20241022': {'name': 'Claude 3.5 Sonnet', 'context': 200000, 'best_for': 'Balanced performance'},
                'claude-3-opus-20240229': {'name': 'Claude 3 Opus', 'context': 200000, 'best_for': 'Complex analysis'},
                'claude-3-haiku-20240307': {'name': 'Claude 3 Haiku', 'context': 200000, 'best_for': 'Fast, efficient processing'}
            },
            'key_name': 'ANTHROPIC_API_KEY',
            'display_name': 'Anthropic'
        },
        'gemini': {
            'models': {
                'gemini-1.5-pro': {'name': 'Gemini 1.5 Pro', 'context': 1000000, 'best_for': 'Large context analysis'},
                'gemini-1.5-flash': {'name': 'Gemini 1.5 Flash', 'context': 1000000, 'best_for': 'Fast processing'},
                'gemini-2.0-flash-thinking-exp': {'name': 'Gemini 2.0 Flash (Experimental)', 'context': 32000, 'best_for': 'Advanced reasoning'}
            },
            'key_name': 'GEMINI_API_KEY',
            'display_name': 'Google Gemini'
        }
    }
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict]:
        """Returns available AI providers and their models based on stored secrets"""
        available = {}
        
        try:
            for provider, config in APIManager.SUPPORTED_PROVIDERS.items():
                # Check if the key exists AND is not empty
                if config['key_name'] in st.secrets:
                    key_value = st.secrets[config['key_name']]
                    # Ensure the key is not empty or just whitespace
                    if key_value and str(key_value).strip():
                        available[provider] = {
                            'models': config['models'],
                            'display_name': config['display_name']
                        }
                        logger.info(f"Found valid API key for {provider}")
                    else:
                        logger.warning(f"Empty API key for {provider}")
                else:
                    logger.info(f"No API key found for {provider}")
        except Exception as e:
            logger.error(f"Error checking available providers: {e}")
            
        return available
    
    @staticmethod
    def get_api_key(provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        try:
            config = APIManager.SUPPORTED_PROVIDERS.get(provider)
            if config and config['key_name'] in st.secrets:
                key_value = st.secrets[config['key_name']]
                # Return key only if it's not empty
                if key_value and str(key_value).strip():
                    return str(key_value).strip()
        except Exception as e:
            logger.error(f"Error retrieving API key for {provider}: {e}")
        return None
    
    @staticmethod
    def has_serper_api() -> bool:
        """Check if Serper API key is available"""
        return 'SERPER_API_KEY' in st.secrets
    
    @staticmethod
    def get_serper_api_key() -> Optional[str]:
        """Get Serper API key"""
        try:
            if 'SERPER_API_KEY' in st.secrets:
                return st.secrets['SERPER_API_KEY']
        except Exception:
            pass
        return None
    
    @staticmethod
    def has_gsc_oauth() -> bool:
        """Check if Google Search Console OAuth credentials are available"""
        return 'gsc_oauth_config' in st.secrets
    
    @staticmethod
    def get_model_info(provider: str, model: str) -> Dict:
        """Get information about a specific model"""
        provider_config = APIManager.SUPPORTED_PROVIDERS.get(provider, {})
        models = provider_config.get('models', {})
        return models.get(model, {})
    
    @staticmethod
    def validate_apis() -> Tuple[bool, List[str]]:
        """Validate all configured APIs and return status"""
        issues = []
        
        # Check for at least one AI provider
        available_providers = APIManager.get_available_providers()
        if not available_providers:
            issues.append("No AI provider API keys found. Please add at least one (OpenAI, Anthropic, or Gemini) in Streamlit secrets.")
        
        # Check optional APIs
        if not APIManager.has_serper_api():
            issues.append("Serper API key not found. SERP overlap analysis will be disabled.")
        
        if not APIManager.has_gsc_oauth():
            issues.append("Google Search Console OAuth not configured. Direct GSC integration will be disabled.")
        
        return len(issues) == 0, issues
