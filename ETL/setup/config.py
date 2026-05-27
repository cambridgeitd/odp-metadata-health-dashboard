"""
Configuration Management for Cambridge ODP Evaluation Pipeline

This module handles:
- Loading environment variables from .env file
- Providing LLM client abstraction (supports multiple providers)
- Validating configuration on import

Usage:
    from config import get_llm_client, get_config

    llm = get_llm_client()
    response = llm.evaluate(prompt)
"""

import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load .env file from project root (two levels up from this script)
# config.py location: dd4g-cambridge-meta-health/ETL/setup/config.py
# .env location: dd4g-cambridge-meta-health/.env
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)


def _resolve_llm_provider() -> str:
    """
    Resolve the active LLM provider from environment configuration.

    If LLM_PROVIDER is unset, fall back to the first configured provider key.
    This keeps local .env usage working while allowing CI environments to
    inject secrets directly without requiring a checked-in .env file.
    """
    configured_provider = os.getenv('LLM_PROVIDER')
    if configured_provider:
        return configured_provider

    provider_keys = (
        ('openai', os.getenv('OPENAI_API_KEY')),
        ('anthropic', os.getenv('ANTHROPIC_API_KEY')),
        ('gemini', os.getenv('GEMINI_API_KEY')),
        ('huggingface', os.getenv('HF_TOKEN')),
    )

    for provider_name, api_key in provider_keys:
        if api_key:
            return provider_name

    return 'gemini'

# ────────────────────────────────────────────────────────────
# CONFIGURATION GETTERS
# ────────────────────────────────────────────────────────────

def get_config() -> Dict[str, Any]:
    """
    Get all configuration values from environment.

    Returns:
        Dictionary with configuration keys including:
        - llm_provider: Which LLM service to use ('gemini', 'openai',
          'anthropic', or 'huggingface')
        - API keys or tokens for each provider
        - Model names/IDs
        - LLM parameters (temperature, max_tokens)
    """
    return {
        'llm_provider': _resolve_llm_provider(),
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
        'hf_token': os.getenv('HF_TOKEN'),
        'gemini_model': os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4'),
        'anthropic_model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
        'hf_model': os.getenv('HF_MODEL', 'meta-llama/Meta-Llama-3-8B-Instruct'),
        'temperature': float(os.getenv('LLM_TEMPERATURE', '0.3')),
        'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1000')),
    }

# ────────────────────────────────────────────────────────────
# LLM CLIENT ABSTRACTION
# ───────────────────────────────────────────────────────────

class BaseLLMClient:
    """
    Base class for LLM clients.

    All LLM provider implementations should inherit from this class
    and implement the evaluate() method.
    """

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Send evaluation prompt to LLM and parse response.

        Args:
            prompt: The evaluation prompt (formatted by evaluate.py)

        Returns:
            Dictionary with evaluation scores:
            {
                'ai_description_score': float (0.0-1.0),
                'ai_tag_relevance_score': float (0.0-1.0),
                'ai_category_fit_score': float (0.0-1.0),
                'ai_suggestions': str
            }

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclass must implement evaluate()")


class OpenAIClient(BaseLLMClient):
    """
    OpenAI GPT client for dataset metadata evaluation.

    This client uses OpenAI's API to evaluate dataset quality.
    Developers should implement the evaluate() method to:
    1. Call the OpenAI API with the provided prompt
    2. Parse the JSON response
    3. Return the structured evaluation result
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., 'gpt-4')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens
        """
        # TODO: Implement OpenAI client initialization
        #
        # IMPLEMENTATION GUIDE:
        # 1. Import the openai library:
        #    from openai import OpenAI
        #
        # 2. Create client instance:
        #    self.client = OpenAI(api_key=api_key)
        #
        # 3. Store model parameters:
        #    self.model = model
        #    self.temperature = temperature
        #    self.max_tokens = max_tokens
        #
        # PLACEHOLDER IMPLEMENTATION (stores parameters for now):

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"OpenAI client initialized (model: {model})")

    def call_llm(self, prompt: str, retries: int = 6) -> str:
        """
        Call OpenAI API with retry logic and exponential backoff.

        Args:
            prompt: The prompt to send to OpenAI
            retries: Number of retry attempts on failure

        Returns:
            Response text from OpenAI

        Raises:
            Exception: If all retries are exhausted
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a metadata quality evaluator. Respond only in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                wait = 20 * (2 ** attempt)  # 20s, 40s, 80s, 160s, 320s, 640s
                logger.warning(f"OpenAI API error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    raise

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API with evaluation prompt.

        Args:
            prompt: The evaluation prompt

        Returns:
            Dictionary with evaluation scores
        """
        # TODO: Implement OpenAI API call
        #
        # IMPLEMENTATION GUIDE:
        # 1. Call the OpenAI ChatCompletion API:
        #    response = self.client.chat.completions.create(
        #        model=self.model,
        #        messages=[
        #            {"role": "system", "content": "You are a metadata quality evaluator."},
        #            {"role": "user", "content": prompt}
        #        ],
        #        temperature=self.temperature,
        #        max_tokens=self.max_tokens,
        #        response_format={"type": "json_object"}  # Request JSON response
        #    )
        #
        # 2. Extract the response content:
        #    content = response.choices[0].message.content
        #
        # 3. Parse the JSON response:
        #    import json
        #    result = json.loads(content)
        #
        # 4. Return the structured result (ensure keys match expected format):
        #    return {
        #        'ai_description_score': result['description_score'],
        #        'ai_tag_relevance_score': result['tag_score'],
        #        'ai_category_fit_score': result['category_score'],
        #        'ai_suggestions': result['suggestions']
        #    }
        #
        # PLACEHOLDER: Return dummy scores until implemented
        logger.debug("OpenAI evaluation called (not yet implemented)")
        return {
            'ai_description_score': 0.75,
            'ai_tag_relevance_score': 0.80,
            'ai_category_fit_score': 0.70,
            'ai_suggestions': "OpenAI evaluation not yet implemented. Replace this with actual LLM analysis."
        }


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude client for dataset metadata evaluation.

    This client uses Anthropic's API to evaluate dataset quality.
    Developers should implement the evaluate() method to:
    1. Call the Anthropic API with the provided prompt
    2. Parse the JSON response
    3. Return the structured evaluation result
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., 'claude-3-sonnet-20240229')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens
        """
        # TODO: Implement Anthropic client initialization
        #
        # IMPLEMENTATION GUIDE:
        # 1. Import the anthropic library:
        #    from anthropic import Anthropic
        #
        # 2. Create client instance:
        #    self.client = Anthropic(api_key=api_key)
        #
        # 3. Store model parameters:
        #    self.model = model
        #    self.temperature = temperature
        #    self.max_tokens = max_tokens
        #
        # PLACEHOLDER IMPLEMENTATION (stores parameters for now):

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Anthropic client initialized (model: {model})")

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Call Anthropic API with evaluation prompt.

        Args:
            prompt: The evaluation prompt

        Returns:
            Dictionary with evaluation scores
        """
        # TODO: Implement Anthropic API call
        #
        # IMPLEMENTATION GUIDE:
        # 1. Call the Anthropic Messages API:
        #    message = self.client.messages.create(
        #        model=self.model,
        #        max_tokens=self.max_tokens,
        #        temperature=self.temperature,
        #        messages=[
        #            {
        #                "role": "user",
        #                "content": prompt
        #            }
        #        ]
        #    )
        #
        # 2. Extract the response content:
        #    content = message.content[0].text
        #
        # 3. Parse the JSON response:
        #    import json
        #    result = json.loads(content)
        #
        # 4. Return the structured result (ensure keys match expected format):
        #    return {
        #        'ai_description_score': result['description_score'],
        #        'ai_tag_relevance_score': result['tag_score'],
        #        'ai_category_fit_score': result['category_score'],
        #        'ai_suggestions': result['suggestions']
        #    }
        #
        # PLACEHOLDER: Return dummy scores until implemented
        logger.debug("Anthropic evaluation called (not yet implemented)")
        return {
            'ai_description_score': 0.75,
            'ai_tag_relevance_score': 0.80,
            'ai_category_fit_score': 0.70,
            'ai_suggestions': "Anthropic evaluation not yet implemented. Replace this with actual LLM analysis."
        }


class GeminiClient(BaseLLMClient):
    """
    Google Gemini client for dataset metadata evaluation.

    This client uses Google's Gemini API to evaluate dataset quality.
    Implements the BaseLLMClient interface with retry logic and exponential backoff.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp",
                 temperature: float = 0.3, max_tokens: int = 1000):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key
            model: Model name (default: gemini-2.0-flash-exp)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Gemini client initialized (model: {model})")

    def call_llm(self, prompt: str, retries: int = 4) -> str:
        """
        Call Gemini API with retry logic and exponential backoff.

        Args:
            prompt: The prompt to send to Gemini
            retries: Number of retry attempts on failure

        Returns:
            Response text from Gemini

        Raises:
            Exception: If all retries are exhausted
        """
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                )
                return response.text.strip()
            except Exception as e:
                wait = 15 * (attempt + 1)  # Exponential backoff: 15s, 30s, 45s, 60s
                logger.warning(f"Gemini API error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    logger.error("Max retries exceeded for Gemini API")
                    raise

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini API with evaluation prompt.

        Note: This method is kept for backward compatibility with the BaseLLMClient
        interface, but the actual implementation will be in evaluate.py using
        call_llm() directly with custom prompts for each evaluation aspect.

        Args:
            prompt: The evaluation prompt

        Returns:
            Dictionary with evaluation scores
        """
        logger.debug("Gemini evaluation called via evaluate() method")
        # Placeholder - actual implementation will be in evaluate.py
        return {
            'ai_description_score': 0.75,
            'ai_tag_relevance_score': 0.80,
            'ai_category_fit_score': 0.70,
            'ai_suggestions': "Use call_llm() directly for custom prompts in evaluate.py"
        }


class HuggingFaceClient(BaseLLMClient):
    """
    Hugging Face Inference API client for dataset metadata evaluation.

    Uses the `huggingface_hub` InferenceClient with a cooldown between calls
    to avoid rate limits. Evaluations are expected to return a JSON string
    containing the same keys as other clients.
    """

    def __init__(self, api_key: str, model: str,
                 temperature: float = 0.3, max_tokens: int = 1000,
                 delay_seconds: int = 10):
        """
        Initialize Hugging Face client.

        Args:
            api_key: HF access token
            model: HF model ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
            temperature: Sampling temperature (ignored by some HF endpoints)
            max_tokens: Maximum tokens to request
            delay_seconds: Number of seconds to sleep between requests
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay_seconds = delay_seconds
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=api_key)
        logger.info(f"HuggingFace client initialized (model: {model})")

    def call_llm(self, prompt: str) -> str:
        """
        Wrapper that returns raw text response from the Hugging Face model.

        This is provided for compatibility with existing code (evaluate.py)
        which uses `llm_client.call_llm()` directly.  Internally this method
        simply invokes `self.client.text_generation` and applies the same
        rate-limit delay as the evaluate() method.
        """
        logger.debug("Sending prompt to HuggingFace model (raw call)")
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data evaluator. Respond only in JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            content = response.choices[0].message.content
        except Exception as e:
            logger.error(f"HuggingFace API call failed: {e}")
            content = ""
        # delay regardless of success to avoid rate limiting
        time.sleep(self.delay_seconds)
        return content

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Call the Hugging Face Inference API with the evaluation prompt.

        A delay of `delay_seconds` is enforced after each call to avoid hitting
        API rate limits.

        Args:
            prompt: The evaluation prompt

        Returns:
            Dictionary with evaluation scores
        """
        logger.debug("Sending prompt to HuggingFace model")
        try:
            response_text = self.call_llm(prompt)
            # try to parse JSON from the model output
            try:
                import json
                result = json.loads(response_text)
            except Exception:
                # if parsing fails, log and return fallbacks
                logger.warning("Failed to parse JSON from HuggingFace response")
                result = {}

            return {
                'ai_description_score': result.get('description_score', 0.0),
                'ai_tag_relevance_score': result.get('tag_score', 0.0),
                'ai_category_fit_score': result.get('category_score', 0.0),
                'ai_suggestions': result.get('suggestions', '')
            }
        except Exception as e:
            logger.error(f"HuggingFace evaluate() failed: {e}")
            return {
                'ai_description_score': 0.0,
                'ai_tag_relevance_score': 0.0,
                'ai_category_fit_score': 0.0,
                'ai_suggestions': ''
            }


def get_llm_client() -> BaseLLMClient:
    """
    Get configured LLM client based on environment settings.

    This factory function reads the LLM_PROVIDER environment variable
    and returns the appropriate client instance (Gemini, OpenAI, Anthropic,
    or HuggingFace).

    Returns:
        Configured LLM client instance

    Raises:
        ValueError: If provider is unsupported or API key is missing

    Example:
        >>> llm = get_llm_client()
        >>> result = llm.evaluate("Evaluate this dataset...")
    """
    config = get_config()
    provider = config['llm_provider'].lower()

    if provider == 'gemini':
        api_key = config['gemini_api_key']
        if not api_key or api_key == 'your-gemini-api-key-here':
            raise ValueError(
                "GEMINI_API_KEY not set in .env file. "
                "Copy .env.example to .env in project root and add your API key."
            )

        return GeminiClient(
            api_key=api_key,
            model=config['gemini_model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

    elif provider == 'openai':
        api_key = config['openai_api_key']
        if not api_key or api_key == 'sk-your-openai-api-key-here':
            raise ValueError(
                "OPENAI_API_KEY not set in .env file. "
                "Copy .env.example to .env in project root and add your API key."
            )

        return OpenAIClient(
            api_key=api_key,
            model=config['openai_model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

    elif provider == 'anthropic':
        api_key = config['anthropic_api_key']
        if not api_key or api_key == 'sk-ant-your-anthropic-api-key-here':
            raise ValueError(
                "ANTHROPIC_API_KEY not set in .env file. "
                "Copy .env.example to .env in project root and add your API key."
            )

        return AnthropicClient(
            api_key=api_key,
            model=config['anthropic_model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

    elif provider == 'huggingface' or provider == 'hf':
        api_key = config['hf_token']
        if not api_key or api_key == 'your-hf-token-here':
            raise ValueError(
                "HF_TOKEN not set in .env file. "
                "Copy .env.example to .env in project root and add your Hugging Face token."
            )

        return HuggingFaceClient(
            api_key=api_key,
            model=config['hf_model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Must be 'gemini', 'openai', 'anthropic', or 'huggingface'."
        )


# ────────────────────────────────────────────────────────────
# CONFIGURATION VALIDATION
# ────────────────────────────────────────────────────────────

# Validate configuration on module import
try:
    config = get_config()
    logger.info(f"Configuration loaded (LLM provider: {config['llm_provider']})")
except Exception as e:
    logger.warning(f"Configuration validation failed: {e}")
    logger.warning("Make sure to create .env file from .env.example in project root before running evaluations")
