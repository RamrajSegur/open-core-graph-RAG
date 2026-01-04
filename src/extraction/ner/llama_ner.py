"""LLaMA-based NER using Ollama for improved accuracy.

DEPRECATED: Use llm_provider.OllamaProvider instead.
This module is kept for backward compatibility only.
"""

import logging
from typing import List

from .entity_models import ExtractedEntity
from .llm_provider import OllamaProvider

logger = logging.getLogger(__name__)

# Check if Ollama is available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLaMaNER(OllamaProvider):
    """Use open-source LLaMA model for NER via Ollama.
    
    DEPRECATED: Use OllamaProvider from llm_provider.py instead.
    
    This class is maintained for backward compatibility.
    New code should use the modular LLMProvider interface.
    
    Requires: ollama service running on localhost:11434
    Install: brew install ollama && ollama pull llama2
    Run: ollama serve
    
    Example (old way - still works):
        ner = LLaMaNER(model_name="llama2")
        entities = ner.extract_entities("Steve Jobs founded Apple.")
        
    Example (new way - recommended):
        from src.extraction.ner.llm_provider import get_llm_provider
        ner = get_llm_provider("ollama", model_name="llama2")
        entities = ner.extract_entities("Steve Jobs founded Apple.")
    """

    def __init__(
        self,
        model_name: str = "llama2",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        """Initialize LLaMA NER (backward compatible wrapper).
        
        Args:
            model_name: Model to use ("llama2", "mistral", "llama2:13b", etc.)
            host: Ollama server host
            temperature: Temperature for generation (0.1 = deterministic)
        """
        if not OLLAMA_AVAILABLE:
            logger.error(
                "ollama package not installed. Install with: pip install ollama"
            )
            raise ImportError("ollama package required for LLaMaNER")

        # Call parent class constructor
        super().__init__(
            model_name=model_name,
            host=host,
            temperature=temperature,
        )
        logger.info(f"Initialized LLaMaNER (via OllamaProvider) with model: {model_name}")


class LLaMaNERFallback:
    """Fallback LLaMaNER that works without ollama installed.
    
    Used for testing and CI/CD environments without ollama service.
    Returns empty list - meant to be used in tests with mocking.
    """

    def __init__(self, *args, **kwargs):
        """Initialize fallback NER."""
        logger.warning(
            "Using LLaMaNERFallback - ollama service not available. "
            "Install ollama to enable LLaMA-based extraction."
        )

    def extract_entities(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        timeout: int = 30,
    ) -> List[ExtractedEntity]:
        """Return empty list (fallback behavior)."""
        return []

    def extract_entities_batch(
        self,
        texts: List[tuple],
        batch_size: int = 1,
    ) -> List[ExtractedEntity]:
        """Return empty list (fallback behavior)."""
        return []


# Use appropriate implementation based on ollama availability
if OLLAMA_AVAILABLE:
    DefaultLLaMaNER = LLaMaNER
else:
    DefaultLLaMaNER = LLaMaNERFallback