"""Abstract base class and implementations for LLM-based NER providers.

This module provides a modular interface for any LLM provider:
- Ollama (open-source: LLaMA, Mistral, Neural Chat, etc.)
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Azure OpenAI
- Any other provider
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .entity_models import EntityType, ExtractedEntity

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM-based NER providers.
    
    Supports two extraction modes:
    1. Guided Mode: Extract predefined entity types (default)
    2. Discovery Mode: Let LLM autonomously infer entities
    """

    # Default entity types (for guided mode)
    DEFAULT_ENTITY_TYPES = [
        "PERSON",
        "ORG",
        "LOCATION",
        "DATE",
        "TIME",
        "MONEY",
        "PERCENT",
        "PRODUCT",
        "EVENT",
        "LANGUAGE",
    ]

    # Map common LLM entity types to our EntityType enum
    # Subclasses can override for model-specific mappings
    ENTITY_TYPE_MAPPING = {
        "PERSON": EntityType.PERSON,
        "PEOPLE": EntityType.PERSON,
        "ORG": EntityType.ORG,
        "ORGANIZATION": EntityType.ORG,
        "COMPANY": EntityType.ORG,
        "LOCATION": EntityType.LOCATION,
        "PLACE": EntityType.LOCATION,
        "LOC": EntityType.LOCATION,
        "CITY": EntityType.LOCATION,
        "COUNTRY": EntityType.LOCATION,
        "STATE": EntityType.LOCATION,
        "REGION": EntityType.LOCATION,
        "DATE": EntityType.DATE,
        "TIME": EntityType.TIME,
        "DATETIME": EntityType.DATE,
        "MONEY": EntityType.MONEY,
        "MONETARY": EntityType.MONEY,
        "CURRENCY": EntityType.MONEY,
        "PERCENT": EntityType.PERCENT,
        "PERCENTAGE": EntityType.PERCENT,
        "PRODUCT": EntityType.PRODUCT,
        "WORK": EntityType.WORK_OF_ART,
        "WORK_OF_ART": EntityType.WORK_OF_ART,
        "EVENT": EntityType.EVENT,
        "LANGUAGE": EntityType.LANGUAGE,
        "LANG": EntityType.LANGUAGE,
        "CUSTOM": EntityType.CUSTOM,
        "OTHER": EntityType.CUSTOM,
    }

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        discovery_mode: bool = False,
    ):
        """Initialize LLM provider with entity configuration.
        
        Args:
            entity_types: List of entity types to extract (e.g., ["PERSON", "ORG"])
                         Only used if discovery_mode=False
                         If None, uses DEFAULT_ENTITY_TYPES
            discovery_mode: If True, let LLM infer entities autonomously
                           If False, only extract specified entity_types
        """
        self.discovery_mode = discovery_mode
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        
        if discovery_mode:
            logger.info("NER in DISCOVERY MODE - LLM will infer entities autonomously")
        else:
            logger.info(f"NER in GUIDED MODE - Entity types: {self.entity_types}")

    @abstractmethod
    def extract_entities(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities from text.
        
        Args:
            text: Text to extract entities from
            chunk_id: ID of the chunk
            source_file: Source file path
            timeout: Request timeout in seconds
            entity_types: Override entity types for this call (guided mode only)
            discovery_mode: Override discovery mode for this call
            
        Returns:
            List of ExtractedEntity objects
        """
        pass

    @abstractmethod
    def extract_entities_batch(
        self,
        texts: List[str],
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[List[ExtractedEntity]]:
        """Extract entities from multiple texts.
        
        Args:
            texts: List of texts to extract from
            timeout: Request timeout in seconds
            entity_types: Override entity types
            discovery_mode: Override discovery mode
            
        Returns:
            List of entity lists
        """
        pass

    def _build_prompt(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> str:
        """Build the NER prompt (guided or discovery mode).
        
        Args:
            text: Text to extract entities from
            entity_types: Entity types to extract (guided mode)
            discovery_mode: Whether to use discovery mode
            
        Returns:
            The prompt to send to the LLM
        """
        # Use provided values or fall back to instance defaults
        mode = discovery_mode if discovery_mode is not None else self.discovery_mode
        types_to_use = entity_types or self.entity_types

        if mode:
            # DISCOVERY MODE - Let LLM infer entities autonomously
            return f"""Analyze the following text and extract ALL named entities you can find.
For each entity, identify:
1. The entity text
2. The most appropriate entity type (e.g., PERSON, ORG, LOCATION, DATE, etc.)
3. Your confidence score (0.0 to 1.0)

Be comprehensive - extract entities of any type you identify, not limited to predefined categories.

Text: "{text}"

Return ONLY valid JSON in this format (no markdown, no explanations):
{{"entities": [
    {{"text": "entity_text", "type": "INFERRED_TYPE", "confidence": 0.95}},
    {{"text": "another_entity", "type": "ANOTHER_TYPE", "confidence": 0.87}}
]}}"""

        else:
            # GUIDED MODE - Extract only specified entity types
            entity_types_str = ", ".join(types_to_use)
            return f"""Extract named entities from this text and return as JSON.
Extract ONLY these entity types: {entity_types_str}

Text: "{text}"

Return ONLY valid JSON in this format (no markdown, no explanations):
{{"entities": [
    {{"text": "entity_text", "type": "ENTITY_TYPE", "confidence": 0.95}},
    {{"text": "another_entity", "type": "ENTITY_TYPE", "confidence": 0.87}}
]}}"""

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse JSON response from LLM.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List of entity dictionaries
            
        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            # Try to extract JSON from response (in case LLM adds extra text)
            if "{" not in response_text:
                logger.warning(f"No JSON found in response: {response_text[:100]}")
                return []

            # Find JSON object
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]

            data = json.loads(json_str)
            entities = data.get("entities", [])
            
            if not isinstance(entities, list):
                logger.warning(f"Entities is not a list: {type(entities)}")
                return []
                
            return entities
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:200]}")
            return []

    def _map_entity_type(self, entity_type_str: str) -> EntityType:
        """Map LLM entity type string to our EntityType enum.
        
        Args:
            entity_type_str: Entity type from LLM
            
        Returns:
            Corresponding EntityType enum value
        """
        entity_type_upper = entity_type_str.upper().strip()
        return self.ENTITY_TYPE_MAPPING.get(entity_type_upper, EntityType.CUSTOM)

    def _create_entity(
        self,
        text: str,
        entity_type: EntityType,
        chunk_id: str = "",
        source_file: str = "",
        confidence: Optional[float] = None,
    ) -> Optional[ExtractedEntity]:
        """Create an ExtractedEntity from parsed data.
        
        Args:
            text: Entity text
            entity_type: Entity type
            chunk_id: Chunk ID
            source_file: Source file
            confidence: Confidence score
            
        Returns:
            ExtractedEntity or None if invalid
        """
        if not text or not text.strip():
            return None

        start_pos = -1
        end_pos = -1

        entity = ExtractedEntity(
            text=text.strip(),
            entity_type=entity_type,
            start_position=start_pos,
            end_position=end_pos,
            chunk_id=chunk_id,
            source_file=source_file,
        )

        if confidence is not None:
            entity.confidence = max(0.0, min(1.0, confidence))

        return entity


class OllamaProvider(LLMProvider):
    """Provider for open-source models via Ollama.
    
    Supports: LLaMA, Mistral, Neural Chat, Orca, and any Ollama model
    Install: brew install ollama && ollama pull llama2
    """

    def __init__(
        self,
        model_name: str = "llama2",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        entity_types: Optional[List[str]] = None,
        discovery_mode: bool = False,
    ):
        """Initialize Ollama provider.
        
        Args:
            model_name: Model name (e.g., "llama2", "mistral", "llama2:13b")
            host: Ollama server host
            temperature: Temperature for generation (0.1 = deterministic)
            entity_types: Entity types to extract (guided mode)
            discovery_mode: If True, let LLM infer entities autonomously
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package required. Install with: pip install ollama"
            )

        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.client = ollama.Client(host=host)

        # Initialize parent with entity types and discovery mode
        super().__init__(entity_types=entity_types, discovery_mode=discovery_mode)

        # Verify model is available
        self._verify_model()
        logger.info(f"Initialized OllamaProvider with model: {model_name}")

    def _verify_model(self) -> bool:
        """Verify that the model is available."""
        try:
            self.client.show(self.model_name)
            logger.debug(f"Model {self.model_name} verified")
            return True
        except Exception as e:
            logger.error(
                f"Model '{self.model_name}' not found at {self.host}. "
                f"Run: ollama pull {self.model_name} && ollama serve"
            )
            raise RuntimeError(
                f"Ollama model '{self.model_name}' not available: {e}"
            )

    def extract_entities(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities using Ollama (guided or discovery mode)."""
        if not text or not text.strip():
            return []

        # Use provided values or fall back to instance defaults
        mode = discovery_mode if discovery_mode is not None else self.discovery_mode
        prompt = self._build_prompt(text, entity_types=entity_types, discovery_mode=mode)

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                temperature=self.temperature,
            )

            response_text = response.get("response", "").strip()
            logger.debug(f"Ollama response: {response_text[:200]}...")

            entities_data = self._parse_response(response_text)

            entities = []
            for item in entities_data:
                entity_text = item.get("text", "").strip()
                entity_type_str = item.get("type", "CUSTOM")
                confidence = item.get("confidence", 0.85)

                entity_type = self._map_entity_type(entity_type_str)
                entity = self._create_entity(
                    text=entity_text,
                    entity_type=entity_type,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    confidence=confidence,
                )

                if entity:
                    entities.append(entity)

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities with Ollama: {e}")
            return []

    def extract_entities_batch(
        self,
        texts: List[str],
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[List[ExtractedEntity]]:
        """Extract entities from multiple texts."""
        return [
            self.extract_entities(
                text,
                timeout=timeout,
                entity_types=entity_types,
                discovery_mode=discovery_mode,
            )
            for text in texts
        ]


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models (GPT-3.5, GPT-4).
    
    Install: pip install openai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        entity_types: Optional[List[str]] = None,
        discovery_mode: bool = False,
    ):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
            entity_types: Entity types to extract (guided mode)
            discovery_mode: If True, let LLM infer entities autonomously
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        if api_key:
            openai.api_key = api_key
        
        self.model = model
        self.client = openai.OpenAI()
        
        # Initialize parent with entity types and discovery mode
        super().__init__(entity_types=entity_types, discovery_mode=discovery_mode)
        logger.info(f"Initialized OpenAIProvider with model: {model}")

    def extract_entities(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities using OpenAI."""
        if not text or not text.strip():
            return []

        # Use provided values or fall back to instance defaults
        mode = discovery_mode if discovery_mode is not None else self.discovery_mode
        prompt = self._build_prompt(text, entity_types=entity_types, discovery_mode=mode)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=timeout,
            )

            response_text = response.choices[0].message.content
            logger.debug(f"OpenAI response: {response_text[:200]}...")

            entities_data = self._parse_response(response_text)

            entities = []
            for item in entities_data:
                entity_text = item.get("text", "").strip()
                entity_type_str = item.get("type", "CUSTOM")
                confidence = item.get("confidence", 0.90)

                entity_type = self._map_entity_type(entity_type_str)
                entity = self._create_entity(
                    text=entity_text,
                    entity_type=entity_type,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    confidence=confidence,
                )

                if entity:
                    entities.append(entity)

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities with OpenAI: {e}")
            return []

    def extract_entities_batch(
        self,
        texts: List[str],
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[List[ExtractedEntity]]:
        """Extract entities from multiple texts."""
        return [
            self.extract_entities(text, timeout=timeout, entity_types=entity_types, discovery_mode=discovery_mode)
            for text in texts
        ]


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models.
    
    Install: pip install anthropic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        entity_types: Optional[List[str]] = None,
        discovery_mode: bool = False,
    ):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name (e.g., "claude-3-sonnet-20240229", "claude-3-opus-20240229")
            entity_types: List of entity types to extract (default: DEFAULT_ENTITY_TYPES)
            discovery_mode: If True, let Claude autonomously infer entities (default: False)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        super().__init__(entity_types=entity_types, discovery_mode=discovery_mode)
        logger.info(f"Initialized AnthropicProvider with model: {model}")

    def extract_entities(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities using Claude."""
        if not text or not text.strip():
            return []

        # Use provided values or fall back to instance defaults
        mode = discovery_mode if discovery_mode is not None else self.discovery_mode
        prompt = self._build_prompt(text, entity_types=entity_types, discovery_mode=mode)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text
            logger.debug(f"Claude response: {response_text[:200]}...")

            entities_data = self._parse_response(response_text)

            entities = []
            for item in entities_data:
                entity_text = item.get("text", "").strip()
                entity_type_str = item.get("type", "CUSTOM")
                confidence = item.get("confidence", 0.93)

                entity_type = self._map_entity_type(entity_type_str)
                entity = self._create_entity(
                    text=entity_text,
                    entity_type=entity_type,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    confidence=confidence,
                )

                if entity:
                    entities.append(entity)

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities with Claude: {e}")
            return []

    def extract_entities_batch(
        self,
        texts: List[str],
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[List[ExtractedEntity]]:
        """Extract entities from multiple texts."""
        return [
            self.extract_entities(text, timeout=timeout, entity_types=entity_types, discovery_mode=discovery_mode)
            for text in texts
        ]


def get_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to get LLM provider by type.
    
    Args:
        provider_type: Type of provider ("ollama", "openai", "anthropic")
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMProvider instance
        
    Example:
        # Using Ollama (local, free)
        provider = get_llm_provider("ollama", model_name="llama2")
        
        # Using OpenAI (API)
        provider = get_llm_provider("openai", api_key="sk-...", model="gpt-4")
        
        # Using Claude (API)
        provider = get_llm_provider("anthropic", api_key="sk-...", model="claude-3-opus")
    """
    provider_type = provider_type.lower().strip()

    if provider_type == "ollama":
        return OllamaProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported: ollama, openai, anthropic"
        )


# Check if ollama is available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
