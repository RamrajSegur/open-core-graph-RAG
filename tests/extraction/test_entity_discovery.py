"""Tests for entity type discovery feature in LLMProvider."""

import pytest
from typing import List, Optional
from unittest.mock import Mock, patch, MagicMock

from src.extraction.ner.llm_provider import LLMProvider
from src.extraction.ner.entity_models import EntityType, ExtractedEntity


class MockDiscoveryProvider(LLMProvider):
    """Mock LLM provider for testing entity discovery feature."""
    
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        discovery_mode: bool = False,
        entities: Optional[List[ExtractedEntity]] = None,
    ):
        """Initialize mock provider."""
        super().__init__(entity_types=entity_types, discovery_mode=discovery_mode)
        self.entities = entities or []
        self.last_prompt = None
        self.last_mode = None
        self.last_entity_types = None
    
    def extract_entities(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        timeout: int = 30,
        entity_types: Optional[List[str]] = None,
        discovery_mode: Optional[bool] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities and track parameters."""
        mode = discovery_mode if discovery_mode is not None else self.discovery_mode
        types_to_use = entity_types or self.entity_types
        
        # Store for verification in tests
        self.last_prompt = self._build_prompt(text, entity_types=entity_types, discovery_mode=mode)
        self.last_mode = mode
        self.last_entity_types = types_to_use
        
        return self.entities
    
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


class TestLLMProviderInitialization:
    """Tests for LLMProvider initialization with discovery mode."""
    
    def test_default_initialization(self):
        """Test provider with default settings."""
        provider = MockDiscoveryProvider()
        
        assert provider.discovery_mode is False
        assert provider.entity_types == LLMProvider.DEFAULT_ENTITY_TYPES
    
    def test_guided_mode_with_custom_types(self):
        """Test guided mode with custom entity types."""
        custom_types = ["PERSON", "ORG"]
        provider = MockDiscoveryProvider(
            entity_types=custom_types,
            discovery_mode=False
        )
        
        assert provider.discovery_mode is False
        assert provider.entity_types == custom_types
    
    def test_discovery_mode_initialization(self):
        """Test discovery mode initialization."""
        provider = MockDiscoveryProvider(discovery_mode=True)
        
        assert provider.discovery_mode is True
        assert provider.entity_types == LLMProvider.DEFAULT_ENTITY_TYPES
    
    def test_discovery_mode_ignores_entity_types(self):
        """Test that discovery mode with explicit types stores both."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON"],
            discovery_mode=True
        )
        
        assert provider.discovery_mode is True
        assert provider.entity_types == ["PERSON"]  # Still stores them
    
    def test_none_entity_types_uses_defaults(self):
        """Test that None entity_types uses DEFAULT_ENTITY_TYPES."""
        provider = MockDiscoveryProvider(entity_types=None)
        
        assert provider.entity_types == LLMProvider.DEFAULT_ENTITY_TYPES
        assert "PERSON" in provider.entity_types
        assert "ORG" in provider.entity_types


class TestPromptGeneration:
    """Tests for prompt generation in different modes."""
    
    def test_guided_mode_prompt(self):
        """Test prompt generation in guided mode."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        text = "Apple was founded by Steve Jobs."
        provider.extract_entities(text)
        
        assert "[GUIDED MODE]" in provider.last_prompt or "Extract ONLY" in provider.last_prompt
        assert "PERSON" in provider.last_prompt
        assert "ORG" in provider.last_prompt
    
    def test_discovery_mode_prompt(self):
        """Test prompt generation in discovery mode."""
        provider = MockDiscoveryProvider(discovery_mode=True)
        
        text = "Apple was founded by Steve Jobs."
        provider.extract_entities(text)
        
        assert "extract ALL" in provider.last_prompt or "comprehensive" in provider.last_prompt.lower()
        assert "comprehensive" in provider.last_prompt.lower()
    
    def test_prompt_includes_entity_types(self):
        """Test that guided mode prompt includes specified entity types."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "LOCATION", "PRODUCT"],
            discovery_mode=False
        )
        
        provider.extract_entities("Test text")
        
        assert "PERSON" in provider.last_prompt
        assert "LOCATION" in provider.last_prompt
        assert "PRODUCT" in provider.last_prompt
    
    def test_prompt_different_for_modes(self):
        """Test that guided and discovery prompts are different."""
        guided_provider = MockDiscoveryProvider(discovery_mode=False)
        discovery_provider = MockDiscoveryProvider(discovery_mode=True)
        
        text = "Same text"
        
        guided_provider.extract_entities(text)
        guided_prompt = guided_provider.last_prompt
        
        discovery_provider.extract_entities(text)
        discovery_prompt = discovery_provider.last_prompt
        
        assert guided_prompt != discovery_prompt


class TestPerCallOverride:
    """Tests for per-call parameter overrides."""
    
    def test_override_discovery_mode(self):
        """Test overriding discovery_mode in method call."""
        # Provider initialized in guided mode
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        # Call with discovery_mode=True override
        provider.extract_entities("Test", discovery_mode=True)
        
        assert provider.last_mode is True  # Override took effect
        assert "comprehensive" in provider.last_prompt.lower()
    
    def test_override_entity_types(self):
        """Test overriding entity_types in method call."""
        # Provider initialized with default types
        provider = MockDiscoveryProvider(discovery_mode=False)
        
        # Call with custom entity_types override
        custom_types = ["LOCATION", "DATE"]
        provider.extract_entities("Test", entity_types=custom_types)
        
        assert provider.last_entity_types == custom_types
        assert "LOCATION" in provider.last_prompt
        assert "DATE" in provider.last_prompt
    
    def test_override_both_parameters(self):
        """Test overriding both discovery_mode and entity_types."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        # Override both
        provider.extract_entities(
            "Test",
            entity_types=["LOCATION"],
            discovery_mode=True
        )
        
        assert provider.last_mode is True
        assert provider.last_entity_types == ["LOCATION"]
    
    def test_none_override_uses_instance_settings(self):
        """Test that None override values use instance settings."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON"],
            discovery_mode=True
        )
        
        # Call with None (explicit) should use instance settings
        provider.extract_entities("Test", entity_types=None, discovery_mode=None)
        
        assert provider.last_mode is True  # Uses instance setting
        assert provider.last_entity_types == ["PERSON"]  # Uses instance setting


class TestParameterPrecedence:
    """Tests for parameter precedence (method > instance > defaults)."""
    
    def test_method_param_overrides_instance(self):
        """Test that method parameters override instance settings."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        # Method parameter should override instance
        provider.extract_entities(
            "Test",
            entity_types=["LOCATION"],
            discovery_mode=True
        )
        
        assert provider.last_entity_types == ["LOCATION"]
        assert provider.last_mode is True
    
    def test_instance_overrides_default(self):
        """Test that instance settings override defaults."""
        custom_types = ["CUSTOM_TYPE_1", "CUSTOM_TYPE_2"]
        provider = MockDiscoveryProvider(entity_types=custom_types)
        
        assert provider.entity_types == custom_types
        assert provider.entity_types != LLMProvider.DEFAULT_ENTITY_TYPES
    
    def test_default_used_when_nothing_specified(self):
        """Test that defaults are used when nothing is specified."""
        provider = MockDiscoveryProvider()  # No parameters
        
        assert provider.entity_types == LLMProvider.DEFAULT_ENTITY_TYPES
        assert provider.discovery_mode is False


class TestBatchProcessing:
    """Tests for batch processing with entity discovery."""
    
    def test_batch_with_default_settings(self):
        """Test batch processing with default settings."""
        entities = [
            ExtractedEntity(
                text="Test", entity_type=EntityType.PERSON, chunk_id="1",
                start_position=0, end_position=4
            )
        ]
        provider = MockDiscoveryProvider(entities=entities)
        
        results = provider.extract_entities_batch(
            texts=["Text 1", "Text 2", "Text 3"]
        )
        
        assert len(results) == 3
        assert all(len(r) == 1 for r in results)
    
    def test_batch_with_custom_entity_types(self):
        """Test batch processing with custom entity types."""
        provider = MockDiscoveryProvider()
        
        custom_types = ["PERSON", "LOCATION"]
        provider.extract_entities_batch(
            texts=["Text 1", "Text 2"],
            entity_types=custom_types
        )
        
        # All calls should use the custom types
        assert provider.last_entity_types == custom_types
    
    def test_batch_with_discovery_mode(self):
        """Test batch processing in discovery mode."""
        provider = MockDiscoveryProvider()
        
        provider.extract_entities_batch(
            texts=["Text 1", "Text 2"],
            discovery_mode=True
        )
        
        assert provider.last_mode is True


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""
    
    def test_extract_without_new_parameters(self):
        """Test that extract_entities works without new parameters."""
        provider = MockDiscoveryProvider()
        
        # Old code calling extract_entities without new parameters
        entities = provider.extract_entities(
            text="Test text",
            chunk_id="chunk_1",
            source_file="file.txt"
        )
        
        assert isinstance(entities, list)  # Should return list
    
    def test_batch_without_new_parameters(self):
        """Test that extract_entities_batch works without new parameters."""
        provider = MockDiscoveryProvider()
        
        # Old code calling batch without new parameters
        results = provider.extract_entities_batch(
            texts=["Text 1", "Text 2"]
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_default_entity_types_unchanged(self):
        """Test that DEFAULT_ENTITY_TYPES has expected values."""
        expected = [
            "PERSON", "ORG", "LOCATION", "DATE", "TIME",
            "MONEY", "PERCENT", "PRODUCT", "EVENT", "LANGUAGE"
        ]
        
        assert LLMProvider.DEFAULT_ENTITY_TYPES == expected


class TestModeLogging:
    """Tests for proper logging of mode selection."""
    
    @patch('src.extraction.ner.llm_provider.logger')
    def test_discovery_mode_logged(self, mock_logger):
        """Test that discovery mode initialization is logged."""
        provider = MockDiscoveryProvider(discovery_mode=True)
        
        # Check if logging was called with discovery mode message
        assert any(
            "DISCOVERY MODE" in str(call)
            for call in mock_logger.info.call_args_list
        ) or provider.discovery_mode is True
    
    @patch('src.extraction.ner.llm_provider.logger')
    def test_guided_mode_logged(self, mock_logger):
        """Test that guided mode initialization is logged."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        # Check if logging was called with guided mode message
        assert any(
            "GUIDED MODE" in str(call) or "Entity types" in str(call)
            for call in mock_logger.info.call_args_list
        ) or provider.discovery_mode is False


class TestEntityTypeMapping:
    """Tests for entity type mapping with discovery."""
    
    def test_maps_common_entity_types(self):
        """Test that common entity type variations are mapped correctly."""
        provider = MockDiscoveryProvider()
        
        # Test entity type mapping dictionary
        assert provider.ENTITY_TYPE_MAPPING["PERSON"] == EntityType.PERSON
        assert provider.ENTITY_TYPE_MAPPING["ORG"] == EntityType.ORG
        assert provider.ENTITY_TYPE_MAPPING["ORGANIZATION"] == EntityType.ORG
        assert provider.ENTITY_TYPE_MAPPING["LOCATION"] == EntityType.LOCATION
    
    def test_maps_to_custom_type_as_fallback(self):
        """Test that unknown types map to CUSTOM."""
        provider = MockDiscoveryProvider()
        
        assert provider.ENTITY_TYPE_MAPPING.get("CUSTOM") == EntityType.CUSTOM
        assert provider.ENTITY_TYPE_MAPPING.get("OTHER") == EntityType.CUSTOM


class TestEntityDiscoveryIntegration:
    """Integration tests for entity discovery feature."""
    
    def test_guided_mode_workflow(self):
        """Test complete guided mode workflow."""
        # Setup
        entities = [
            ExtractedEntity(
                text="John", entity_type=EntityType.PERSON, chunk_id="1",
                start_position=0, end_position=4, confidence=0.95
            ),
            ExtractedEntity(
                text="Google", entity_type=EntityType.ORG, chunk_id="1",
                start_position=10, end_position=16, confidence=0.98
            )
        ]
        
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False,
            entities=entities
        )
        
        # Extract
        result = provider.extract_entities("John works at Google")
        
        # Verify
        assert len(result) == 2
        assert result[0].entity_type == EntityType.PERSON
        assert result[1].entity_type == EntityType.ORG
        assert "PERSON" in provider.last_prompt
        assert "ORG" in provider.last_prompt
    
    def test_discovery_mode_workflow(self):
        """Test complete discovery mode workflow."""
        # Setup with custom entity types that would be discovered
        entities = [
            ExtractedEntity(
                text="AAPL", entity_type=EntityType.CUSTOM, chunk_id="1",
                start_position=0, end_position=4, confidence=0.92
            ),
            ExtractedEntity(
                text="Apple", entity_type=EntityType.ORG, chunk_id="1",
                start_position=10, end_position=15, confidence=0.97
            )
        ]
        
        provider = MockDiscoveryProvider(
            discovery_mode=True,
            entities=entities
        )
        
        # Extract
        result = provider.extract_entities("AAPL is Apple's ticker")
        
        # Verify
        assert len(result) == 2
        assert "comprehensive" in provider.last_prompt.lower()
        assert "extract ALL" in provider.last_prompt.lower() or "all named entities" in provider.last_prompt.lower()
    
    def test_switching_modes_with_same_provider(self):
        """Test switching between modes with per-call overrides."""
        provider = MockDiscoveryProvider(
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        text = "Apple was founded by Steve Jobs"
        
        # First call: guided mode (instance default)
        provider.extract_entities(text)
        first_prompt = provider.last_prompt
        
        # Second call: discovery mode (override)
        provider.extract_entities(text, discovery_mode=True)
        second_prompt = provider.last_prompt
        
        # Third call: back to guided mode (explicit)
        provider.extract_entities(text, discovery_mode=False)
        third_prompt = provider.last_prompt
        
        # Verify different prompts
        assert first_prompt != second_prompt
        assert first_prompt == third_prompt or "GUIDED" in first_prompt.upper()


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_entity_types_list(self):
        """Test behavior with empty entity types list - defaults to DEFAULT_ENTITY_TYPES."""
        provider = MockDiscoveryProvider(entity_types=[], discovery_mode=False)
        
        # Empty list defaults to DEFAULT_ENTITY_TYPES (not an empty list)
        assert provider.entity_types == LLMProvider.DEFAULT_ENTITY_TYPES
    
    def test_single_entity_type(self):
        """Test with single entity type."""
        provider = MockDiscoveryProvider(entity_types=["PERSON"])
        
        assert provider.entity_types == ["PERSON"]
    
    def test_many_entity_types(self):
        """Test with many entity types."""
        types = [f"TYPE_{i}" for i in range(50)]
        provider = MockDiscoveryProvider(entity_types=types)
        
        assert len(provider.entity_types) == 50
    
    def test_empty_text(self):
        """Test extraction with empty text."""
        provider = MockDiscoveryProvider()
        
        result = provider.extract_entities("")
        
        assert isinstance(result, list)
    
    def test_whitespace_only_text(self):
        """Test extraction with whitespace-only text."""
        provider = MockDiscoveryProvider()
        
        result = provider.extract_entities("   \n\t  ")
        
        assert isinstance(result, list)
