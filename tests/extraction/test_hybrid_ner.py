"""Tests for Hybrid NER (SpaCy + LLM Provider System)."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.extraction import EntityType, ExtractedEntity, TextChunk
from src.extraction.ner.hybrid_extraction import HybridNER, HybridSpaCyLLaMA
from src.extraction.ner.llm_provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, entities=None, model_name="mock-model"):
        self.entities = entities or []
        self.model_name = model_name
    
    def extract_entities(self, text, chunk_id="", source_file="", timeout=30):
        return self.entities
    
    def extract_entities_batch(self, texts, timeout=30):
        return [self.entities for _ in texts]


class TestHybridNERInitialization:
    """Test HybridNER initialization."""
    
    def test_initialization_spacy_only(self):
        """Test initialization with SpaCy only."""
        hybrid = HybridNER(use_llm=False)
        assert hybrid.spacy_model is not None
        assert hybrid.llm_provider is None
        assert hybrid.confidence_threshold == 0.75
    
    def test_initialization_with_custom_threshold(self):
        """Test initialization with custom confidence threshold."""
        hybrid = HybridNER(
            spacy_model="en_core_web_sm",
            confidence_threshold=0.8,
            use_llm=False
        )
        assert hybrid.confidence_threshold == 0.8
        assert hybrid.llm_provider is None
    
    def test_initialization_with_custom_provider(self):
        """Test initialization with custom LLM provider."""
        mock_provider = MockLLMProvider(model_name="test-model")
        hybrid = HybridNER(
            llm_provider=mock_provider,
            use_llm=True
        )
        assert hybrid.llm_provider == mock_provider
        assert hybrid.spacy_model is not None
    
    def test_initialization_with_strategy(self):
        """Test initialization with specific strategy."""
        hybrid = HybridNER(
            strategy="llm_default",
            use_llm=False
        )
        assert hybrid.strategy == "llm_default"
    
    def test_invalid_strategy_defaults_to_llm_default(self):
        """Test invalid strategy defaults to llm_default."""
        hybrid = HybridNER(
            strategy="invalid_strategy",
            use_llm=False
        )
        assert hybrid.strategy == "llm_default"


class TestHybridNERExtraction:
    """Test HybridNER extraction functionality."""
    
    def test_extract_from_empty_chunk(self):
        """Test extraction from empty chunk."""
        hybrid = HybridNER(use_llm=False)
        empty_chunk = TextChunk(
            content="",
            chunk_id="empty",
            source_file="test.txt"
        )
        entities = hybrid.extract_from_chunk(empty_chunk)
        assert entities == []
    
    def test_extract_from_chunk_spacy_only(self):
        """Test extraction from chunk with SpaCy only."""
        hybrid = HybridNER(use_llm=False)
        chunk = TextChunk(
            content="Apple Inc. was founded by Steve Jobs.",
            chunk_id="chunk1",
            source_file="test.txt"
        )
        entities = hybrid.extract_from_chunk(chunk)
        
        # Should extract some entities using SpaCy
        assert len(entities) > 0
        assert all(isinstance(e, ExtractedEntity) for e in entities)
    
    def test_extract_from_chunk_with_stats(self):
        """Test extraction returns stats when requested."""
        hybrid = HybridNER(use_llm=False)
        chunk = TextChunk(
            content="Apple Inc. is in Cupertino.",
            chunk_id="chunk1",
            source_file="test.txt"
        )
        entities, stats = hybrid.extract_from_chunk(chunk, return_stats=True)
        
        assert isinstance(entities, list)
        assert isinstance(stats, dict)
        assert "spacy_time_ms" in stats or "model_used" in stats
    
    def test_extract_from_multiple_chunks(self):
        """Test extraction from multiple chunks."""
        hybrid = HybridNER(use_llm=False)
        chunks = [
            TextChunk(
                content="Apple Inc. was founded by Steve Jobs.",
                chunk_id="chunk1",
                source_file="test.txt"
            ),
            TextChunk(
                content="Microsoft was founded by Bill Gates.",
                chunk_id="chunk2",
                source_file="test.txt"
            ),
        ]
        
        entities = hybrid.extract_from_chunks(chunks)
        
        assert isinstance(entities, list)
        assert len(entities) > 0
    
    def test_extract_from_chunks_with_stats(self):
        """Test extraction from chunks with stats."""
        hybrid = HybridNER(use_llm=False)
        chunks = [
            TextChunk(
                content="Apple Inc. is in Cupertino.",
                chunk_id="chunk1",
                source_file="test.txt"
            ),
            TextChunk(
                content="Google is in Mountain View.",
                chunk_id="chunk2",
                source_file="test.txt"
            ),
        ]
        
        entities, stats = hybrid.extract_from_chunks(chunks, include_stats=True)
        
        assert isinstance(entities, list)
        # stats can be dict or NERStats object
        assert stats is not None
        # Check for key attributes regardless of type
        if hasattr(stats, 'total_entities'):
            assert hasattr(stats, 'chunks_processed')
        else:
            assert isinstance(stats, dict)
            assert "total_entities" in stats
            assert "chunks_processed" in stats


class TestHybridNERBackwardCompatibility:
    """Test backward compatibility with HybridSpaCyLLaMA."""
    
    def test_hybrid_spacy_llama_alias_exists(self):
        """Test that HybridSpaCyLLaMA alias exists."""
        assert HybridSpaCyLLaMA is not None
    
    def test_hybrid_spacy_llama_initialization_spacy_only(self):
        """Test HybridSpaCyLLaMA initialization in SpaCy-only mode."""
        hybrid = HybridSpaCyLLaMA(use_llama=False)
        assert hybrid.spacy_model is not None
        assert hybrid.llama_model is None
        assert hybrid.confidence_threshold == 0.75
    
    def test_hybrid_spacy_llama_initialization_with_options(self):
        """Test HybridSpaCyLLaMA initialization with old parameter names."""
        hybrid = HybridSpaCyLLaMA(
            spacy_model="en_core_web_sm",
            confidence_threshold=0.8,
            use_llama=False
        )
        assert hybrid.confidence_threshold == 0.8
        assert hybrid.llama_model is None
    
    def test_hybrid_spacy_llama_extract(self):
        """Test HybridSpaCyLLaMA extraction."""
        hybrid = HybridSpaCyLLaMA(use_llama=False)
        chunk = TextChunk(
            content="Apple Inc. was founded by Steve Jobs.",
            chunk_id="chunk1",
            source_file="test.txt"
        )
        entities = hybrid.extract_from_chunk(chunk)
        
        assert isinstance(entities, list)
        assert all(isinstance(e, ExtractedEntity) for e in entities)
    
    def test_hybrid_spacy_llama_old_parameter_mapping(self):
        """Test that old parameter names are mapped correctly."""
        # Old style: use_llama=False
        hybrid_old = HybridSpaCyLLaMA(use_llama=False)
        
        # New style: use_llm=False
        hybrid_new = HybridNER(use_llm=False)
        
        # Both should have same behavior
        assert hybrid_old.use_llm == hybrid_new.use_llm
        assert hybrid_old.confidence_threshold == hybrid_new.confidence_threshold


class TestHybridNEREdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extract_with_special_characters(self):
        """Test extraction with special characters."""
        hybrid = HybridNER(use_llm=False)
        chunk = TextChunk(
            content="Dr. Jane Smith & Co. (Est. 2020) @ 123 Main St.",
            chunk_id="chunk1",
            source_file="test.txt"
        )
        entities = hybrid.extract_from_chunk(chunk)
        
        assert isinstance(entities, list)
    
    def test_extract_very_long_text(self):
        """Test extraction from very long text."""
        hybrid = HybridNER(use_llm=False)
        long_text = "Apple Inc. " * 1000  # Repeat text 1000 times
        chunk = TextChunk(
            content=long_text,
            chunk_id="chunk1",
            source_file="test.txt"
        )
        entities = hybrid.extract_from_chunk(chunk)
        
        assert isinstance(entities, list)
    
    def test_extract_empty_chunks_list(self):
        """Test extraction with empty chunks list."""
        hybrid = HybridNER(use_llm=False)
        entities = hybrid.extract_from_chunks([])
        
        assert entities == []
    
    def test_get_stats(self):
        """Test getting statistics."""
        # Use HybridSpaCyLLaMA for backward compat test
        hybrid = HybridSpaCyLLaMA(use_llama=False, strategy="llm_default")
        stats = hybrid.get_stats()
        
        assert isinstance(stats, dict)
        assert "spacy_model" in stats
        assert "confidence_threshold" in stats
        assert "strategy" in stats


class TestHybridNERWithCustomProvider:
    """Test HybridNER with custom LLM provider."""
    
    def test_extract_with_custom_mock_provider(self):
        """Test extraction with custom mock provider."""
        mock_entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        mock_provider = MockLLMProvider(entities=mock_entities, model_name="test")
        hybrid = HybridNER(llm_provider=mock_provider, use_llm=True)
        
        # Just verify initialization works
        assert hybrid.llm_provider is not None


class TestHybridNERConfiguration:
    """Test various configuration options."""
    
    def test_all_strategies(self):
        """Test all available strategies."""
        strategies = ["spacy_default", "llm_default", "llm_only"]
        
        for strategy in strategies:
            hybrid = HybridNER(strategy=strategy, use_llm=False)
            assert hybrid.strategy == strategy
    
    def test_confidence_threshold_range(self):
        """Test various confidence thresholds."""
        thresholds = [0.0, 0.5, 0.75, 0.9, 1.0]
        
        for threshold in thresholds:
            hybrid = HybridNER(confidence_threshold=threshold, use_llm=False)
            assert hybrid.confidence_threshold == threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
