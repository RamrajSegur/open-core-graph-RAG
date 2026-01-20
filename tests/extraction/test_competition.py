"""Tests for competitive multi-LLM NER system."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.extraction.ner.competition import (
    LLMCompetitor,
    CompetitiveNER,
    EntityAgreement,
    run_competition,
)
from src.extraction.ner.entity_models import EntityType, ExtractedEntity
from src.extraction.ner.llm_provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, entities=None, entity_types=None, discovery_mode=False):
        """Initialize mock provider with optional discovery mode."""
        super().__init__(entity_types=entity_types, discovery_mode=discovery_mode)
        self.entities = entities or []
    
    def extract_entities(
        self,
        text,
        chunk_id="",
        source_file="",
        timeout=30,
        entity_types=None,
        discovery_mode=None,
    ):
        """Extract entities with support for discovery mode."""
        return self.entities
    
    def extract_entities_batch(
        self,
        texts,
        timeout=30,
        entity_types=None,
        discovery_mode=None,
    ):
        """Extract entities from batch with discovery mode support."""
        return [self.entities for _ in texts]


class TestLLMCompetitor:
    """Tests for LLMCompetitor class."""
    
    def test_competitor_initialization(self):
        """Test competitor initialization."""
        provider = MockLLMProvider()
        competitor = LLMCompetitor(name="test_model", provider=provider)
        
        assert competitor.name == "test_model"
        assert competitor.provider == provider
        assert competitor.entities == []
        assert competitor.confidence == 0.0
        assert competitor.execution_time_ms == 0.0
    
    def test_competitor_extract_with_entities(self):
        """Test extraction with entities."""
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
            ExtractedEntity(
                text="Steve Jobs",
                entity_type=EntityType.PERSON,
                chunk_id="1",
                start_position=10,
                end_position=21,
                confidence=0.92,
            ),
        ]
        
        provider = MockLLMProvider(entities=entities)
        competitor = LLMCompetitor(name="test", provider=provider)
        
        competitor.extract("Apple was founded by Steve Jobs")
        
        assert len(competitor.entities) == 2
        assert competitor.confidence == pytest.approx(0.935, rel=0.01)
        assert competitor.execution_time_ms > 0
    
    def test_competitor_extract_empty(self):
        """Test extraction with no entities."""
        provider = MockLLMProvider(entities=[])
        competitor = LLMCompetitor(name="test", provider=provider)
        
        competitor.extract("No entities here")
        
        assert competitor.entities == []
        assert competitor.confidence == 0.0
    
    def test_competitor_to_dict(self):
        """Test conversion to dictionary."""
        entities = [
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=6,
                confidence=0.98,
            ),
        ]
        
        provider = MockLLMProvider(entities=entities)
        competitor = LLMCompetitor(name="gpt4", provider=provider)
        competitor.extract("Google was founded")
        
        result = competitor.to_dict()
        
        assert result["name"] == "gpt4"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "Google"
        assert result["entities"][0]["type"] == "ORG"
        assert result["entities"][0]["confidence"] == 0.98
        assert result["avg_confidence"] == 0.98


class TestEntityAgreement:
    """Tests for EntityAgreement class."""
    
    def test_agreement_initialization(self):
        """Test agreement initialization."""
        agreement = EntityAgreement(
            text="Apple Inc.",
            entity_type=EntityType.ORG,
        )
        
        assert agreement.text == "Apple Inc."
        assert agreement.entity_type == EntityType.ORG
        assert agreement.agreement_count == 0
        assert agreement.avg_confidence == 0.0
    
    def test_agreement_with_models(self):
        """Test agreement with multiple models."""
        agreement = EntityAgreement(
            text="Apple Inc.",
            entity_type=EntityType.ORG,
            found_by=["mistral", "gpt4", "claude"],
            confidences=[0.95, 0.98, 0.92],
        )
        
        assert agreement.agreement_count == 3
        assert agreement.avg_confidence == pytest.approx(0.95, rel=0.01)
    
    def test_agreement_to_dict(self):
        """Test conversion to dictionary."""
        agreement = EntityAgreement(
            text="Steve Jobs",
            entity_type=EntityType.PERSON,
            found_by=["mistral", "gpt4"],
            confidences=[0.92, 0.95],
        )
        
        result = agreement.to_dict()
        
        assert result["text"] == "Steve Jobs"
        assert result["type"] == "PERSON"
        assert result["agreement_count"] == 2
        assert "confidences_by_model" in result


class TestCompetitiveNER:
    """Tests for CompetitiveNER class."""
    
    def test_initialization(self):
        """Test CompetitiveNER initialization."""
        competitors = [
            LLMCompetitor("model1", MockLLMProvider()),
            LLMCompetitor("model2", MockLLMProvider()),
        ]
        
        ner = CompetitiveNER(competitors, voting_strategy="weighted")
        
        assert len(ner.competitors) == 2
        assert ner.voting_strategy == "weighted"
    
    def test_invalid_voting_strategy(self):
        """Test invalid voting strategy raises error."""
        competitors = [LLMCompetitor("model1", MockLLMProvider())]
        
        with pytest.raises(ValueError):
            CompetitiveNER(competitors, voting_strategy="invalid")
    
    def test_consensus_voting_all_agree(self):
        """Test consensus voting when all models agree."""
        # All 3 models extract same entities
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        competitors = [
            LLMCompetitor("mistral", MockLLMProvider(entities=entities)),
            LLMCompetitor("gpt4", MockLLMProvider(entities=entities)),
            LLMCompetitor("claude", MockLLMProvider(entities=entities)),
        ]
        
        ner = CompetitiveNER(competitors, voting_strategy="consensus")
        final_entities, stats = ner.extract_with_competition("Apple Inc.")
        
        assert len(final_entities) == 1
        assert final_entities[0].text == "Apple"
        assert stats["voting_strategy"] == "consensus"
    
    def test_consensus_voting_partial_agreement(self):
        """Test consensus voting when not all agree."""
        entities1 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        entities2 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.92,
            ),
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=20,
                end_position=26,
                confidence=0.90,
            ),
        ]
        
        competitors = [
            LLMCompetitor("mistral", MockLLMProvider(entities=entities1)),
            LLMCompetitor("gpt4", MockLLMProvider(entities=entities2)),
        ]
        
        ner = CompetitiveNER(competitors, voting_strategy="consensus")
        final_entities, stats = ner.extract_with_competition("Apple and Google")
        
        # Only "Apple" should be in consensus (both found it)
        assert len(final_entities) == 1
        assert final_entities[0].text == "Apple"
    
    def test_majority_voting(self):
        """Test majority voting strategy."""
        entities1 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        entities2 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.92,
            ),
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=20,
                end_position=26,
                confidence=0.90,
            ),
        ]
        entities3 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.93,
            ),
        ]
        
        competitors = [
            LLMCompetitor("mistral", MockLLMProvider(entities=entities1)),
            LLMCompetitor("gpt4", MockLLMProvider(entities=entities2)),
            LLMCompetitor("claude", MockLLMProvider(entities=entities3)),
        ]
        
        ner = CompetitiveNER(competitors, voting_strategy="majority")
        final_entities, stats = ner.extract_with_competition("Apple and Google")
        
        # Apple: 3/3 models (include)
        # Google: 1/3 models (exclude)
        assert len(final_entities) == 1
        assert final_entities[0].text == "Apple"
    
    def test_weighted_voting(self):
        """Test weighted voting strategy."""
        entities1 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        entities2 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.92,
            ),
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=20,
                end_position=26,
                confidence=0.90,
            ),
        ]
        
        competitors = [
            LLMCompetitor("mistral", MockLLMProvider(entities=entities1)),
            LLMCompetitor("gpt4", MockLLMProvider(entities=entities2)),
        ]
        
        ner = CompetitiveNER(competitors, voting_strategy="weighted")
        final_entities, stats = ner.extract_with_competition("Apple and Google")
        
        assert len(final_entities) == 2
        
        # Apple: 2/2 models, confidence = avg(0.95, 0.92) * (2/2) = 0.935
        apple = [e for e in final_entities if e.text == "Apple"][0]
        assert apple.confidence == pytest.approx(0.935, rel=0.01)
        
        # Google: 1/2 models, confidence = 0.90 * (1/2) = 0.45
        google = [e for e in final_entities if e.text == "Google"][0]
        assert google.confidence == pytest.approx(0.45, rel=0.01)
    
    def test_best_voting(self):
        """Test best voting strategy (use best model)."""
        entities1 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.70,
            ),
        ]
        entities2 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=20,
                end_position=26,
                confidence=0.92,
            ),
        ]
        
        competitors = [
            LLMCompetitor("weak_model", MockLLMProvider(entities=entities1)),
            LLMCompetitor("strong_model", MockLLMProvider(entities=entities2)),
        ]
        
        ner = CompetitiveNER(competitors, voting_strategy="best")
        final_entities, stats = ner.extract_with_competition("Apple and Google")
        
        # Should use strong_model's results
        assert len(final_entities) == 2
        assert any(e.text == "Google" for e in final_entities)
    
    def test_parallel_execution(self):
        """Test that competitors run in parallel."""
        import time
        
        # Create slow providers
        class SlowProvider(LLMProvider):
            def __init__(self, delay=0.1):
                self.delay = delay
            
            def extract_entities(self, text, chunk_id="", source_file="", timeout=30):
                time.sleep(self.delay)
                return [
                    ExtractedEntity(
                        text="Test",
                        entity_type=EntityType.ORG,
                        chunk_id=chunk_id or "1",
                        start_position=0,
                        end_position=4,
                        confidence=0.9,
                    )
                ]
            
            def extract_entities_batch(self, texts, timeout=30):
                return [self.extract_entities(t) for t in texts]
        
        competitors = [
            LLMCompetitor("slow1", SlowProvider(delay=0.1)),
            LLMCompetitor("slow2", SlowProvider(delay=0.1)),
            LLMCompetitor("slow3", SlowProvider(delay=0.1)),
        ]
        
        ner = CompetitiveNER(competitors)
        
        start = time.time()
        final_entities, stats = ner.extract_with_competition("Test text")
        elapsed = time.time() - start
        
        # If truly parallel, should take ~0.1s (not 0.3s sequential)
        # Allow some overhead
        assert elapsed < 0.3, f"Parallel execution too slow: {elapsed}s"
    
    def test_agreement_analysis(self):
        """Test agreement analysis."""
        entities1 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
            ExtractedEntity(
                text="Steve Jobs",
                entity_type=EntityType.PERSON,
                chunk_id="1",
                start_position=20,
                end_position=31,
                confidence=0.92,
            ),
        ]
        entities2 = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.93,
            ),
        ]
        
        competitors = [
            LLMCompetitor("model1", MockLLMProvider(entities=entities1)),
            LLMCompetitor("model2", MockLLMProvider(entities=entities2)),
        ]
        
        ner = CompetitiveNER(competitors)
        final_entities, stats = ner.extract_with_competition("Test text")
        
        # Check agreement stats
        assert "agreement_analysis" in stats
        assert "total_unique_entities_found" in stats
        assert stats["total_unique_entities_found"] == 2  # Apple and Steve Jobs


class TestRunCompetition:
    """Tests for convenience function."""
    
    def test_run_competition(self):
        """Test quick competition function."""
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        competitors = [
            LLMCompetitor("model1", MockLLMProvider(entities=entities)),
            LLMCompetitor("model2", MockLLMProvider(entities=entities)),
        ]
        
        entities, stats = run_competition(
            "Apple Inc.",
            competitors,
            voting_strategy="consensus"
        )
        
        assert len(entities) == 1
        assert entities[0].text == "Apple"
        assert stats["voting_strategy"] == "consensus"


class TestEntityDiscoveryInCompetition:
    """Tests for entity discovery modes in competition."""
    
    def test_competitor_with_guided_mode(self):
        """Test competitor using guided mode."""
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        provider = MockLLMProvider(
            entities=entities,
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        competitor = LLMCompetitor(name="guided_model", provider=provider)
        
        assert competitor.provider.discovery_mode is False
        assert "PERSON" in competitor.provider.entity_types or "ORG" in competitor.provider.entity_types
    
    def test_competitor_with_discovery_mode(self):
        """Test competitor using discovery mode."""
        entities = [
            ExtractedEntity(
                text="AAPL",
                entity_type=EntityType.CUSTOM,
                chunk_id="1",
                start_position=0,
                end_position=4,
                confidence=0.92,
            ),
        ]
        
        provider = MockLLMProvider(
            entities=entities,
            discovery_mode=True
        )
        competitor = LLMCompetitor(name="discovery_model", provider=provider)
        
        assert competitor.provider.discovery_mode is True
    
    def test_mixed_mode_competition(self):
        """Test competition with mixed guided and discovery modes."""
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        guided_provider = MockLLMProvider(
            entities=entities,
            entity_types=["PERSON", "ORG"],
            discovery_mode=False
        )
        
        discovery_provider = MockLLMProvider(
            entities=entities,
            discovery_mode=True
        )
        
        competitors = [
            LLMCompetitor("model_guided", guided_provider),
            LLMCompetitor("model_discovery", discovery_provider),
        ]
        
        # Both competitors should work in competition
        assert competitors[0].provider.discovery_mode is False
        assert competitors[1].provider.discovery_mode is True
        
        # Extract with both
        text = "Apple Inc."
        competitors[0].extract(text)
        competitors[1].extract(text)
        
        assert len(competitors[0].entities) > 0
        assert len(competitors[1].entities) > 0
    
    def test_competition_extraction_preserves_modes(self):
        """Test that competition doesn't change individual modes."""
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        competitor1 = LLMCompetitor(
            "model1",
            MockLLMProvider(entities=entities, discovery_mode=False)
        )
        competitor2 = LLMCompetitor(
            "model2",
            MockLLMProvider(entities=entities, discovery_mode=True)
        )
        
        text = "Apple Inc."
        competitor1.extract(text)
        competitor2.extract(text)
        
        # Modes should be preserved
        assert competitor1.provider.discovery_mode is False
        assert competitor2.provider.discovery_mode is True
    
    def test_competitor_with_custom_entity_types(self):
        """Test competitor with custom entity types."""
        entities = []
        custom_types = ["COMPANY", "FOUNDER", "TICKER"]
        
        provider = MockLLMProvider(
            entities=entities,
            entity_types=custom_types,
            discovery_mode=False
        )
        competitor = LLMCompetitor("custom_types", provider)
        
        assert competitor.provider.entity_types == custom_types
