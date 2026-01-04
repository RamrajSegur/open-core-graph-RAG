"""Unit tests for competitive relationship extraction."""

import pytest
from typing import List

from src.extraction.chunking import TextChunk
from src.extraction.ner import ExtractedEntity, EntityType
from src.extraction.relationships import (
    RelationshipCompetitor,
    RelationshipAgreement,
    CompetitiveRelationshipExtractor,
    RelationshipProvider,
    DefaultRelationshipProvider,
    RelationshipExtractor,
    ExtractedRelationship,
    RelationshipType,
)


class MockRelationshipProvider(RelationshipProvider):
    """Mock relationship provider for testing."""
    
    def __init__(self, name: str, relationships: List[ExtractedRelationship] = None):
        """Initialize mock provider with predefined relationships."""
        self.name = name
        self._relationships = relationships or []
    
    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str = "",
        source_file: str = "",
    ) -> List[ExtractedRelationship]:
        """Return predefined relationships."""
        return self._relationships


class TestRelationshipCompetitor:
    """Tests for RelationshipCompetitor."""
    
    def test_competitor_initialization(self):
        """Test competitor initialization."""
        provider = MockRelationshipProvider("test")
        competitor = RelationshipCompetitor(name="TestModel", provider=provider)
        
        assert competitor.name == "TestModel"
        assert competitor.provider == provider
        assert competitor.relationships == []
        assert competitor.confidence == 0.0
        assert competitor.execution_time_ms == 0.0
    
    def test_competitor_extract_with_entities(self):
        """Test competitor extraction with relationships."""
        # Create mock relationships
        rels = [
            ExtractedRelationship(
                source_entity="Apple",
                target_entity="Steve Jobs",
                relationship_type=RelationshipType.FOUNDED_BY,
                source_chunk_id="chunk1",
                confidence=0.9,
            ),
            ExtractedRelationship(
                source_entity="Google",
                target_entity="Mountain View",
                relationship_type=RelationshipType.LOCATED_IN,
                source_chunk_id="chunk1",
                confidence=0.85,
            ),
        ]
        
        provider = MockRelationshipProvider("mistral", rels)
        competitor = RelationshipCompetitor(name="Mistral", provider=provider)
        
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="chunk1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
            ExtractedEntity(
                text="Steve Jobs",
                entity_type=EntityType.PERSON,
                chunk_id="chunk1",
                start_position=25,
                end_position=36,
                confidence=0.95,
            ),
        ]
        
        competitor.extract("Apple was founded by Steve Jobs", entities, "chunk1")
        
        assert len(competitor.relationships) == 2
        assert competitor.confidence == pytest.approx(0.875)  # (0.9 + 0.85) / 2
        assert competitor.execution_time_ms > 0
    
    def test_competitor_to_dict(self):
        """Test competitor to_dict conversion."""
        rels = [
            ExtractedRelationship(
                source_entity="Apple",
                target_entity="Steve Jobs",
                relationship_type=RelationshipType.FOUNDED_BY,
                source_chunk_id="chunk1",
                confidence=0.9,
            ),
        ]
        
        provider = MockRelationshipProvider("test", rels)
        competitor = RelationshipCompetitor(name="TestModel", provider=provider)
        
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="chunk1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
        ]
        
        competitor.extract("Test text", entities)
        result = competitor.to_dict()
        
        assert result["name"] == "TestModel"
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["source_entity"] == "Apple"
        assert result["avg_confidence"] == pytest.approx(0.9)


class TestRelationshipAgreement:
    """Tests for RelationshipAgreement."""
    
    def test_agreement_initialization(self):
        """Test agreement initialization."""
        agreement = RelationshipAgreement(
            source_entity="Apple",
            target_entity="Steve Jobs",
            relationship_type=RelationshipType.FOUNDED_BY,
        )
        
        assert agreement.source_entity == "Apple"
        assert agreement.target_entity == "Steve Jobs"
        assert agreement.agreement_count == 0
        assert agreement.avg_confidence == 0.0
    
    def test_agreement_metrics(self):
        """Test agreement metric calculations."""
        agreement = RelationshipAgreement(
            source_entity="Apple",
            target_entity="Steve Jobs",
            relationship_type=RelationshipType.FOUNDED_BY,
        )
        
        agreement.found_by = ["mistral", "gpt-4", "claude"]
        agreement.confidences = [0.95, 0.88, 0.92]
        
        assert agreement.agreement_count == 3
        assert agreement.avg_confidence == pytest.approx(0.9167, abs=0.001)
    
    def test_agreement_to_dict(self):
        """Test agreement to_dict conversion."""
        agreement = RelationshipAgreement(
            source_entity="Apple",
            target_entity="Steve Jobs",
            relationship_type=RelationshipType.FOUNDED_BY,
        )
        
        agreement.found_by = ["mistral", "gpt-4"]
        agreement.confidences = [0.95, 0.88]
        
        result = agreement.to_dict()
        
        assert result["source_entity"] == "Apple"
        assert result["target_entity"] == "Steve Jobs"
        assert result["agreement_count"] == 2
        assert result["avg_confidence"] == pytest.approx(0.915)


class TestCompetitiveRelationshipExtractor:
    """Tests for CompetitiveRelationshipExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create relationships for different models
        self.apple_founded_rels = [
            ExtractedRelationship(
                source_entity="Apple",
                target_entity="Steve Jobs",
                relationship_type=RelationshipType.FOUNDED_BY,
                source_chunk_id="chunk1",
                confidence=0.95,
            ),
        ]
        
        self.google_located_rels = [
            ExtractedRelationship(
                source_entity="Google",
                target_entity="Mountain View",
                relationship_type=RelationshipType.LOCATED_IN,
                source_chunk_id="chunk1",
                confidence=0.92,
            ),
        ]
        
        self.apple_located_rels = [
            ExtractedRelationship(
                source_entity="Apple",
                target_entity="Cupertino",
                relationship_type=RelationshipType.LOCATED_IN,
                source_chunk_id="chunk1",
                confidence=0.90,
            ),
        ]
        
        # Create competitors with different relationships
        self.mistral = RelationshipCompetitor(
            name="Mistral",
            provider=MockRelationshipProvider(
                "mistral", self.apple_founded_rels + self.google_located_rels
            ),
        )
        
        self.gpt4 = RelationshipCompetitor(
            name="GPT-4",
            provider=MockRelationshipProvider(
                "gpt4", self.apple_founded_rels + self.apple_located_rels
            ),
        )
        
        self.claude = RelationshipCompetitor(
            name="Claude",
            provider=MockRelationshipProvider(
                "claude", self.apple_founded_rels
            ),
        )
    
    def test_competitive_extractor_initialization(self):
        """Test initialization with valid strategy."""
        competitors = [self.mistral, self.gpt4]
        ner = CompetitiveRelationshipExtractor(competitors, voting_strategy="weighted")
        
        assert len(ner.competitors) == 2
        assert ner.voting_strategy == "weighted"
    
    def test_invalid_voting_strategy(self):
        """Test that invalid strategy raises error."""
        competitors = [self.mistral, self.gpt4]
        
        with pytest.raises(ValueError):
            CompetitiveRelationshipExtractor(competitors, voting_strategy="invalid")
    
    def test_consensus_voting(self):
        """Test consensus voting strategy."""
        competitors = [self.mistral, self.gpt4, self.claude]
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="consensus"
        )
        
        entities = []
        relationships, stats = ner.extract_with_competition(
            "Apple was founded by Steve Jobs",
            entities,
            "chunk1",
        )
        
        # Only FOUNDED_BY should appear (all 3 models found it)
        assert len(relationships) == 1
        assert relationships[0].relationship_type == RelationshipType.FOUNDED_BY
        assert relationships[0].source_entity == "Apple"
    
    def test_majority_voting(self):
        """Test majority voting strategy."""
        competitors = [self.mistral, self.gpt4, self.claude]
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="majority"
        )
        
        entities = []
        relationships, stats = ner.extract_with_competition(
            "Test text",
            entities,
            "chunk1",
        )
        
        # FOUNDED_BY appears 3 times (all agree)
        # LOCATED_IN appears 1 time from mistral, 1 from gpt4 (2 models)
        # Should include both since they have 2+ agreements
        founded_by_rels = [
            r for r in relationships
            if r.relationship_type == RelationshipType.FOUNDED_BY
        ]
        assert len(founded_by_rels) > 0
    
    def test_weighted_voting(self):
        """Test weighted voting strategy."""
        competitors = [self.mistral, self.gpt4, self.claude]
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="weighted"
        )
        
        entities = []
        relationships, stats = ner.extract_with_competition(
            "Test text",
            entities,
            "chunk1",
        )
        
        # Weighted voting includes all relationships but weights by confidence
        assert len(relationships) > 0
        
        # Relationships with more agreement should have higher confidence
        founded_by_rels = [
            r for r in relationships
            if r.relationship_type == RelationshipType.FOUNDED_BY
        ]
        located_in_rels = [
            r for r in relationships
            if r.relationship_type == RelationshipType.LOCATED_IN
        ]
        
        if founded_by_rels and located_in_rels:
            # FOUNDED_BY has 3 agreements (all models), LOCATED_IN has 1-2
            assert founded_by_rels[0].confidence > located_in_rels[0].confidence
    
    def test_best_voting(self):
        """Test best voting strategy."""
        competitors = [self.mistral, self.gpt4, self.claude]
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="best"
        )
        
        entities = []
        relationships, stats = ner.extract_with_competition(
            "Test text",
            entities,
            "chunk1",
        )
        
        # Should use only the best competitor's relationships
        # All three competitors have equal confidence in this setup
        assert len(relationships) > 0
    
    def test_agreement_analysis(self):
        """Test agreement analysis."""
        competitors = [self.mistral, self.gpt4, self.claude]
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="consensus"
        )
        
        entities = []
        relationships, stats = ner.extract_with_competition(
            "Test text",
            entities,
            "chunk1",
            return_agreement=True,
        )
        
        # Check that agreement details are in stats
        assert "agreement_summary" in stats
        assert "consensus_count" in stats["agreement_summary"]
        assert "majority_count" in stats["agreement_summary"]
    
    def test_statistics_collection(self):
        """Test statistics are properly collected."""
        competitors = [self.mistral, self.gpt4]
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="weighted"
        )
        
        entities = []
        relationships, stats = ner.extract_with_competition(
            "Test text",
            entities,
            "chunk1",
            return_agreement=True,
        )
        
        assert "voting_strategy" in stats
        assert stats["voting_strategy"] == "weighted"
        assert "num_competitors" in stats
        assert stats["num_competitors"] == 2
        assert "competitors" in stats
        assert len(stats["competitors"]) == 2
        assert "avg_competitor_confidence" in stats


class TestRelationshipProviders:
    """Tests for relationship provider implementations."""
    
    def test_default_relationship_provider(self):
        """Test DefaultRelationshipProvider wrapping RelationshipExtractor."""
        # Create a real RelationshipExtractor
        extractor = RelationshipExtractor(use_patterns=True, use_semantic=False)
        
        # Wrap it in DefaultRelationshipProvider
        provider = DefaultRelationshipProvider(extractor)
        
        # Create test entities
        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="chunk1",
                start_position=0,
                end_position=5,
                confidence=0.95,
            ),
            ExtractedEntity(
                text="Cupertino",
                entity_type=EntityType.GPE,
                chunk_id="chunk1",
                start_position=23,
                end_position=32,
                confidence=0.95,
            ),
        ]
        
        # Extract relationships
        relationships = provider.extract_relationships(
            "Apple Inc. is located in Cupertino.",
            entities,
            "chunk1",
        )
        
        # Should return list of relationships
        assert isinstance(relationships, list)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_competitor_results(self):
        """Test handling of competitors with no results."""
        empty_provider = MockRelationshipProvider("empty", [])
        competitor = RelationshipCompetitor(name="Empty", provider=empty_provider)
        
        entities = []
        competitor.extract("Test text", entities, "chunk1")
        
        assert len(competitor.relationships) == 0
        assert competitor.confidence == 0.0
    
    def test_disagreement_across_models(self):
        """Test when models completely disagree."""
        provider1 = MockRelationshipProvider("p1", [
            ExtractedRelationship(
                source_entity="Apple",
                target_entity="Steve Jobs",
                relationship_type=RelationshipType.FOUNDED_BY,
                source_chunk_id="c1",
                confidence=0.9,
            ),
        ])
        
        provider2 = MockRelationshipProvider("p2", [
            ExtractedRelationship(
                source_entity="Google",
                target_entity="Mountain View",
                relationship_type=RelationshipType.LOCATED_IN,
                source_chunk_id="c1",
                confidence=0.9,
            ),
        ])
        
        competitors = [
            RelationshipCompetitor(name="Model1", provider=provider1),
            RelationshipCompetitor(name="Model2", provider=provider2),
        ]
        
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="consensus"
        )
        
        relationships, stats = ner.extract_with_competition(
            "Test text", [], "c1"
        )
        
        # Consensus voting should return empty since no agreement
        assert len(relationships) == 0
    
    def test_single_competitor(self):
        """Test with single competitor."""
        rels = [
            ExtractedRelationship(
                source_entity="Apple",
                target_entity="Steve Jobs",
                relationship_type=RelationshipType.FOUNDED_BY,
                source_chunk_id="c1",
                confidence=0.9,
            ),
        ]
        
        provider = MockRelationshipProvider("test", rels)
        competitors = [RelationshipCompetitor(name="Single", provider=provider)]
        
        ner = CompetitiveRelationshipExtractor(
            competitors, voting_strategy="weighted"
        )
        
        relationships, stats = ner.extract_with_competition(
            "Test text", [], "c1"
        )
        
        # Should return the single relationship
        assert len(relationships) == 1
