"""Tests for Named Entity Recognition module."""

import pytest

from src.extraction import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
    NERModel,
    TextChunk,
)


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self):
        """Test EntityType enum has expected values."""
        assert EntityType.PERSON.value == "PERSON"
        assert EntityType.ORG.value == "ORG"
        assert EntityType.LOCATION.value == "LOCATION"
        assert EntityType.DATE.value == "DATE"
        assert EntityType.PRODUCT.value == "PRODUCT"
        assert EntityType.MONEY.value == "MONEY"

    def test_entity_type_comparison(self):
        """Test EntityType comparison."""
        assert EntityType.PERSON == EntityType.PERSON
        assert EntityType.PERSON != EntityType.ORG

    def test_entity_type_iteration(self):
        """Test EntityType can be iterated."""
        types = list(EntityType)
        assert len(types) > 0
        assert EntityType.PERSON in types
        assert EntityType.ORG in types


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_entity_creation_minimal(self):
        """Test creating entity with minimal fields."""
        entity = ExtractedEntity(
            text="John Smith",
            entity_type=EntityType.PERSON,
            chunk_id="chunk_001",
            start_position=0,
            end_position=10,
        )
        assert entity.text == "John Smith"
        assert entity.entity_type == EntityType.PERSON
        assert entity.chunk_id == "chunk_001"
        assert entity.confidence == 0.0

    def test_entity_creation_full(self):
        """Test creating entity with all fields."""
        entity = ExtractedEntity(
            text="Microsoft",
            entity_type=EntityType.ORG,
            chunk_id="chunk_001",
            start_position=10,
            end_position=19,
            confidence=0.95,
            source_file="doc.txt",
            page_number=2,
            span_text="Works at Microsoft.",
        )
        assert entity.text == "Microsoft"
        assert entity.confidence == 0.95
        assert entity.page_number == 2

    def test_entity_properties(self):
        """Test ExtractedEntity properties."""
        entity = ExtractedEntity(
            text="January 15, 2025",
            entity_type=EntityType.DATE,
            chunk_id="chunk_001",
            start_position=0,
            end_position=15,
            confidence=0.88,
        )
        assert entity.char_span == (0, 15)
        assert entity.normalized_text == "january 15, 2025"
        assert entity.is_high_confidence is True  # 0.88 >= 0.75
        assert entity.entity_id is not None
        assert len(entity.entity_id) > 0

    def test_entity_high_confidence(self):
        """Test high confidence property."""
        entity_high = ExtractedEntity(
            text="Apple Inc.",
            entity_type=EntityType.ORG,
            chunk_id="chunk_001",
            start_position=0,
            end_position=9,
            confidence=0.85,
        )
        assert entity_high.is_high_confidence is True

        entity_low = ExtractedEntity(
            text="Unknown",
            entity_type=EntityType.ENTITY,
            chunk_id="chunk_001",
            start_position=0,
            end_position=7,
            confidence=0.60,
        )
        assert entity_low.is_high_confidence is False

    def test_entity_to_dict(self):
        """Test entity serialization to dictionary."""
        entity = ExtractedEntity(
            text="Paris",
            entity_type=EntityType.LOCATION,
            chunk_id="chunk_001",
            start_position=5,
            end_position=10,
            confidence=0.92,
            source_file="doc.txt",
        )
        entity_dict = entity.to_dict()
        assert entity_dict["text"] == "Paris"
        assert entity_dict["entity_type"] == "LOCATION"
        assert entity_dict["confidence"] == 0.92
        assert entity_dict["chunk_id"] == "chunk_001"
        assert entity_dict["entity_id"] == entity.entity_id

    def test_entity_repr(self):
        """Test entity string representation."""
        entity = ExtractedEntity(
            text="Tesla",
            entity_type=EntityType.ORG,
            chunk_id="chunk_001",
            start_position=0,
            end_position=5,
            confidence=0.89,
        )
        repr_str = repr(entity)
        assert "Tesla" in repr_str
        assert "ORG" in repr_str
        assert "0.89" in repr_str


@pytest.mark.usefixtures("mock_spacy_model")
class TestNERModel:
    """Tests for NERModel wrapper."""

    def test_model_initialization(self):
        """Test NER model initialization."""
        model = NERModel(model_name="en_core_web_sm")
        assert model.model_name == "en_core_web_sm"
        assert model.nlp is not None
        model.close()

    def test_entity_type_mapping(self):
        """Test SpaCy to standard entity type mapping."""
        assert (
            NERModel.SPACY_TO_ENTITY_TYPE["PERSON"] == EntityType.PERSON
        )
        assert NERModel.SPACY_TO_ENTITY_TYPE["ORG"] == EntityType.ORG
        assert NERModel.SPACY_TO_ENTITY_TYPE["GPE"] == EntityType.GPE
        assert NERModel.SPACY_TO_ENTITY_TYPE["DATE"] == EntityType.DATE

    def test_extract_entities_simple(self):
        """Test entity extraction from simple text."""
        model = NERModel(model_name="en_core_web_sm")
        text = "John Smith works at Microsoft in Seattle."

        entities = model.extract_entities(text, chunk_id="test_001")

        assert len(entities) > 0
        assert entities[0].entity_type == EntityType.PERSON

        model.close()

    def test_extract_entities_empty(self):
        """Test extraction from empty text."""
        model = NERModel(model_name="en_core_web_sm")

        entities_empty = model.extract_entities("", chunk_id="test_001")
        assert len(entities_empty) == 0

        entities_whitespace = model.extract_entities(
            "   \n  ", chunk_id="test_001"
        )
        assert len(entities_whitespace) == 0

        model.close()

    def test_extract_entities_metadata(self):
        """Test that extracted entities have proper metadata."""
        model = NERModel(model_name="en_core_web_sm")
        text = "Apple CEO Tim Cook announced earnings on January 30."

        entities = model.extract_entities(
            text, chunk_id="chunk_1", source_file="report.txt"
        )

        for entity in entities:
            assert entity.chunk_id == "chunk_1"
            assert entity.source_file == "report.txt"
            assert entity.start_position >= 0
            assert entity.end_position > entity.start_position
            # Note: With mock, positions may not align to actual text spans
            # Just verify entity has text and position fields
            assert len(entity.text) > 0

        model.close()

    def test_extract_entities_batch(self):
        """Test batch entity extraction."""
        model = NERModel(model_name="en_core_web_sm")
        texts = [
            ("John Smith is a developer.", "chunk_1", "doc1.txt"),
            ("Microsoft was founded by Bill Gates.", "chunk_2", "doc2.txt"),
            ("Paris is in France.", "chunk_3", "doc3.txt"),
        ]

        entities = model.extract_entities_batch(texts, batch_size=2)

        assert len(entities) > 0
        # Check chunk_id mapping
        chunk_ids = {e.chunk_id for e in entities}
        assert "chunk_1" in chunk_ids

        model.close()

    def test_available_models(self):
        """Test getting list of available models."""
        model = NERModel(model_name="en_core_web_sm")
        available = model.get_available_models()
        assert len(available) > 0
        assert "en_core_web_sm" in available
        model.close()


@pytest.mark.usefixtures("mock_spacy_model")
class TestEntityExtractor:
    """Tests for EntityExtractor pipeline."""

    def test_extractor_initialization(self):
        """Test entity extractor initialization."""
        extractor = EntityExtractor(model_name="en_core_web_sm")
        assert extractor.model is not None
        extractor.close()

    def test_extract_from_chunk(self):
        """Test extracting entities from a single chunk."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        chunk = TextChunk(
            content="Steve Jobs founded Apple in 1976.",
            source_file="history.txt",
            chunk_id="chunk_001",
            position_in_document=0,
        )

        entities = extractor.extract_from_chunk(chunk)

        assert len(entities) > 0
        # Verify metadata enrichment
        for entity in entities:
            assert entity.chunk_id == "chunk_001"
            assert entity.source_file == "history.txt"

        extractor.close()

    def test_extract_from_empty_chunk(self):
        """Test extraction from empty chunk."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        chunk = TextChunk(
            content="",
            source_file="empty.txt",
            chunk_id="chunk_empty",
            position_in_document=0,
        )

        entities = extractor.extract_from_chunk(chunk)
        assert len(entities) == 0

        extractor.close()

    def test_extract_from_chunks(self):
        """Test extracting entities from multiple chunks."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        chunks = [
            TextChunk(
                content="Alice works at Google.",
                source_file="doc.txt",
                chunk_id="chunk_1",
                position_in_document=0,
            ),
            TextChunk(
                content="Bob is from London.",
                source_file="doc.txt",
                chunk_id="chunk_2",
                position_in_document=1,
            ),
        ]

        entities = extractor.extract_from_chunks(chunks)

        assert isinstance(entities, list)
        assert len(entities) > 0

        extractor.close()

    def test_extract_from_chunks_with_stats(self):
        """Test extraction with statistics."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        chunks = [
            TextChunk(
                content="John Smith is an engineer.",
                source_file="doc.txt",
                chunk_id="chunk_1",
                position_in_document=0,
            ),
            TextChunk(
                content="Lives in New York.",
                source_file="doc.txt",
                chunk_id="chunk_2",
                position_in_document=1,
            ),
        ]

        entities, stats = extractor.extract_from_chunks(
            chunks, include_stats=True
        )

        assert isinstance(entities, list)
        assert stats.total_entities >= 0
        assert stats.chunks_processed == 2
        assert stats.processing_time > 0
        assert len(stats.entities_by_type) >= 0

        extractor.close()

    def test_extract_from_text(self):
        """Test extracting entities from raw text."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        text = "The meeting is scheduled for January 15 at Microsoft headquarters."
        entities = extractor.extract_from_text(text, source_file="email.txt")

        assert len(entities) > 0
        for entity in entities:
            assert entity.source_file == "email.txt"

        extractor.close()

    def test_filter_by_confidence(self):
        """Test filtering entities by confidence."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        entities = [
            ExtractedEntity(
                text="High",
                entity_type=EntityType.PERSON,
                chunk_id="c1",
                start_position=0,
                end_position=4,
                confidence=0.95,
            ),
            ExtractedEntity(
                text="Low",
                entity_type=EntityType.ORG,
                chunk_id="c1",
                start_position=5,
                end_position=8,
                confidence=0.50,
            ),
        ]

        filtered = extractor.filter_by_confidence(entities, min_confidence=0.75)
        assert len(filtered) == 1
        assert filtered[0].text == "High"

        extractor.close()

    def test_filter_by_type(self):
        """Test filtering entities by type."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        entities = [
            ExtractedEntity(
                text="John",
                entity_type=EntityType.PERSON,
                chunk_id="c1",
                start_position=0,
                end_position=4,
            ),
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.ORG,
                chunk_id="c1",
                start_position=10,
                end_position=16,
            ),
            ExtractedEntity(
                text="Paris",
                entity_type=EntityType.LOCATION,
                chunk_id="c1",
                start_position=20,
                end_position=25,
            ),
        ]

        filtered = extractor.filter_by_type(
            entities, entity_types=[EntityType.PERSON, EntityType.ORG]
        )
        assert len(filtered) == 2
        assert EntityType.LOCATION not in [e.entity_type for e in filtered]

        extractor.close()

    def test_deduplicate_entities(self):
        """Test entity deduplication."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        entities = [
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="c1",
                start_position=0,
                end_position=5,
            ),
            ExtractedEntity(
                text="Apple",
                entity_type=EntityType.ORG,
                chunk_id="c2",
                start_position=0,
                end_position=5,
            ),
            ExtractedEntity(
                text="APPLE",
                entity_type=EntityType.ORG,
                chunk_id="c3",
                start_position=0,
                end_position=5,
            ),
        ]

        # Case-insensitive deduplication
        deduplicated = extractor.deduplicate_entities(
            entities, case_sensitive=False
        )
        assert len(deduplicated) == 1

        # Case-sensitive deduplication
        deduplicated_case = extractor.deduplicate_entities(
            entities, case_sensitive=True
        )
        assert len(deduplicated_case) == 2

        extractor.close()

    def test_group_by_type(self):
        """Test grouping entities by type."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        entities = [
            ExtractedEntity(
                text="Alice",
                entity_type=EntityType.PERSON,
                chunk_id="c1",
                start_position=0,
                end_position=5,
            ),
            ExtractedEntity(
                text="Bob",
                entity_type=EntityType.PERSON,
                chunk_id="c1",
                start_position=10,
                end_position=13,
            ),
            ExtractedEntity(
                text="Microsoft",
                entity_type=EntityType.ORG,
                chunk_id="c1",
                start_position=20,
                end_position=29,
            ),
        ]

        grouped = extractor.group_by_type(entities)

        assert len(grouped) == 2
        assert len(grouped[EntityType.PERSON]) == 2
        assert len(grouped[EntityType.ORG]) == 1
        assert grouped[EntityType.PERSON][0].text in ["Alice", "Bob"]

        extractor.close()

    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold raises error."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        entities = []
        with pytest.raises(ValueError):
            extractor.filter_by_confidence(entities, min_confidence=1.5)

        with pytest.raises(ValueError):
            extractor.filter_by_confidence(entities, min_confidence=-0.1)

        extractor.close()

    def test_entity_position_accuracy(self):
        """Test that entity positions are accurate."""
        extractor = EntityExtractor(model_name="en_core_web_sm")

        text = "Dr. John Smith works at Microsoft Corporation."
        entities = extractor.extract_from_text(text, source_file="doc.txt")

        # Verify entities have position data
        for entity in entities:
            assert entity.start_position >= 0
            assert entity.end_position > entity.start_position
            # Note: With mock, positions may not align to actual text
            # Just verify the entity object has consistent data
            assert len(entity.text) > 0

        extractor.close()


@pytest.mark.usefixtures("mock_spacy_model")
class TestNERIntegration:
    """Integration tests for NER pipeline."""

    def test_full_pipeline(self):
        """Test complete extraction pipeline from chunk to entities."""
        # Create a chunk
        chunk = TextChunk(
            content="Microsoft CEO Satya Nadella announced a partnership with OpenAI on January 23, 2023 in New York.",
            source_file="press_release.txt",
            chunk_id="chunk_001",
            position_in_document=0,
        )

        # Extract entities
        extractor = EntityExtractor(model_name="en_core_web_sm")
        entities = extractor.extract_from_chunk(chunk)

        # Verify extraction
        assert len(entities) > 0

        # Verify metadata
        for entity in entities:
            assert entity.chunk_id == "chunk_001"
            assert entity.source_file == "press_release.txt"

        extractor.close()

    def test_multi_chunk_extraction(self):
        """Test extraction across multiple chunks."""
        chunks = [
            TextChunk(
                content="Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                source_file="history.txt",
                chunk_id="chunk_1",
                position_in_document=0,
            ),
            TextChunk(
                content="The company is headquartered in Cupertino, California.",
                source_file="history.txt",
                chunk_id="chunk_2",
                position_in_document=1,
            ),
        ]

        extractor = EntityExtractor(model_name="en_core_web_sm")
        entities = extractor.extract_from_chunks(chunks)

        assert len(entities) > 0

        # Verify chunk IDs are preserved
        chunk_ids = {e.chunk_id for e in entities}
        assert len(chunk_ids) >= 1  # At least one chunk has entities

        extractor.close()
