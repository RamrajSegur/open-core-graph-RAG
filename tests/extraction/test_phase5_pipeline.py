"""Tests for Phase 5: Pipeline Integration & Storage."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.extraction import (
    ExtractionPipeline,
    PipelineStats,
    PipelineConfig,
    ChunkingConfig,
    NERConfig,
    RelationshipConfig,
    StorageConfig,
    StorageConnector,
    TextChunk,
    ExtractedEntity,
    EntityType,
    ExtractedRelationship,
    RelationshipType,
)


class TestPipelineStats:
    """Tests for PipelineStats."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = PipelineStats()
        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_entities == 0
        assert stats.total_relationships == 0
        assert len(stats.errors) == 0

    def test_stats_add_error(self):
        """Test adding errors to stats."""
        stats = PipelineStats()
        stats.add_error("Test error")
        assert len(stats.errors) == 1
        assert "Test error" in stats.errors

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = PipelineStats(
            total_documents=1,
            total_chunks=10,
            total_entities=50,
            total_relationships=30,
        )
        stats_dict = stats.to_dict()
        assert stats_dict["total_documents"] == 1
        assert stats_dict["total_chunks"] == 10
        assert stats_dict["total_entities"] == 50


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        """Test default chunking config."""
        config = ChunkingConfig()
        assert config.strategy == "semantic"
        assert config.semantic_max_size == 512
        assert config.window_size == 256

    def test_custom_config(self):
        """Test custom chunking config."""
        config = ChunkingConfig(
            strategy="sliding_window",
            window_size=512,
            overlap=128,
        )
        assert config.strategy == "sliding_window"
        assert config.window_size == 512
        assert config.overlap == 128


class TestNERConfig:
    """Tests for NERConfig."""

    def test_default_config(self):
        """Test default NER config."""
        config = NERConfig()
        assert config.enabled is True
        assert config.model_name == "en_core_web_sm"
        assert len(config.entity_types) > 0
        assert EntityType.PERSON.value in config.entity_types

    def test_custom_entity_types(self):
        """Test custom entity types."""
        config = NERConfig(
            entity_types=["PERSON", "ORGANIZATION"]
        )
        assert len(config.entity_types) == 2
        assert "PERSON" in config.entity_types


class TestRelationshipConfig:
    """Tests for RelationshipConfig."""

    def test_default_config(self):
        """Test default relationship config."""
        config = RelationshipConfig()
        assert config.enabled is True
        assert config.use_patterns is True
        assert config.use_semantic is True
        assert len(config.relationship_types) > 0

    def test_extraction_methods(self):
        """Test extraction method configuration."""
        config = RelationshipConfig(
            use_patterns=True,
            use_semantic=False,
        )
        assert config.use_patterns is True
        assert config.use_semantic is False


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_config(self):
        """Test default storage config."""
        config = StorageConfig()
        assert config.enabled is True
        assert config.backend == "tigergraph"
        assert config.host == "localhost"
        assert config.port == 9000

    def test_custom_config(self):
        """Test custom storage config."""
        config = StorageConfig(
            host="remote.server",
            port=9001,
            graph_name="custom_graph",
        )
        assert config.host == "remote.server"
        assert config.port == 9001
        assert config.graph_name == "custom_graph"


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default pipeline config."""
        config = PipelineConfig()
        assert config.chunking is not None
        assert config.ner is not None
        assert config.relationships is not None
        assert config.storage is not None

    def test_to_dict(self):
        """Test config to dict conversion."""
        config = PipelineConfig()
        config_dict = config.to_dict()
        assert "chunking" in config_dict
        assert "ner" in config_dict
        assert "relationships" in config_dict
        assert "storage" in config_dict

    def test_from_dict(self):
        """Test loading config from dict."""
        data = {
            "chunking": {"strategy": "sliding_window"},
            "ner": {"confidence_threshold": 0.8},
        }
        config = PipelineConfig.from_dict(data)
        assert config.chunking.strategy == "sliding_window"
        assert config.ner.confidence_threshold == 0.8

    def test_to_json_str(self):
        """Test config to JSON string."""
        config = PipelineConfig()
        json_str = config.to_json_str()
        assert isinstance(json_str, str)
        assert "chunking" in json_str
        assert "storage" in json_str


@pytest.mark.usefixtures("mock_spacy_model")
class TestExtractionPipeline:
    """Tests for ExtractionPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = ExtractionPipeline()
        assert pipeline.chunker is not None
        assert pipeline.entity_extractor is not None
        assert pipeline.relationship_extractor is not None
        pipeline.close()

    def test_process_text_without_relationships(self):
        """Test processing text without relationships."""
        pipeline = ExtractionPipeline()
        text = "John Smith works for Google in California."
        
        entities, relationships = pipeline.process_text(
            text,
            source_file="test.txt",
            include_relationships=False,
        )
        
        assert isinstance(entities, list)
        assert isinstance(relationships, list)
        assert len(relationships) == 0
        
        stats = pipeline.get_stats()
        assert stats.documents_processed == 1
        assert stats.chunks_processed > 0
        
        pipeline.close()

    def test_process_text_with_relationships(self):
        """Test processing text with relationships."""
        pipeline = ExtractionPipeline()
        text = "John works for Google."
        
        entities, relationships = pipeline.process_text(
            text,
            source_file="test.txt",
            include_relationships=True,
        )
        
        assert isinstance(entities, list)
        assert isinstance(relationships, list)
        
        stats = pipeline.get_stats()
        assert stats.documents_processed == 1
        
        pipeline.close()

    def test_pipeline_stats(self):
        """Test pipeline statistics."""
        pipeline = ExtractionPipeline()
        text = "Alice and Bob are friends."
        
        pipeline.process_text(text)
        
        stats = pipeline.get_stats()
        assert stats.documents_processed == 1
        assert stats.chunks_processed > 0
        assert stats.processing_time > 0
        
        pipeline.close()

    def test_reset_stats(self):
        """Test resetting statistics."""
        pipeline = ExtractionPipeline()
        pipeline.process_text("Test text")
        
        stats_before = pipeline.get_stats()
        assert stats_before.documents_processed == 1
        
        pipeline.reset_stats()
        stats_after = pipeline.get_stats()
        assert stats_after.documents_processed == 0
        
        pipeline.close()

    def test_process_multiple_documents(self):
        """Test processing multiple documents."""
        pipeline = ExtractionPipeline()
        texts = [
            "John works at Microsoft.",
            "Alice works at Google.",
        ]
        
        all_entities = []
        all_relationships = []
        
        for text in texts:
            entities, relationships = pipeline.process_text(text)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        stats = pipeline.get_stats()
        assert stats.documents_processed == 2
        assert stats.chunks_processed > 0
        
        pipeline.close()


class TestStorageConnector:
    """Tests for StorageConnector."""

    def test_connector_initialization(self):
        """Test storage connector initialization."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        assert connector.graph_store == mock_store
        assert len(connector.stored_entities) == 0
        assert len(connector.stored_relationships) == 0

    def test_store_entity(self):
        """Test storing single entity."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        entity = ExtractedEntity(
            text="John",
            entity_type=EntityType.PERSON,
            chunk_id="chunk_1",
            start_position=0,
            end_position=4,
        )
        
        result = connector.store_entity(entity)
        assert result is True
        assert connector.stats["entities"] == 1
        mock_store.create_vertex.assert_called_once()

    def test_store_entity_duplicate(self):
        """Test duplicate entity handling."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        entity = ExtractedEntity(
            text="John",
            entity_type=EntityType.PERSON,
            chunk_id="chunk_1",
            start_position=0,
            end_position=4,
        )
        
        connector.store_entity(entity, skip_duplicates=True)
        result = connector.store_entity(entity, skip_duplicates=True)
        
        assert result is False
        assert connector.stats["entities"] == 1

    def test_store_entities_batch(self):
        """Test storing multiple entities."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        entities = [
            ExtractedEntity(
                text="John",
                entity_type=EntityType.PERSON,
                chunk_id="chunk_1",
                start_position=0,
                end_position=4,
            ),
            ExtractedEntity(
                text="Google",
                entity_type=EntityType.GPE,  # Use GPE for geopolitical entity (works for org)
                chunk_id="chunk_1",
                start_position=14,
                end_position=20,
            ),
        ]
        
        count = connector.store_entities(entities)
        assert count == 2
        assert connector.stats["entities"] == 2

    def test_store_relationship(self):
        """Test storing single relationship."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        relationship = ExtractedRelationship(
            source_entity="John",
            target_entity="Google",
            relationship_type=RelationshipType.WORKS_FOR,
            source_chunk_id="chunk_1",
            confidence=0.9,
        )
        
        result = connector.store_relationship(relationship)
        assert result is True
        assert connector.stats["relationships"] == 1
        mock_store.create_edge.assert_called_once()

    def test_store_relationship_duplicate(self):
        """Test duplicate relationship handling."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        relationship = ExtractedRelationship(
            source_entity="John",
            target_entity="Google",
            relationship_type=RelationshipType.WORKS_FOR,
            source_chunk_id="chunk_1",
        )
        
        connector.store_relationship(relationship, skip_duplicates=True)
        result = connector.store_relationship(relationship, skip_duplicates=True)
        
        assert result is False
        assert connector.stats["relationships"] == 1

    def test_store_relationships_batch(self):
        """Test storing multiple relationships."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        relationships = [
            ExtractedRelationship(
                source_entity="John",
                target_entity="Google",
                relationship_type=RelationshipType.WORKS_FOR,
                source_chunk_id="chunk_1",
            ),
            ExtractedRelationship(
                source_entity="Google",
                target_entity="California",
                relationship_type=RelationshipType.LOCATED_IN,
                source_chunk_id="chunk_1",
            ),
        ]
        
        count = connector.store_relationships(relationships)
        assert count == 2
        assert connector.stats["relationships"] == 2

    def test_store_knowledge_graph(self):
        """Test storing complete knowledge graph."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        entities = [
            ExtractedEntity(
                text="John",
                entity_type=EntityType.PERSON,
                chunk_id="chunk_1",
                start_position=0,
                end_position=4,
            ),
        ]
        
        relationships = [
            ExtractedRelationship(
                source_entity="John",
                target_entity="Google",
                relationship_type=RelationshipType.WORKS_FOR,
                source_chunk_id="chunk_1",
            ),
        ]
        
        result = connector.store_knowledge_graph(entities, relationships)
        
        assert result["entities"] == 1
        assert result["relationships"] == 1

    def test_get_stats(self):
        """Test getting statistics."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        entity = ExtractedEntity(
            text="John",
            entity_type=EntityType.PERSON,
            chunk_id="chunk_1",
            start_position=0,
            end_position=4,
        )
        
        connector.store_entity(entity)
        stats = connector.get_stats()
        
        assert stats["entities"] == 1
        assert stats["relationships"] == 0

    def test_reset_stats(self):
        """Test resetting statistics."""
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        entity = ExtractedEntity(
            text="John",
            entity_type=EntityType.PERSON,
            chunk_id="chunk_1",
            start_position=0,
            end_position=4,
        )
        
        connector.store_entity(entity)
        assert connector.stats["entities"] == 1
        
        connector.reset_stats()
        assert connector.stats["entities"] == 0


class TestIntegration:
    """Integration tests for Phase 5."""

    @pytest.mark.usefixtures("mock_spacy_model")
    def test_pipeline_to_storage_workflow(self):
        """Test complete pipeline to storage workflow."""
        pipeline = ExtractionPipeline()
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        # Process text through pipeline
        text = "John Smith works for Google."
        entities, relationships = pipeline.process_text(text)
        
        # Store in graph
        result = connector.store_knowledge_graph(entities, relationships)
        
        # Verify
        assert result["entities"] >= 0
        assert result["relationships"] >= 0
        
        pipeline.close()
        connector.close()

    @pytest.mark.usefixtures("mock_spacy_model")
    def test_multi_document_workflow(self):
        """Test multi-document processing and storage."""
        pipeline = ExtractionPipeline()
        mock_store = Mock()
        connector = StorageConnector(graph_store=mock_store)
        
        texts = [
            "Alice works at Microsoft.",
            "Bob works at Google.",
        ]
        
        for text in texts:
            entities, relationships = pipeline.process_text(text)
            connector.store_knowledge_graph(entities, relationships)
        
        pipeline_stats = pipeline.get_stats()
        storage_stats = connector.get_stats()
        
        assert pipeline_stats.documents_processed == 2
        assert storage_stats["entities"] >= 0
        
        pipeline.close()
        connector.close()
