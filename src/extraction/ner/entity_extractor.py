"""Entity extractor pipeline combining chunked text with NER."""

import logging
import time
from typing import Dict, List, Optional

from ..chunking import TextChunk
from .entity_models import ExtractedEntity, EntityType, NERStats
from .ner_model import NERModel

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Pipeline for extracting entities from chunks using NER."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize entity extractor.

        Args:
            model_name: SpaCy model to use for NER
        """
        self.model = NERModel(model_name=model_name)
        logger.info(f"Initialized EntityExtractor with model: {model_name}")

    def extract_from_chunk(
        self, chunk: TextChunk
    ) -> List[ExtractedEntity]:
        """Extract entities from a single chunk.

        Args:
            chunk: TextChunk to extract entities from

        Returns:
            List of ExtractedEntity objects
        """
        if not chunk.content or not chunk.content.strip():
            return []

        entities = self.model.extract_entities(
            text=chunk.content,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
        )

        # Enrich entities with chunk metadata
        for entity in entities:
            entity.page_number = chunk.page_number
            if "position_in_document" in chunk.metadata:
                entity.metadata["position_in_document"] = chunk.metadata[
                    "position_in_document"
                ]

        return entities

    def extract_from_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 128,
        include_stats: bool = False,
    ) -> tuple:
        """Extract entities from multiple chunks efficiently.

        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for processing
            include_stats: Whether to return extraction statistics

        Returns:
            Tuple of (entities, stats) if include_stats=True,
            else just entities list
        """
        start_time = time.time()
        all_entities = []

        # Prepare batch data: (text, chunk_id, source_file)
        batch_data = [
            (chunk.content, chunk.chunk_id, chunk.source_file)
            for chunk in chunks
            if chunk.content and chunk.content.strip()
        ]

        if not batch_data:
            if include_stats:
                return [], NERStats(total_entities=0, chunks_processed=0)
            return [] if not include_stats else ([], NERStats(0))

        # Extract entities in batch
        entities = self.model.extract_entities_batch(
            texts=batch_data, batch_size=batch_size
        )

        # Map entities back to chunks for metadata enrichment
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        for entity in entities:
            if entity.chunk_id in chunk_map:
                chunk = chunk_map[entity.chunk_id]
                entity.page_number = chunk.page_number
                if "position_in_document" in chunk.metadata:
                    entity.metadata["position_in_document"] = chunk.metadata[
                        "position_in_document"
                    ]

        processing_time = time.time() - start_time

        if include_stats:
            stats = self._calculate_stats(
                entities, len(chunks), processing_time
            )
            return entities, stats

        return entities

    def extract_from_text(
        self,
        text: str,
        source_file: str = "",
        chunk_id: str = "single_chunk",
    ) -> List[ExtractedEntity]:
        """Extract entities from raw text (without chunking).

        Args:
            text: Raw text to extract entities from
            source_file: Source file path
            chunk_id: Chunk ID to assign

        Returns:
            List of ExtractedEntity objects
        """
        if not text or not text.strip():
            return []

        return self.model.extract_entities(
            text=text,
            chunk_id=chunk_id,
            source_file=source_file,
        )

    def filter_by_confidence(
        self,
        entities: List[ExtractedEntity],
        min_confidence: float = 0.75,
    ) -> List[ExtractedEntity]:
        """Filter entities by confidence threshold.

        Args:
            entities: List of entities to filter
            min_confidence: Minimum confidence score (0.0-1.0)

        Returns:
            Filtered list of entities
        """
        if not 0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        return [
            e for e in entities if e.confidence >= min_confidence
        ]

    def filter_by_type(
        self,
        entities: List[ExtractedEntity],
        entity_types: List[EntityType],
    ) -> List[ExtractedEntity]:
        """Filter entities by type.

        Args:
            entities: List of entities to filter
            entity_types: Types to keep

        Returns:
            Filtered list of entities
        """
        type_set = set(entity_types)
        return [e for e in entities if e.entity_type in type_set]

    def deduplicate_entities(
        self,
        entities: List[ExtractedEntity],
        case_sensitive: bool = False,
    ) -> List[ExtractedEntity]:
        """Remove duplicate entities by normalized text.

        Args:
            entities: List of entities
            case_sensitive: Whether text comparison is case-sensitive

        Returns:
            List of unique entities (keeps first occurrence)
        """
        seen = set()
        unique = []

        for entity in entities:
            key = (
                entity.text
                if case_sensitive
                else entity.normalized_text
            )

            if key not in seen:
                seen.add(key)
                unique.append(entity)

        logger.debug(
            f"Deduplicated {len(entities)} entities to {len(unique)}"
        )
        return unique

    def group_by_type(
        self,
        entities: List[ExtractedEntity],
    ) -> Dict[EntityType, List[ExtractedEntity]]:
        """Group entities by their type.

        Args:
            entities: List of entities

        Returns:
            Dictionary mapping EntityType to list of entities
        """
        grouped: Dict[EntityType, List[ExtractedEntity]] = {}

        for entity in entities:
            if entity.entity_type not in grouped:
                grouped[entity.entity_type] = []
            grouped[entity.entity_type].append(entity)

        return grouped

    def _calculate_stats(
        self,
        entities: List[ExtractedEntity],
        chunks_processed: int,
        processing_time: float,
    ) -> NERStats:
        """Calculate extraction statistics.

        Args:
            entities: Extracted entities
            chunks_processed: Number of chunks processed
            processing_time: Time taken for extraction

        Returns:
            NERStats object
        """
        entities_by_type: Dict[str, int] = {}
        total_confidence = 0.0
        high_confidence_count = 0

        for entity in entities:
            # Count by type
            type_key = entity.entity_type.value
            entities_by_type[type_key] = entities_by_type.get(type_key, 0) + 1

            # Confidence stats
            total_confidence += entity.confidence
            if entity.is_high_confidence:
                high_confidence_count += 1

        average_confidence = (
            total_confidence / len(entities) if entities else 0.0
        )

        return NERStats(
            total_entities=len(entities),
            entities_by_type=entities_by_type,
            average_confidence=average_confidence,
            high_confidence_count=high_confidence_count,
            chunks_processed=chunks_processed,
            processing_time=processing_time,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self.model:
            self.model.close()
            logger.debug("Closed EntityExtractor")
