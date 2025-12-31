"""Relationship extraction pipeline."""

import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from ..chunking import TextChunk
from ..ner import EntityExtractor, ExtractedEntity, EntityType
from .relationship_models import (
    ExtractedRelationship,
    RelationshipExtractionStats,
    RelationshipType,
)

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Pipeline for extracting relationships between entities."""

    # Pattern-based relationship patterns
    RELATIONSHIP_PATTERNS = {
        RelationshipType.WORKS_FOR: [
            r"(?P<source>\w+\s+\w+)\s+(?:works|worked)\s+for\s+(?P<target>.+?)(?:\.|,|;)",
            r"(?P<source>\w+\s+\w+)\s+(?:is|was)\s+(?:an?|the)\s+(?:\w+\s+)*(?:at|in)\s+(?P<target>.+?)(?:\.|,|;)",
        ],
        RelationshipType.FOUNDED_BY: [
            r"(?P<target>.+?)\s+(?:was\s+)?founded\s+by\s+(?P<source>\w+\s+\w+)",
            r"(?P<source>\w+\s+\w+)\s+founded\s+(?P<target>.+?)(?:\.|,|;)",
        ],
        RelationshipType.LOCATED_IN: [
            r"(?P<source>.+?)\s+(?:is\s+)?(?:located\s+)?in\s+(?P<target>[A-Z]\w+(?:\s+[A-Z]\w+)*)",
            r"(?P<source>.+?)\s+(?:based|headquartered)\s+in\s+(?P<target>[A-Z]\w+(?:\s+[A-Z]\w+)*)",
        ],
        RelationshipType.OWNED_BY: [
            r"(?P<source>.+?)\s+(?:is\s+)?owned\s+by\s+(?P<target>\w+\s+\w+|\w+)",
            r"(?P<source>.+?)\s+(?:belongs\s+)?to\s+(?P<target>\w+\s+\w+|\w+)",
        ],
        RelationshipType.CREATED_BY: [
            r"(?P<source>.+?)\s+(?:was\s+)?created\s+by\s+(?P<target>\w+\s+\w+|\w+)",
            r"(?P<source>.+?)\s+created\s+by\s+(?P<target>.+?)(?:\.|,|;)",
        ],
        RelationshipType.PARENT_OF: [
            r"(?P<source>\w+\s+\w+)\s+(?:is\s+)?(?:the\s+)?(?:father|mother|parent)\s+of\s+(?P<target>\w+\s+\w+)",
        ],
    }

    def __init__(
        self,
        use_patterns: bool = True,
        use_semantic: bool = False,
        ner_model: str = "en_core_web_sm",
    ):
        """Initialize relationship extractor.

        Args:
            use_patterns: Whether to use pattern-based extraction
            use_semantic: Whether to use semantic/LLM-based extraction
            ner_model: SpaCy model for NER
        """
        self.use_patterns = use_patterns
        self.use_semantic = use_semantic
        self.entity_extractor = EntityExtractor(model_name=ner_model)
        logger.info(
            f"Initialized RelationshipExtractor "
            f"(patterns={use_patterns}, semantic={use_semantic})"
        )

    def extract_from_chunk(
        self,
        chunk: TextChunk,
        entities: Optional[List[ExtractedEntity]] = None,
    ) -> List[ExtractedRelationship]:
        """Extract relationships from a single chunk.

        Args:
            chunk: TextChunk to extract from
            entities: Pre-extracted entities (will extract if not provided)

        Returns:
            List of ExtractedRelationship objects
        """
        if entities is None:
            entities = self.entity_extractor.extract_from_chunk(chunk)

        relationships = []

        # Pattern-based extraction
        if self.use_patterns:
            pattern_rels = self._extract_patterns(
                chunk.content, chunk.chunk_id, chunk.source_file, entities
            )
            relationships.extend(pattern_rels)

        # Semantic extraction
        if self.use_semantic:
            semantic_rels = self._extract_semantic(
                chunk.content, chunk.chunk_id, chunk.source_file, entities
            )
            relationships.extend(semantic_rels)

        return relationships

    def extract_from_chunks(
        self,
        chunks: List[TextChunk],
        entities_by_chunk: Optional[Dict[str, List[ExtractedEntity]]] = None,
        include_stats: bool = False,
    ) -> tuple:
        """Extract relationships from multiple chunks.

        Args:
            chunks: List of TextChunk objects
            entities_by_chunk: Pre-extracted entities by chunk_id
            include_stats: Whether to return statistics

        Returns:
            Tuple of (relationships, stats) if include_stats=True, else relationships
        """
        start_time = time.time()
        all_relationships = []

        # Extract entities if not provided
        if entities_by_chunk is None:
            entities_by_chunk = {}
            for chunk in chunks:
                entities_by_chunk[chunk.chunk_id] = (
                    self.entity_extractor.extract_from_chunk(chunk)
                )

        # Extract relationships from each chunk
        for chunk in chunks:
            entities = entities_by_chunk.get(chunk.chunk_id, [])
            rels = self.extract_from_chunk(chunk, entities)
            all_relationships.extend(rels)

        processing_time = time.time() - start_time

        if include_stats:
            stats = self._calculate_stats(
                all_relationships, len(chunks), processing_time
            )
            return all_relationships, stats

        return all_relationships

    def filter_by_confidence(
        self,
        relationships: List[ExtractedRelationship],
        min_confidence: float = 0.75,
    ) -> List[ExtractedRelationship]:
        """Filter relationships by confidence threshold.

        Args:
            relationships: List of relationships
            min_confidence: Minimum confidence (0.0-1.0)

        Returns:
            Filtered list of relationships
        """
        if not 0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        return [
            r for r in relationships if r.confidence >= min_confidence
        ]

    def filter_by_type(
        self,
        relationships: List[ExtractedRelationship],
        relationship_types: List[RelationshipType],
    ) -> List[ExtractedRelationship]:
        """Filter relationships by type.

        Args:
            relationships: List of relationships
            relationship_types: Types to keep

        Returns:
            Filtered list of relationships
        """
        type_set = set(relationship_types)
        return [r for r in relationships if r.relationship_type in type_set]

    def deduplicate_relationships(
        self,
        relationships: List[ExtractedRelationship],
        case_sensitive: bool = False,
    ) -> List[ExtractedRelationship]:
        """Remove duplicate relationships.

        Args:
            relationships: List of relationships
            case_sensitive: Whether comparison is case-sensitive

        Returns:
            List of unique relationships (keeps first occurrence)
        """
        seen: Set[Tuple] = set()
        unique = []

        for rel in relationships:
            key = (
                (rel.source_entity, rel.target_entity, rel.relationship_type)
                if case_sensitive
                else (
                    rel.normalized_source,
                    rel.normalized_target,
                    rel.relationship_type,
                )
            )

            if key not in seen:
                seen.add(key)
                unique.append(rel)

        logger.debug(
            f"Deduplicated {len(relationships)} relationships to {len(unique)}"
        )
        return unique

    def group_by_type(
        self,
        relationships: List[ExtractedRelationship],
    ) -> Dict[RelationshipType, List[ExtractedRelationship]]:
        """Group relationships by type.

        Args:
            relationships: List of relationships

        Returns:
            Dictionary mapping RelationshipType to relationships
        """
        grouped: Dict[RelationshipType, List[ExtractedRelationship]] = {}

        for rel in relationships:
            if rel.relationship_type not in grouped:
                grouped[rel.relationship_type] = []
            grouped[rel.relationship_type].append(rel)

        return grouped

    def group_by_entity(
        self,
        relationships: List[ExtractedRelationship],
    ) -> Dict[str, List[ExtractedRelationship]]:
        """Group relationships by source entity.

        Args:
            relationships: List of relationships

        Returns:
            Dictionary mapping entity to relationships it's source of
        """
        grouped: Dict[str, List[ExtractedRelationship]] = {}

        for rel in relationships:
            if rel.source_entity not in grouped:
                grouped[rel.source_entity] = []
            grouped[rel.source_entity].append(rel)

        return grouped

    def _extract_patterns(
        self,
        text: str,
        chunk_id: str,
        source_file: str,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedRelationship]:
        """Extract relationships using regex patterns.

        Args:
            text: Text to extract from
            chunk_id: Chunk ID
            source_file: Source file path
            entities: List of extracted entities

        Returns:
            List of extracted relationships
        """
        relationships = []

        for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            source = match.group("source").strip()
                            target = match.group("target").strip()
                            span_start = match.start()
                            span_end = match.end()

                            rel = ExtractedRelationship(
                                source_entity=source,
                                target_entity=target,
                                relationship_type=rel_type,
                                source_chunk_id=chunk_id,
                                confidence=0.7,  # Pattern-based default
                                source_file=source_file,
                                start_char=span_start,
                                end_char=span_end,
                                supporting_text=match.group(0),
                                metadata={"extraction_method": "pattern"},
                            )
                            relationships.append(rel)
                        except (IndexError, AttributeError):
                            continue
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern}")
                    continue

        return relationships

    def _extract_semantic(
        self,
        text: str,
        chunk_id: str,
        source_file: str,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedRelationship]:
        """Extract relationships using semantic/context analysis.

        Args:
            text: Text to extract from
            chunk_id: Chunk ID
            source_file: Source file path
            entities: List of extracted entities

        Returns:
            List of extracted relationships
        """
        relationships = []

        # For now, implement basic co-occurrence based extraction
        # This can be enhanced with LLM-based extraction later

        if len(entities) < 2:
            return relationships

        # Find entity pairs mentioned close together
        entity_positions = {}
        for entity in entities:
            if entity.text not in entity_positions:
                entity_positions[entity.text] = []
            entity_positions[entity.text].append(
                (entity.start_position, entity.end_position)
            )

        # Check for nearby entity pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # Check if entities are within 50 characters
                if (
                    abs(entity1.end_position - entity2.start_position) < 50
                    or abs(entity2.end_position - entity1.start_position) < 50
                ):
                    # Extract supporting text
                    start = min(entity1.start_position, entity2.start_position)
                    end = max(entity1.end_position, entity2.end_position)
                    supporting_text = text[max(0, start - 20) : min(len(text), end + 20)]

                    rel = ExtractedRelationship(
                        source_entity=entity1.text,
                        target_entity=entity2.text,
                        relationship_type=RelationshipType.RELATED_TO,
                        source_chunk_id=chunk_id,
                        confidence=0.5,  # Lower confidence for semantic
                        source_file=source_file,
                        start_char=start,
                        end_char=end,
                        supporting_text=supporting_text,
                        metadata={"extraction_method": "semantic"},
                    )
                    relationships.append(rel)

        return relationships

    def _calculate_stats(
        self,
        relationships: List[ExtractedRelationship],
        chunks_processed: int,
        processing_time: float,
    ) -> RelationshipExtractionStats:
        """Calculate extraction statistics.

        Args:
            relationships: Extracted relationships
            chunks_processed: Number of chunks processed
            processing_time: Time taken

        Returns:
            RelationshipExtractionStats object
        """
        rels_by_type: Dict[str, int] = {}
        total_confidence = 0.0
        high_confidence_count = 0
        unique_entities: Set[str] = set()

        for rel in relationships:
            # Count by type
            type_key = rel.relationship_type.value
            rels_by_type[type_key] = rels_by_type.get(type_key, 0) + 1

            # Confidence stats
            total_confidence += rel.confidence
            if rel.is_high_confidence:
                high_confidence_count += 1

            # Unique entities
            unique_entities.add(rel.source_entity)
            unique_entities.add(rel.target_entity)

        average_confidence = (
            total_confidence / len(relationships)
            if relationships
            else 0.0
        )

        return RelationshipExtractionStats(
            total_relationships=len(relationships),
            relationships_by_type=rels_by_type,
            average_confidence=average_confidence,
            high_confidence_count=high_confidence_count,
            entities_involved=len(unique_entities),
            chunks_processed=chunks_processed,
            processing_time=processing_time,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self.entity_extractor:
            self.entity_extractor.close()
            logger.debug("Closed RelationshipExtractor")
