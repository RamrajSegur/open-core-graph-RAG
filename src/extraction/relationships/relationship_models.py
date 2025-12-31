"""Data models for relationship extraction."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


class RelationshipType(Enum):
    """Standard relationship types between entities."""

    # Professional/Employment
    WORKS_FOR = "WORKS_FOR"
    """Person works for an organization"""

    EMPLOYED_BY = "EMPLOYED_BY"
    """Person is employed by an organization"""

    FOUNDED_BY = "FOUNDED_BY"
    """Organization founded by a person/organization"""

    LOCATED_IN = "LOCATED_IN"
    """Entity located in a geographic location"""

    HEADQUARTERS = "HEADQUARTERS"
    """Organization headquartered in a location"""

    # Personal
    PERSON_OF = "PERSON_OF"
    """Generic person-of relationship"""

    MEMBER_OF = "MEMBER_OF"
    """Entity is member of a group/organization"""

    PARENT_OF = "PARENT_OF"
    """Person is parent of another person"""

    CHILD_OF = "CHILD_OF"
    """Person is child of another person"""

    SPOUSE_OF = "SPOUSE_OF"
    """Person is spouse of another person"""

    SIBLING_OF = "SIBLING_OF"
    """Person is sibling of another person"""

    # Organizational
    OWNS = "OWNS"
    """Entity owns another entity"""

    OWNED_BY = "OWNED_BY"
    """Entity is owned by another entity"""

    PARTNER_OF = "PARTNER_OF"
    """Entities are partners"""

    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    """Organization is subsidiary of another"""

    # Temporal
    OCCURRED_ON = "OCCURRED_ON"
    """Event occurred on a date"""

    OCCURRED_AT = "OCCURRED_AT"
    """Event occurred at a location"""

    OCCURRED_DURING = "OCCURRED_DURING"
    """Event occurred during a time period"""

    # Product/Creation
    CREATED_BY = "CREATED_BY"
    """Product created by person/organization"""

    AUTHORED_BY = "AUTHORED_BY"
    """Work authored by a person"""

    PUBLISHED_BY = "PUBLISHED_BY"
    """Work published by an organization"""

    # Semantic
    RELATED_TO = "RELATED_TO"
    """Generic relationship between entities"""

    SAME_AS = "SAME_AS"
    """Entities refer to same thing (co-reference)"""

    SYNONYM_OF = "SYNONYM_OF"
    """One entity is synonym of another"""

    # Custom
    CUSTOM = "CUSTOM"
    """Custom domain-specific relationship"""


@dataclass
class ExtractedRelationship:
    """Represents a relationship between two entities.

    Captures semantic relationships extracted from text with
    source information, confidence, and supporting context.
    """

    source_entity: str
    """Text of the source entity"""

    target_entity: str
    """Text of the target entity"""

    relationship_type: RelationshipType
    """Type of relationship between entities"""

    source_chunk_id: str
    """Chunk ID where relationship was found"""

    confidence: float = 0.0
    """Confidence score (0.0-1.0) for the relationship"""

    supporting_text: str = ""
    """Text snippet supporting this relationship"""

    relationship_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this relationship"""

    source_file: str = ""
    """Source document file path"""

    start_char: int = 0
    """Start character position in chunk"""

    end_char: int = 0
    """End character position in chunk"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the relationship"""

    created_at: datetime = field(default_factory=datetime.now)
    """Timestamp when relationship was extracted"""

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is above 0.75."""
        return self.confidence >= 0.75

    @property
    def normalized_source(self) -> str:
        """Return normalized source entity text."""
        return self.source_entity.lower().strip()

    @property
    def normalized_target(self) -> str:
        """Return normalized target entity text."""
        return self.target_entity.lower().strip()

    def __repr__(self) -> str:
        return (
            f"ExtractedRelationship("
            f"'{self.source_entity}' "
            f"{self.relationship_type.value} "
            f"'{self.target_entity}', "
            f"conf={self.confidence:.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "relationship_id": self.relationship_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship_type": self.relationship_type.value,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
            "supporting_text": self.supporting_text,
            "source_file": self.source_file,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RelationshipExtractionStats:
    """Statistics about a relationship extraction operation."""

    total_relationships: int
    """Total relationships extracted."""

    relationships_by_type: Dict[str, int] = field(default_factory=dict)
    """Count of relationships by type."""

    average_confidence: float = 0.0
    """Average confidence across all relationships."""

    high_confidence_count: int = 0
    """Count of relationships with confidence >= 0.75."""

    entities_involved: int = 0
    """Count of unique entities in relationships."""

    chunks_processed: int = 0
    """Number of chunks processed."""

    processing_time: float = 0.0
    """Time taken for extraction."""

    @property
    def density(self) -> float:
        """Relationship density (relationships per chunk)."""
        if self.chunks_processed == 0:
            return 0.0
        return self.total_relationships / self.chunks_processed

    def __repr__(self) -> str:
        return (
            f"RelationshipExtractionStats("
            f"total={self.total_relationships}, "
            f"types={len(self.relationships_by_type)}, "
            f"high_conf={self.high_confidence_count})"
        )
