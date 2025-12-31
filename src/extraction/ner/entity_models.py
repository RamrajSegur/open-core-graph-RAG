"""Entity data models for NER extraction."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


class EntityType(Enum):
    """Standard entity types based on common NER models."""

    # Person-related
    PERSON = "PERSON"
    """Individual person entity (e.g., John Smith, Mary Johnson)"""

    # Organization/Group
    ORG = "ORG"
    """Organization entity (e.g., Microsoft, Apple Inc.)"""

    # Location/Geography
    LOCATION = "LOCATION"
    """Geographic location (e.g., New York, United States)"""

    GPE = "GPE"
    """Geopolitical entity (e.g., country, city)"""

    # Product/Work
    PRODUCT = "PRODUCT"
    """Product entity (e.g., iPhone, Windows 10)"""

    WORK_OF_ART = "WORK_OF_ART"
    """Creative work (e.g., Mona Lisa, Harry Potter)"""

    # Time/Date
    DATE = "DATE"
    """Date entity (e.g., January 15, 2025)"""

    TIME = "TIME"
    """Time entity (e.g., 3:30 PM, midnight)"""

    # Quantity/Money
    QUANTITY = "QUANTITY"
    """Quantity measurement (e.g., 5 kilograms, three dozen)"""

    MONEY = "MONEY"
    """Monetary value (e.g., $100, 50 euros)"""

    PERCENT = "PERCENT"
    """Percentage value (e.g., 25%, 90 percent)"""

    # Events
    EVENT = "EVENT"
    """Named event (e.g., World Cup, Olympics)"""

    # Language
    LANGUAGE = "LANGUAGE"
    """Language name (e.g., English, Spanish)"""

    # Generic
    ENTITY = "ENTITY"
    """Generic entity for unknown type"""

    # Custom/Domain-specific
    CUSTOM = "CUSTOM"
    """Custom domain-specific entity type"""


@dataclass
class ExtractedEntity:
    """Represents an entity extracted from text via NER.

    Contains the entity text, type, position, confidence score,
    and associated metadata.
    """

    text: str
    """The actual entity text from the document."""

    entity_type: EntityType
    """Classification of the entity."""

    chunk_id: str
    """ID of the chunk where entity was found."""

    start_position: int
    """Start character position in the chunk."""

    end_position: int
    """End character position in the chunk."""

    confidence: float = 0.0
    """Confidence score (0.0-1.0) for the extraction."""

    entity_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this entity."""

    source_file: str = ""
    """Source document file path."""

    page_number: int = 1
    """Page number in source document."""

    span_text: str = ""
    """Surrounding context (for disambiguation)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the entity."""

    created_at: datetime = field(default_factory=datetime.now)
    """Timestamp when entity was extracted."""

    @property
    def char_span(self) -> tuple:
        """Return (start, end) character positions."""
        return (self.start_position, self.end_position)

    @property
    def normalized_text(self) -> str:
        """Return normalized entity text (lowercased)."""
        return self.text.lower().strip()

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is above 0.75."""
        return self.confidence >= 0.75

    def __repr__(self) -> str:
        return (
            f"ExtractedEntity("
            f"text='{self.text}', "
            f"type={self.entity_type.value}, "
            f"conf={self.confidence:.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type.value,
            "chunk_id": self.chunk_id,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "confidence": self.confidence,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "span_text": self.span_text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class NERStats:
    """Statistics about a NER extraction operation."""

    total_entities: int
    """Total entities extracted."""

    entities_by_type: Dict[str, int] = field(default_factory=dict)
    """Count of entities by type."""

    average_confidence: float = 0.0
    """Average confidence across all entities."""

    high_confidence_count: int = 0
    """Count of entities with confidence >= 0.75."""

    chunks_processed: int = 0
    """Number of chunks processed."""

    processing_time: float = 0.0
    """Time taken for extraction."""

    @property
    def coverage(self) -> float:
        """Percentage of chunks with entities."""
        if self.chunks_processed == 0:
            return 0.0
        chunks_with_entities = len(self.entities_by_type)
        return (chunks_with_entities / self.chunks_processed) * 100

    def __repr__(self) -> str:
        return (
            f"NERStats("
            f"total={self.total_entities}, "
            f"high_conf={self.high_confidence_count}, "
            f"types={len(self.entities_by_type)})"
        )
