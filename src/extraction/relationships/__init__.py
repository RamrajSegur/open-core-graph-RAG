"""Relationship Extraction module.

Provides extraction and classification of semantic relationships
between entities extracted from text.
"""

from .relationship_extractor import RelationshipExtractor
from .relationship_models import (
    ExtractedRelationship,
    RelationshipExtractionStats,
    RelationshipType,
)

__all__ = [
    "RelationshipType",
    "ExtractedRelationship",
    "RelationshipExtractionStats",
    "RelationshipExtractor",
]
