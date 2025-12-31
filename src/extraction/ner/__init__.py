"""Named Entity Recognition module.

Provides entity extraction from text chunks using SpaCy NER models.
"""

from .entity_extractor import EntityExtractor
from .entity_models import EntityType, ExtractedEntity, NERStats
from .ner_model import NERModel

__all__ = [
    "EntityType",
    "ExtractedEntity",
    "NERStats",
    "NERModel",
    "EntityExtractor",
]
