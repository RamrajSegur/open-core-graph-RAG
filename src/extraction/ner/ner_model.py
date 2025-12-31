"""SpaCy-based NER model wrapper."""

import logging
from typing import List, Optional, Tuple

import spacy
from spacy.language import Language

from .entity_models import EntityType, ExtractedEntity

logger = logging.getLogger(__name__)


class NERModel:
    """Wrapper for SpaCy NER models with entity extraction."""

    # Default entity type mappings from SpaCy to standard types
    SPACY_TO_ENTITY_TYPE = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORG,
        "GPE": EntityType.GPE,
        "LOC": EntityType.LOCATION,
        "DATE": EntityType.DATE,
        "TIME": EntityType.TIME,
        "MONEY": EntityType.MONEY,
        "PERCENT": EntityType.PERCENT,
        "PRODUCT": EntityType.PRODUCT,
        "EVENT": EntityType.EVENT,
        "LANGUAGE": EntityType.LANGUAGE,
        "LAW": EntityType.CUSTOM,
        "FAC": EntityType.LOCATION,
        "NORP": EntityType.ORG,
        "CARDINAL": EntityType.QUANTITY,
        "ORDINAL": EntityType.QUANTITY,
        "QUANTITY": EntityType.QUANTITY,
    }

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable_components: Optional[List[str]] = None,
    ):
        """Initialize NER model.

        Args:
            model_name: SpaCy model name (default: en_core_web_sm)
            disable_components: Pipeline components to disable (e.g., ['ner'])
                               If None, only keep 'ner' component
        """
        self.model_name = model_name
        self.nlp: Optional[Language] = None

        try:
            # Load model
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded SpaCy model: {model_name}")

            # Disable unnecessary components for speed
            if disable_components is None:
                # Keep only NER, remove others for speed
                components_to_disable = [
                    comp for comp in self.nlp.pipe_names if comp != "ner"
                ]
            else:
                components_to_disable = disable_components

            if components_to_disable:
                self.nlp.disable_pipes(*components_to_disable)
                logger.debug(f"Disabled components: {components_to_disable}")

        except OSError as e:
            logger.error(f"Failed to load SpaCy model '{model_name}': {e}")
            logger.error(
                "Install with: python -m spacy download en_core_web_sm"
            )
            raise

    def extract_entities(
        self, text: str, chunk_id: str = "", source_file: str = ""
    ) -> List[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from
            chunk_id: ID of the chunk this text comes from
            source_file: Source file path

        Returns:
            List of ExtractedEntity objects
        """
        if not self.nlp:
            raise RuntimeError("NER model not initialized")

        if not text or not text.strip():
            return []

        entities = []
        doc = self.nlp(text)

        for ent in doc.ents:
            # Map SpaCy entity label to standard type
            entity_type = self.SPACY_TO_ENTITY_TYPE.get(
                ent.label_, EntityType.ENTITY
            )

            # Extract entity with metadata
            entity = ExtractedEntity(
                text=ent.text,
                entity_type=entity_type,
                chunk_id=chunk_id,
                start_position=ent.start_char,
                end_position=ent.end_char,
                confidence=1.0,  # SpaCy doesn't provide confidence for NER
                source_file=source_file,
                span_text=text[
                    max(0, ent.start_char - 20) : min(
                        len(text), ent.end_char + 20
                    )
                ],
                metadata={"spacy_label": ent.label_, "token_count": len(ent)},
            )
            entities.append(entity)

        return entities

    def extract_entities_batch(
        self,
        texts: List[Tuple[str, str, str]],
        batch_size: int = 128,
    ) -> List[ExtractedEntity]:
        """Extract entities from multiple texts efficiently.

        Args:
            texts: List of (text, chunk_id, source_file) tuples
            batch_size: Batch size for processing

        Returns:
            List of ExtractedEntity objects from all texts
        """
        if not self.nlp:
            raise RuntimeError("NER model not initialized")

        all_entities = []

        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            texts_only = [t[0] for t in batch]

            # Process batch with SpaCy
            for doc, (text, chunk_id, source_file) in zip(
                self.nlp.pipe(texts_only), batch
            ):
                for ent in doc.ents:
                    entity_type = self.SPACY_TO_ENTITY_TYPE.get(
                        ent.label_, EntityType.ENTITY
                    )

                    entity = ExtractedEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        chunk_id=chunk_id,
                        start_position=ent.start_char,
                        end_position=ent.end_char,
                        confidence=1.0,
                        source_file=source_file,
                        span_text=text[
                            max(0, ent.start_char - 20) : min(
                                len(text), ent.end_char + 20
                            )
                        ],
                        metadata={
                            "spacy_label": ent.label_,
                            "token_count": len(ent),
                        },
                    )
                    all_entities.append(entity)

        return all_entities

    def get_available_models(self) -> List[str]:
        """Get list of commonly available SpaCy models."""
        return [
            "en_core_web_sm",
            "en_core_web_md",
            "en_core_web_lg",
            "en_core_web_trf",
        ]

    def close(self) -> None:
        """Clean up resources."""
        if self.nlp:
            self.nlp = None
            logger.debug(f"Closed NER model: {self.model_name}")
