"""End-to-end extraction pipeline orchestrating all phases."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.extraction import (
    BaseChunker,
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelationship,
    RelationshipExtractor,
    SemanticChunker,
    TextChunk,
)
from src.extraction.ner.hybrid_extraction import HybridSpaCyLLaMA
from src.extraction.parsers import BaseParser, ParserFactory


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""

    total_documents: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    documents_processed: int = 0
    chunks_processed: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    average_chunk_time: float = 0.0
    average_entity_time: float = 0.0
    average_relationship_time: float = 0.0

    def add_error(self, error: str) -> None:
        """Add error to stats."""
        self.errors.append(error)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
            "documents_processed": self.documents_processed,
            "chunks_processed": self.chunks_processed,
            "entities_extracted": self.entities_extracted,
            "relationships_extracted": self.relationships_extracted,
            "error_count": len(self.errors),
            "processing_time": self.processing_time,
            "average_chunk_time": self.average_chunk_time,
            "average_entity_time": self.average_entity_time,
            "average_relationship_time": self.average_relationship_time,
        }


class ExtractionPipeline:
    """Orchestrates end-to-end document processing pipeline."""

    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        relationship_extractor: Optional[RelationshipExtractor] = None,
        parser_factory: Optional[ParserFactory] = None,
        use_hybrid_ner: bool = True,
    ):
        """Initialize pipeline with extractors.

        Args:
            chunker: Text chunking strategy (default: SemanticChunker)
            entity_extractor: NER pipeline (default: HybridSpaCyLLaMA if use_hybrid_ner=True)
            relationship_extractor: Relationship extraction (default: RelationshipExtractor)
            parser_factory: Document parser factory (default: ParserFactory)
            use_hybrid_ner: Use hybrid SpaCy+LLaMA NER (default: True)
                           If False, uses traditional EntityExtractor (SpaCy only)
        """
        self.chunker = chunker or SemanticChunker()
        
        # Initialize NER based on use_hybrid_ner flag
        if entity_extractor:
            self.entity_extractor = entity_extractor
            self.hybrid_ner = False
        elif use_hybrid_ner:
            # Use hybrid SpaCy+LLaMA approach
            self.entity_extractor = HybridSpaCyLLaMA()
            self.hybrid_ner = True
        else:
            # Use traditional SpaCy-only approach
            self.entity_extractor = EntityExtractor()
            self.hybrid_ner = False
        
        self.relationship_extractor = relationship_extractor or RelationshipExtractor()
        self.parser_factory = parser_factory or ParserFactory()
        self.stats = PipelineStats()

    def process_document(
        self,
        file_path: Union[str, Path],
        include_relationships: bool = True,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Process single document through full pipeline.

        Args:
            file_path: Path to document file
            include_relationships: Whether to extract relationships

        Returns:
            Tuple of (entities, relationships)
        """
        self.stats.total_documents += 1
        start_time = time.time()

        try:
            file_path = Path(file_path)

            # Phase 1: Parse document
            parser = self.parser_factory.create_parser(str(file_path))
            text = parser.parse(str(file_path))

            # Phase 2: Chunk text
            chunk_start = time.time()
            chunks = self.chunker.chunk(text, str(file_path))
            self.stats.total_chunks += len(chunks)
            self.stats.chunks_processed += len(chunks)
            chunk_time = time.time() - chunk_start

            # Phase 3: Extract entities
            entity_start = time.time()
            all_entities = []
            entities_by_chunk = {}

            for chunk in chunks:
                chunk_entities = self.entity_extractor.extract_from_chunk(chunk)
                all_entities.extend(chunk_entities)
                entities_by_chunk[chunk.chunk_id] = chunk_entities
                self.stats.total_entities += len(chunk_entities)
                self.stats.entities_extracted += len(chunk_entities)

            entity_time = time.time() - entity_start

            # Phase 4: Extract relationships (optional)
            all_relationships = []
            relationship_time = 0.0

            if include_relationships:
                rel_start = time.time()
                all_relationships = self.relationship_extractor.extract_from_chunks(
                    chunks, entities_by_chunk=entities_by_chunk
                )
                self.stats.total_relationships += len(all_relationships)
                self.stats.relationships_extracted += len(all_relationships)
                relationship_time = time.time() - rel_start

            self.stats.documents_processed += 1
            total_time = time.time() - start_time

            # Update timing stats
            if self.stats.chunks_processed > 0:
                self.stats.average_chunk_time = chunk_time / len(chunks)
            if self.stats.entities_extracted > 0:
                self.stats.average_entity_time = entity_time / self.stats.entities_extracted
            if self.stats.relationships_extracted > 0:
                self.stats.average_relationship_time = (
                    relationship_time / self.stats.relationships_extracted
                )

            return all_entities, all_relationships

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats.add_error(error_msg)
            return [], []

    def process_documents(
        self,
        file_paths: List[Union[str, Path]],
        include_relationships: bool = True,
        stop_on_error: bool = False,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Process multiple documents through pipeline.

        Args:
            file_paths: List of document paths
            include_relationships: Whether to extract relationships
            stop_on_error: Stop processing on first error

        Returns:
            Tuple of (all_entities, all_relationships)
        """
        start_time = time.time()
        all_entities = []
        all_relationships = []

        for file_path in file_paths:
            entities, relationships = self.process_document(
                file_path, include_relationships=include_relationships
            )
            all_entities.extend(entities)
            all_relationships.extend(relationships)

            if stop_on_error and self.stats.errors:
                break

        self.stats.processing_time = time.time() - start_time
        return all_entities, all_relationships

    def process_text(
        self,
        text: str,
        source_file: str = "input",
        include_relationships: bool = True,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Process raw text through pipeline.

        Args:
            text: Input text
            source_file: Source identifier
            include_relationships: Whether to extract relationships

        Returns:
            Tuple of (entities, relationships)
        """
        start_time = time.time()
        self.stats.total_documents += 1

        try:
            # Phase 2: Chunk text
            chunk_start = time.time()
            chunks = self.chunker.chunk(text, source_file)
            self.stats.total_chunks += len(chunks)
            self.stats.chunks_processed += len(chunks)
            chunk_time = time.time() - chunk_start

            # Phase 3: Extract entities
            entity_start = time.time()
            all_entities = []
            entities_by_chunk = {}

            for chunk in chunks:
                chunk_entities = self.entity_extractor.extract_from_chunk(chunk)
                all_entities.extend(chunk_entities)
                entities_by_chunk[chunk.chunk_id] = chunk_entities
                self.stats.total_entities += len(chunk_entities)
                self.stats.entities_extracted += len(chunk_entities)

            entity_time = time.time() - entity_start

            # Phase 4: Extract relationships
            all_relationships = []
            relationship_time = 0.0

            if include_relationships:
                rel_start = time.time()
                all_relationships = self.relationship_extractor.extract_from_chunks(
                    chunks, entities_by_chunk=entities_by_chunk
                )
                self.stats.total_relationships += len(all_relationships)
                self.stats.relationships_extracted += len(all_relationships)
                relationship_time = time.time() - rel_start

            self.stats.documents_processed += 1
            total_time = time.time() - start_time
            self.stats.processing_time = total_time

            # Update timing stats
            if self.stats.chunks_processed > 0:
                self.stats.average_chunk_time = chunk_time / len(chunks)
            if self.stats.entities_extracted > 0:
                self.stats.average_entity_time = entity_time / self.stats.entities_extracted
            if self.stats.relationships_extracted > 0:
                self.stats.average_relationship_time = (
                    relationship_time / self.stats.relationships_extracted
                )

            return all_entities, all_relationships

        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            self.stats.add_error(error_msg)
            return [], []

    def get_stats(self) -> PipelineStats:
        """Get pipeline execution statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self.stats = PipelineStats()

    def close(self) -> None:
        """Cleanup resources."""
        if hasattr(self.entity_extractor, "close"):
            self.entity_extractor.close()
        if hasattr(self.relationship_extractor, "close"):
            self.relationship_extractor.close()
