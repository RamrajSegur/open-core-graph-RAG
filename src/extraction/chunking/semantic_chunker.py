"""Semantic text chunker that preserves sentence and paragraph boundaries."""

import logging
import re
from typing import List

from .base_chunker import BaseChunker
from .text_chunk import TextChunk

logger = logging.getLogger(__name__)

# Sentence ending patterns (simplified)
SENTENCE_ENDINGS = re.compile(r'[.!?]+(?=\s|$)')


class SemanticChunker(BaseChunker):
    """Chunk text while preserving sentence and paragraph boundaries.

    This chunker respects natural text boundaries (sentences, paragraphs)
    to maintain semantic coherence within chunks.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        preserve_paragraphs: bool = True,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            overlap: Overlap between chunks in tokens (default: 50)
            preserve_paragraphs: Keep paragraphs together (default: True)
        """
        super().__init__(chunk_size, overlap)
        self.preserve_paragraphs = preserve_paragraphs

    def chunk(
        self,
        text: str,
        source_file: str = "unknown",
        metadata: dict | None = None,
    ) -> List[TextChunk]:
        """
        Split text into semantically coherent chunks.

        Respects sentence and paragraph boundaries while aiming for
        target chunk size. Overlapping chunks help maintain context.

        Args:
            text: Text to chunk
            source_file: Source file path
            metadata: Additional metadata

        Returns:
            List of TextChunk objects
        """
        self._validate_text(text)

        if metadata is None:
            metadata = {}

        # Clean text
        text = self.clean_text(text)

        # Split into logical units
        if self.preserve_paragraphs:
            paragraphs = self._split_paragraphs(text)
        else:
            paragraphs = [text]

        # Split paragraphs into sentences
        sentences = []
        for para in paragraphs:
            para_sentences = self._split_sentences(para)
            sentences.extend(para_sentences)

        if not sentences:
            logger.warning("No sentences found in text")
            return []

        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(
            sentences, source_file, metadata
        )

        logger.info(
            f"Created {len(chunks)} chunks from {len(sentences)} sentences"
        )

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\n+', text)
        # Filter empty paragraphs and strip whitespace
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []

        # Split on sentence boundaries (., !, ?)
        # Handle ellipsis and abbreviations
        text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', '|||', text)
        sentences = text.split('|||')

        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _group_sentences_into_chunks(
        self,
        sentences: List[str],
        source_file: str,
        metadata: dict,
    ) -> List[TextChunk]:
        """Group sentences into chunks respecting size constraints."""
        chunks: List[TextChunk] = []
        current_chunk_sentences: List[str] = []
        current_chunk_text = ""
        chunk_index = 0
        position_in_document = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Calculate what the chunk would be if we add this sentence
            potential_text = (
                current_chunk_text + " " + sentence
                if current_chunk_text
                else sentence
            )
            potential_tokens = self.estimate_tokens(potential_text)

            # Check if adding this sentence would exceed chunk size
            if (
                current_chunk_text
                and potential_tokens > self.chunk_size
            ):
                # Create chunk from current sentences
                chunk = self._create_chunk(
                    content=current_chunk_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    position_in_document=position_in_document,
                    metadata=metadata,
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk_sentences:
                    # Keep last sentences for overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk_sentences
                    )
                    overlap_text = " ".join(overlap_sentences)
                    current_chunk_text = overlap_text + " " + sentence
                else:
                    current_chunk_text = sentence

                position_in_document += len(chunk.content)
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + sentence
                else:
                    current_chunk_text = sentence

            current_chunk_sentences.append(sentence)

        # Add final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                content=current_chunk_text,
                source_file=source_file,
                chunk_index=chunk_index,
                position_in_document=position_in_document,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on token count."""
        overlap_sentences = []
        overlap_tokens = 0

        # Go backwards through sentences collecting overlap
        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences
