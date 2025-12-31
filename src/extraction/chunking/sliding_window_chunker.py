"""Simple sliding window text chunker."""

import logging
from typing import List

from .base_chunker import BaseChunker
from .text_chunk import TextChunk

logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    """Chunk text using a simple sliding window approach.

    This chunker splits text into fixed-size chunks with configurable
    overlap, without respecting sentence or paragraph boundaries.
    Fast and simple, but may split mid-sentence.
    """

    def chunk(
        self,
        text: str,
        source_file: str = "unknown",
        metadata: dict | None = None,
    ) -> List[TextChunk]:
        """
        Split text using a sliding window.

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

        # Calculate character-based window sizes
        chunk_size_chars = self.chunk_size_chars
        overlap_chars = self.overlap_chars
        step_size = max(1, chunk_size_chars - overlap_chars)

        chunks: List[TextChunk] = []
        chunk_index = 0

        # Create sliding windows
        for start in range(0, len(text), step_size):
            end = min(start + chunk_size_chars, len(text))

            # Extract chunk
            chunk_text = text[start:end].strip()

            if not chunk_text:
                continue

            chunk = self._create_chunk(
                content=chunk_text,
                source_file=source_file,
                chunk_index=chunk_index,
                position_in_document=start,
                metadata=metadata,
            )
            chunks.append(chunk)
            chunk_index += 1

            # Stop if we've reached the end
            if end == len(text):
                break

        logger.info(
            f"Created {len(chunks)} chunks using sliding window"
        )

        return chunks
