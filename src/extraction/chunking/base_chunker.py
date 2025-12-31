"""Base chunker interface and utility functions."""

import logging
import re
from abc import ABC, abstractmethod
from typing import List

from .text_chunk import ChunkingStats, TextChunk

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            overlap: Overlap between chunks in tokens (default: 50)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.char_per_token = 4  # Rough estimate: 1 token ≈ 4 characters

    @property
    def chunk_size_chars(self) -> int:
        """Convert token-based chunk size to approximate character count."""
        return self.chunk_size * self.char_per_token

    @property
    def overlap_chars(self) -> int:
        """Convert token-based overlap to approximate character count."""
        return self.overlap * self.char_per_token

    @abstractmethod
    def chunk(
        self,
        text: str,
        source_file: str = "unknown",
        metadata: dict | None = None,
    ) -> List[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            source_file: Source file path for metadata
            metadata: Additional metadata to attach to chunks

        Returns:
            List of TextChunk objects

        Raises:
            ValueError: If text is empty or invalid
        """
        pass

    def _validate_text(self, text: str) -> None:
        """Validate input text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

    def _calculate_stats(
        self,
        original_text: str,
        chunks: List[TextChunk],
    ) -> ChunkingStats:
        """Calculate statistics about the chunking."""
        if not chunks:
            return ChunkingStats(
                total_characters=len(original_text),
                total_tokens_estimate=len(original_text) // self.char_per_token,
                chunk_count=0,
                average_chunk_size=0,
                average_chunk_tokens=0,
                max_chunk_size=0,
                min_chunk_size=0,
            )

        chunk_sizes = [chunk.character_count for chunk in chunks]
        chunk_tokens = [chunk.token_estimate for chunk in chunks]

        return ChunkingStats(
            total_characters=len(original_text),
            total_tokens_estimate=len(original_text) // self.char_per_token,
            chunk_count=len(chunks),
            average_chunk_size=sum(chunk_sizes) // len(chunk_sizes),
            average_chunk_tokens=sum(chunk_tokens) // len(chunk_tokens),
            max_chunk_size=max(chunk_sizes),
            min_chunk_size=min(chunk_sizes),
        )

    def _create_chunk(
        self,
        content: str,
        source_file: str,
        chunk_index: int,
        position_in_document: int,
        page_number: int = 1,
        metadata: dict | None = None,
    ) -> TextChunk:
        """Create a TextChunk with proper metadata."""
        if metadata is None:
            metadata = {}

        return TextChunk(
            content=content,
            source_file=source_file,
            chunk_index=chunk_index,
            position_in_document=position_in_document,
            page_number=page_number,
            metadata=metadata,
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text (rough estimate)."""
        return max(1, len(text) // 4)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace."""
        # Remove extra spaces, tabs
        text = re.sub(r"[ \t]+", " ", text)
        # Remove extra newlines
        text = re.sub(r"\n\n+", "\n\n", text)
        # Strip leading/trailing whitespace
        return text.strip()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )
