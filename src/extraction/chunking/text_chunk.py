"""Text chunk data models for document processing."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class TextChunk:
    """Represents a chunk of text extracted from a document.

    A chunk is a logical unit of text with metadata tracking its source,
    position, and characteristics useful for processing pipelines.
    """

    content: str
    """The actual text content of the chunk."""

    source_file: str
    """Path to the source document."""

    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this chunk."""

    page_number: int = 1
    """Page number in the source document (1-indexed)."""

    position_in_document: int = 0
    """Character position in the original document."""

    chunk_index: int = 0
    """Index of this chunk in the sequence of chunks from the document."""

    character_count: int = field(init=False)
    """Total characters in this chunk (auto-calculated)."""

    token_estimate: int = field(init=False)
    """Estimated token count (roughly chars / 4)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (source file metadata, extraction method, etc)."""

    created_at: datetime = field(default_factory=datetime.now)
    """Timestamp when chunk was created."""

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.character_count = len(self.content)
        # Rough token estimation: ~4 characters per token
        self.token_estimate = max(1, len(self.content) // 4)

    @property
    def word_count(self) -> int:
        """Estimate word count by splitting on whitespace."""
        return len(self.content.split())

    @property
    def line_count(self) -> int:
        """Count lines in the chunk."""
        return len(self.content.splitlines())

    def __repr__(self) -> str:
        return (
            f"TextChunk("
            f"id={self.chunk_id[:8]}..., "
            f"chars={self.character_count}, "
            f"tokens≈{self.token_estimate}, "
            f"page={self.page_number})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "position_in_document": self.position_in_document,
            "chunk_index": self.chunk_index,
            "character_count": self.character_count,
            "token_estimate": self.token_estimate,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChunkingStats:
    """Statistics about a chunking operation."""

    total_characters: int
    """Total characters in source text."""

    total_tokens_estimate: int
    """Estimated total tokens in source."""

    chunk_count: int
    """Number of chunks created."""

    average_chunk_size: int
    """Average characters per chunk."""

    average_chunk_tokens: int
    """Average estimated tokens per chunk."""

    max_chunk_size: int
    """Largest chunk size."""

    min_chunk_size: int
    """Smallest chunk size."""

    @property
    def coverage(self) -> float:
        """Calculate coverage percentage (with overlap)."""
        if self.total_characters == 0:
            return 0.0
        total_chunk_chars = self.average_chunk_size * self.chunk_count
        return (total_chunk_chars / self.total_characters) * 100

    def __repr__(self) -> str:
        return (
            f"ChunkingStats("
            f"chunks={self.chunk_count}, "
            f"avg_size={self.average_chunk_size}, "
            f"coverage={self.coverage:.1f}%)"
        )
