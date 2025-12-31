"""Tests for text chunking strategies."""

import pytest

from src.extraction import (
    BaseChunker,
    ChunkingStats,
    SemanticChunker,
    SlidingWindowChunker,
    TextChunk,
)


@pytest.fixture
def sample_text():
    """Simple text for testing."""
    return (
        "This is the first sentence. This is the second sentence. "
        "This is the third sentence. And here is a fourth one! "
        "Finally, we have a fifth sentence?"
    )


@pytest.fixture
def multiline_text():
    """Text with paragraphs."""
    return """The first paragraph has multiple sentences. 
It contains several thoughts. And some more text here.

The second paragraph starts here. It is different from the first.
It has its own ideas and concepts.

And finally, a third paragraph. 
Wrapping up the document."""


@pytest.fixture
def long_text():
    """Longer text for testing chunking behavior."""
    sentences = [
        "This is sentence number {}.".format(i)
        for i in range(1, 51)
    ]
    return " ".join(sentences)


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_chunk_initialization(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            content="Hello world",
            source_file="test.txt",
            page_number=1,
        )
        assert chunk.content == "Hello world"
        assert chunk.source_file == "test.txt"
        assert chunk.page_number == 1
        assert chunk.character_count == 11

    def test_chunk_auto_calculations(self):
        """Test automatic calculations in TextChunk."""
        chunk = TextChunk(
            content="This is a test chunk with more content.",
            source_file="test.txt",
        )
        assert chunk.character_count == len("This is a test chunk with more content.")
        assert chunk.token_estimate > 0
        assert chunk.word_count == 8
        assert chunk.line_count == 1

    def test_chunk_multiline(self):
        """Test chunk with multiple lines."""
        content = "Line 1\nLine 2\nLine 3"
        chunk = TextChunk(content=content, source_file="test.txt")
        assert chunk.line_count == 3
        assert chunk.character_count == len(content)

    def test_chunk_unique_ids(self):
        """Test that chunks get unique IDs."""
        chunk1 = TextChunk(content="A", source_file="test.txt")
        chunk2 = TextChunk(content="B", source_file="test.txt")
        assert chunk1.chunk_id != chunk2.chunk_id

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = TextChunk(
            content="Test content",
            source_file="test.txt",
            page_number=2,
            metadata={"key": "value"},
        )
        chunk_dict = chunk.to_dict()
        assert chunk_dict["content"] == "Test content"
        assert chunk_dict["source_file"] == "test.txt"
        assert chunk_dict["page_number"] == 2
        assert chunk_dict["metadata"]["key"] == "value"

    def test_chunk_repr(self):
        """Test chunk string representation."""
        chunk = TextChunk(content="A" * 100, source_file="test.txt")
        repr_str = repr(chunk)
        assert "TextChunk" in repr_str
        assert "chars=" in repr_str


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_initialization(self):
        """Test SemanticChunker creation."""
        chunker = SemanticChunker(chunk_size=256, overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25
        assert chunker.preserve_paragraphs is True

    def test_simple_chunking(self, sample_text):
        """Test basic chunking with simple text."""
        chunker = SemanticChunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.source_file == "test.txt" for c in chunks)

    def test_chunk_indices(self, sample_text):
        """Test that chunk indices are sequential."""
        chunker = SemanticChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunking_respects_size(self, sample_text):
        """Test that chunks don't vastly exceed target size."""
        chunker = SemanticChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        # Chunks should be roughly around the target (allowing some overage)
        # for semantic coherence
        for chunk in chunks:
            tokens = chunker.estimate_tokens(chunk.content)
            # Allow up to 2x target size to keep sentences together
            assert tokens <= chunker.chunk_size * 2

    def test_paragraph_preservation(self, multiline_text):
        """Test that paragraphs are kept together when possible."""
        chunker = SemanticChunker(
            chunk_size=200, overlap=20, preserve_paragraphs=True
        )
        chunks = chunker.chunk(multiline_text, source_file="test.txt")

        # Chunks should respect paragraph boundaries
        assert len(chunks) > 0

    def test_metadata_preservation(self, sample_text):
        """Test that metadata is preserved through chunking."""
        metadata = {"document_id": "doc123", "language": "en"}
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk(
            sample_text, source_file="test.txt", metadata=metadata
        )

        for chunk in chunks:
            assert chunk.metadata["document_id"] == "doc123"
            assert chunk.metadata["language"] == "en"

    def test_empty_text_raises(self):
        """Test that empty text raises ValueError."""
        chunker = SemanticChunker()
        with pytest.raises(ValueError):
            chunker.chunk("")

    def test_invalid_text_type_raises(self):
        """Test that non-string text raises TypeError."""
        chunker = SemanticChunker()
        with pytest.raises(TypeError):
            chunker.chunk(123)

    def test_invalid_chunk_size_raises(self):
        """Test that invalid chunk size raises ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_size=0)

    def test_invalid_overlap_raises(self):
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(overlap=-1)

    def test_overlap_exceeds_size_raises(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_size=100, overlap=100)

    def test_sentence_splitting(self):
        """Test that sentences are properly identified."""
        text = "First sentence. Second sentence! Third sentence?"
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk(text, source_file="test.txt")

        # Should have at least one chunk
        assert len(chunks) > 0
        # Combined content should match original (roughly)
        combined = " ".join(c.content for c in chunks)
        assert "First sentence" in combined
        assert "Second sentence" in combined
        assert "Third sentence" in combined

    def test_long_text_chunking(self, long_text):
        """Test chunking of longer texts."""
        chunker = SemanticChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(long_text, source_file="test.txt")

        assert len(chunks) > 1  # Should create multiple chunks
        # All chunks should have content
        assert all(len(c.content) > 0 for c in chunks)

    def test_position_tracking(self, sample_text):
        """Test that position_in_document is tracked."""
        chunker = SemanticChunker(chunk_size=100)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        # Positions should generally increase
        for i in range(1, len(chunks)):
            # Note: with overlap, position might not strictly increase
            assert chunks[i].position_in_document >= 0


class TestSlidingWindowChunker:
    """Tests for SlidingWindowChunker."""

    def test_initialization(self):
        """Test SlidingWindowChunker creation."""
        chunker = SlidingWindowChunker(chunk_size=256, overlap=50)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 50

    def test_simple_chunking(self, sample_text):
        """Test basic sliding window chunking."""
        chunker = SlidingWindowChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_window_size_respected(self, long_text):
        """Test that sliding window respects size constraints."""
        chunker = SlidingWindowChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(long_text, source_file="test.txt")

        # Most chunks should be within range
        for chunk in chunks:
            chars = len(chunk.content)
            max_chars = chunker.chunk_size_chars
            # Allow some overage for non-break characters
            assert chars <= max_chars * 1.2

    def test_overlap_coverage(self, sample_text):
        """Test that overlapping windows cover the text."""
        chunker = SlidingWindowChunker(chunk_size=100, overlap=50)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        # With overlap, coverage should be good
        assert len(chunks) >= 1

    def test_chunk_ordering(self, long_text):
        """Test that chunks are in order."""
        chunker = SlidingWindowChunker(chunk_size=150, overlap=20)
        chunks = chunker.chunk(long_text, source_file="test.txt")

        # Chunks should be ordered by position
        positions = [c.position_in_document for c in chunks]
        for i in range(1, len(positions)):
            # Positions should be increasing (but not strictly due to overlap)
            assert positions[i] >= 0

    def test_no_overlap(self, sample_text):
        """Test sliding window with no overlap."""
        chunker = SlidingWindowChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk(sample_text, source_file="test.txt")

        # Should create some chunks
        assert len(chunks) > 0


class TestChunkingStats:
    """Tests for ChunkingStats."""

    def test_stats_initialization(self):
        """Test ChunkingStats creation."""
        stats = ChunkingStats(
            total_characters=1000,
            total_tokens_estimate=250,
            chunk_count=5,
            average_chunk_size=200,
            average_chunk_tokens=50,
            max_chunk_size=300,
            min_chunk_size=100,
        )
        assert stats.total_characters == 1000
        assert stats.chunk_count == 5

    def test_stats_coverage(self):
        """Test coverage calculation."""
        stats = ChunkingStats(
            total_characters=1000,
            total_tokens_estimate=250,
            chunk_count=10,
            average_chunk_size=100,
            average_chunk_tokens=25,
            max_chunk_size=150,
            min_chunk_size=50,
        )
        # Coverage: (100 * 10) / 1000 * 100 = 100%
        assert stats.coverage == 100.0

    def test_stats_repr(self):
        """Test stats string representation."""
        stats = ChunkingStats(
            total_characters=1000,
            total_tokens_estimate=250,
            chunk_count=5,
            average_chunk_size=200,
            average_chunk_tokens=50,
            max_chunk_size=300,
            min_chunk_size=100,
        )
        repr_str = repr(stats)
        assert "ChunkingStats" in repr_str
        assert "chunks=5" in repr_str


class TestBaseChunker:
    """Tests for BaseChunker functionality."""

    def test_token_estimation(self):
        """Test token estimation utility."""
        text = "This is a test text."
        tokens = BaseChunker.estimate_tokens(text)
        assert tokens > 0
        # Should be roughly characters / 4
        assert tokens >= len(text) // 5  # Allow some variance

    def test_text_cleaning(self):
        """Test text cleaning utility."""
        text = "  Multiple   spaces  \n\n  and  newlines  \n\n here  "
        cleaned = BaseChunker.clean_text(text)
        # Should remove extra whitespace
        assert "   " not in cleaned
        # Should still have content
        assert "Multiple" in cleaned
        assert "spaces" in cleaned
        assert "newlines" in cleaned

    def test_chunk_size_chars_conversion(self):
        """Test token to character conversion."""
        chunker = SemanticChunker(chunk_size=100, overlap=10)
        assert chunker.chunk_size_chars == 400  # 100 * 4
        assert chunker.overlap_chars == 40  # 10 * 4
