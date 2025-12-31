# Phase 2: Text Chunking

Split text into manageable chunks with configurable strategies and metadata preservation.

## Overview

The chunking module divides large text documents into smaller chunks while preserving metadata (position, source file, encoding) and maintaining context. It supports multiple chunking strategies optimized for different use cases.

## Chunking Strategies

### 1. Semantic Chunker

**Best for:** General document processing, semantic understanding

Chunks text at sentence boundaries to preserve meaning:

- **Method:** Sentence tokenization using NLTK
- **Granularity:** Sentence-level
- **Boundary Detection:** Sentence end markers (., !, ?)
- **Context Preservation:** Excellent
- **Config:** `max_size` (default: 512 characters)

```python
from src.extraction.chunking import SemanticChunker

chunker = SemanticChunker(max_size=512)
chunks = chunker.chunk(text, source_file="doc.txt")

for chunk in chunks:
    print(f"Position {chunk.position_in_document}: {chunk.content[:50]}...")
```

### 2. Sliding Window Chunker

**Best for:** Token-level analysis, overlapping context

Uses fixed-size token windows with overlap:

- **Method:** Token-based windowing
- **Granularity:** Token-level
- **Overlap:** Configurable (default: 64 tokens)
- **Consistency:** Fixed-size windows
- **Config:** `window_size`, `overlap` (in tokens)

```python
from src.extraction.chunking import SlidingWindowChunker

chunker = SlidingWindowChunker(window_size=256, overlap=64)
chunks = chunker.chunk(text, source_file="doc.txt")

for chunk in chunks:
    print(f"Tokens: {len(chunk.content.split())} | Position: {chunk.position_in_document}")
```

## Architecture

```
Raw Text
   │
   ├─ SemanticChunker
   │  ├─ Tokenize sentences
   │  ├─ Group by size
   │  └─ Preserve boundaries
   │
   └─ SlidingWindowChunker
      ├─ Tokenize text
      ├─ Create windows
      └─ Add overlaps
   │
   ▼
TextChunk[] (with metadata)
```

## Components

### TextChunk (Dataclass)

Represents a single chunk with metadata:

```python
@dataclass
class TextChunk:
    content: str                        # Chunk text content
    source_file: str                    # Source document
    chunk_id: str                       # Unique identifier
    position_in_document: int = 0       # Order in document
    encoding: str = "utf-8"             # Text encoding
    language: str = "en"                # Language code
    word_count: int = 0                 # Word count
    char_count: int = 0                 # Character count
    custom_attributes: Dict = None      # Extensible metadata
```

**Properties:**
- `is_high_quality` - Checks if chunk meets quality thresholds
- `normalized_content` - Lowercase version for comparison

**Methods:**
- `to_dict()` - Serialize to dictionary
- `from_dict()` - Deserialize from dictionary

### BaseChunker (Abstract)

Abstract interface for chunking strategies:

```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, source_file: str = "input") -> List[TextChunk]:
        """Chunk text and return list of TextChunk objects."""
        
    def get_stats(self) -> ChunkingStats:
        """Get statistics about chunking operation."""
```

### SemanticChunker

Sentence-aware chunking implementation:

```python
class SemanticChunker(BaseChunker):
    def __init__(self, max_size: int = 512):
        """Initialize with max chunk size in characters."""
        self.max_size = max_size
        
    def chunk(self, text: str, source_file: str = "input") -> List[TextChunk]:
        """Chunk by sentences up to max_size."""
```

### SlidingWindowChunker

Token-based windowing implementation:

```python
class SlidingWindowChunker(BaseChunker):
    def __init__(self, window_size: int = 256, overlap: int = 64):
        """Initialize with window size and overlap in tokens."""
        self.window_size = window_size
        self.overlap = overlap
        
    def chunk(self, text: str, source_file: str = "input") -> List[TextChunk]:
        """Create overlapping windows of tokens."""
```

### ChunkingStats

Statistics about chunking operation:

```python
@dataclass
class ChunkingStats:
    total_text_length: int = 0
    total_chunks: int = 0
    average_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    processing_time: float = 0.0
```

## Usage Examples

### Semantic Chunking

```python
from src.extraction.chunking import SemanticChunker

text = """
Machine learning is a branch of artificial intelligence. 
It focuses on the development of algorithms and statistical models. 
These models enable computers to learn from data without being explicitly programmed.
"""

chunker = SemanticChunker(max_size=100)
chunks = chunker.chunk(text, source_file="article.txt")

for chunk in chunks:
    print(f"ID: {chunk.chunk_id}")
    print(f"Position: {chunk.position_in_document}")
    print(f"Content: {chunk.content}")
    print(f"Words: {chunk.word_count}")
    print("---")
```

### Sliding Window Chunking

```python
from src.extraction.chunking import SlidingWindowChunker

chunker = SlidingWindowChunker(window_size=64, overlap=16)
chunks = chunker.chunk(text, source_file="document.txt")

# Chunks will overlap by 16 tokens
print(f"Created {len(chunks)} chunks with overlaps")
```

### Batch Chunking

```python
from src.extraction.chunking import SemanticChunker
from pathlib import Path

chunker = SemanticChunker()
all_chunks = []

for file_path in Path("documents/").glob("*.txt"):
    with open(file_path, 'r') as f:
        text = f.read()
    
    chunks = chunker.chunk(text, source_file=str(file_path))
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")
```

### Statistics

```python
chunker = SemanticChunker()
chunks = chunker.chunk(text, source_file="doc.txt")
stats = chunker.get_stats()

print(f"Average chunk size: {stats.average_chunk_size:.0f} characters")
print(f"Total chunks: {stats.total_chunks}")
print(f"Processing time: {stats.processing_time:.3f}s")
```

## File Structure

```
chunking/
├── __init__.py                  # Module exports
├── base_chunker.py              # Abstract base class
├── text_chunk.py                # TextChunk dataclass
├── semantic_chunker.py          # Semantic chunking
├── sliding_window_chunker.py    # Sliding window chunking
└── README.md                    # This file
```

## Testing

```bash
# Run all chunking tests
pytest tests/extraction/test_chunking.py -v

# Run specific test
pytest tests/extraction/test_chunking.py::TestSemanticChunker -v

# With coverage
pytest tests/extraction/test_chunking.py --cov=src/extraction/chunking
```

## Test Coverage

- ✅ Semantic sentence tokenization
- ✅ Sliding window mechanics
- ✅ Boundary condition handling
- ✅ Metadata preservation
- ✅ Edge cases (empty text, single sentence)
- ✅ Large document processing
- ✅ Chunk statistics
- ✅ Overlapping window verification
- ✅ Position tracking
- ✅ Encoding handling

## Performance

| Operation | Speed | Memory |
|-----------|-------|--------|
| Semantic Chunking | ~1000 chunks/sec | Minimal |
| Sliding Window | ~500 chunks/sec | Minimal |
| Large Document | ~100K chars/sec | Streaming |
| Batch Processing | Scales linearly | ~1MB overhead |

## Configuration Reference

### Semantic Chunker

```python
from src.extraction.chunking import SemanticChunker

# Small chunks (for fine-grained analysis)
chunker = SemanticChunker(max_size=256)

# Medium chunks (default)
chunker = SemanticChunker(max_size=512)

# Large chunks (for context preservation)
chunker = SemanticChunker(max_size=1024)
```

### Sliding Window Chunker

```python
from src.extraction.chunking import SlidingWindowChunker

# Small window with overlap
chunker = SlidingWindowChunker(window_size=128, overlap=32)

# Default configuration
chunker = SlidingWindowChunker(window_size=256, overlap=64)

# Large window with minimal overlap
chunker = SlidingWindowChunker(window_size=512, overlap=64)
```

## Best Practices

1. **Choose Strategy Based on Use Case:**
   - Semantic: General document processing, NER, classification
   - Sliding Window: Token analysis, embeddings, fine-grained analysis

2. **Batch Processing:**
   ```python
   chunks = []
   for text in texts:
       chunks.extend(chunker.chunk(text))
   ```

3. **Error Handling:**
   ```python
   try:
       chunks = chunker.chunk(text)
   except ValueError as e:
       logger.error(f"Chunking failed: {e}")
   ```

4. **Memory Management:**
   - Use streaming for large documents
   - Process batches incrementally
   - Clear chunks after processing

## Dependencies

- **nltk** (3.8.1) - Sentence tokenization
- **Python** (3.10+) - Built-in tokenization

## Integration with Next Phase

Chunks are typically passed to NER (Phase 3):

```python
from src.extraction.chunking import SemanticChunker
from src.extraction.ner import EntityExtractor

# Chunk
chunker = SemanticChunker()
chunks = chunker.chunk(text, source_file="doc.txt")

# Extract entities from chunks
extractor = EntityExtractor()
for chunk in chunks:
    entities = extractor.extract_from_chunk(chunk)
    # Process entities...
```

## Troubleshooting

### Problem: Chunks are too small
**Solution:** Increase `max_size` parameter for semantic chunker

### Problem: Chunks lose important context
**Solution:** Use sliding window with larger `overlap`

### Problem: Memory issues with large documents
**Solution:** Process in batches, use generator pattern

## See Also

- [Phase 1: Document Parsing](../parsers/README.md)
- [Phase 3: Named Entity Recognition](../ner/README.md)
- [Extraction Pipeline Overview](../README.md)
- [Main Documentation](../../../PHASES_1_5_SUMMARY.md)
