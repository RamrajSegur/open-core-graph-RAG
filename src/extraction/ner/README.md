# Phase 3: Named Entity Recognition (NER)

Extract named entities (people, organizations, locations, etc.) from text chunks with SpaCy-based NLP.

## Overview

The NER module identifies and classifies named entities in text using SpaCy's pre-trained models. It provides confidence scoring, batch processing, deduplication, and comprehensive statistics tracking.

## Entity Types

### Supported Entity Types (16+)

| Category | Types | Examples |
|----------|-------|----------|
| **Person** | PERSON | John Smith, Jane Doe |
| **Organization** | GPE, NORP | Apple Inc, Google |
| **Location** | LOCATION, GPE | New York, San Francisco |
| **Temporal** | DATE, TIME | December 31, 2025; 9:30 AM |
| **Monetary** | MONEY | $100, €50 |
| **Numeric** | PERCENT, CARDINAL, ORDINAL | 50%, 100, first |
| **Facility** | FACILITY | Times Square, Empire State Building |
| **Product** | PRODUCT | iPhone, Windows 10 |
| **Event** | EVENT | World War II, Olympic Games |
| **Law** | LAW | Magna Carta, GDPR |
| **Language** | LANGUAGE | English, Mandarin |
| **Custom** | CUSTOM | User-defined types |

## Architecture

```
TextChunk
   │
   ├─ Text extraction
   │
   ├─ SpaCy NLP processing
   │  ├─ Tokenization
   │  ├─ POS tagging
   │  └─ NER tagging
   │
   ├─ Confidence scoring
   │
   ├─ Deduplication
   │
   └─ Statistics collection
   │
   ▼
ExtractedEntity[] (with confidence and metadata)
```

## Components

### EntityType (Enum)

All supported entity types:

```python
class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    FACILITY = "FACILITY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    GPE = "GPE"
    NORP = "NORP"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    CUSTOM = "CUSTOM"
```

### ExtractedEntity (Dataclass)

Represents a single extracted entity:

```python
@dataclass
class ExtractedEntity:
    text: str                       # Entity text
    entity_type: EntityType         # Type of entity
    chunk_id: str                   # Source chunk
    start_position: int = 0         # Position in chunk
    end_position: int = 0           # Position in chunk
    confidence: float = 0.0         # Confidence (0.0-1.0)
    source_file: str = ""           # Source document
```

**Properties:**
- `normalized_text` - Lowercase version for comparison
- `is_high_confidence` - True if confidence ≥ 0.75
- `entity_id` - Unique identifier (UUID)

**Methods:**
- `to_dict()` - Serialize to dictionary

### NERModel (Wrapper)

Wraps SpaCy NLP model:

```python
class NERModel:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize with SpaCy model."""
        
    def extract_entities(self, text: str) -> List[dict]:
        """Extract entities from text."""
```

### EntityExtractor (Pipeline)

Main NER pipeline orchestrator:

```python
class EntityExtractor:
    def extract_from_chunk(self, chunk: TextChunk, 
                          entity_types: List[EntityType] = None) -> List[ExtractedEntity]:
        """Extract entities from a single chunk."""
        
    def extract_from_chunks(self, chunks: List[TextChunk],
                           include_stats: bool = False) -> Union[List, Tuple]:
        """Extract entities from multiple chunks."""
        
    def filter_by_confidence(self, entities: List[ExtractedEntity],
                            min_confidence: float = 0.75) -> List[ExtractedEntity]:
        """Filter entities by confidence threshold."""
        
    def deduplicate_entities(self, entities: List[ExtractedEntity],
                            case_sensitive: bool = False) -> List[ExtractedEntity]:
        """Remove duplicate entities."""
```

### NERStats (Dataclass)

Statistics about extraction operation:

```python
@dataclass
class NERStats:
    total_entities: int = 0
    entities_by_type: Dict[EntityType, int] = None
    average_confidence: float = 0.0
    high_confidence_count: int = 0
    unique_entities: int = 0
    chunks_processed: int = 0
    processing_time: float = 0.0
```

## Usage Examples

### Single Chunk Extraction

```python
from src.extraction.ner import EntityExtractor
from src.extraction.chunking import TextChunk

chunk = TextChunk(
    content="John Smith works for Google in California.",
    source_file="doc.txt",
    chunk_id="chunk_001",
    position_in_document=0
)

extractor = EntityExtractor()
entities = extractor.extract_from_chunk(chunk)

for entity in entities:
    print(f"{entity.text} ({entity.entity_type.value})")
    print(f"  Confidence: {entity.confidence:.2f}")
    print(f"  Position: {entity.start_position}-{entity.end_position}")
```

### Batch Processing

```python
from src.extraction.ner import EntityExtractor

extractor = EntityExtractor()
chunks = [...]  # List of TextChunk objects

# Extract from all chunks
entities = extractor.extract_from_chunks(chunks)
print(f"Extracted {len(entities)} entities from {len(chunks)} chunks")

# With statistics
entities, stats = extractor.extract_from_chunks(chunks, include_stats=True)
print(f"Average confidence: {stats.average_confidence:.3f}")
print(f"High-confidence entities: {stats.high_confidence_count}")
```

### Filtering and Deduplication

```python
extractor = EntityExtractor()
entities = extractor.extract_from_chunks(chunks)

# Filter by confidence
high_conf = extractor.filter_by_confidence(entities, min_confidence=0.8)

# Remove duplicates
unique = extractor.deduplicate_entities(high_conf, case_sensitive=False)

print(f"Kept {len(unique)} entities after filtering")
```

### Specific Entity Type Extraction

```python
from src.extraction.ner import EntityExtractor, EntityType

extractor = EntityExtractor()

# Extract only PERSON entities
persons = extractor.extract_from_chunk(chunk, entity_types=[EntityType.PERSON])

# Multiple types
entities = extractor.extract_from_chunk(
    chunk,
    entity_types=[EntityType.PERSON, EntityType.ORGANIZATION]
)
```

## File Structure

```
ner/
├── __init__.py                  # Module exports
├── entity_models.py             # Data structures
├── ner_model.py                 # SpaCy wrapper
├── entity_extractor.py          # Pipeline orchestrator
└── README.md                    # This file
```

## Testing

```bash
# Run all NER tests
pytest tests/extraction/test_ner.py -v

# Run specific test
pytest tests/extraction/test_ner.py::TestEntityExtractor -v

# With coverage
pytest tests/extraction/test_ner.py --cov=src/extraction/ner
```

## Test Coverage

- ✅ Entity extraction accuracy
- ✅ All 16+ entity types
- ✅ Confidence scoring
- ✅ Deduplication logic (case-sensitive/insensitive)
- ✅ Batch processing
- ✅ Statistics computation
- ✅ Filtering by confidence
- ✅ Type-specific extraction
- ✅ Edge cases (empty text, special characters)
- ✅ Large document processing

## Performance

| Operation | Speed | Throughput |
|-----------|-------|-----------|
| Entity Extraction | ~2ms/entity | 500+ entities/sec |
| Batch Processing | ~100ms/100 entities | 1000+ entities/sec |
| Deduplication | <1ms/100 entities | Negligible |
| Filtering | <1ms/100 entities | Negligible |

## Configuration

### SpaCy Model Selection

```python
from src.extraction.ner import EntityExtractor

# Small, fast model (default)
extractor = EntityExtractor(model_name="en_core_web_sm")

# Medium model (more accurate)
extractor = EntityExtractor(model_name="en_core_web_md")

# Large model (most accurate)
extractor = EntityExtractor(model_name="en_core_web_lg")
```

### Confidence Threshold

```python
# Extract all entities (no filtering)
entities = extractor.extract_from_chunks(chunks)

# Filter during extraction
high_conf = extractor.filter_by_confidence(entities, min_confidence=0.7)

# Very high confidence only
very_high = extractor.filter_by_confidence(entities, min_confidence=0.95)
```

## Best Practices

### 1. Use Appropriate Model

```python
# For speed: use small model
extractor = EntityExtractor(model_name="en_core_web_sm")

# For accuracy: use large model
extractor = EntityExtractor(model_name="en_core_web_lg")
```

### 2. Filter by Confidence

```python
# High-confidence entities only
entities = extractor.filter_by_confidence(entities, min_confidence=0.8)
```

### 3. Deduplicate

```python
# Remove duplicate mentions
entities = extractor.deduplicate_entities(entities, case_sensitive=False)
```

### 4. Batch Processing

```python
# Process multiple chunks at once
entities, stats = extractor.extract_from_chunks(chunks, include_stats=True)
```

### 5. Type Filtering

```python
# Extract specific entity types
persons = extractor.extract_from_chunk(chunk, 
                                      entity_types=[EntityType.PERSON])
```

## Dependencies

- **spacy** (3.7.2) - NLP and entity recognition
- **en_core_web_sm** - SpaCy English model

## Installation

```bash
# Install spaCy
pip install spacy

# Download model
python -m spacy download en_core_web_sm
```

## Troubleshooting

### Problem: Poor entity recognition accuracy
**Solution:** Try larger SpaCy model (en_core_web_md or lg) or increase confidence threshold

### Problem: Missing entities
**Solution:** Adjust confidence threshold lower (default 0.0) or try different model

### Problem: Duplicate entities
**Solution:** Use deduplication with case_sensitive=False

### Problem: Memory issues
**Solution:** Process in smaller batches, use generator pattern

## Integration with Next Phase

Extracted entities are typically passed to relationship extraction (Phase 4):

```python
from src.extraction.ner import EntityExtractor
from src.extraction.relationships import RelationshipExtractor

# Extract entities
extractor = EntityExtractor()
entities = extractor.extract_from_chunks(chunks)

# Extract relationships
rel_extractor = RelationshipExtractor()
relationships = rel_extractor.extract_from_chunks(chunks, entities)
```

## Advanced: Custom Entity Types

```python
from src.extraction.ner import EntityType

# Use CUSTOM type for domain-specific entities
custom_entity = ExtractedEntity(
    text="OpenCoreGraphRAG",
    entity_type=EntityType.CUSTOM,
    chunk_id="chunk_001",
    start_position=0,
    end_position=16,
    confidence=0.95
)
```

## See Also

- [Phase 2: Text Chunking](../chunking/README.md)
- [Phase 4: Relationship Extraction](../relationships/README.md)
- [Extraction Pipeline Overview](../README.md)
- [Main Documentation](../../../PHASES_1_5_SUMMARY.md)
