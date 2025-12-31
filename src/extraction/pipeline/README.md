# Phase 5: Extraction Pipeline & Storage Integration

Unified orchestration of all extraction phases with TigerGraph storage backend and comprehensive configuration management.

## Overview

Phase 5 brings together all previous phases (Parsing → Chunking → NER → Relationships) into a single production-ready pipeline. It provides:

1. **Unified Orchestration** - Single interface for end-to-end extraction
2. **Configuration Management** - YAML/JSON-based configuration with validation
3. **TigerGraph Integration** - Seamless storage of entities and relationships
4. **Batch Processing** - Efficient handling of multiple documents
5. **Statistics & Monitoring** - Comprehensive tracking of all operations

## Architecture

```
Raw Documents
   │
   ├─ Phase 1: Parsing
   │  └─ Extracted text
   │
   ├─ Phase 2: Chunking
   │  └─ TextChunk[]
   │
   ├─ Phase 3: NER
   │  └─ ExtractedEntity[]
   │
   ├─ Phase 4: Relationships
   │  └─ ExtractedRelationship[]
   │
   ├─ Phase 5: Storage
   │  ├─ TigerGraph Vertices (entities)
   │  └─ TigerGraph Edges (relationships)
   │
   └─ Statistics & Monitoring
```

## Components

### 1. ExtractionPipeline (orchestrator.py)

Main interface for running the complete extraction workflow:

```python
class ExtractionPipeline:
    def __init__(self, chunker=None, entity_extractor=None, 
                 relationship_extractor=None, parser_factory=None):
        """Initialize pipeline with extractors."""
        
    def process_document(self, file_path: str) -> Tuple[entities, relationships]:
        """Process a single document."""
        
    def process_documents(self, file_paths: List[str]) -> Tuple[entities, relationships]:
        """Process multiple documents."""
        
    def process_text(self, text: str, source: str = "text") -> Tuple[entities, relationships]:
        """Process raw text directly."""
        
    def get_stats(self) -> PipelineStats:
        """Get overall pipeline statistics."""
```

### 2. PipelineConfig (config.py)

Hierarchical configuration system with YAML/JSON support:

```python
@dataclass
class PipelineConfig:
    chunking: ChunkingConfig = None
    ner: NERConfig = None
    relationships: RelationshipConfig = None
    storage: StorageConfig = None
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig': ...
    
    @classmethod
    def from_json(cls, path: str) -> 'PipelineConfig': ...
    
    def to_yaml(self, path: str) -> None: ...
    def to_json(self, path: str) -> None: ...
```

### 3. StorageConnector (storage.py)

Handles all interactions with TigerGraph:

```python
class StorageConnector:
    def __init__(self, graph_store=None, graph_name: str = "RAG"):
        """Initialize TigerGraph connection."""
        
    def store_entity(self, entity: ExtractedEntity) -> bool:
        """Save entity as vertex."""
        
    def store_entities(self, entities: List[ExtractedEntity]) -> int:
        """Save multiple entities."""
        
    def store_relationship(self, relationship: ExtractedRelationship) -> bool:
        """Save relationship as edge."""
        
    def store_relationships(self, relationships: List) -> int:
        """Save multiple relationships."""
        
    def store_knowledge_graph(self, entities, relationships) -> Dict:
        """Save complete knowledge graph."""
```

### 4. PipelineStats (orchestrator.py)

Comprehensive statistics tracking:

```python
@dataclass
class PipelineStats:
    total_documents: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    documents_processed: int = 0
    chunks_processed: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    errors: List[str] = None
    processing_time: float = 0.0
```

## Usage Examples

### Basic Pipeline Execution

```python
from src.extraction.pipeline import ExtractionPipeline

# Use default configuration
pipeline = ExtractionPipeline()

# Process single document
entities, relationships = pipeline.process_document("document.pdf")
print(f"Extracted {len(entities)} entities")
print(f"Extracted {len(relationships)} relationships")

# Get statistics
stats = pipeline.get_stats()
print(f"Processing time: {stats.processing_time:.2f}s")
```

### Batch Processing

```python
pipeline = ExtractionPipeline()

# Process multiple documents
entities, relationships = pipeline.process_documents([
    "report1.pdf",
    "report2.docx",
    "report3.txt"
])

stats = pipeline.get_stats()
print(f"Documents processed: {stats.documents_processed}")
print(f"Total entities: {stats.total_entities}")
print(f"Total relationships: {stats.total_relationships}")
```

### Processing Raw Text

```python
pipeline = ExtractionPipeline()

text = """
John Smith works at Google in Mountain View.
He manages a team of 10 engineers.
"""

entities, relationships = pipeline.process_text(text, source="manual")
```

### Configuration from YAML

```yaml
# pipeline_config.yaml
chunking:
  strategy: semantic
  semantic_max_size: 512

ner:
  enabled: true
  model_name: en_core_web_sm
  confidence_threshold: 0.7

relationships:
  enabled: true
  use_patterns: true
  use_semantic: true
  confidence_threshold: 0.6

storage:
  enabled: true
  backend: tigergraph
  host: localhost
  port: 9000
  graph_name: RAG
```

```python
from src.extraction.pipeline import PipelineConfig, ExtractionPipeline

config = PipelineConfig.from_yaml("pipeline_config.yaml")
pipeline = ExtractionPipeline()
entities, relationships = pipeline.process_documents(["doc1.pdf", "doc2.pdf"])
```

## File Structure

```
pipeline/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── orchestrator.py             # ExtractionPipeline orchestrator
├── config.py                   # Configuration management
└── storage.py                  # StorageConnector (TigerGraph)
```

## Testing

```bash
# Run all Phase 5 tests
pytest tests/extraction/test_phase5_pipeline.py -v

# Run specific test class
pytest tests/extraction/test_phase5_pipeline.py::TestExtractionPipeline -v

# Run with coverage
pytest tests/extraction/test_phase5_pipeline.py --cov=src/extraction/pipeline
```

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Single Doc (1-5 pages) | ~500ms | Includes all phases |
| Batch Save (100 entities) | ~100ms | TigerGraph batch operation |
| Full Pipeline (5 docs) | ~10s | End-to-end with storage |

## Configuration Reference

### ChunkingConfig
- `strategy` (str): "semantic" or "sliding_window"
- `semantic_max_size` (int): Max chars per chunk (default: 512)
- `window_size` (int): Token window size (default: 256)
- `overlap` (int): Token overlap (default: 64)

### NERConfig
- `enabled` (bool): Enable NER extraction (default: True)
- `model_name` (str): SpaCy model (default: en_core_web_sm)
- `confidence_threshold` (float): Min confidence to keep (default: 0.0)

### RelationshipConfig
- `enabled` (bool): Enable relationship extraction (default: True)
- `use_patterns` (bool): Pattern-based extraction (default: True)
- `use_semantic` (bool): Semantic extraction (default: True)
- `confidence_threshold` (float): Min confidence (default: 0.0)
- `deduplicate` (bool): Remove duplicates (default: True)

### StorageConfig
- `enabled` (bool): Enable storage (default: True)
- `backend` (str): "tigergraph" or "neo4j" (default: tigergraph)
- `host` (str): Server host (default: localhost)
- `port` (int): Server port (default: 9000)
- `graph_name` (str): Graph name (default: RAG)
- `username` (str): Username (default: tigergraph)
- `password` (str): Password (default: tigergraph)

## Best Practices

### 1. Use Configuration Files

```python
config = PipelineConfig.from_yaml("config.yaml")
pipeline = ExtractionPipeline()
```

### 2. Monitor Statistics

```python
stats = pipeline.get_stats()
if stats.errors:
    print(f"Errors: {stats.errors}")
```

### 3. Error Handling

```python
try:
    entities, relationships = pipeline.process_document(file_path)
except Exception as e:
    print(f"Processing error: {e}")
    stats = pipeline.get_stats()
    print(f"Errors: {stats.errors}")
```

### 4. Batch Processing

```python
# Process large document sets efficiently
docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
all_entities, all_relationships = pipeline.process_documents(docs)
```

## Integration with Previous Phases

Phase 5 orchestrates all previous phases:

```
parsers/ → chunking/ → ner/ → relationships/ → pipeline/
  Phase 1    Phase 2    Phase 3   Phase 4      Phase 5
```

Each phase output feeds into the next, coordinated by the ExtractionPipeline.

## See Also

- [Phase 1: Document Parsing](../parsers/README.md)
- [Phase 2: Text Chunking](../chunking/README.md)
- [Phase 3: Named Entity Recognition](../ner/README.md)
- [Phase 4: Relationship Extraction](../relationships/README.md)
- [Extraction Module Overview](../README.md)
