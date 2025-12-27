# Extraction Pipeline Component

Data extraction and ingestion pipeline for building the knowledge graph from raw documents.

## 📋 Overview

The Extraction Pipeline handles:
- Document parsing (PDF, TXT, HTML, etc.)
- Text chunking and preprocessing
- Entity extraction (people, organizations, locations, etc.)
- Relationship extraction (connections between entities)
- Metadata tracking in PostgreSQL
- Population of TigerGraph with extracted data

## 🏗️ Architecture

```
Raw Documents
     │
     ├─────────────────────────────────────────────────────┐
     │                                                      │
     ▼                                                      ▼
┌──────────────────┐                          ┌──────────────────┐
│ Document Parser  │                          │  Text Chunking   │
│ - Read files     │                          │ - Split text     │
│ - Extract text   │                          │ - Preserve context
│ - Handle formats │                          └──────────────────┘
└────────┬─────────┘                                       │
         │                                                 │
         └──────────────────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Entity Extraction           │
                    │  (spaCy + LLM)                │
                    │ - Identify entities           │
                    │ - Classify types              │
                    │ - Extract properties          │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Relation Extraction          │
                    │  (LLM-based)                  │
                    │ - Find connections            │
                    │ - Classify relationships      │
                    │ - Extract properties          │
                    └───────────────┬───────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────────┐
         │                                                          │
         ▼                                                          ▼
┌─────────────────────────┐                          ┌──────────────────────┐
│  Graph Population       │                          │  Metadata Tracking   │
│  (TigerGraph)           │                          │  (PostgreSQL)        │
│ - Create entities       │                          │ - Job status         │
│ - Create relationships  │                          │ - Document refs      │
│ - Add properties        │                          │ - Extraction metrics │
└─────────────────────────┘                          └──────────────────────┘
```

## 📁 Files

- **`document_parser.py`** - Document parsing and text extraction (to be created)
- **`entity_extractor.py`** - Entity recognition and extraction (to be created)
- **`relation_extractor.py`** - Relationship extraction (to be created)
- **`pipeline.py`** - Orchestrates the full extraction workflow (to be created)

## 🚀 Getting Started

### Prerequisites

- Docker containers running (TigerGraph, PostgreSQL)
- Ollama service with Mistral/Llama2 model
- Python environment configured

### Basic Usage

```python
from src.extraction.pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Ingest documents
job_id = pipeline.ingest(
    source_path="data/raw/",
    source_type="pdf",
    source_name="Technical Docs"
)

# Check status
status = pipeline.get_job_status(job_id)
print(f"Status: {status['status']}")
print(f"Entities extracted: {status['num_entities']}")
print(f"Relations extracted: {status['num_relations']}")
```

## 🔧 Configuration

Configuration is managed in `src/config.py`:

```python
from src.config import config

# Access extraction config
extraction_cfg = config.extraction
print(extraction_cfg.chunk_size)  # 512
print(extraction_cfg.entity_model)  # en_core_web_trf
```

### Environment Variables

```bash
EXTRACTION_CHUNK_SIZE=512
EXTRACTION_CHUNK_OVERLAP=128
EXTRACTION_ENTITY_CONFIDENCE_THRESHOLD=0.5
EXTRACTION_RELATION_CONFIDENCE_THRESHOLD=0.4
```

## 📚 Components

### Document Parser

Reads and parses documents in various formats:

```python
from src.extraction.document_parser import DocumentParser

parser = DocumentParser()

# Parse a PDF
text = parser.parse("document.pdf")

# Parse with chunking
chunks = parser.parse_and_chunk(
    "document.pdf",
    chunk_size=512,
    overlap=128
)
```

**Supported formats:**
- `.pdf` - PDF documents (PyPDF2)
- `.txt` - Plain text
- `.html` - HTML documents
- `.docx` - Word documents (python-docx)
- `.csv` - CSV files (pandas)

### Entity Extractor

Extracts entities using spaCy + LLM:

```python
from src.extraction.entity_extractor import EntityExtractor

extractor = EntityExtractor()

text = "John Doe works at Apple Inc in San Francisco."

# Extract entities
entities = extractor.extract(text)
# Output:
# [
#     {"name": "John Doe", "type": "PERSON", "confidence": 0.95},
#     {"name": "Apple Inc", "type": "ORGANIZATION", "confidence": 0.92},
#     {"name": "San Francisco", "type": "LOCATION", "confidence": 0.88}
# ]
```

**Entity types:**
- `PERSON` - People, individuals
- `ORGANIZATION` - Companies, groups, institutions
- `LOCATION` - Places, cities, countries
- `EVENT` - Named events, dates
- `PRODUCT` - Products, services
- `CONCEPT` - Abstract concepts, topics

### Relation Extractor

Extracts relationships between entities:

```python
from src.extraction.relation_extractor import RelationExtractor

extractor = RelationExtractor()

# With entities already extracted
entities = [
    {"name": "John Doe", "type": "PERSON"},
    {"name": "Apple Inc", "type": "ORGANIZATION"}
]

text = "John Doe works at Apple Inc."

# Extract relationships
relations = extractor.extract(text, entities)
# Output:
# [
#     {
#         "source": "John Doe",
#         "type": "WORKS_FOR",
#         "target": "Apple Inc",
#         "confidence": 0.91
#     }
# ]
```

**Relationship types:**
- `WORKS_FOR` - Employment relationship
- `LOCATED_IN` - Geographic containment
- `FOUNDED_BY` - Founding relationship
- `OWNS` - Ownership relationship
- `RELATED_TO` - Generic relationship

### Data Pipeline

Orchestrates the complete ingestion workflow:

```python
from src.extraction.pipeline import DataPipeline

pipeline = DataPipeline()

# Ingest from directory
job_id = pipeline.ingest(
    source_path="data/raw/",
    source_type="pdf",
    source_name="Q4 2024 Reports"
)

# Monitor progress
while True:
    status = pipeline.get_job_status(job_id)
    print(f"Progress: {status['status']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)

# Get results
results = pipeline.get_job_results(job_id)
print(f"Extracted {results['num_entities']} entities")
print(f"Extracted {results['num_relations']} relations")
```

## 📊 Example: Complete Extraction

```python
from src.extraction.pipeline import DataPipeline

# Setup pipeline
pipeline = DataPipeline()

# Configure extraction
config = {
    "entity_confidence_threshold": 0.6,
    "relation_confidence_threshold": 0.5,
    "chunk_size": 512,
    "overlap": 128
}

# Start ingestion job
job_id = pipeline.ingest(
    source_path="data/raw/company_docs/",
    source_type="pdf",
    source_name="Company Documents",
    config=config
)

# Monitor job
job_status = pipeline.get_job_status(job_id)

# Query results from metadata
results = pipeline.query_metadata(job_id)
print(f"Documents processed: {results['num_documents']}")
print(f"Entities: {results['num_entities']}")
print(f"Relations: {results['num_relations']}")
```

## 🔄 Metadata Tracking

All ingestion activities are tracked in PostgreSQL:

### Tables

- `ingestion_jobs` - Tracks ingestion batches
- `documents` - Tracks source documents
- `entities` - Tracks extracted entities
- `relations` - Tracks extracted relationships

### Query Examples

```python
from src.utils.db import get_db_session

session = get_db_session()

# Get job status
job = session.query(IngestionJob).filter_by(job_id=job_id).first()
print(f"Status: {job.status}")
print(f"Entities: {job.num_entities}")
print(f"Relations: {job.num_relations}")

# List recent jobs
jobs = session.query(IngestionJob).order_by(
    IngestionJob.started_at.desc()
).limit(10).all()

for job in jobs:
    print(f"{job.source_name}: {job.num_entities} entities")
```

## 🧪 Testing

```bash
# Run extraction tests
docker-compose -f docker/docker-compose.yml exec app pytest tests/test_extraction.py -v

# Run with specific test
docker-compose -f docker/docker-compose.yml exec app pytest tests/test_extraction.py::test_entity_extraction -v
```

## 💡 Design Decisions

1. **LLM for Extraction**: Uses open-source LLMs (Mistral/Llama2) for flexibility
2. **Confidence Scores**: All extractions include confidence metrics
3. **Metadata Tracking**: Full audit trail in PostgreSQL
4. **Incremental Processing**: Can process large document collections in batches
5. **Error Handling**: Continues on errors, logs failures for retry

## 🚧 To-Do

- [ ] Implement DocumentParser
  - [ ] PDF support
  - [ ] TXT support
  - [ ] HTML support
  - [ ] Chunking logic
- [ ] Implement EntityExtractor
  - [ ] spaCy integration
  - [ ] LLM-based extraction
  - [ ] Confidence scoring
- [ ] Implement RelationExtractor
  - [ ] Relationship detection
  - [ ] Type classification
  - [ ] Confidence scoring
- [ ] Implement DataPipeline
  - [ ] Job orchestration
  - [ ] Error handling
  - [ ] Progress tracking
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Create example notebooks

## 📖 Related Documentation

- [Knowledge Graph Component](../core/README.md) - Where extracted data goes
- [Configuration](../config.py) - Setting up extraction parameters
- [Main Architecture](../../ARCHITECTURE.md) - System-wide context
- [Database Schema](../../docker/init/init_db.sql) - Metadata tables

## 🔗 Related Components

- **Knowledge Graph** - Stores extracted entities and relationships
- **Retrieval Layer** - Uses the populated graph for searching
- **Reasoning Engine** - Performs inference over the graph

## 📝 Notes

- This component is **in progress**
- Extraction uses open-source models and LLMs for cost-effectiveness
- All extracted data is tracked in PostgreSQL for audit and replay
- The pipeline is designed for incremental processing of large datasets
