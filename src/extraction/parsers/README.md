# Phase 1: Document Parsing

Extract text from various document formats with automatic format detection and metadata preservation.

## Overview

The parsers module provides a unified interface for extracting text from multiple document formats. It uses the factory pattern to automatically detect file format and route to the appropriate parser implementation.

## Supported Formats

| Format | Parser | Features |
|--------|--------|----------|
| **PDF** | `PDFParser` | Multi-page, table extraction, image handling |
| **DOCX** | `DOCXParser` | Text formatting, tables, headers/footers |
| **CSV** | `CSVParser` | Delimiter detection, type handling |
| **TXT** | `TXTParser` | Encoding detection, raw text |
| **HTML** | `WebpageParser` | URLs, local HTML files, tag removal |
| **JSON** | `JSONParser` | Structured data preservation |
| **Binary** | Generic handling | Fallback for unknown formats |

## Architecture

```
Input File
    │
    ├─ File extension check
    │
    ├─ PDF → PDFParser → Extract pages
    ├─ DOCX → DOCXParser → Parse structure
    ├─ CSV → CSVParser → Parse delimiters
    ├─ TXT → TXTParser → Detect encoding
    └─ JSON → JSONParser → Parse structure
    │
    ▼
ParsedDocument (text + metadata)
```

## Components

### BaseParser (Abstract)

Abstract base class defining the parser interface:

```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parse document and return extracted text."""
        
    @abstractmethod
    def get_metadata(self) -> ParsedDocument:
        """Get document metadata."""
```

### Concrete Parsers

#### PDFParser
- Extracts text from all pages
- Handles tables and multi-column layouts
- Preserves page ordering
- Example: `PDFParser().parse("document.pdf")`

#### DOCXParser
- Preserves document structure
- Extracts tables
- Handles headers and footers
- Example: `DOCXParser().parse("document.docx")`

#### CSVParser
- Auto-detects delimiters
- Handles quoted fields
- Preserves headers
- Example: `CSVParser().parse("data.csv")`

#### TXTParser
- Detects file encoding (UTF-8, ASCII, Latin-1, etc.)
- Handles various line endings
- Preserves formatting
- Example: `TXTParser().parse("document.txt")`

#### WebpageParser
- Fetches and parses webpages from URLs
- Parses local HTML files
- Removes HTML tags and scripts
- Extracts metadata (title, description)
- Example: `WebpageParser().parse("https://example.com")` or `WebpageParser().parse("page.html")`

#### JSONParser
- Parses structured JSON data
- Converts to readable text format
- Preserves hierarchy
- Example: `JSONParser().parse("data.json")`

### ParserFactory

Automatic format detection and parser creation:

```python
factory = ParserFactory()
parser = factory.create_parser("document.pdf")  # Auto-detects PDF
text = parser.parse("document.pdf")
```

### ParsedDocument (Dataclass)

Metadata about parsed document:

```python
@dataclass
class ParsedDocument:
    text: str                       # Extracted text content
    file_path: str                  # Source file path
    file_size: int                  # File size in bytes
    total_pages: int = 0            # For multi-page formats
    encoding: str = "utf-8"         # Text encoding
    extracted_at: str = ""          # Extraction timestamp
```

## Usage Examples

### Single Document Parsing

```python
from src.extraction.parsers import ParserFactory

# Create factory
factory = ParserFactory()

# Parse PDF
parser = factory.get_parser("report.pdf")
text = parser.parse("report.pdf")

# Parse DOCX
parser = factory.get_parser("contract.docx")
text = parser.parse("contract.docx")

# Parse CSV
parser = factory.get_parser("data.csv")
text = parser.parse("data.csv")

# Parse webpage from URL
parser = factory.get_parser("https://example.com")
text = parser.parse("https://example.com")

# Parse local HTML file
parser = factory.get_parser("page.html")
text = parser.parse("page.html")
```

### Batch Processing

```python
from src.extraction.parsers import ParserFactory
from pathlib import Path

factory = ParserFactory()
results = []

for file_path in Path("documents/").glob("*"):
    try:
        parser = factory.get_parser(str(file_path))
        text = parser.parse(str(file_path))
        results.append({
            "file": str(file_path),
            "text": text,
            "success": True
        })
    except Exception as e:
        results.append({
            "file": str(file_path),
            "error": str(e),
            "success": False
        })
```

### Error Handling

```python
from src.extraction.parsers import ParserFactory, DocumentType

factory = ParserFactory()

try:
    parser = factory.get_parser("document.pdf")
    text = parser.parse("document.pdf")
except FileNotFoundError:
    print("Document not found")
except ValueError:
    print("Format not supported")
except Exception as e:
    print(f"Parsing error: {e}")
```

## File Structure

```
parsers/
├── __init__.py              # Module exports
├── base_parser.py           # Abstract base class
├── pdf_parser.py            # PDF implementation
├── docx_parser.py           # DOCX implementation
├── csv_parser.py            # CSV implementation
├── txt_parser.py            # TXT implementation
├── webpage_parser.py        # Webpage/HTML implementation
├── parser_factory.py        # Factory pattern
└── README.md                # This file
```

## Testing

```bash
# Run all parser tests
pytest tests/extraction/test_parsers.py -v

# Run specific parser test
pytest tests/extraction/test_parsers.py::TestPDFParser -v

# With coverage
pytest tests/extraction/test_parsers.py --cov=src/extraction/parsers
```

## Test Coverage

- ✅ PDF extraction (single and multi-page)
- ✅ DOCX with tables and formatting
- ✅ CSV with various delimiters
- ✅ TXT with encoding detection
- ✅ HTML parsing from URLs
- ✅ Local HTML file parsing
- ✅ HTML tag removal and text extraction
- ✅ JSON structure preservation
- ✅ Factory pattern auto-detection
- ✅ Error handling for corrupted files
- ✅ Metadata preservation
- ✅ Large file handling
- ✅ Edge cases (empty files, special characters)
- ✅ Network error handling (URL parsing)

## Performance

| Format | Speed | Memory |
|--------|-------|--------|
| PDF | ~10ms/page | ~1MB/page |
| DOCX | ~5ms | ~100KB |
| CSV | ~1ms/1000 rows | ~50KB/1000 rows |
| TXT | <1ms | Minimal |
| JSON | ~2ms | Minimal |

## Dependencies

- **PyPDF2** (4.1.1) - PDF parsing
- **python-docx** (0.8.11) - DOCX parsing
- **requests** (2.31.0) - URL fetching for webpage parsing
- **beautifulsoup4** (4.12.2) - HTML parsing and tag removal
- **csv** (stdlib) - CSV handling
- **json** (stdlib) - JSON parsing

## Configuration

No explicit configuration needed - parsers auto-detect formats based on file extensions.

## Error Handling

All parsers implement graceful error handling:

```python
try:
    text = parser.parse("document.pdf")
except FileNotFoundError:
    # Handle missing file
    pass
except Exception as e:
    # Handle parsing error
    logger.error(f"Parsing failed: {e}")
```

## Next Steps

Parsed text is typically passed to the chunking module (Phase 2) for further processing:

```python
from src.extraction.parsers import ParserFactory
from src.extraction.chunking import SemanticChunker

# Parse
factory = ParserFactory()
parser = factory.create_parser("document.pdf")
text = parser.parse("document.pdf")

# Chunk
chunker = SemanticChunker()
chunks = chunker.chunk(text, source_file="document.pdf")

# Further processing...
```

## See Also

- [Phase 2: Text Chunking](../chunking/README.md)
- [Extraction Pipeline Overview](../README.md)
- [Main Documentation](../../../PHASES_1_5_SUMMARY.md)
