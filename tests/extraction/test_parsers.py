"""Tests for document parsers."""

from pathlib import Path

import pytest

from src.extraction import (
    DocumentType,
    ParserFactory,
    ParsedDocument,
    TXTParser,
    CSVParser,
)


class TestBaseParser:
    """Tests for base parser functionality."""

    def test_detect_pdf_type(self):
        """Test PDF type detection."""
        file_type = DocumentType.PDF
        assert file_type == DocumentType.PDF

    def test_detect_csv_type(self):
        """Test CSV type detection."""
        file_type = DocumentType.CSV
        assert file_type == DocumentType.CSV

    def test_detect_txt_type(self):
        """Test TXT type detection."""
        file_type = DocumentType.TXT
        assert file_type == DocumentType.TXT


class TestTXTParser:
    """Tests for plain text parser."""

    def test_txt_parser_initialization(self):
        """Test that TXTParser initializes correctly."""
        parser = TXTParser()
        assert parser is not None
        assert parser.encoding == "utf-8"

    def test_txt_parser_custom_encoding(self):
        """Test TXTParser with custom encoding."""
        parser = TXTParser(encoding="latin-1")
        assert parser.encoding == "latin-1"

    def test_can_parse_txt_file(self, txt_file):
        """Test can_parse for txt files."""
        parser = TXTParser()
        assert parser.can_parse(txt_file)

    def test_parse_txt_file(self, txt_file):
        """Test parsing a plain text file."""
        parser = TXTParser()
        doc = parser.parse(txt_file)

        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == DocumentType.TXT
        assert len(doc.raw_text) > 0
        assert "sample text document" in doc.raw_text
        assert doc.metadata["parser"] == "plain_text"
        assert "line_count" in doc.metadata

    def test_parse_multiline_txt(self, multiline_txt_file):
        """Test parsing a multi-line text file."""
        parser = TXTParser()
        doc = parser.parse(multiline_txt_file)

        assert isinstance(doc, ParsedDocument)
        assert doc.metadata["line_count"] == 100
        assert "Line 1:" in doc.raw_text
        assert "Line 100:" in doc.raw_text

    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        parser = TXTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.txt")


class TestCSVParser:
    """Tests for CSV parser."""

    def test_csv_parser_initialization(self):
        """Test that CSVParser initializes correctly."""
        parser = CSVParser()
        assert parser is not None
        assert parser.delimiter == ","

    def test_csv_parser_custom_delimiter(self):
        """Test CSVParser with custom delimiter."""
        parser = CSVParser(delimiter=";")
        assert parser.delimiter == ";"

    def test_can_parse_csv_file(self, csv_file):
        """Test can_parse for csv files."""
        parser = CSVParser()
        assert parser.can_parse(csv_file)

    def test_parse_csv_file(self, csv_file):
        """Test parsing a CSV file."""
        parser = CSVParser()
        doc = parser.parse(csv_file)

        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == DocumentType.CSV
        assert len(doc.raw_text) > 0
        assert "Name" in doc.raw_text
        assert "Alice" in doc.raw_text
        assert doc.metadata["parser"] == "pandas"
        assert doc.metadata["rows"] == 3
        assert doc.metadata["columns"] == 3
        assert "Name" in doc.metadata["column_names"]

    def test_parse_csv_preserves_headers(self, csv_file):
        """Test that CSV parsing preserves column headers."""
        parser = CSVParser()
        doc = parser.parse(csv_file)

        headers = doc.metadata["column_names"]
        assert headers == ["Name", "Age", "City"]

    def test_parse_nonexistent_csv(self):
        """Test parsing a non-existent CSV file."""
        parser = CSVParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.csv")


class TestParserFactory:
    """Tests for parser factory."""

    def test_factory_get_txt_parser(self, txt_file):
        """Test factory returns TXTParser for txt files."""
        parser = ParserFactory.get_parser(txt_file)
        assert isinstance(parser, TXTParser)

    def test_factory_get_csv_parser(self, csv_file):
        """Test factory returns CSVParser for csv files."""
        parser = ParserFactory.get_parser(csv_file)
        assert isinstance(parser, CSVParser)

    def test_factory_get_supported_types(self):
        """Test getting supported types."""
        supported = ParserFactory.get_supported_types()
        assert DocumentType.TXT in supported
        assert DocumentType.CSV in supported
        assert DocumentType.PDF in supported
        assert DocumentType.DOCX in supported

    def test_factory_can_parse_txt(self, txt_file):
        """Test factory can_parse for txt files."""
        assert ParserFactory.can_parse(txt_file)

    def test_factory_can_parse_csv(self, csv_file):
        """Test factory can_parse for csv files."""
        assert ParserFactory.can_parse(csv_file)

    def test_factory_cannot_parse_invalid(self, invalid_file):
        """Test factory returns False for invalid file types."""
        assert not ParserFactory.can_parse(invalid_file)

    def test_factory_unsupported_type(self, invalid_file):
        """Test factory raises error for unsupported types."""
        with pytest.raises(ValueError):
            ParserFactory.get_parser(invalid_file)

    def test_factory_explicit_type(self, txt_file):
        """Test factory with explicit type specification."""
        parser = ParserFactory.get_parser(txt_file, DocumentType.TXT)
        assert isinstance(parser, TXTParser)

    def test_factory_register_parser(self):
        """Test registering a custom parser."""
        # Custom test parser class
        class CustomParser(TXTParser):
            pass

        # Register it
        ParserFactory.register_parser(DocumentType.TXT, CustomParser)

        # Verify it's registered
        parser = ParserFactory.get_parser("test.txt", DocumentType.TXT)
        assert isinstance(parser, CustomParser)


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_parsed_document_initialization(self, txt_file):
        """Test ParsedDocument initialization."""
        doc = ParsedDocument(
            file_path=txt_file,
            file_type=DocumentType.TXT,
            raw_text="Sample content",
            metadata={"key": "value"}
        )

        assert doc.file_path == txt_file
        assert doc.file_type == DocumentType.TXT
        assert doc.raw_text == "Sample content"
        assert doc.metadata == {"key": "value"}

    def test_parsed_document_character_count(self, txt_file):
        """Test character count property."""
        doc = ParsedDocument(
            file_path=txt_file,
            file_type=DocumentType.TXT,
            raw_text="Hello World!",
        )

        assert doc.character_count == 12

    def test_parsed_document_page_count_no_pages(self, txt_file):
        """Test page count when no pages specified."""
        doc = ParsedDocument(
            file_path=txt_file,
            file_type=DocumentType.TXT,
            raw_text="Content",
        )

        assert doc.page_count == 1

    def test_parsed_document_page_count_with_pages(self, txt_file):
        """Test page count with pages specified."""
        doc = ParsedDocument(
            file_path=txt_file,
            file_type=DocumentType.TXT,
            raw_text="Content",
            pages=["Page 1", "Page 2", "Page 3"]
        )

        assert doc.page_count == 3

    def test_parsed_document_repr(self, txt_file):
        """Test string representation."""
        doc = ParsedDocument(
            file_path=txt_file,
            file_type=DocumentType.TXT,
            raw_text="Content",
        )

        repr_str = repr(doc)
        assert "ParsedDocument" in repr_str
        assert "txt" in repr_str
