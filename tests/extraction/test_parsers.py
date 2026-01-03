"""Tests for document parsers."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.extraction import (
    DocumentType,
    ParserFactory,
    ParsedDocument,
    TXTParser,
    CSVParser,
    WebpageParser,
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


class TestWebpageParser:
    """Tests for webpage parser."""

    def test_webpage_parser_initialization(self):
        """Test that WebpageParser initializes correctly."""
        parser = WebpageParser()
        assert parser is not None
        assert parser.timeout == 10

    def test_webpage_parser_custom_timeout(self):
        """Test WebpageParser with custom timeout."""
        parser = WebpageParser(timeout=30)
        assert parser.timeout == 30

    def test_can_parse_url_http(self):
        """Test can_parse for HTTP URLs."""
        parser = WebpageParser()
        assert parser.can_parse("http://example.com")
        assert parser.can_parse("https://example.com/page")

    def test_can_parse_html_file(self, tmp_path):
        """Test can_parse for HTML files."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")
        
        parser = WebpageParser()
        assert parser.can_parse(str(html_file))

    def test_cannot_parse_non_html(self):
        """Test can_parse returns False for non-HTML files."""
        parser = WebpageParser()
        assert not parser.can_parse("document.pdf")
        assert not parser.can_parse("data.csv")
        assert not parser.can_parse("/path/to/file.txt")

    @patch('requests.get')
    def test_parse_url_success(self, mock_get):
        """Test parsing a URL successfully."""
        # Mock the response
        mock_response = Mock()
        mock_response.content = b"<html><head><title>Test Page</title></head><body>Hello World</body></html>"
        mock_response.headers = {
            'content-type': 'text/html; charset=utf-8'
        }
        mock_response.encoding = 'utf-8'
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Parse URL
        parser = WebpageParser()
        doc = parser.parse("https://example.com")

        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == DocumentType.HTML
        assert "Hello World" in doc.raw_text
        assert doc.metadata['url'] == "https://example.com"
        assert doc.metadata['status_code'] == 200

    def test_parse_html_file(self, tmp_path):
        """Test parsing a local HTML file."""
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><head><title>Test</title></head>"
            "<body><h1>Header</h1><p>Content here</p></body></html>"
        )

        parser = WebpageParser()
        doc = parser.parse(str(html_file))

        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == DocumentType.HTML
        assert "Header" in doc.raw_text
        assert "Content here" in doc.raw_text
        assert doc.metadata['filename'] == "test.html"

    def test_parse_html_file_nonexistent(self):
        """Test parsing a non-existent HTML file."""
        parser = WebpageParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.html")

    def test_parse_invalid_input(self):
        """Test parsing with invalid input."""
        parser = WebpageParser()
        with pytest.raises(ValueError):
            parser.parse("document.pdf")

    def test_parse_url_network_error(self):
        """Test handling network errors when fetching URL."""
        parser = WebpageParser()
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            with pytest.raises(ValueError):
                parser.parse("https://invalid-domain-12345.com")

    def test_parse_html_preserves_text_content(self, tmp_path):
        """Test that HTML parsing extracts text without HTML tags."""
        html_file = tmp_path / "content.html"
        html_file.write_text(
            "<html><body>"
            "<script>var x = 1;</script>"
            "<style>.class { color: red; }</style>"
            "<p>Useful content</p>"
            "<div>More content</div>"
            "</body></html>"
        )

        parser = WebpageParser()
        doc = parser.parse(str(html_file))

        # Should have content but not scripts/styles
        assert "Useful content" in doc.raw_text
        assert "More content" in doc.raw_text
        assert "var x = 1;" not in doc.raw_text
        assert ".class { color: red; }" not in doc.raw_text

    def test_factory_get_webpage_parser(self):
        """Test factory returns WebpageParser for HTML."""
        parser = ParserFactory.get_parser("https://example.com")
        assert isinstance(parser, WebpageParser)

    def test_factory_register_webpage_parser(self):
        """Test that WebpageParser is registered in factory."""
        supported = ParserFactory.get_supported_types()
        assert DocumentType.HTML in supported

    def test_factory_can_parse_url(self):
        """Test factory can_parse for URLs."""
        assert ParserFactory.can_parse("https://example.com")

    def test_factory_can_parse_html_file(self, tmp_path):
        """Test factory can_parse for HTML files."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html></html>")
        
        assert ParserFactory.can_parse(str(html_file))

