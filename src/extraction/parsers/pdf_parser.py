"""PDF document parser using PyPDF2 with fallback to pdfplumber."""

import logging
from typing import List, Optional

from .base_parser import BaseParser, DocumentType, ParsedDocument

logger = logging.getLogger(__name__)

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class PDFParser(BaseParser):
    """Parser for PDF documents."""

    def __init__(self, use_pdfplumber: bool = False):
        """
        Initialize the PDF parser.

        Args:
            use_pdfplumber: If True, use pdfplumber instead of PyPDF2.
                           Useful for complex PDFs with tables/formatting.
        """
        super().__init__()
        self.use_pdfplumber = use_pdfplumber

        if use_pdfplumber and not HAS_PDFPLUMBER:
            logger.warning(
                "pdfplumber not available, falling back to PyPDF2. "
                "Install pdfplumber for better table extraction."
            )
            self.use_pdfplumber = False

        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            raise ImportError(
                "Neither PyPDF2 nor pdfplumber is installed. "
                "Install one with: pip install PyPDF2 pdfplumber"
            )

    def can_parse(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return self.detect_file_type(file_path) == DocumentType.PDF

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract text.

        Args:
            file_path: Path to the PDF file

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If PDF cannot be parsed
        """
        path = self._validate_file_exists(file_path)

        try:
            if self.use_pdfplumber:
                return self._parse_with_pdfplumber(str(path))
            else:
                return self._parse_with_pypdf2(str(path))
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise ValueError(f"Cannot parse PDF: {e}") from e

    def _parse_with_pypdf2(self, file_path: str) -> ParsedDocument:
        """Parse PDF using PyPDF2."""
        logger.debug(f"Parsing PDF with PyPDF2: {file_path}")

        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            num_pages = len(reader.pages)

            # Extract text from all pages
            pages: List[str] = []
            full_text_parts: List[str] = []

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                pages.append(text)
                full_text_parts.append(f"--- Page {page_num} ---\n{text}")

        raw_text = "\n\n".join(full_text_parts)

        # Extract metadata
        metadata = {
            "total_pages": num_pages,
            "parser": "PyPDF2",
        }

        if reader.metadata:
            metadata.update({
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            })

        logger.info(f"Successfully parsed PDF: {num_pages} pages")

        return ParsedDocument(
            file_path=file_path,
            file_type=DocumentType.PDF,
            raw_text=raw_text,
            pages=pages,
            metadata=metadata,
        )

    def _parse_with_pdfplumber(self, file_path: str) -> ParsedDocument:
        """Parse PDF using pdfplumber for better table extraction."""
        logger.debug(f"Parsing PDF with pdfplumber: {file_path}")

        pages: List[str] = []
        full_text_parts: List[str] = []
        metadata_dict = {}

        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""

                # Try to extract tables if they exist
                tables = page.extract_tables()
                if tables:
                    text += "\n\n[Tables found on page]\n"
                    for table in tables:
                        text += "\n".join(
                            [" | ".join(str(cell) for cell in row) for row in table]
                        )
                        text += "\n"

                pages.append(text)
                full_text_parts.append(f"--- Page {page_num} ---\n{text}")

            # Extract document metadata
            if pdf.metadata:
                metadata_dict.update(pdf.metadata)

        raw_text = "\n\n".join(full_text_parts)

        # Extract metadata
        metadata = {
            "total_pages": num_pages,
            "parser": "pdfplumber",
        }
        metadata.update(metadata_dict)

        logger.info(
            f"Successfully parsed PDF with pdfplumber: {num_pages} pages"
        )

        return ParsedDocument(
            file_path=file_path,
            file_type=DocumentType.PDF,
            raw_text=raw_text,
            pages=pages,
            metadata=metadata,
        )
