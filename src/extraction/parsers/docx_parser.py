"""DOCX (Word) document parser using python-docx."""

import logging
from typing import List

from .base_parser import BaseParser, DocumentType, ParsedDocument

logger = logging.getLogger(__name__)

try:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False


class DOCXParser(BaseParser):
    """Parser for DOCX (Microsoft Word) documents."""

    def __init__(self):
        """Initialize the DOCX parser."""
        super().__init__()
        if not HAS_PYTHON_DOCX:
            raise ImportError(
                "python-docx is not installed. "
                "Install it with: pip install python-docx"
            )

    def can_parse(self, file_path: str) -> bool:
        """Check if file is a DOCX."""
        file_type = self.detect_file_type(file_path)
        return file_type == DocumentType.DOCX

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a DOCX file and extract text.

        Args:
            file_path: Path to the DOCX file

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If DOCX cannot be parsed
        """
        path = self._validate_file_exists(file_path)

        try:
            doc = Document(str(path))

            # Extract text from paragraphs and tables
            text_parts: List[str] = []
            page_breaks: List[int] = []

            for element in doc.element.body:
                if element.tag.endswith("p"):
                    # Paragraph
                    para = Paragraph(element, doc.element.body)
                    text = para.text.strip()
                    if text:
                        text_parts.append(text)

                    # Check for page break
                    if self._has_page_break(para):
                        page_breaks.append(len(text_parts))

                elif element.tag.endswith("tbl"):
                    # Table
                    table = Table(element, doc.element.body)
                    table_text = self._extract_table_text(table)
                    if table_text:
                        text_parts.append(f"[Table]\n{table_text}")

            raw_text = "\n".join(text_parts)

            # Extract metadata
            metadata = {
                "parser": "python-docx",
                "element_count": len(doc.element.body),
            }

            # Add document properties if available
            if doc.core_properties:
                core_props = doc.core_properties
                metadata.update({
                    "title": core_props.title or "",
                    "author": core_props.author or "",
                    "subject": core_props.subject or "",
                    "keywords": core_props.keywords or "",
                    "created": core_props.created,
                    "modified": core_props.modified,
                })

            logger.info(f"Successfully parsed DOCX: {len(text_parts)} elements")

            return ParsedDocument(
                file_path=file_path,
                file_type=DocumentType.DOCX,
                raw_text=raw_text,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}")
            raise ValueError(f"Cannot parse DOCX: {e}") from e

    @staticmethod
    def _has_page_break(paragraph: Paragraph) -> bool:
        """Check if paragraph contains a page break."""
        for run in paragraph.runs:
            if run._r.br_count:
                return True
        return False

    @staticmethod
    def _extract_table_text(table: Table) -> str:
        """Extract text content from a table."""
        rows_text: List[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows_text.append(" | ".join(cells))
        return "\n".join(rows_text)
