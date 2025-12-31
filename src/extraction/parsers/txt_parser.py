"""Plain text document parser."""

import logging

from .base_parser import BaseParser, DocumentType, ParsedDocument

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files."""

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the TXT parser.

        Args:
            encoding: File encoding (default: utf-8)
        """
        super().__init__()
        self.encoding = encoding

    def can_parse(self, file_path: str) -> bool:
        """Check if file is plain text."""
        return self.detect_file_type(file_path) == DocumentType.TXT

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a plain text file.

        Args:
            file_path: Path to the text file

        Returns:
            ParsedDocument with text content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        path = self._validate_file_exists(file_path)

        try:
            # Read file with specified encoding
            with open(path, "r", encoding=self.encoding) as f:
                raw_text = f.read()

            logger.info(f"Successfully parsed TXT: {len(raw_text)} characters")

            metadata = {
                "parser": "plain_text",
                "encoding": self.encoding,
                "line_count": len(raw_text.splitlines()),
            }

            return ParsedDocument(
                file_path=file_path,
                file_type=DocumentType.TXT,
                raw_text=raw_text,
                metadata=metadata,
            )

        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {file_path}: {e}")
            # Try with latin-1 as fallback
            try:
                with open(path, "r", encoding="latin-1") as f:
                    raw_text = f.read()
                logger.warning(
                    f"Fell back to latin-1 encoding for {file_path}"
                )
                metadata = {
                    "parser": "plain_text",
                    "encoding": "latin-1 (fallback)",
                    "line_count": len(raw_text.splitlines()),
                }
                return ParsedDocument(
                    file_path=file_path,
                    file_type=DocumentType.TXT,
                    raw_text=raw_text,
                    metadata=metadata,
                )
            except Exception as e2:
                raise ValueError(
                    f"Cannot parse TXT with either encoding: {e2}"
                ) from e2

        except Exception as e:
            logger.error(f"Failed to parse TXT {file_path}: {e}")
            raise ValueError(f"Cannot parse TXT: {e}") from e
