"""Base classes and data models for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    TXT = "txt"
    HTML = "html"
    XLSX = "xlsx"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    """Represents a parsed document with extracted content and metadata."""

    file_path: str
    file_type: DocumentType
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.now)

    @property
    def character_count(self) -> int:
        """Return the total character count."""
        return len(self.raw_text)

    @property
    def page_count(self) -> int:
        """Return the number of pages."""
        return len(self.pages) if self.pages else 1

    def __repr__(self) -> str:
        return (
            f"ParsedDocument("
            f"file_type={self.file_type.value}, "
            f"chars={self.character_count}, "
            f"pages={self.page_count})"
        )


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self):
        """Initialize the parser."""
        pass

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this parser can handle the file, False otherwise
        """
        pass

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document and return its content.

        Args:
            file_path: Path to the document to parse

        Returns:
            ParsedDocument with extracted content and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be parsed
        """
        pass

    def _validate_file_exists(self, file_path: str) -> Path:
        """
        Validate that a file exists and return a Path object.

        Args:
            file_path: Path to validate

        Returns:
            Path object if file exists

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        return path

    @staticmethod
    def detect_file_type(file_path: str) -> DocumentType:
        """
        Detect document type from file extension.

        Args:
            file_path: Path to the file

        Returns:
            DocumentType enum value
        """
        suffix = Path(file_path).suffix.lower()
        type_map = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".csv": DocumentType.CSV,
            ".txt": DocumentType.TXT,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".xlsx": DocumentType.XLSX,
            ".xls": DocumentType.XLSX,
        }
        return type_map.get(suffix, DocumentType.UNKNOWN)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
