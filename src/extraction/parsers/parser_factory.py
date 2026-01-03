"""Factory for creating appropriate parsers based on file type."""

import logging
from typing import Dict, Type

from .base_parser import BaseParser, DocumentType
from .csv_parser import CSVParser
from .docx_parser import DOCXParser
from .pdf_parser import PDFParser
from .txt_parser import TXTParser
from .webpage_parser import WebpageParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """Factory for creating document parsers."""

    # Map of document types to parser classes
    _parser_map: Dict[DocumentType, Type[BaseParser]] = {
        DocumentType.PDF: PDFParser,
        DocumentType.DOCX: DOCXParser,
        DocumentType.CSV: CSVParser,
        DocumentType.TXT: TXTParser,
        DocumentType.HTML: WebpageParser,
    }

    @classmethod
    def get_parser(
        cls,
        file_path: str,
        file_type: DocumentType | None = None,
        **kwargs
    ) -> BaseParser:
        """
        Get an appropriate parser for the given file.

        Args:
            file_path: Path to the document
            file_type: DocumentType override (auto-detect if None)
            **kwargs: Additional arguments to pass to parser constructor

        Returns:
            Instantiated parser for the document type

        Raises:
            ValueError: If file type is unsupported
        """
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = BaseParser.detect_file_type(file_path)

        if file_type not in cls._parser_map:
            raise ValueError(
                f"Unsupported document type: {file_type.value}. "
                f"Supported types: {', '.join(t.value for t in cls._parser_map.keys())}"
            )

        parser_class = cls._parser_map[file_type]
        logger.debug(f"Creating {parser_class.__name__} for {file_path}")

        try:
            return parser_class(**kwargs)
        except ImportError as e:
            logger.error(f"Cannot create parser: {e}")
            raise ValueError(
                f"Required dependencies not installed for {file_type.value} parsing: {e}"
            ) from e

    @classmethod
    def register_parser(
        cls,
        file_type: DocumentType,
        parser_class: Type[BaseParser]
    ) -> None:
        """
        Register a custom parser for a document type.

        Args:
            file_type: DocumentType to register for
            parser_class: Parser class to use
        """
        if not issubclass(parser_class, BaseParser):
            raise TypeError(
                f"{parser_class} must inherit from BaseParser"
            )
        cls._parser_map[file_type] = parser_class
        logger.info(f"Registered {parser_class.__name__} for {file_type.value}")

    @classmethod
    def get_supported_types(cls) -> list[DocumentType]:
        """Get list of supported document types."""
        return list(cls._parser_map.keys())

    @classmethod
    def can_parse(cls, file_path: str) -> bool:
        """
        Check if any registered parser can handle the file.

        Args:
            file_path: Path to check

        Returns:
            True if a parser exists for this file type
        """
        file_type = BaseParser.detect_file_type(file_path)
        return file_type in cls._parser_map
