"""CSV document parser using pandas."""

import logging
from io import StringIO
from typing import Dict, List

from .base_parser import BaseParser, DocumentType, ParsedDocument

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class CSVParser(BaseParser):
    """Parser for CSV files."""

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        """
        Initialize the CSV parser.

        Args:
            delimiter: Column delimiter (default: comma)
            encoding: File encoding (default: utf-8)
        """
        super().__init__()
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is not installed. "
                "Install it with: pip install pandas"
            )
        self.delimiter = delimiter
        self.encoding = encoding

    def can_parse(self, file_path: str) -> bool:
        """Check if file is a CSV."""
        return self.detect_file_type(file_path) == DocumentType.CSV

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a CSV file and extract data.

        Args:
            file_path: Path to the CSV file

        Returns:
            ParsedDocument with tabular data as text

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV cannot be parsed
        """
        path = self._validate_file_exists(file_path)

        try:
            # Read CSV with pandas
            df = pd.read_csv(
                str(path),
                delimiter=self.delimiter,
                encoding=self.encoding,
            )

            logger.debug(f"Loaded CSV with shape: {df.shape}")

            # Convert to formatted text
            raw_text = self._dataframe_to_text(df)

            # Extract metadata
            metadata = {
                "parser": "pandas",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
            }

            logger.info(f"Successfully parsed CSV: {metadata['rows']} rows, "
                       f"{metadata['columns']} columns")

            return ParsedDocument(
                file_path=file_path,
                file_type=DocumentType.CSV,
                raw_text=raw_text,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse CSV {file_path}: {e}")
            raise ValueError(f"Cannot parse CSV: {e}") from e

    @staticmethod
    def _dataframe_to_text(df) -> str:
        """
        Convert a pandas DataFrame to formatted text.

        Args:
            df: pandas DataFrame

        Returns:
            Formatted text representation
        """
        # Use tabulate-like formatting
        text_parts: List[str] = []

        # Add header
        header = " | ".join(str(col) for col in df.columns)
        text_parts.append(header)
        text_parts.append("-" * len(header))

        # Add rows
        for _, row in df.iterrows():
            row_text = " | ".join(str(val) for val in row.values)
            text_parts.append(row_text)

        return "\n".join(text_parts)
