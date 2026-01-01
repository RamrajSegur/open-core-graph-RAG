"""Fixtures for extraction tests."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def txt_file():
    """Create a temporary plain text file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write("This is a sample text document.\n")
        f.write("It contains multiple lines.\n")
        f.write("And should be parsed correctly.\n")
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink()


@pytest.fixture
def csv_file():
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age", "City"])
        writer.writerow(["Alice", "30", "New York"])
        writer.writerow(["Bob", "25", "Los Angeles"])
        writer.writerow(["Charlie", "35", "Chicago"])
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink()


@pytest.fixture
def multiline_txt_file():
    """Create a multi-page text file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8"
    ) as f:
        for i in range(100):
            f.write(f"Line {i + 1}: This is a sample line.\n")
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink()


@pytest.fixture
def invalid_file():
    """Create a temporary file with invalid type."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xyz",
        delete=False
    ) as f:
        f.write("Invalid content")
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink()


@pytest.fixture
def mock_spacy_model():
    """Create a mock SpaCy model for NER tests.
    
    This fixture mocks the SpaCy NLP pipeline to avoid
    needing the actual model installed during tests.
    """
    # Create a mock entity
    mock_entity = MagicMock()
    mock_entity.text = "John Smith"
    mock_entity.label_ = "PERSON"
    mock_entity.start_char = 0
    mock_entity.end_char = 10
    mock_entity.__len__ = lambda self: 2  # 2 tokens
    
    # Create a mock doc
    mock_doc = MagicMock()
    mock_doc.ents = [mock_entity]
    
    # Create a mock nlp
    mock_nlp = MagicMock()
    mock_nlp.pipe_names = ["tok2vec", "ner"]
    mock_nlp.return_value = mock_doc
    mock_nlp.__call__ = lambda text: mock_doc
    
    def mock_pipe(texts):
        """Mock the pipe method to return docs."""
        for _ in texts:
            yield mock_doc
    
    mock_nlp.pipe = mock_pipe
    
    # Patch spacy.load to return our mock
    with patch("src.extraction.ner.ner_model.spacy.load") as mock_load:
        mock_load.return_value = mock_nlp
        yield mock_nlp
