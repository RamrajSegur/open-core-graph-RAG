"""Pipeline configuration and settings."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from src.extraction import RelationshipType, EntityType


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    strategy: str = "semantic"  # "semantic" or "sliding_window"
    semantic_max_size: int = 512  # Max characters per semantic chunk
    window_size: int = 256  # Token window size for sliding window
    overlap: int = 64  # Token overlap for sliding window


@dataclass
class NERConfig:
    """Configuration for Named Entity Recognition."""

    enabled: bool = True
    model_name: str = "en_core_web_sm"  # SpaCy model
    entity_types: List[str] = None  # Specific types to extract (None = all)
    confidence_threshold: float = 0.0  # Minimum confidence to keep

    def __post_init__(self):
        """Initialize default entity types."""
        if self.entity_types is None:
            self.entity_types = [t.value for t in EntityType]


@dataclass
class RelationshipConfig:
    """Configuration for relationship extraction."""

    enabled: bool = True
    use_patterns: bool = True  # Pattern-based extraction
    use_semantic: bool = True  # Semantic co-occurrence
    relationship_types: List[str] = None  # Specific types to extract (None = all)
    confidence_threshold: float = 0.0  # Minimum confidence to keep
    deduplicate: bool = True  # Remove duplicate relationships
    case_sensitive: bool = False  # Case sensitivity for deduplication

    def __post_init__(self):
        """Initialize default relationship types."""
        if self.relationship_types is None:
            self.relationship_types = [t.value for t in RelationshipType]


@dataclass
class StorageConfig:
    """Configuration for knowledge graph storage."""

    enabled: bool = True
    backend: str = "tigergraph"  # "tigergraph" or "neo4j"
    host: str = "localhost"
    port: int = 9000
    graph_name: str = "RAG"
    username: str = "tigergraph"
    password: str = "tigergraph"
    skip_duplicates: bool = True  # Skip duplicate entities/relationships


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    chunking: ChunkingConfig = None
    ner: NERConfig = None
    relationships: RelationshipConfig = None
    storage: StorageConfig = None

    def __post_init__(self):
        """Initialize default subconfigs."""
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.ner is None:
            self.ner = NERConfig()
        if self.relationships is None:
            self.relationships = RelationshipConfig()
        if self.storage is None:
            self.storage = StorageConfig()

    @classmethod
    def from_yaml(cls, file_path: str) -> "PipelineConfig":
        """Load configuration from YAML file.

        Args:
            file_path: Path to YAML config file

        Returns:
            Loaded PipelineConfig
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, file_path: str) -> "PipelineConfig":
        """Load configuration from JSON file.

        Args:
            file_path: Path to JSON config file

        Returns:
            Loaded PipelineConfig
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Load configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Loaded PipelineConfig
        """
        chunking_data = data.get("chunking", {})
        chunking = ChunkingConfig(**chunking_data) if chunking_data else ChunkingConfig()

        ner_data = data.get("ner", {})
        ner = NERConfig(**ner_data) if ner_data else NERConfig()

        relationships_data = data.get("relationships", {})
        relationships = (
            RelationshipConfig(**relationships_data)
            if relationships_data
            else RelationshipConfig()
        )

        storage_data = data.get("storage", {})
        storage = StorageConfig(**storage_data) if storage_data else StorageConfig()

        return cls(
            chunking=chunking,
            ner=ner,
            relationships=relationships,
            storage=storage,
        )

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            file_path: Path to save YAML file
        """
        data = self.to_dict()
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_json(self, file_path: str) -> None:
        """Save configuration to JSON file.

        Args:
            file_path: Path to save JSON file
        """
        data = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "chunking": asdict(self.chunking),
            "ner": asdict(self.ner),
            "relationships": asdict(self.relationships),
            "storage": asdict(self.storage),
        }

    def to_json_str(self) -> str:
        """Convert configuration to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)
