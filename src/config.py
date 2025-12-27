"""
Configuration management for the application
"""

from typing import Optional
from pydantic_settings import BaseSettings


class TigerGraphConfig(BaseSettings):
    """TigerGraph connection configuration"""

    host: str = "localhost"
    port: int = 9000
    username: str = "tigergraph"
    password: str = "tigergraph"
    graph_name: str = "graph_rag"
    protocol: str = "http"

    @property
    def url(self) -> str:
        """Get the full TigerGraph URL"""
        return f"{self.protocol}://{self.host}:{self.port}"

    class Config:
        env_prefix = "TIGERGRAPH_"


class PostgresConfig(BaseSettings):
    """PostgreSQL connection configuration"""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "graph_rag"

    @property
    def connection_string(self) -> str:
        """Get the PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "POSTGRES_"


class LLMConfig(BaseSettings):
    """LLM configuration"""

    provider: str = "ollama"  # ollama, openai, huggingface
    model_name: str = "mistral"  # mistral, llama2, etc
    base_url: Optional[str] = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024

    class Config:
        env_prefix = "LLM_"


class ExtractionConfig(BaseSettings):
    """Data extraction configuration"""

    # Entity extraction
    entity_model: str = "en_core_web_trf"  # spaCy model
    entity_confidence_threshold: float = 0.5

    # Relation extraction
    relation_model: str = "distilbert-base-uncased"
    relation_confidence_threshold: float = 0.4

    # Text chunking
    chunk_size: int = 512
    chunk_overlap: int = 128

    class Config:
        env_prefix = "EXTRACTION_"


class AppConfig(BaseSettings):
    """Application configuration"""

    debug: bool = False
    env: str = "development"
    log_level: str = "INFO"
    port: int = 8000

    # Component configurations
    tigergraph: TigerGraphConfig = TigerGraphConfig()
    postgres: PostgresConfig = PostgresConfig()
    llm: LLMConfig = LLMConfig()
    extraction: ExtractionConfig = ExtractionConfig()

    class Config:
        env_prefix = "APP_"


# Global config instance
config = AppConfig()
