"""
Abstract base class for graph storage backends.

This module defines the GraphStore interface that all implementations must follow.
This allows for pluggable backends (TigerGraph, local in-memory, Neo4j, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class EdgeDirection(str, Enum):
    """Direction for graph traversal"""
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass
class Entity:
    """Represents a graph entity (node)"""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class Relationship:
    """Represents a graph relationship (edge)"""
    id: str
    source_id: str
    target_id: str
    rel_type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return self.id == other.id


@dataclass
class QueryResult:
    """Result from a graph query"""
    query: str
    results: List[Dict[str, Any]]
    execution_time_ms: int
    result_count: int = None
    
    def __post_init__(self):
        if self.result_count is None:
            self.result_count = len(self.results)


class GraphStore(ABC):
    """
    Abstract base class for graph storage implementations.
    
    Defines the interface that all graph storage backends must implement.
    Allows for pluggable implementations: TigerGraph, NetworkX, Neo4j, etc.
    """
    
    # ==================== Connection & Lifecycle ====================
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the graph backend.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to the graph backend.
        
        Returns:
            bool: True if disconnection successful
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if currently connected to the graph backend.
        
        Returns:
            bool: True if connected
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the health of the graph backend.
        
        Returns:
            bool: True if backend is healthy and responsive
        """
        pass
    
    # ==================== Schema Management ====================
    
    @abstractmethod
    def create_schema(self, schema_definition: Dict[str, Any]) -> bool:
        """
        Create or initialize the graph schema.
        
        Args:
            schema_definition: Schema definition (entities, relationships, properties)
            
        Returns:
            bool: True if schema created successfully
        """
        pass
    
    @abstractmethod
    def clear_all_data(self) -> bool:
        """
        Clear all entities and relationships from the graph.
        WARNING: This deletes all data!
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dict with keys:
            - num_entities: Total number of entities
            - num_relationships: Total number of relationships
            - entity_types: Dict of entity type counts
            - relationship_types: Dict of relationship type counts
        """
        pass
    
    # ==================== Entity Operations ====================
    
    @abstractmethod
    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """
        Add an entity (node) to the graph.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Human-readable name
            entity_type: Type/classification of entity (PERSON, ORGANIZATION, etc.)
            properties: Optional additional properties
            
        Returns:
            Entity: The created entity
            
        Raises:
            ValueError: If entity_id already exists
        """
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve an entity by ID.
        
        Args:
            entity_id: The entity ID to retrieve
            
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> Optional[Entity]:
        """
        Update an entity's properties.
        
        Args:
            entity_id: The entity to update
            properties: Properties to update/add
            
        Returns:
            Updated Entity if successful, None if not found
        """
        pass
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the graph.
        
        Args:
            entity_id: The entity to delete
            
        Returns:
            bool: True if entity was deleted
        """
        pass
    
    @abstractmethod
    def list_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Entity]:
        """
        List entities, optionally filtered by type.
        
        Args:
            entity_type: Filter by entity type (optional)
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of Entity objects
        """
        pass
    
    @abstractmethod
    def entity_exists(self, entity_id: str) -> bool:
        """
        Check if an entity exists.
        
        Args:
            entity_id: The entity to check
            
        Returns:
            bool: True if entity exists
        """
        pass
    
    # ==================== Relationship Operations ====================
    
    @abstractmethod
    def add_relationship(
        self,
        relationship_id: str,
        source_id: str,
        relation_type: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Relationship:
        """
        Add a relationship (edge) to the graph.
        
        Args:
            relationship_id: Unique identifier for the relationship
            source_id: ID of the source entity
            relation_type: Type of relationship (WORKS_FOR, KNOWS, etc.)
            target_id: ID of the target entity
            properties: Optional additional properties
            
        Returns:
            Relationship: The created relationship
            
        Raises:
            ValueError: If entities don't exist or relationship_id exists
        """
        pass
    
    @abstractmethod
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Retrieve a relationship by ID.
        
        Args:
            relationship_id: The relationship ID
            
        Returns:
            Relationship if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship from the graph.
        
        Args:
            relationship_id: The relationship to delete
            
        Returns:
            bool: True if relationship was deleted
        """
        pass
    
    @abstractmethod
    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Relationship]:
        """
        Get relationships, optionally filtered.
        
        Args:
            source_id: Filter by source entity (optional)
            target_id: Filter by target entity (optional)
            relation_type: Filter by relationship type (optional)
            limit: Maximum number of results
            
        Returns:
            List of Relationship objects
        """
        pass
    
    @abstractmethod
    def relationship_exists(
        self,
        source_id: str,
        relation_type: str,
        target_id: str
    ) -> bool:
        """
        Check if a relationship exists.
        
        Args:
            source_id: Source entity ID
            relation_type: Relationship type
            target_id: Target entity ID
            
        Returns:
            bool: True if relationship exists
        """
        pass
    
    # ==================== Traversal & Path Finding ====================
    
    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: EdgeDirection = EdgeDirection.OUTGOING
    ) -> List[Tuple[Entity, Relationship]]:
        """
        Get neighboring entities (one hop).
        
        Args:
            entity_id: Starting entity
            relation_type: Filter by relationship type (optional)
            direction: Direction of traversal (out, in, both)
            
        Returns:
            List of (Entity, Relationship) tuples
        """
        pass
    
    @abstractmethod
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relation_type: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities.
        
        Args:
            source_id: Starting entity
            target_id: Target entity
            max_depth: Maximum path length
            relation_type: Filter paths by relationship type (optional)
            
        Returns:
            List of paths, where each path is a list of {'entity': Entity, 'relationship': Relationship}
        """
        pass
    
    @abstractmethod
    def traverse(
        self,
        start_id: str,
        depth: int = 2,
        relation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Traverse the graph from a starting entity.
        
        Args:
            start_id: Starting entity ID
            depth: Traversal depth (number of hops)
            relation_type: Filter by relationship type (optional)
            
        Returns:
            Dict with:
            - entities: All entities found
            - relationships: All relationships in the traversal
            - depth_map: Dict mapping entity ID to discovery depth
        """
        pass
    
    # ==================== Querying ====================
    
    @abstractmethod
    def query(self, query_string: str, **kwargs) -> QueryResult:
        """
        Execute a query against the graph.
        
        The query format depends on the backend:
        - TigerGraphStore: GSQL queries
        - LocalGraphStore: Simple filter-based queries
        
        Args:
            query_string: Query in backend-specific format
            **kwargs: Additional query parameters
            
        Returns:
            QueryResult with entities, relationships, and metadata
            
        Raises:
            ValueError: If query is invalid
        """
        pass
    
    @abstractmethod
    def search_entities(
        self,
        name_pattern: Optional[str] = None,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Entity]:
        """
        Search for entities by various criteria.
        
        Args:
            name_pattern: Substring to search in entity names (case-insensitive)
            entity_type: Filter by entity type
            properties: Filter by properties (AND conditions)
            limit: Maximum results
            
        Returns:
            List of matching Entity objects
        """
        pass
    
    # ==================== Batch Operations ====================
    
    @abstractmethod
    def batch_add_entities(self, entities: List[Dict[str, Any]]) -> List[Entity]:
        """
        Add multiple entities efficiently.
        
        Args:
            entities: List of dicts with keys: id, name, entity_type, properties
            
        Returns:
            List of created Entity objects
        """
        pass
    
    @abstractmethod
    def batch_add_relationships(self, relationships: List[Dict[str, Any]]) -> List[Relationship]:
        """
        Add multiple relationships efficiently.
        
        Args:
            relationships: List of dicts with keys: id, source_id, relation_type, target_id, properties
            
        Returns:
            List of created Relationship objects
        """
        pass
    
    # ==================== Transaction Support (Optional) ====================
    
    @abstractmethod
    def begin_transaction(self) -> str:
        """
        Begin a transaction (if supported by backend).
        
        Returns:
            str: Transaction ID
        """
        pass
    
    @abstractmethod
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction to commit
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction to rollback
            
        Returns:
            bool: True if successful
        """
        pass
