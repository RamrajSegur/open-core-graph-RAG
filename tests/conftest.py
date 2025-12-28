"""
Pytest configuration and fixtures for graph store tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.core.graph_store import Entity, Relationship, QueryResult


class MockTigerGraphStore:
    """Mock TigerGraphStore for testing without requiring actual TigerGraph server"""
    
    def __init__(self, *args, **kwargs):
        """Initialize mock store with in-memory storage"""
        self.entities = {}  # {entity_id: Entity}
        self.relationships = {}  # {rel_id: Relationship}
        self._connected = True
        self._health = True
        
    def connect(self) -> bool:
        """Mock connection"""
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        """Mock disconnect"""
        self._connected = False
        return True
    
    def is_connected(self) -> bool:
        """Check mock connection status"""
        return self._connected
    
    def health_check(self) -> bool:
        """Mock health check"""
        return self._health
    
    def create_schema(self, schema_definition) -> bool:
        """Mock schema creation"""
        return True
    
    def clear_all_data(self) -> bool:
        """Mock clear all data"""
        self.entities.clear()
        self.relationships.clear()
        return True
    
    def get_stats(self):
        """Mock get stats"""
        return {
            "num_entities": len(self.entities),
            "entity_count": len(self.entities),
            "vertices_count": len(self.entities),
            "num_relationships": len(self.relationships),
            "entity_types": {},
            "relationship_types": {}
        }
    
    def add_entity(self, entity_id=None, name=None, entity_type=None, 
                   properties=None, entity=None) -> bool:
        """Mock add entity"""
        # Handle positional argument as Entity object
        if isinstance(entity_id, Entity):
            entity = entity_id
            entity_id = None
        
        if entity is not None:
            entity_id = entity.id
            name = entity.name
            entity_type = entity.entity_type
            properties = entity.properties
        
        if entity_id in self.entities:
            return False  # Already exists
        
        self.entities[entity_id] = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {}
        )
        return True
    
    def get_entity(self, entity_id: str):
        """Mock get entity"""
        return self.entities.get(entity_id)
    
    def update_entity(self, entity_id: str, properties):
        """Mock update entity"""
        if entity_id in self.entities:
            self.entities[entity_id].properties.update(properties)
            return self.entities[entity_id]
        return None
    
    def delete_entity(self, entity_id: str) -> bool:
        """Mock delete entity"""
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False
    
    def list_entities(self, entity_type=None, limit=1000, offset=0):
        """Mock list entities"""
        entities = list(self.entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return entities[offset:offset+limit]
    
    def entity_exists(self, entity_id: str) -> bool:
        """Check if entity exists"""
        return entity_id in self.entities
    
    def add_relationship(self, relationship_id=None, source_id=None, relation_type=None,
                        target_id=None, properties=None, relationship=None) -> bool:
        """Mock add relationship"""
        # Handle positional argument as Relationship object
        if isinstance(relationship_id, Relationship):
            relationship = relationship_id
            relationship_id = None
        
        if relationship is not None:
            relationship_id = relationship.id
            source_id = relationship.source_id
            target_id = relationship.target_id
            relation_type = relationship.rel_type
            properties = relationship.properties
        
        if relationship_id in self.relationships:
            return False  # Already exists
        
        self.relationships[relationship_id] = Relationship(
            id=relationship_id,
            source_id=source_id,
            target_id=target_id,
            rel_type=relation_type,
            properties=properties or {}
        )
        return True
    
    def get_relationship(self, relationship_id: str):
        """Mock get relationship"""
        return self.relationships.get(relationship_id)
    
    def delete_relationship(self, relationship_id: str) -> bool:
        """Mock delete relationship"""
        if relationship_id in self.relationships:
            del self.relationships[relationship_id]
            return True
        return False
    
    def get_relationships(self, source_id=None, target_id=None, 
                         relation_type=None, limit=1000):
        """Mock get relationships"""
        rels = list(self.relationships.values())
        
        if source_id:
            rels = [r for r in rels if r.source_id == source_id]
        if target_id:
            rels = [r for r in rels if r.target_id == target_id]
        if relation_type:
            rels = [r for r in rels if r.rel_type == relation_type]
        
        return rels[:limit]
    
    def relationship_exists(self, source_id: str, relation_type: str, 
                           target_id: str) -> bool:
        """Check if relationship exists"""
        for rel in self.relationships.values():
            if (rel.source_id == source_id and 
                rel.target_id == target_id and 
                rel.rel_type == relation_type):
                return True
        return False
    
    def get_neighbors(self, entity_id: str, relation_type=None, direction=None):
        """Mock get neighbors"""
        neighbors = []
        
        for rel in self.relationships.values():
            if rel.source_id == entity_id:
                neighbor = self.get_entity(rel.target_id)
                if neighbor:
                    neighbors.append((neighbor, rel))
            elif rel.target_id == entity_id:
                neighbor = self.get_entity(rel.source_id)
                if neighbor:
                    neighbors.append((neighbor, rel))
        
        return neighbors
    
    def find_paths(self, source_id: str, target_id: str, max_depth=5, 
                   relation_type=None):
        """Mock find paths"""
        return []
    
    def traverse(self, start_id: str, depth=2, relation_type=None):
        """Mock traverse"""
        return {
            "entities": [],
            "relationships": [],
            "depth_map": {}
        }
    
    def query(self, query_string: str, **kwargs) -> QueryResult:
        """Mock query execution"""
        return QueryResult(
            query=query_string,
            results=[],
            execution_time_ms=0,
            result_count=0
        )
    
    def search_entities(self, name_pattern=None, entity_type=None, 
                       properties=None, limit=100):
        """Mock search entities"""
        return list(self.entities.values())[:limit]
    
    def batch_add_entities(self, entities):
        """Mock batch add entities"""
        results = []
        for entity in entities:
            success = self.add_entity(entity=entity)
            results.append(success)
        return results
    
    def batch_add_relationships(self, relationships):
        """Mock batch add relationships"""
        results = []
        for rel in relationships:
            success = self.add_relationship(relationship=rel)
            results.append(success)
        return results
    
    def begin_transaction(self) -> str:
        """Mock begin transaction"""
        import uuid
        return str(uuid.uuid4())
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """Mock commit transaction"""
        return True
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """Mock rollback transaction"""
        return True


@pytest.fixture
def mock_store():
    """Provide a mock TigerGraphStore for testing"""
    return MockTigerGraphStore()


@pytest.fixture
def mock_tigergraph_store(monkeypatch):
    """
    Fixture that patches TigerGraphStore with mock implementation
    for integration tests
    """
    # Import here to avoid circular imports
    from src.core import tiger_graph_store
    
    # Store original class
    original_class = tiger_graph_store.TigerGraphStore
    
    # Replace with mock
    monkeypatch.setattr(tiger_graph_store, 'TigerGraphStore', MockTigerGraphStore)
    
    # Also patch in the test module scope
    yield MockTigerGraphStore
    
    # Restore original (cleanup)
    monkeypatch.setattr(tiger_graph_store, 'TigerGraphStore', original_class)
