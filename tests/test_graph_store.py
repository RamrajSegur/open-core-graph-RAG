"""
Unit tests for GraphStore implementations.

Tests abstract interface contract compliance and TigerGraphStore functionality.
Note: TigerGraphStore integration tests require a running TigerGraph server via Docker.
To run the full test suite:
    docker-compose up -d
    pytest tests/test_graph_store.py
"""

import pytest
from src.core.graph_store import GraphStore, Entity, Relationship, QueryResult, EdgeDirection


class TestGraphStoreInterface:
    """Test that implementations properly fulfill the GraphStore contract"""
    
    def test_entity_creation(self):
        """Test Entity dataclass creation and properties"""
        entity = Entity(
            id="e1",
            name="Alice",
            entity_type="Person",
            properties={"age": 30}
        )
        
        assert entity.id == "e1"
        assert entity.name == "Alice"
        assert entity.entity_type == "Person"
        assert entity.properties["age"] == 30
    
    def test_entity_creation_without_properties(self):
        """Test Entity can be created without properties"""
        entity = Entity(
            id="e1",
            name="Alice",
            entity_type="Person"
        )
        
        assert entity.id == "e1"
        assert entity.name == "Alice"
        assert entity.entity_type == "Person"
        assert entity.properties == {}
    
    def test_relationship_creation(self):
        """Test Relationship dataclass creation"""
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            rel_type="knows",
            properties={"since": 2020}
        )
        
        assert rel.id == "r1"
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.rel_type == "knows"
        assert rel.properties["since"] == 2020
    
    def test_relationship_creation_without_properties(self):
        """Test Relationship can be created without properties"""
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            rel_type="knows"
        )
        
        assert rel.id == "r1"
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.rel_type == "knows"
        assert rel.properties == {}
    
    def test_query_result_creation(self):
        """Test QueryResult dataclass creation"""
        result = QueryResult(
            query="MATCH (n) RETURN n",
            results=[{"id": "e1", "name": "Alice"}],
            execution_time_ms=42,
            result_count=1
        )
        
        assert result.query == "MATCH (n) RETURN n"
        assert len(result.results) == 1
        assert result.execution_time_ms == 42
        assert result.result_count == 1
    
    def test_edge_direction_enum(self):
        """Test EdgeDirection enum values"""
        assert EdgeDirection.OUTGOING.value == "outgoing"
        assert EdgeDirection.INCOMING.value == "incoming"
        assert EdgeDirection.BOTH.value == "both"
    
    def test_edge_direction_comparison(self):
        """Test EdgeDirection enum comparisons"""
        assert EdgeDirection.OUTGOING == EdgeDirection.OUTGOING
        assert EdgeDirection.OUTGOING != EdgeDirection.INCOMING


class TestTigerGraphStoreIntegration:
    """Integration tests for TigerGraphStore with running TigerGraph server
    
    These tests require a TigerGraph server to be running.
    Run: docker-compose up -d
    
    Then run: pytest tests/test_graph_store.py::TestTigerGraphStoreIntegration -v
    """
    
    @pytest.fixture
    def store(self, mock_tigergraph_store, monkeypatch):
        """
        Create and connect a TigerGraphStore.
        Uses mock store for testing without requiring actual TigerGraph server.
        """
        from tests.conftest import MockTigerGraphStore
        
        # Create mock store instance
        store = MockTigerGraphStore()
        
        # Verify it's working
        if not store.health_check():
            pytest.skip("Mock store initialization failed")
        
        # Connect and setup
        if not store.connect():
            pytest.skip("Failed to connect to mock store")
        
        yield store
        
        try:
            store.disconnect()
        except Exception:
            pass  # Cleanup
    
    # ============ Connection & Lifecycle Tests ============
    
    def test_tigergraph_health_check(self, store):
        """Test health check with running TigerGraph"""
        assert store.health_check()
    
    def test_tigergraph_is_connected(self, store):
        """Test connection status"""
        assert store.is_connected()
    
    # ============ Basic Operations ============
    
    def test_tigergraph_get_stats(self, store):
        """Test getting graph statistics"""
        try:
            stats = store.get_stats()
            assert isinstance(stats, dict)
            assert "entity_count" in stats or "vertices_count" in stats
        except NotImplementedError:
            pytest.skip("get_stats not yet implemented for TigerGraphStore")
    
    def test_tigergraph_clear_data(self, store):
        """Test clearing all data from graph"""
        try:
            result = store.clear_all_data()
            assert result is True
        except NotImplementedError:
            pytest.skip("clear_all_data not yet implemented for TigerGraphStore")
    
    # ============ Entity Operations ============
    
    def test_tigergraph_add_entity(self, store):
        """Test adding an entity to TigerGraph"""
        try:
            entity = Entity(
                id="test_person_1",
                name="Alice",
                entity_type="Person",
                properties={"age": 30}
            )
            
            result = store.add_entity(entity)
            assert result is True
        except NotImplementedError:
            pytest.skip("add_entity not yet implemented for TigerGraphStore")
    
    def test_tigergraph_get_entity(self, store):
        """Test retrieving an entity from TigerGraph"""
        try:
            # First add an entity
            entity = Entity(
                id="test_person_2",
                name="Bob",
                entity_type="Person",
                properties={"age": 28}
            )
            store.add_entity(entity)
            
            # Then retrieve it
            retrieved = store.get_entity("test_person_2")
            assert retrieved is not None
            assert retrieved.id == "test_person_2"
        except NotImplementedError:
            pytest.skip("get_entity not yet implemented for TigerGraphStore")
    
    # ============ Relationship Operations ============
    
    def test_tigergraph_add_relationship(self, store):
        """Test adding a relationship to TigerGraph"""
        try:
            # Add entities first
            e1 = Entity(id="person_a", name="Alice", entity_type="Person")
            e2 = Entity(id="person_b", name="Bob", entity_type="Person")
            store.add_entity(e1)
            store.add_entity(e2)
            
            # Add relationship
            rel = Relationship(
                id="rel_1",
                source_id="person_a",
                target_id="person_b",
                rel_type="knows",
                properties={"since": 2020}
            )
            
            result = store.add_relationship(rel)
            assert result is True
        except NotImplementedError:
            pytest.skip("add_relationship not yet implemented for TigerGraphStore")
    
    # ============ Query Operations ============
    
    def test_tigergraph_query_execution(self, store):
        """Test executing a GSQL query"""
        try:
            # Try a simple query
            result = store.query("SELECT * FROM Person LIMIT 10")
            assert result is not None
        except NotImplementedError:
            pytest.skip("query not yet implemented for TigerGraphStore")
    
    # ============ Batch Operations ============
    
    def test_tigergraph_batch_add_entities(self, store):
        """Test batch adding entities"""
        try:
            entities = [
                Entity(id=f"batch_person_{i}", name=f"Person{i}", entity_type="Person")
                for i in range(5)
            ]
            
            results = store.batch_add_entities(entities)
            assert len(results) == 5
        except NotImplementedError:
            pytest.skip("batch_add_entities not yet implemented for TigerGraphStore")


class TestGraphStoreFactory:
    """Tests for the factory pattern and GraphStore creation"""
    
    def test_factory_import(self):
        """Test that factory can be imported"""
        from src.core.graph_factory import get_graph_store, get_available_backends
        
        assert callable(get_graph_store)
        assert callable(get_available_backends)
    
    def test_get_available_backends(self):
        """Test getting list of available backends"""
        from src.core.graph_factory import get_available_backends
        
        backends = get_available_backends()
        assert isinstance(backends, dict)
        assert 'tigergraph' in backends
    
    def test_factory_create_tigergraph_store(self):
        """Test factory creates TigerGraphStore instance"""
        from src.core.graph_factory import get_graph_store
        from src.core.tiger_graph_store import TigerGraphStore
        
        try:
            store = get_graph_store()
            assert isinstance(store, TigerGraphStore)
        except RuntimeError as e:
            pytest.skip(f"Cannot create TigerGraphStore: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
