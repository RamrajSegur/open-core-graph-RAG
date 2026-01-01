"""Storage connector for TigerGraph."""

import logging
from typing import Dict, List, Optional, Set

from src.core.graph_store import GraphStore
from src.core.graph_factory import get_graph_store
from src.extraction import ExtractedEntity, ExtractedRelationship, EntityType, RelationshipType

logger = logging.getLogger(__name__)


class StorageConnector:
    """Connects extracted entities and relationships to graph storage."""

    def __init__(self, graph_store: Optional[GraphStore] = None, graph_name: str = "RAG"):
        """Initialize storage connector.

        Args:
            graph_store: GraphStore instance (default: TigerGraph via factory)
            graph_name: Name of the graph in TigerGraph
        """
        if graph_store is None:
            self.graph_store = get_graph_store(graph_name=graph_name)
        else:
            self.graph_store = graph_store

        self.graph_name = graph_name
        self.stored_entities: Set[str] = set()
        self.stored_relationships: Set[str] = set()
        self.stats = {"entities": 0, "relationships": 0, "errors": []}

    def store_entity(self, entity: ExtractedEntity, skip_duplicates: bool = True) -> bool:
        """Store extracted entity as vertex in graph.

        Args:
            entity: Extracted entity to store
            skip_duplicates: Skip if entity already stored

        Returns:
            True if stored successfully
        """
        try:
            # Create unique entity key
            entity_key = f"{entity.entity_type.value}:{entity.normalized_text}"

            if skip_duplicates and entity_key in self.stored_entities:
                return False

            # Create vertex in graph
            vertex_data = {
                "id": entity.entity_id,
                "text": entity.text,
                "type": entity.entity_type.value,
                "confidence": entity.confidence,
                "source_file": entity.source_file,
                "chunk_id": entity.chunk_id,
            }

            self.graph_store.create_vertex(
                vertex_type=entity.entity_type.value,
                vertex_id=entity.entity_id,
                attributes=vertex_data,
            )

            self.stored_entities.add(entity_key)
            self.stats["entities"] += 1
            return True

        except Exception as e:
            error_msg = f"Error storing entity {entity.text}: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def store_entities(
        self,
        entities: List[ExtractedEntity],
        skip_duplicates: bool = True,
    ) -> int:
        """Store multiple entities.

        Args:
            entities: List of entities to store
            skip_duplicates: Skip duplicates

        Returns:
            Count of successfully stored entities
        """
        count = 0
        for entity in entities:
            if self.store_entity(entity, skip_duplicates=skip_duplicates):
                count += 1
        return count

    def store_relationship(
        self,
        relationship: ExtractedRelationship,
        skip_duplicates: bool = True,
    ) -> bool:
        """Store extracted relationship as edge in graph.

        Args:
            relationship: Extracted relationship to store
            skip_duplicates: Skip if relationship already stored

        Returns:
            True if stored successfully
        """
        try:
            # Create unique relationship key
            rel_key = f"{relationship.source_entity}:{relationship.relationship_type.value}:{relationship.target_entity}"

            if skip_duplicates and rel_key in self.stored_relationships:
                return False

            # Map relationship type to edge type in graph
            edge_type = relationship.relationship_type.value

            # Create edge in graph
            edge_data = {
                "relationship_type": relationship.relationship_type.value,
                "confidence": relationship.confidence,
                "supporting_text": relationship.supporting_text,
                "source_chunk_id": relationship.source_chunk_id,
            }

            self.graph_store.create_edge(
                edge_type=edge_type,
                from_vertex_id=relationship.source_entity,
                to_vertex_id=relationship.target_entity,
                attributes=edge_data,
            )

            self.stored_relationships.add(rel_key)
            self.stats["relationships"] += 1
            return True

        except Exception as e:
            error_msg = f"Error storing relationship {rel_key}: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False

    def store_relationships(
        self,
        relationships: List[ExtractedRelationship],
        skip_duplicates: bool = True,
    ) -> int:
        """Store multiple relationships.

        Args:
            relationships: List of relationships to store
            skip_duplicates: Skip duplicates

        Returns:
            Count of successfully stored relationships
        """
        count = 0
        for relationship in relationships:
            if self.store_relationship(relationship, skip_duplicates=skip_duplicates):
                count += 1
        return count

    def store_knowledge_graph(
        self,
        entities: List[ExtractedEntity],
        relationships: List[ExtractedRelationship],
        skip_duplicates: bool = True,
    ) -> Dict[str, int]:
        """Store complete knowledge graph (entities + relationships).

        Args:
            entities: Entities to store
            relationships: Relationships to store
            skip_duplicates: Skip duplicates

        Returns:
            Dictionary with counts: {"entities": int, "relationships": int}
        """
        entity_count = self.store_entities(entities, skip_duplicates=skip_duplicates)
        relationship_count = self.store_relationships(
            relationships, skip_duplicates=skip_duplicates
        )

        return {
            "entities": entity_count,
            "relationships": relationship_count,
        }

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Retrieve entity from graph.

        Args:
            entity_id: Entity vertex ID

        Returns:
            Entity attributes or None if not found
        """
        try:
            return self.graph_store.get_vertex(entity_id)
        except Exception as e:
            logger.error(f"Error retrieving entity {entity_id}: {str(e)}")
            return None

    def get_relationships_for_entity(
        self, entity_id: str
    ) -> List[Dict]:
        """Get all relationships for an entity.

        Args:
            entity_id: Entity vertex ID

        Returns:
            List of relationships involving entity
        """
        try:
            # Query edges where entity is source or target
            edges = []
            outgoing = self.graph_store.get_edges_from_vertex(entity_id)
            incoming = self.graph_store.get_edges_to_vertex(entity_id)

            edges.extend(outgoing)
            edges.extend(incoming)
            return edges

        except Exception as e:
            logger.error(f"Error retrieving relationships for {entity_id}: {str(e)}")
            return []

    def query_entities_by_type(self, entity_type: EntityType) -> List[Dict]:
        """Query entities by type.

        Args:
            entity_type: EntityType to filter

        Returns:
            List of entities of that type
        """
        try:
            # Query vertices of specific type
            return self.graph_store.get_vertices(entity_type.value)
        except Exception as e:
            logger.error(f"Error querying entities by type {entity_type}: {str(e)}")
            return []

    def query_relationships_by_type(
        self, relationship_type: RelationshipType
    ) -> List[Dict]:
        """Query relationships by type.

        Args:
            relationship_type: RelationshipType to filter

        Returns:
            List of relationships of that type
        """
        try:
            # Query edges of specific type
            return self.graph_store.get_edges(relationship_type.value)
        except Exception as e:
            logger.error(
                f"Error querying relationships by type {relationship_type}: {str(e)}"
            )
            return []

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset storage statistics."""
        self.stats = {"entities": 0, "relationships": 0, "errors": []}
        self.stored_entities.clear()
        self.stored_relationships.clear()

    def close(self) -> None:
        """Cleanup resources."""
        if hasattr(self.graph_store, "close"):
            self.graph_store.close()
