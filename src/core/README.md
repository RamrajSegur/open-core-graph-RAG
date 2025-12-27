# Knowledge Graph Component

Core graph storage and operations for the Open Core Graph RAG system using TigerGraph.

## 📋 Overview

The Knowledge Graph component handles:
- Graph database connection and management
- Entity and relationship storage
- Graph queries and traversals
- Multi-hop reasoning support
- Graph schema management

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│    TigerGraph Database (Graph Storage)   │
│   - Entities (nodes)                     │
│   - Relationships (edges)                │
│   - Graph queries                        │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│     Knowledge Graph Interface             │
│  (KnowledgeGraph class)                   │
│   - add_entity()                         │
│   - add_relationship()                   │
│   - query()                              │
│   - traverse()                           │
│   - get_neighbors()                      │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│    Reasoning Engine & Retrieval Layer    │
│  (Uses graph operations)                  │
└──────────────────────────────────────────┘
```

## 📁 Files

- **`knowledge_graph.py`** - Main KnowledgeGraph class (to be created)
- **`reasoning_engine.py`** - Inference and reasoning logic (to be created)
- **`query_processor.py`** - Query parsing and processing (to be created)

## 🚀 Getting Started

### Prerequisites

- Docker containers running (TigerGraph service)
- Python environment configured
- TigerGraph admin credentials

### Basic Usage

```python
from src.core.knowledge_graph import KnowledgeGraph

# Connect to TigerGraph
kg = KnowledgeGraph(
    host="localhost",
    port=9000,
    username="tigergraph",
    password="tigergraph"
)

# Add entities
kg.add_entity("John", entity_type="PERSON")
kg.add_entity("Apple", entity_type="ORGANIZATION")

# Add relationships
kg.add_relationship(
    source="John",
    relation_type="WORKS_FOR",
    target="Apple"
)

# Query the graph
results = kg.query("MATCH (p:PERSON)-[r:WORKS_FOR]->(o:ORGANIZATION) RETURN p, r, o")
```

## 🔧 Configuration

Configuration is managed in `src/config.py`:

```python
from src.config import config

# Access TigerGraph config
tg_config = config.tigergraph
print(tg_config.url)  # http://localhost:9000
```

### Environment Variables

```bash
TIGERGRAPH_HOST=localhost
TIGERGRAPH_PORT=9000
TIGERGRAPH_ADMIN=tigergraph
TIGERGRAPH_PASSWORD=tigergraph
TIGERGRAPH_GRAPH_NAME=graph_rag
```

See `docker/.env` for all configuration options.

## 📚 Data Model

### Entity Types

Common entity types in the system:

| Type | Example | Properties |
|------|---------|-----------|
| PERSON | John Doe | name, title, bio |
| ORGANIZATION | Apple Inc | name, industry, founded |
| LOCATION | New York | name, country, coordinates |
| CONCEPT | Machine Learning | name, description |
| EVENT | WWDC 2024 | name, date, location |

### Relationship Types

Common relationships:

| Type | Description | Entities |
|------|-------------|----------|
| WORKS_FOR | Person works at organization | PERSON → ORGANIZATION |
| LOCATED_IN | Place is located in another | LOCATION → LOCATION |
| FOUNDED_BY | Company founded by person | ORGANIZATION → PERSON |
| KNOWS | Person knows another | PERSON → PERSON |
| RELATED_TO | Generic relationship | Any → Any |

## 🔍 Key Methods

### Entity Operations

```python
# Add entity
kg.add_entity(name, entity_type, properties={})

# Get entity
entity = kg.get_entity(name)

# Update entity
kg.update_entity(name, properties={})

# Delete entity
kg.delete_entity(name)

# List entities by type
people = kg.list_entities(entity_type="PERSON")
```

### Relationship Operations

```python
# Add relationship
kg.add_relationship(
    source, 
    relation_type, 
    target, 
    properties={}
)

# Get relationships
rels = kg.get_relationships(source, relation_type)

# Delete relationship
kg.delete_relationship(source, relation_type, target)
```

### Graph Queries

```python
# Execute GSQL query
results = kg.query(gsql_query)

# Find neighbors (one hop)
neighbors = kg.get_neighbors(entity, direction="out")

# Find paths (multi-hop)
paths = kg.find_paths(source, target, max_depth=3)

# Traverse from entity
traversal = kg.traverse(start_entity, depth=2)
```

## 📊 Example: Building a Simple Graph

```python
from src.core.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()

# Create entities
entities = [
    ("Alice", "PERSON"),
    ("Bob", "PERSON"),
    ("TechCorp", "ORGANIZATION"),
    ("San Francisco", "LOCATION")
]

for name, entity_type in entities:
    kg.add_entity(name, entity_type)

# Create relationships
relationships = [
    ("Alice", "WORKS_FOR", "TechCorp"),
    ("Bob", "WORKS_FOR", "TechCorp"),
    ("Alice", "KNOWS", "Bob"),
    ("TechCorp", "LOCATED_IN", "San Francisco")
]

for source, rel_type, target in relationships:
    kg.add_relationship(source, rel_type, target)

# Query: Find all people who work at TechCorp
results = kg.query("""
    MATCH (p:PERSON)-[r:WORKS_FOR]->(o:ORGANIZATION {name: "TechCorp"})
    RETURN p.name as person, o.name as company
""")

print(results)
# Output: [{"person": "Alice", "company": "TechCorp"}, ...]
```

## 🧪 Testing

```bash
# Run knowledge graph tests
docker-compose -f docker/docker-compose.yml exec app pytest tests/test_core.py -v
```

See `tests/test_core.py` for test examples.

## 📖 Additional Resources

- [TigerGraph Documentation](https://docs.tigergraph.com/)
- [GSQL Query Language](https://docs.tigergraph.com/gsql-ref/current/intro)
- [Graph Database Concepts](https://www.tigergraph.com/what-is-a-graph-database/)
- [Main Architecture](../ARCHITECTURE.md)

## 🔗 Related Components

- **Extraction Pipeline** - Populates the knowledge graph with entities and relationships
- **Reasoning Engine** - Performs inference over the graph
- **Retrieval Layer** - Queries the graph for relevant information

## 💡 Design Notes

1. **TigerGraph as Backend**: All graph operations go through TigerGraph's REST API
2. **Type Safety**: Use Pydantic models for entity and relationship data
3. **Connection Pooling**: Reuse connections to TigerGraph for efficiency
4. **Error Handling**: Graceful handling of TigerGraph connection errors
5. **Logging**: Detailed logging of graph operations for debugging

## 🚧 To-Do

- [ ] Implement KnowledgeGraph class
- [ ] Create TigerGraph schema initialization
- [ ] Add entity operations
- [ ] Add relationship operations
- [ ] Implement query methods
- [ ] Add traversal/path-finding
- [ ] Create unit tests
- [ ] Add integration tests
- [ ] Create example notebooks

## 📝 Notes

- This component is **in progress**
- See [extraction/README.md](../extraction/README.md) for how entities and relationships are populated
- See [ARCHITECTURE.md](../../ARCHITECTURE.md) for system-wide context
