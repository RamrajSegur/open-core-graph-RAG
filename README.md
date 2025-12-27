# Open Core Graph RAG

> A reference architecture and runtime for **reasoning-first RAG systems** with graph-based knowledge representation.

## 🎯 Project Overview

Open Core Graph RAG is an open-source system that combines knowledge graphs with large language models to build reasoning-capable retrieval-augmented generation (RAG) systems. Unlike traditional RAG systems that simply retrieve relevant documents, this system performs multi-hop reasoning over a knowledge graph to answer complex queries.

**Key Features:**
- 🧠 **Reasoning-First**: Multi-hop inference over knowledge graphs
- 📊 **Graph-Based**: TigerGraph for scalable, distributed graph storage
- 🔓 **Open Source**: Uses open-source LLMs (Mistral, Llama2 via Ollama)
- 🐳 **Containerized**: Fully Docker-based for easy deployment
- 📈 **Extensible**: Modular architecture for easy customization
- 📝 **Well-Documented**: Comprehensive guides and examples

## 🏗️ Architecture

The system consists of several integrated layers:

```
┌─────────────────────────────────────────┐
│        User Interface (API/CLI)         │
└────────────────────┬────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────────┐ ┌────▼──────┐ ┌──────▼──┐
│   Query    │ │ Reasoning  │ │Retrieval│
│ Processor  │ │   Engine   │ │  Layer  │
└───┬────────┘ └────┬──────┘ └──────┬──┘
    │                │                │
    └────────────────┼────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼──────────┐ ┌──▼─────────┐ ┌───▼────┐
│   Knowledge  │ │   Vector   │ │  LLM   │
│    Graph     │ │  Database  │ │ Engine │
│  (TigerGraph)│ │(PostgreSQL)│ │(Ollama)│
└──────────────┘ └────────────┘ └────────┘
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ available RAM
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/RamrajSegur/open-core-graph-RAG.git
cd open-core-graph-RAG
```

### 2. Configure (Optional)
```bash
# Customize Docker settings if needed
nano docker/.env
```

### 3. Start Services
```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

### 4. Verify
```bash
# Check services are running
docker-compose -f docker/docker-compose.yml ps

# Test TigerGraph
curl -u tigergraph:tigergraph http://localhost:9000/echo
```

See [docker/README.md](./docker/README.md) for detailed Docker instructions.

## 📚 Component Documentation

### Core Components

| Component | Status | Description |
|-----------|--------|-------------|
| [Knowledge Graph](./src/core/README.md) | 🚧 In Progress | TigerGraph integration, graph operations, querying |
| [Extraction Pipeline](./src/extraction/README.md) | 🚧 In Progress | Entity/relation extraction, document processing |
| [Retrieval](./src/retrieval/) | ⏳ Planned | Hybrid graph + semantic search |
| [Reasoning Engine](./src/core/) | ⏳ Planned | Multi-hop inference, constraint satisfaction |
| [LLM Integration](./src/llm/) | ⏳ Planned | Ollama, prompt management, response generation |
| [REST API](./src/api/) | ⏳ Planned | FastAPI endpoints, request handling |

### Infrastructure

| Component | Documentation |
|-----------|---------------|
| **Docker Setup** | [docker/README.md](./docker/README.md) |
| **Configuration** | [src/config.py](./src/config.py) |
| **Database Schema** | [docker/init/init_db.sql](./docker/init/init_db.sql) |

## 📁 Project Structure

```
open_core_graph_RAG/
│
├── README.md                    # This file
├── ARCHITECTURE.md              # Detailed system design
├── LICENSE                      # MIT License
│
├── docker/                      # Docker configuration
│   ├── Dockerfile              # Application image
│   ├── docker-compose.yml      # Service orchestration
│   ├── .env                    # Configuration (not committed)
│   ├── .dockerignore          # Build exclusions
│   ├── init/                  # Database initialization
│   │   └── init_db.sql
│   └── README.md              # Docker guide
│
├── src/                        # Source code
│   ├── config.py              # Configuration management
│   │
│   ├── core/                  # Core graph components
│   │   ├── README.md
│   │   ├── knowledge_graph.py
│   │   ├── reasoning_engine.py
│   │   └── query_processor.py
│   │
│   ├── extraction/            # Data extraction pipeline
│   │   ├── README.md
│   │   ├── entity_extractor.py
│   │   ├── relation_extractor.py
│   │   ├── document_parser.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/             # Retrieval components
│   │   ├── graph_retriever.py
│   │   ├── semantic_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── llm/                   # LLM integration
│   │   ├── base_llm.py
│   │   ├── ollama_client.py
│   │   └── prompt_templates.py
│   │
│   ├── api/                   # REST API (future)
│   │   └── app.py
│   │
│   └── utils/                 # Utilities
│       ├── logger.py
│       ├── metrics.py
│       └── helpers.py
│
├── tests/                     # Test suite
│   ├── test_core.py
│   ├── test_extraction.py
│   └── test_integration.py
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploration.ipynb
│   └── ingestion_example.ipynb
│
├── data/                      # Data directories
│   ├── raw/                   # Input documents
│   └── processed/             # Processed data
│
├── scripts/                   # Utility scripts
│   ├── build_graph.py        # Build knowledge graph
│   └── evaluate.py           # Evaluation tools
│
└── requirements.txt           # Python dependencies
```

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Graph DB** | TigerGraph Community | Distributed graph storage |
| **Metadata DB** | PostgreSQL | Ingestion tracking, audit trails |
| **NLP** | spaCy + Transformers | Entity & relation extraction |
| **LLM** | Ollama (Mistral/Llama2) | Local, open-source inference |
| **Framework** | LangChain | LLM abstraction & prompts |
| **API** | FastAPI | REST endpoints (future) |
| **Testing** | pytest | Unit & integration tests |
| **Language** | Python 3.10 | Primary development language |

## 🔧 Development Workflow

### 1. Install Dependencies
```bash
docker-compose -f docker/docker-compose.yml exec app pip install -e .
```

### 2. Run Tests
```bash
docker-compose -f docker/docker-compose.yml exec app pytest tests/
```

### 3. Run Notebooks
```bash
docker-compose -f docker/docker-compose.yml exec app jupyter notebook --ip=0.0.0.0
```

### 4. Interactive Shell
```bash
docker-compose -f docker/docker-compose.yml exec app ipython
```

See component READMEs for specific development guides.

## 📖 Getting Started

1. **First time?** Start with [Quick Start](#-quick-start) above
2. **Setting up Docker?** Read [docker/README.md](./docker/README.md)
3. **Building components?** See individual README files:
   - [Knowledge Graph](./src/core/README.md)
   - [Extraction Pipeline](./src/extraction/README.md)
4. **Understanding architecture?** Read [ARCHITECTURE.md](./ARCHITECTURE.md)

## 🤝 Contributing

This is an open-core project. We welcome contributions! 

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write tests
5. Submit a pull request

See [ARCHITECTURE.md](./ARCHITECTURE.md) for development guidelines.

## 📝 Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Detailed system design and component interactions
- **[docker/README.md](./docker/README.md)** - Docker setup and configuration
- **[src/core/README.md](./src/core/README.md)** - Knowledge graph documentation
- **[src/extraction/README.md](./src/extraction/README.md)** - Extraction pipeline documentation

## 🐛 Troubleshooting

### Services Not Starting?
```bash
docker-compose -f docker/docker-compose.yml logs -f tigergraph
```

### Connection Errors?
```bash
docker-compose -f docker/docker-compose.yml ps
docker-compose -f docker/docker-compose.yml exec app ping tigergraph
```

See [docker/README.md](./docker/README.md#troubleshooting) for more troubleshooting tips.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

## 👨‍💻 Author

**Ramraj Segur Mahadevaraja**
- GitHub: [@RamrajSegur](https://github.com/RamrajSegur)
- Project: [open-core-graph-RAG](https://github.com/RamrajSegur/open-core-graph-RAG)

## 🙏 Acknowledgments

This system builds on excellent open-source technologies:
- [TigerGraph](https://www.tigergraph.com/) - Graph database
- [spaCy](https://spacy.io/) - NLP library
- [LangChain](https://www.langchain.com/) - LLM framework
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Hugging Face](https://huggingface.co/) - Transformers & models

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/RamrajSegur/open-core-graph-RAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RamrajSegur/open-core-graph-RAG/discussions)
- **Documentation**: See README files in each component directory

---

**Happy building! 🚀**
