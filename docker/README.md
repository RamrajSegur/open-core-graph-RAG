# Docker Environment

This directory contains all Docker-related configuration and scripts for the Open Core Graph RAG system.

## Directory Structure

```
docker/
├── Dockerfile              # Python application image definition
├── docker-compose.yml      # Service orchestration (TigerGraph, PostgreSQL, Python app)
├── .env                    # Environment variables (customize here)
├── .dockerignore          # Files to exclude from Docker build
├── init/                  # Database initialization scripts
│   └── init_db.sql       # PostgreSQL schema and tables
└── README.md             # This file
```

## Quick Start

### 1. Configure Environment (Optional)

Edit `.env` to customize settings:
```bash
nano docker/.env
```

Common settings:
- `TIGERGRAPH_PASSWORD` - Change TigerGraph admin password
- `POSTGRES_PASSWORD` - Change PostgreSQL password
- `TIGERGRAPH_PORT` - Change TigerGraph port (default 9000)
- `POSTGRES_PORT` - Change PostgreSQL port (default 5432)

### 2. Build and Start Services

Run from the repository root:
```bash
cd docker
docker-compose up -d --build
```

Or from root:
```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

### 3. Verify Services

```bash
# Check container status
docker-compose -f docker/docker-compose.yml ps

# Test TigerGraph (wait 30-60 seconds for startup)
curl -u tigergraph:tigergraph http://localhost:9000/echo

# Test PostgreSQL
docker-compose -f docker/docker-compose.yml exec postgres psql -U postgres -d graph_rag -c "SELECT 1"
```

## Services

### TigerGraph (Port 9000, 14240)
- **REST API**: http://localhost:9000
- **Admin UI**: http://localhost:14240
- **Type**: Graph Database (Community Edition)
- **Purpose**: Stores entities and relationships
- **Startup**: ~1-2 minutes

### PostgreSQL (Port 5432)
- **Host**: localhost
- **Database**: graph_rag
- **User**: postgres
- **Purpose**: Metadata and audit trails
- **Startup**: ~10 seconds

### Python App (Port 8000)
- **Type**: Application container
- **Purpose**: Extraction pipeline, graph operations
- **Startup**: ~5 seconds

## Common Commands

### View Logs
```bash
# All services
docker-compose -f docker/docker-compose.yml logs -f

# Specific service
docker-compose -f docker/docker-compose.yml logs -f tigergraph
docker-compose -f docker/docker-compose.yml logs -f postgres
docker-compose -f docker/docker-compose.yml logs -f app
```

### Interactive Shell
```bash
# Enter Python app container
docker-compose -f docker/docker-compose.yml exec app bash

# Start IPython REPL
docker-compose -f docker/docker-compose.yml exec app ipython

# PostgreSQL CLI
docker-compose -f docker/docker-compose.yml exec postgres psql -U postgres -d graph_rag
```

### Stop Services
```bash
# Stop containers (keep volumes)
docker-compose -f docker/docker-compose.yml down

# Stop and remove volumes (reset all data)
docker-compose -f docker/docker-compose.yml down -v

# Restart services
docker-compose -f docker/docker-compose.yml restart
```

### Update Dependencies
```bash
# Edit requirements.txt, then rebuild
docker-compose -f docker/docker-compose.yml build --no-cache app
docker-compose -f docker/docker-compose.yml restart app
```

## Files Explained

### `Dockerfile`
Defines the Python application container image:
- Base: Python 3.10 slim
- Installs system dependencies
- Installs Python packages from requirements.txt
- Creates necessary directories
- Sets up working environment

### `docker-compose.yml`
Orchestrates three services:
- **tigergraph**: Graph database
- **postgres**: Metadata and audit database
- **app**: Python application

Key features:
- Health checks for automatic dependency management
- Volume persistence for data
- Network isolation
- Environment variable injection

### `.env`
Configuration file for environment variables. **Do NOT commit to git** with production credentials.

**Important**: Add to `.gitignore`:
```bash
docker/.env
docker/.env.local
```

### `.dockerignore`
Excludes unnecessary files from Docker build context to reduce image size.

### `init/init_db.sql`
PostgreSQL initialization script that automatically runs when postgres container starts.

**What it does:**
- Creates `graph_metadata` schema
- Creates 4 tables:
  - `ingestion_jobs` - tracks data ingestion batches
  - `documents` - tracks source documents
  - `entities` - metadata about extracted entities
  - `relations` - metadata about extracted relationships
- Creates indexes for performance
- Sets up foreign keys and constraints

**Note**: This stores metadata and audit trails only. The actual graph data (entities and relationships) lives in TigerGraph.

## Troubleshooting

### TigerGraph Not Starting
```bash
# Wait longer (first startup can take 1-2 minutes)
sleep 60
docker-compose -f docker/docker-compose.yml logs tigergraph

# Check if port is already in use
lsof -i :9000

# Restart TigerGraph
docker-compose -f docker/docker-compose.yml restart tigergraph
```

### Connection Refused
```bash
# Verify containers are running
docker-compose -f docker/docker-compose.yml ps

# Check network connectivity
docker-compose -f docker/docker-compose.yml exec app ping tigergraph
docker-compose -f docker/docker-compose.yml exec app ping postgres
```

### Memory Issues
- Increase Docker memory limit (Docker Desktop settings)
- Reduce TigerGraph cache size
- Use smaller language models

### Permission Errors
```bash
# Ensure user can run docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

## Helpful Aliases

Add to your shell config (`.bashrc`, `.zshrc`):

```bash
alias docker-up='docker-compose -f docker/docker-compose.yml up -d'
alias docker-down='docker-compose -f docker/docker-compose.yml down'
alias docker-logs='docker-compose -f docker/docker-compose.yml logs -f'
alias docker-shell='docker-compose -f docker/docker-compose.yml exec app bash'
```

## Security Notes

1. **Don't commit `.env`** with real credentials to git
2. **Change default passwords** in `.env` for production
3. **Use secrets management** (Docker Secrets, Vault) in production
4. **Restrict network access** in production deployments
5. **Keep images updated** - rebuild periodically

## Next Steps

1. Start the services: `docker-compose -f docker/docker-compose.yml up -d`
2. Wait for health checks to pass
3. Access TigerGraph Admin: http://localhost:14240
4. Run extraction pipeline: `docker-compose exec app python scripts/build_graph.py`
5. Check metadata in PostgreSQL: `docker-compose exec postgres psql -U postgres -d graph_rag`
