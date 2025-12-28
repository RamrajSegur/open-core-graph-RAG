"""
Factory pattern for creating GraphStore instances.

Provides a unified interface for creating and configuring TigerGraph
instances for both local (Docker) and remote deployments.
"""

from typing import Optional, Dict
from src.core.graph_store import GraphStore
from src.core.tiger_graph_store import TigerGraphStore
from src.config import config


def get_graph_store(**kwargs) -> GraphStore:
    """
    Factory function to create TigerGraph GraphStore instance.
    
    TigerGraph can be deployed locally via Docker or on a remote server.
    This factory handles both cases transparently.
    
    Args:
        **kwargs: TigerGraph-specific configuration
                  - host: TigerGraph server host (default: from config)
                  - port: TigerGraph server port (default: from config)
                  - username: Authentication username (default: from config)
                  - password: Authentication password (default: from config)
                  - graph_name: Graph name (default: from config)
                  - protocol: http or https (default: from config)
                  - timeout: Request timeout in seconds (default: 30)
        
    Returns:
        GraphStore instance (TigerGraphStore)
        
    Examples:
        # Use TigerGraph with configuration from config.py
        graph = get_graph_store()
        
        # Use TigerGraph with custom credentials
        graph = get_graph_store(
            host='remote.server.com',
            port=9000,
            username='admin',
            password='password'
        )
        
        # Use local Docker instance with custom port
        graph = get_graph_store(host='localhost', port=14240)
    """
    return _create_tigergraph_store(**kwargs)


def _create_tigergraph_store(**kwargs) -> TigerGraphStore:
    """Create and configure TigerGraphStore for local or remote deployment"""
    
    # Get configuration from kwargs or config
    tg_config = config.tigergraph
    
    # Override with kwargs if provided
    host = kwargs.get('host', tg_config.host)
    port = kwargs.get('port', tg_config.port)
    username = kwargs.get('username', tg_config.username)
    password = kwargs.get('password', tg_config.password)
    graph_name = kwargs.get('graph_name', tg_config.graph_name)
    protocol = kwargs.get('protocol', tg_config.protocol)
    timeout = kwargs.get('timeout', 30)
    
    store = TigerGraphStore(
        host=host,
        port=port,
        username=username,
        password=password,
        graph_name=graph_name,
        protocol=protocol,
        timeout=timeout
    )
    
    # Try to connect
    if not store.connect():
        raise RuntimeError(
            f"Failed to connect to TigerGraph at {protocol}://{host}:{port}"
        )
    
    return store


def get_available_backends() -> Dict[str, str]:
    """
    Get available backends and their descriptions.
    
    Returns:
        Dict mapping backend names to descriptions
    """
    return {
        'tigergraph': 'TigerGraph server. Can be deployed locally via Docker or on remote servers.',
    }


def create_graph_store_from_config() -> GraphStore:
    """
    Create GraphStore using application configuration.
    
    Uses TigerGraph configuration from config.py.
    
    Returns:
        Configured GraphStore instance (TigerGraphStore)
    """
    return get_graph_store()
