-- Initialize database schema for auxiliary storage
-- This script runs automatically when the PostgreSQL container starts
-- Purpose: Track data ingestion, entities, and relationships metadata
-- Note: Actual graph data is stored in TigerGraph, not PostgreSQL

-- Create schema for graph metadata
CREATE SCHEMA IF NOT EXISTS graph_metadata;

-- Table for tracking ingestion jobs
-- Tracks batches of documents being ingested into the system
CREATE TABLE IF NOT EXISTS graph_metadata.ingestion_jobs (
    id SERIAL PRIMARY KEY,
    job_id UUID UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    source_type VARCHAR(100),                        -- pdf, txt, website, etc.
    source_name VARCHAR(255),                        -- name/url of data source
    num_documents INT DEFAULT 0,                     -- count of documents processed
    num_entities INT DEFAULT 0,                      -- count of entities extracted
    num_relations INT DEFAULT 0,                     -- count of relations extracted
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT                               -- error details if failed
);

-- Table for tracking documents
-- Each document that goes through the pipeline
CREATE TABLE IF NOT EXISTS graph_metadata.documents (
    id SERIAL PRIMARY KEY,
    doc_id UUID UNIQUE NOT NULL,
    job_id UUID NOT NULL,
    filename VARCHAR(255),
    content TEXT,                                    -- original text content
    source_url VARCHAR(500),
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES graph_metadata.ingestion_jobs(job_id)
);

-- Table for extracted entities (for reference and audit)
-- Metadata about entities extracted from documents
-- Actual entities are stored in TigerGraph
CREATE TABLE IF NOT EXISTS graph_metadata.entities (
    id SERIAL PRIMARY KEY,
    entity_id UUID UNIQUE NOT NULL,
    doc_id UUID NOT NULL,
    entity_name VARCHAR(255),
    entity_type VARCHAR(100),                        -- PERSON, ORGANIZATION, LOCATION, etc.
    confidence FLOAT,                                -- extraction confidence score
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES graph_metadata.documents(doc_id)
);

-- Table for extracted relations (for reference and audit)
-- Metadata about relationships extracted from documents
-- Actual relationships are stored in TigerGraph
CREATE TABLE IF NOT EXISTS graph_metadata.relations (
    id SERIAL PRIMARY KEY,
    relation_id UUID UNIQUE NOT NULL,
    doc_id UUID NOT NULL,
    source_entity_id UUID,
    relation_type VARCHAR(100),                      -- WORKS_FOR, LOCATED_IN, KNOWS, etc.
    target_entity_id UUID,
    confidence FLOAT,                                -- extraction confidence score
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES graph_metadata.documents(doc_id),
    FOREIGN KEY (source_entity_id) REFERENCES graph_metadata.entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES graph_metadata.entities(entity_id)
);

-- Create indexes for performance
CREATE INDEX idx_ingestion_jobs_status ON graph_metadata.ingestion_jobs(status);
CREATE INDEX idx_documents_job_id ON graph_metadata.documents(job_id);
CREATE INDEX idx_entities_doc_id ON graph_metadata.entities(doc_id);
CREATE INDEX idx_entities_type ON graph_metadata.entities(entity_type);
CREATE INDEX idx_relations_doc_id ON graph_metadata.relations(doc_id);
CREATE INDEX idx_relations_type ON graph_metadata.relations(relation_type);

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA graph_metadata TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA graph_metadata TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA graph_metadata TO postgres;
