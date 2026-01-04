#!/usr/bin/env python3
"""
Demo script showing webpage extraction through the full pipeline.

This script demonstrates:
1. Parsing a text file using the extraction pipeline
2. Chunking the content into semantic chunks
3. Extracting entities using NER
4. Extracting relationships between entities
5. Displaying the results

Usage:
    python demo_webpage_extraction.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from src.extraction import (
    ParserFactory,
    SemanticChunker,
    EntityExtractor,
    RelationshipExtractor,
    DocumentType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_text_file() -> str:
    """Create a sample text file for extraction."""
    sample_content = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is the field of computer science that aims to create intelligent machines.
    AI has been developed by researchers like Andrew Ng and Yann LeCun at major tech companies such as Google, 
    Facebook, and OpenAI.

    Machine Learning is a subset of AI. It enables computers to learn from data without being explicitly programmed.
    Deep Learning, a subfield of machine learning, uses neural networks developed by Geoffrey Hinton.
    
    Popular AI companies include:
    - OpenAI: Founded by Sam Altman, develops GPT models
    - DeepMind: Owned by Google, created AlphaGo
    - Tesla: Led by Elon Musk, develops autonomous vehicles
    
    Key AI Applications:
    Natural Language Processing (NLP) is used for language translation and chatbots.
    Computer Vision helps in image recognition and object detection.
    Robotics combines AI with mechanical engineering.
    
    The future of AI looks promising with advances in transformer architectures and large language models.
    Companies like Microsoft and Apple are investing heavily in AI research.
    """
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write sample content to file
    sample_file = data_dir / "ai_sample.txt"
    sample_file.write_text(sample_content)
    
    logger.info(f"Created sample text file: {sample_file}")
    return str(sample_file)


def extract_and_process(file_path: str) -> Dict[str, Any]:
    """
    Extract text from file and process through full pipeline.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Dictionary with extraction results
    """
    results = {
        "file_path": file_path,
        "document": None,
        "chunks": [],
        "entities": [],
        "relationships": [],
        "stats": {}
    }
    
    try:
        # ========== PHASE 1: PARSING ==========
        logger.info("=" * 60)
        logger.info("PHASE 1: Document Parsing")
        logger.info("=" * 60)
        
        parser = ParserFactory.get_parser(file_path)
        logger.info(f"Using parser: {parser.__class__.__name__}")
        
        parsed_doc = parser.parse(file_path)
        logger.info(f"✓ Parsed document: {parsed_doc.file_type.value}")
        logger.info(f"  - Characters: {parsed_doc.character_count:,}")
        logger.info(f"  - Pages: {parsed_doc.page_count}")
        logger.info(f"  - Metadata: {parsed_doc.metadata}")
        
        results["document"] = {
            "file_path": parsed_doc.file_path,
            "file_type": parsed_doc.file_type.value,
            "character_count": parsed_doc.character_count,
            "page_count": parsed_doc.page_count,
            "metadata": parsed_doc.metadata,
            "text_preview": parsed_doc.raw_text[:200] + "..."
        }
        
        # ========== PHASE 2: CHUNKING ==========
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Text Chunking")
        logger.info("=" * 60)
        
        chunker = SemanticChunker()
        chunks = chunker.chunk(parsed_doc.raw_text)
        logger.info(f"✓ Created {len(chunks)} semantic chunks")
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"  Chunk {i}: {chunk.character_count} chars, "
                       f"word_count: {chunk.word_count}")
            content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            results["chunks"].append({
                "chunk_id": i,
                "character_count": chunk.character_count,
                "word_count": chunk.word_count,
                "content_preview": content_preview
            })
        
        # ========== PHASE 3: NAMED ENTITY RECOGNITION ==========
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Named Entity Recognition (NER)")
        logger.info("=" * 60)
        
        all_entities = []
        try:
            entity_extractor = EntityExtractor()
            
            for i, chunk in enumerate(chunks, 1):
                entities = entity_extractor.extract_from_chunk(chunk)
                logger.info(f"  Chunk {i}: Extracted {len(entities)} entities")
                
                for entity in entities:
                    logger.info(f"    - {entity.text} ({entity.entity_type.value})")
                    all_entities.append(entity)
                    results["entities"].append({
                        "text": entity.text,
                        "type": entity.entity_type.value,
                        "confidence": entity.confidence if hasattr(entity, 'confidence') else None
                    })
            
            logger.info(f"✓ Total entities extracted: {len(all_entities)}")
        except OSError as e:
            logger.warning(f"⚠ NER model not available: {e}")
            logger.info("  Note: Install with: python -m spacy download en_core_web_sm")
            logger.info("  Skipping NER phase for demo")
        
        # ========== PHASE 4: RELATIONSHIP EXTRACTION ==========
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: Relationship Extraction")
        logger.info("=" * 60)
        
        all_relationships = []
        try:
            relationship_extractor = RelationshipExtractor()
            
            for i, chunk in enumerate(chunks, 1):
                relationships = relationship_extractor.extract_from_chunk(chunk)
                logger.info(f"  Chunk {i}: Extracted {len(relationships)} relationships")
                
                for rel in relationships:
                    logger.info(f"    - {rel.source_entity} -> {rel.relationship_type.value} -> "
                               f"{rel.target_entity}")
                    all_relationships.append(rel)
                    results["relationships"].append({
                        "source": rel.source_entity,
                        "relationship_type": rel.relationship_type.value,
                        "target": rel.target_entity,
                        "confidence": rel.confidence if hasattr(rel, 'confidence') else None
                    })
            
            logger.info(f"✓ Total relationships extracted: {len(all_relationships)}")
        except Exception as e:
            logger.warning(f"⚠ Relationship extraction note: {e}")
            logger.info("  Relationships require entity extraction from NER")
        
        # ========== SUMMARY STATISTICS ==========
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        
        stats = {
            "total_characters": parsed_doc.character_count,
            "total_chunks": len(chunks),
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "unique_entity_types": len(set(e.entity_type.value for e in all_entities)) if all_entities else 0,
            "unique_relationship_types": len(set(r.relationship_type.value for r in all_relationships)) if all_relationships else 0,
        }
        
        results["stats"] = stats
        
        logger.info(f"✓ Document Characters: {stats['total_characters']:,}")
        logger.info(f"✓ Total Chunks: {stats['total_chunks']}")
        logger.info(f"✓ Total Entities: {stats['total_entities']}")
        logger.info(f"✓ Total Relationships: {stats['total_relationships']}")
        logger.info(f"✓ Entity Types: {stats['unique_entity_types']}")
        logger.info(f"✓ Relationship Types: {stats['unique_relationship_types']}")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        results["error"] = str(e)
    
    return results


def save_results(results: Dict[str, Any], output_file: str = "data/processed/extraction_results.json") -> None:
    """
    Save extraction results to JSON file.
    
    Args:
        results: Dictionary with extraction results
        output_file: Path to save results
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")


def main():
    """Run the demo extraction pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("OpenCoreGraphRAG - Webpage Extraction Demo")
    logger.info("=" * 60 + "\n")
    
    # Create sample text file
    sample_file = create_sample_text_file()
    
    # Process through pipeline
    results = extract_and_process(sample_file)
    
    # Save results
    save_results(results)
    
    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Results saved to: data/processed/extraction_results.json")
    logger.info("\nYou can now:")
    logger.info("1. Review the extraction_results.json file")
    logger.info("2. Use the extracted entities and relationships for knowledge graph construction")
    logger.info("3. Process the data with TigerGraph for graph storage")


if __name__ == "__main__":
    main()
