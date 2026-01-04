"""Multi-LLM competition system for relationship extraction.

This module implements a competitive extraction approach where multiple LLM
models extract relationships from the same text in parallel, and their results
are combined using various voting strategies.

Similar to CompetitiveNER but focused on relationship extraction.

Benefits:
- Improved accuracy through consensus on relationships
- Diverse model strengths in understanding semantic relationships
- Confidence estimation through agreement levels
- Cost optimization (choose strategy based on needs)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..chunking import TextChunk
from ..ner import ExtractedEntity
from .relationship_models import ExtractedRelationship, RelationshipType

if TYPE_CHECKING:
    from .relationship_extractor import RelationshipExtractor

logger = logging.getLogger(__name__)


@dataclass
class RelationshipCompetitor:
    """Wrapper for an LLM that participates in relationship extraction competition.
    
    Tracks execution metrics and results for a single LLM provider
    used for relationship extraction.
    """
    
    name: str
    """Name/identifier for this competitor (e.g., 'mistral', 'gpt-4')"""
    
    provider: "RelationshipProvider"
    """The underlying relationship extraction provider"""
    
    # Execution results
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    """Relationships extracted by this model"""
    
    confidence: float = 0.0
    """Average confidence of extracted relationships"""
    
    execution_time_ms: float = 0.0
    """Time taken to extract (milliseconds)"""
    
    def extract(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str = "",
        source_file: str = "",
    ) -> None:
        """Extract relationships from text and record metrics.
        
        Args:
            text: Text to extract relationships from
            entities: Already extracted entities to relate
            chunk_id: ID of the chunk
            source_file: Source file path
        """
        start_time = time.time()
        
        try:
            self.relationships = self.provider.extract_relationships(
                text=text,
                entities=entities,
                chunk_id=chunk_id,
                source_file=source_file,
            )
            
            # Calculate average confidence
            self.confidence = (
                sum(r.confidence for r in self.relationships) / len(self.relationships)
                if self.relationships
                else 0.0
            )
        except Exception as e:
            logger.error(f"{self.name} relationship extraction failed: {e}")
            self.relationships = []
            self.confidence = 0.0
        finally:
            self.execution_time_ms = (time.time() - start_time) * 1000
    
    def to_dict(self) -> Dict:
        """Convert competitor results to dictionary.
        
        Returns:
            Dictionary representation of competitor results
        """
        return {
            "name": self.name,
            "relationships": [
                {
                    "source_entity": r.source_entity,
                    "target_entity": r.target_entity,
                    "type": r.relationship_type.value,
                    "confidence": r.confidence,
                }
                for r in self.relationships
            ],
            "avg_confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class RelationshipAgreement:
    """Tracks agreement on a specific relationship across competitors."""
    
    source_entity: str
    """Source entity text"""
    
    target_entity: str
    """Target entity text"""
    
    relationship_type: RelationshipType
    """Relationship type"""
    
    found_by: List[str] = field(default_factory=list)
    """Names of models that found this relationship"""
    
    confidences: List[float] = field(default_factory=list)
    """Confidence scores from each model"""
    
    @property
    def agreement_count(self) -> int:
        """Number of models that found this relationship."""
        return len(self.found_by)
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across models that found it."""
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "type": self.relationship_type.value,
            "found_by": self.found_by,
            "agreement_count": self.agreement_count,
            "avg_confidence": self.avg_confidence,
            "confidences_by_model": {
                name: conf
                for name, conf in zip(self.found_by, self.confidences)
            },
        }


class CompetitiveRelationshipExtractor:
    """Run multiple LLM models in competition for relationship extraction.
    
    Orchestrates parallel extraction from multiple models and combines
    results using voting strategies.
    """
    
    def __init__(
        self,
        competitors: List[RelationshipCompetitor],
        voting_strategy: str = "weighted",
        max_workers: Optional[int] = None,
    ):
        """Initialize competitive relationship extraction system.
        
        Args:
            competitors: List of RelationshipCompetitor instances
            voting_strategy: How to combine results
                - "consensus": Only relationships all models agree on
                - "majority": Relationships 2+ models agree on
                - "weighted": Weight by confidence scores (RECOMMENDED)
                - "best": Use results from model with highest avg confidence
            max_workers: Max concurrent model executions (default: num competitors)
        """
        self.competitors = competitors
        self.voting_strategy = voting_strategy.lower()
        self.max_workers = max_workers or len(competitors)
        
        if self.voting_strategy not in ["consensus", "majority", "weighted", "best"]:
            raise ValueError(
                f"Unknown voting strategy: {voting_strategy}. "
                f"Must be one of: consensus, majority, weighted, best"
            )
        
        logger.info(
            f"Initialized CompetitiveRelationshipExtractor with {len(competitors)} competitors "
            f"and '{self.voting_strategy}' voting strategy"
        )
    
    def extract_with_competition(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str = "",
        source_file: str = "",
        return_agreement: bool = True,
    ) -> Tuple[List[ExtractedRelationship], Dict]:
        """Run competition and return winning relationships.
        
        Args:
            text: Text to extract relationships from
            entities: Pre-extracted entities to relate
            chunk_id: ID of the chunk
            source_file: Source file path
            return_agreement: Include agreement analysis in stats
            
        Returns:
            Tuple of (final_relationships, stats_dict)
        """
        # Step 1: Run all competitors in parallel
        self._run_all_competitors(text, entities, chunk_id, source_file)
        
        # Step 2: Analyze agreement
        agreement = self._analyze_agreement(chunk_id)
        
        # Step 3: Apply voting strategy
        final_relationships = self._apply_voting_strategy(agreement, chunk_id)
        
        # Step 4: Prepare statistics
        stats = self._prepare_stats(agreement, return_agreement)
        
        return final_relationships, stats
    
    def _run_all_competitors(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str,
        source_file: str,
    ) -> None:
        """Run all competitors in parallel.
        
        Args:
            text: Text to extract from
            entities: Pre-extracted entities
            chunk_id: Chunk ID
            source_file: Source file path
        """
        overall_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    competitor.extract,
                    text,
                    entities,
                    chunk_id,
                    source_file,
                ): competitor
                for competitor in self.competitors
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                competitor = futures[future]
                try:
                    future.result()
                    logger.debug(
                        f"{competitor.name}: {len(competitor.relationships)} relationships "
                        f"({competitor.execution_time_ms:.0f}ms, "
                        f"{competitor.confidence:.2f} confidence)"
                    )
                except Exception as e:
                    logger.error(f"{competitor.name} execution failed: {e}")
        
        overall_time = (time.time() - overall_start) * 1000
        logger.debug(f"All relationship extractors completed in {overall_time:.0f}ms")
    
    def _analyze_agreement(
        self, chunk_id: str
    ) -> Dict[Tuple[str, str, str], RelationshipAgreement]:
        """Analyze which relationships different models agree on.
        
        Args:
            chunk_id: Chunk ID for relationship creation
            
        Returns:
            Dictionary mapping (source_entity, target_entity, type) -> RelationshipAgreement
        """
        agreement = {}
        
        for competitor in self.competitors:
            for rel in competitor.relationships:
                # Use (source_entity, target_entity, type) as unique key
                key = (rel.source_entity, rel.target_entity, rel.relationship_type)
                
                if key not in agreement:
                    agreement[key] = RelationshipAgreement(
                        source_entity=rel.source_entity,
                        target_entity=rel.target_entity,
                        relationship_type=rel.relationship_type,
                    )
                
                agreement[key].found_by.append(competitor.name)
                agreement[key].confidences.append(rel.confidence)
        
        return agreement
    
    def _apply_voting_strategy(
        self,
        agreement: Dict[Tuple[str, str, str], RelationshipAgreement],
        chunk_id: str,
    ) -> List[ExtractedRelationship]:
        """Apply voting strategy to determine final relationships.
        
        Args:
            agreement: Agreement analysis from all competitors
            chunk_id: Chunk ID for creating relationships
            
        Returns:
            List of final relationships
        """
        if self.voting_strategy == "consensus":
            return self._consensus_voting(agreement, chunk_id)
        elif self.voting_strategy == "majority":
            return self._majority_voting(agreement, chunk_id)
        elif self.voting_strategy == "weighted":
            return self._weighted_voting(agreement, chunk_id)
        elif self.voting_strategy == "best":
            return self._best_voting(chunk_id)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def _consensus_voting(
        self,
        agreement: Dict[Tuple[str, str, str], RelationshipAgreement],
        chunk_id: str,
    ) -> List[ExtractedRelationship]:
        """Consensus voting: only include relationships ALL models agree on.
        
        Highest precision, lowest recall. Use when accuracy is critical.
        """
        final_rels = []
        
        for rel_agreement in agreement.values():
            # All competitors must agree
            if rel_agreement.agreement_count == len(self.competitors):
                final_rels.append(
                    ExtractedRelationship(
                        source_entity=rel_agreement.source_entity,
                        target_entity=rel_agreement.target_entity,
                        relationship_type=rel_agreement.relationship_type,
                        source_chunk_id=chunk_id,
                        confidence=rel_agreement.avg_confidence,
                        metadata={
                            "agreement_count": rel_agreement.agreement_count,
                            "found_by": rel_agreement.found_by,
                        }
                    )
                )
        
        return final_rels
    
    def _majority_voting(
        self,
        agreement: Dict[Tuple[str, str, str], RelationshipAgreement],
        chunk_id: str,
    ) -> List[ExtractedRelationship]:
        """Majority voting: include relationships 2+ models agree on.
        
        Balanced precision/recall. Use for most applications.
        """
        final_rels = []
        
        for rel_agreement in agreement.values():
            # At least 2 competitors must agree
            if rel_agreement.agreement_count >= 2:
                final_rels.append(
                    ExtractedRelationship(
                        source_entity=rel_agreement.source_entity,
                        target_entity=rel_agreement.target_entity,
                        relationship_type=rel_agreement.relationship_type,
                        source_chunk_id=chunk_id,
                        confidence=rel_agreement.avg_confidence,
                        metadata={
                            "agreement_count": rel_agreement.agreement_count,
                            "found_by": rel_agreement.found_by,
                        }
                    )
                )
        
        return final_rels
    
    def _weighted_voting(
        self,
        agreement: Dict[Tuple[str, str, str], RelationshipAgreement],
        chunk_id: str,
    ) -> List[ExtractedRelationship]:
        """Weighted voting: weight by confidence and agreement count.
        
        RECOMMENDED: Balances precision and recall with confidence weighting.
        Formula: score = avg_confidence * (agreement_count / total_competitors)
        """
        final_rels = []
        
        for rel_agreement in agreement.values():
            # Weight by agreement percentage and confidence
            agreement_weight = rel_agreement.agreement_count / len(self.competitors)
            confidence_score = rel_agreement.avg_confidence * agreement_weight
            
            final_rels.append(
                ExtractedRelationship(
                    source_entity=rel_agreement.source_entity,
                    target_entity=rel_agreement.target_entity,
                    relationship_type=rel_agreement.relationship_type,
                    source_chunk_id=chunk_id,
                    confidence=confidence_score,
                    metadata={
                        "agreement_count": rel_agreement.agreement_count,
                        "found_by": rel_agreement.found_by,
                    }
                )
            )
        
        return final_rels
    
    def _best_voting(self, chunk_id: str) -> List[ExtractedRelationship]:
        """Best model voting: use results from model with highest avg confidence.
        
        Fastest but ignores consensus. Use for speed-critical applications.
        """
        # Find competitor with highest avg confidence
        best_competitor = max(
            self.competitors, key=lambda c: c.confidence
        )
        
        # Return relationships from best competitor with metadata
        final_rels = []
        for rel in best_competitor.relationships:
            # Create a copy with updated metadata
            rel_copy = ExtractedRelationship(
                source_entity=rel.source_entity,
                target_entity=rel.target_entity,
                relationship_type=rel.relationship_type,
                source_chunk_id=chunk_id,
                confidence=rel.confidence,
                metadata={
                    "agreement_count": 1,
                    "found_by": [best_competitor.name],
                }
            )
            final_rels.append(rel_copy)
        
        return final_rels
    
    def _prepare_stats(
        self,
        agreement: Dict[Tuple[str, str, str], RelationshipAgreement],
        return_agreement: bool,
    ) -> Dict:
        """Prepare comprehensive statistics.
        
        Args:
            agreement: Agreement analysis
            return_agreement: Whether to include detailed agreement data
            
        Returns:
            Dictionary with statistics
        """
        competitor_stats = [c.to_dict() for c in self.competitors]
        
        stats = {
            "voting_strategy": self.voting_strategy,
            "num_competitors": len(self.competitors),
            "competitors": competitor_stats,
            "total_unique_relationships": len(agreement),
            "avg_competitor_confidence": (
                sum(c.confidence for c in self.competitors) / len(self.competitors)
                if self.competitors
                else 0.0
            ),
        }
        
        if return_agreement:
            stats["agreement_details"] = {
                str(key): agreement[key].to_dict()
                for key in agreement.keys()
            }
            stats["agreement_summary"] = {
                "consensus_count": sum(
                    1 for a in agreement.values()
                    if a.agreement_count == len(self.competitors)
                ),
                "majority_count": sum(
                    1 for a in agreement.values()
                    if a.agreement_count >= 2
                ),
                "single_model_count": sum(
                    1 for a in agreement.values()
                    if a.agreement_count == 1
                ),
            }
        
        return stats


class RelationshipProvider:
    """Abstract base class for relationship extraction providers.
    
    Used by RelationshipCompetitor to extract relationships from text.
    Implementations can use pattern-based, LLM-based, or hybrid approaches.
    """
    
    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str = "",
        source_file: str = "",
    ) -> List[ExtractedRelationship]:
        """Extract relationships from text.
        
        Args:
            text: Text to extract from
            entities: Pre-extracted entities to relate
            chunk_id: Chunk identifier
            source_file: Source file path
            
        Returns:
            List of ExtractedRelationship objects
        """
        raise NotImplementedError("Subclasses must implement extract_relationships()")


class DefaultRelationshipProvider(RelationshipProvider):
    """Default provider using pattern-based relationship extraction.
    
    Wraps the standard RelationshipExtractor for use in competitions.
    """
    
    def __init__(self, extractor: "RelationshipExtractor"):
        """Initialize with a RelationshipExtractor instance.
        
        Args:
            extractor: The RelationshipExtractor to use
        """
        self.extractor = extractor
    
    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        chunk_id: str = "",
        source_file: str = "",
    ) -> List[ExtractedRelationship]:
        """Extract relationships using the wrapped extractor."""
        chunk = TextChunk(
            content=text,
            chunk_id=chunk_id,
            source_file=source_file,
        )
        return self.extractor.extract_from_chunk(chunk, entities)
