"""Multi-LLM competition system for NER extraction.

This module implements a competitive extraction approach where multiple LLM
models extract entities from the same text in parallel, and their results
are combined using various voting strategies.

Benefits:
- Improved accuracy through consensus
- Diverse model strengths (speed, reasoning, accuracy)
- Confidence estimation through agreement levels
- Cost optimization (choose strategy based on needs)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .entity_models import EntityType, ExtractedEntity
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMCompetitor:
    """Wrapper for an LLM that participates in extraction competition.
    
    Tracks execution metrics and results for a single LLM provider.
    """
    
    name: str
    """Name/identifier for this competitor (e.g., 'mistral', 'gpt-4')"""
    
    provider: LLMProvider
    """The underlying LLM provider"""
    
    # Execution results
    entities: List[ExtractedEntity] = field(default_factory=list)
    """Entities extracted by this model"""
    
    confidence: float = 0.0
    """Average confidence of extracted entities"""
    
    execution_time_ms: float = 0.0
    """Time taken to extract (milliseconds)"""
    
    def extract(self, text: str, chunk_id: str = "", source_file: str = "") -> None:
        """Extract entities from text and record metrics.
        
        Args:
            text: Text to extract entities from
            chunk_id: ID of the chunk
            source_file: Source file path
        """
        start_time = time.time()
        
        try:
            self.entities = self.provider.extract_entities(
                text=text,
                chunk_id=chunk_id,
                source_file=source_file,
            )
            
            # Calculate average confidence
            self.confidence = (
                sum(e.confidence for e in self.entities) / len(self.entities)
                if self.entities
                else 0.0
            )
        except Exception as e:
            logger.error(f"{self.name} extraction failed: {e}")
            self.entities = []
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
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type.value,
                    "confidence": e.confidence,
                    "position": (e.start_position, e.end_position),
                }
                for e in self.entities
            ],
            "avg_confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class EntityAgreement:
    """Tracks agreement on a specific entity across competitors."""
    
    text: str
    """The entity text"""
    
    entity_type: EntityType
    """Entity type"""
    
    found_by: List[str] = field(default_factory=list)
    """Names of models that found this entity"""
    
    confidences: List[float] = field(default_factory=list)
    """Confidence scores from each model"""
    
    @property
    def agreement_count(self) -> int:
        """Number of models that found this entity."""
        return len(self.found_by)
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across models that found it."""
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "found_by": self.found_by,
            "agreement_count": self.agreement_count,
            "avg_confidence": self.avg_confidence,
            "confidences_by_model": {
                name: conf
                for name, conf in zip(self.found_by, self.confidences)
            },
        }


class CompetitiveNER:
    """Run multiple LLM models in competition for entity extraction.
    
    Orchestrates parallel extraction from multiple models and combines
    results using voting strategies.
    """
    
    def __init__(
        self,
        competitors: List[LLMCompetitor],
        voting_strategy: str = "weighted",
        max_workers: Optional[int] = None,
    ):
        """Initialize competitive NER system.
        
        Args:
            competitors: List of LLMCompetitor instances
            voting_strategy: How to combine results
                - "consensus": Only entities all models agree on
                - "majority": Entities 2+ models agree on
                - "weighted": Weight by confidence scores
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
            f"Initialized CompetitiveNER with {len(competitors)} competitors "
            f"and '{self.voting_strategy}' voting strategy"
        )
    
    def extract_with_competition(
        self,
        text: str,
        chunk_id: str = "",
        source_file: str = "",
        return_agreement: bool = True,
    ) -> Tuple[List[ExtractedEntity], Dict]:
        """Run competition and return winning entities.
        
        Args:
            text: Text to extract entities from
            chunk_id: ID of the chunk
            source_file: Source file path
            return_agreement: Include agreement analysis in stats
            
        Returns:
            Tuple of (final_entities, stats_dict)
        """
        # Step 1: Run all competitors in parallel
        self._run_all_competitors(text, chunk_id, source_file)
        
        # Step 2: Analyze agreement
        agreement = self._analyze_agreement(chunk_id)
        
        # Step 3: Apply voting strategy
        final_entities = self._apply_voting_strategy(agreement, chunk_id)
        
        # Step 4: Prepare statistics
        stats = self._prepare_stats(agreement, return_agreement)
        
        return final_entities, stats
    
    def _run_all_competitors(
        self,
        text: str,
        chunk_id: str,
        source_file: str,
    ) -> None:
        """Run all competitors in parallel.
        
        Args:
            text: Text to extract from
            chunk_id: Chunk ID
            source_file: Source file path
        """
        overall_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    competitor.extract,
                    text,
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
                        f"{competitor.name}: {len(competitor.entities)} entities "
                        f"({competitor.execution_time_ms:.0f}ms, "
                        f"{competitor.confidence:.2f} confidence)"
                    )
                except Exception as e:
                    logger.error(f"{competitor.name} execution failed: {e}")
        
        overall_time = (time.time() - overall_start) * 1000
        logger.debug(f"All competitors completed in {overall_time:.0f}ms")
    
    def _analyze_agreement(self, chunk_id: str) -> Dict[Tuple[str, str], EntityAgreement]:
        """Analyze which entities different models agree on.
        
        Args:
            chunk_id: Chunk ID for entity creation
            
        Returns:
            Dictionary mapping (text, type) -> EntityAgreement
        """
        agreement = {}
        
        for competitor in self.competitors:
            for entity in competitor.entities:
                # Use (text, type) as unique key
                key = (entity.text, entity.entity_type)
                
                if key not in agreement:
                    agreement[key] = EntityAgreement(
                        text=entity.text,
                        entity_type=entity.entity_type,
                    )
                
                agreement[key].found_by.append(competitor.name)
                agreement[key].confidences.append(entity.confidence)
        
        return agreement
    
    def _apply_voting_strategy(
        self,
        agreement: Dict[Tuple[str, str], EntityAgreement],
        chunk_id: str,
    ) -> List[ExtractedEntity]:
        """Apply voting strategy to determine final entities.
        
        Args:
            agreement: Agreement analysis from all competitors
            chunk_id: Chunk ID for creating entities
            
        Returns:
            List of final entities
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
        agreement: Dict[Tuple[str, str], EntityAgreement],
        chunk_id: str,
    ) -> List[ExtractedEntity]:
        """Keep only entities ALL models agree on.
        
        Most conservative approach - highest precision.
        
        Args:
            agreement: Agreement analysis
            chunk_id: Chunk ID
            
        Returns:
            Entities with unanimous agreement
        """
        num_competitors = len(self.competitors)
        entities = []
        
        for entity_agreement in agreement.values():
            if entity_agreement.agreement_count == num_competitors:
                # All models agree
                entity = ExtractedEntity(
                    text=entity_agreement.text,
                    entity_type=entity_agreement.entity_type,
                    chunk_id=chunk_id,
                    start_position=0,
                    end_position=len(entity_agreement.text),
                    confidence=entity_agreement.avg_confidence,
                )
                entities.append(entity)
                logger.debug(
                    f"Consensus: {entity_agreement.text} "
                    f"({entity.entity_type.value}, {entity.confidence:.2f})"
                )
        
        return entities
    
    def _majority_voting(
        self,
        agreement: Dict[Tuple[str, str], EntityAgreement],
        chunk_id: str,
    ) -> List[ExtractedEntity]:
        """Keep entities that 2+ models agree on.
        
        Balanced approach - good precision and recall.
        
        Args:
            agreement: Agreement analysis
            chunk_id: Chunk ID
            
        Returns:
            Entities with majority agreement (2+)
        """
        entities = []
        threshold = 2  # At least 2 models must agree
        
        for entity_agreement in agreement.values():
            if entity_agreement.agreement_count >= threshold:
                entity = ExtractedEntity(
                    text=entity_agreement.text,
                    entity_type=entity_agreement.entity_type,
                    chunk_id=chunk_id,
                    start_position=0,
                    end_position=len(entity_agreement.text),
                    confidence=entity_agreement.avg_confidence,
                )
                entities.append(entity)
                logger.debug(
                    f"Majority ({entity_agreement.agreement_count}/{len(self.competitors)}): "
                    f"{entity_agreement.text} ({entity.entity_type.value})"
                )
        
        return entities
    
    def _weighted_voting(
        self,
        agreement: Dict[Tuple[str, str], EntityAgreement],
        chunk_id: str,
    ) -> List[ExtractedEntity]:
        """Weight entities by confidence scores and agreement.
        
        Entities with higher confidence and more agreement get higher final confidence.
        
        Args:
            agreement: Agreement analysis
            chunk_id: Chunk ID
            
        Returns:
            Entities weighted by confidence and agreement
        """
        entities = []
        num_competitors = len(self.competitors)
        
        for entity_agreement in agreement.values():
            # Confidence boost from agreement
            agreement_factor = entity_agreement.agreement_count / num_competitors
            
            # Final confidence = average confidence * agreement factor
            final_confidence = entity_agreement.avg_confidence * agreement_factor
            
            entity = ExtractedEntity(
                text=entity_agreement.text,
                entity_type=entity_agreement.entity_type,
                chunk_id=chunk_id,
                start_position=0,
                end_position=len(entity_agreement.text),
                confidence=final_confidence,
            )
            entities.append(entity)
            
            logger.debug(
                f"Weighted ({entity_agreement.agreement_count}/{num_competitors}): "
                f"{entity_agreement.text} "
                f"(avg: {entity_agreement.avg_confidence:.2f}, "
                f"final: {final_confidence:.2f})"
            )
        
        return entities
    
    def _best_voting(self, chunk_id: str) -> List[ExtractedEntity]:
        """Use results from the model with highest average confidence.
        
        Simplest approach - trust the best performing model.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Entities from best-performing competitor
        """
        best_competitor = max(self.competitors, key=lambda c: c.confidence)
        
        logger.debug(
            f"Best voting: Using {best_competitor.name} "
            f"(confidence: {best_competitor.confidence:.2f})"
        )
        
        return best_competitor.entities
    
    def _prepare_stats(
        self,
        agreement: Dict[Tuple[str, str], EntityAgreement],
        return_agreement: bool = True,
    ) -> Dict:
        """Prepare statistics about the competition.
        
        Args:
            agreement: Agreement analysis
            return_agreement: Include detailed agreement stats
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "voting_strategy": self.voting_strategy,
            "num_competitors": len(self.competitors),
            "execution_times_ms": {
                c.name: c.execution_time_ms
                for c in self.competitors
            },
            "avg_execution_time_ms": sum(
                c.execution_time_ms for c in self.competitors
            ) / len(self.competitors),
            "max_execution_time_ms": max(
                c.execution_time_ms for c in self.competitors
            ),
            "competitor_results": {
                c.name: c.to_dict()
                for c in self.competitors
            },
        }
        
        if return_agreement:
            # Group agreements by count
            agreement_by_count = {}
            for entity_agreement in agreement.values():
                count = entity_agreement.agreement_count
                if count not in agreement_by_count:
                    agreement_by_count[count] = []
                agreement_by_count[count].append(entity_agreement.to_dict())
            
            stats["agreement_analysis"] = {
                f"{count}_models": agreements
                for count, agreements in sorted(agreement_by_count.items())
            }
            stats["total_unique_entities_found"] = sum(
                len(entities) for entities in agreement_by_count.values()
            )
        
        return stats


# Convenience function for quick competition
def run_competition(
    text: str,
    competitors: List[LLMCompetitor],
    voting_strategy: str = "weighted",
) -> Tuple[List[ExtractedEntity], Dict]:
    """Quick wrapper to run competition.
    
    Args:
        text: Text to extract entities from
        competitors: List of LLMCompetitor instances
        voting_strategy: Voting strategy to use
        
    Returns:
        Tuple of (final_entities, stats)
    """
    ner = CompetitiveNER(competitors, voting_strategy=voting_strategy)
    return ner.extract_with_competition(text)
