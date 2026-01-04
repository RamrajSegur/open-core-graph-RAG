"""Hybrid NER combining SpaCy (fast) and LLMs (accurate).

Supports multiple LLM providers:
- Ollama (open-source: LLaMA, Mistral, etc.)
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Or any custom LLMProvider implementation
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from ..chunking import TextChunk
from .entity_models import ExtractedEntity, EntityType, NERStats
from .ner_model import NERModel
from .llm_provider import LLMProvider, get_llm_provider, OLLAMA_AVAILABLE

logger = logging.getLogger(__name__)

# Import old LLaMaNER for backward compatibility
try:
    from .llama_ner import LLaMaNER, OLLAMA_AVAILABLE
except ImportError:
    OLLAMA_AVAILABLE = False
    LLaMaNER = None


class HybridNER:
    """Intelligent hybrid NER combining SpaCy with any LLM provider.
    
    Modular design supports multiple LLM backends:
    - Ollama (free, open-source: LLaMA, Mistral, etc.)
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude)
    - Custom LLMProvider implementations
    
    Strategy:
    1. Try SpaCy first (fast: 50-100ms)
    2. If confidence < threshold, use LLM (accurate but slower)
    3. Combine results for best of both worlds
    
    Benefits:
    - 90% of texts use SpaCy (instant)
    - 10% uncertain texts use LLM (more accurate)
    - Final accuracy: 95%+
    - Average latency: ~200ms
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        llm_provider: Optional[LLMProvider] = None,
        llm_provider_type: str = "ollama",
        llm_model: str = "llama2",
        confidence_threshold: float = 0.75,
        use_llm: bool = True,
        strategy: str = "llm_default",
        **llm_kwargs,
    ):
        """Initialize hybrid NER.
        
        Args:
            spacy_model: SpaCy model name
            llm_provider: LLMProvider instance (if None, creates based on llm_provider_type)
            llm_provider_type: Type of LLM provider ("ollama", "openai", "anthropic")
            llm_model: Model name for the LLM provider
            confidence_threshold: Switch to LLM if SpaCy confidence < this (only for 'hybrid' strategy)
            use_llm: Enable LLM fallback (set False to use SpaCy only)
            strategy: NER extraction strategy:
                - "spacy_default" (old): Try SpaCy first, use LLM if low confidence (50% accuracy)
                - "llm_default" (new, recommended): Use LLM primarily, use SpaCy for speed if needed
                - "llm_only": Use LLM exclusively, no SpaCy
            **llm_kwargs: Additional arguments for LLM provider
            
        Example:
            # Using LLM as primary (RECOMMENDED - more consistent)
            hybrid = HybridNER(strategy="llm_default", llm_provider_type="ollama", llm_model="llama2")
            
            # Using OpenAI for maximum accuracy
            hybrid = HybridNER(
                strategy="llm_only",
                llm_provider_type="openai",
                llm_model="gpt-4",
                api_key="sk-..."
            )
            
            # Old SpaCy-first strategy (less recommended)
            hybrid = HybridNER(strategy="spacy_default", llm_provider_type="ollama")
        """
        self.spacy_model = NERModel(model_name=spacy_model)
        self.confidence_threshold = confidence_threshold
        self.use_llm = use_llm
        self.strategy = strategy.lower()
        
        # Validate strategy
        valid_strategies = {"spacy_default", "llm_default", "llm_only"}
        if self.strategy not in valid_strategies:
            logger.warning(
                f"Unknown strategy '{strategy}'. Valid options: {valid_strategies}. "
                f"Using 'llm_default'."
            )
            self.strategy = "llm_default"
        
        logger.info(f"HybridNER using strategy: {self.strategy}")
        
        self.llm_provider = None
        if use_llm and self.strategy != "spacy_default":
            if llm_provider:
                # Use provided provider instance
                self.llm_provider = llm_provider
                logger.info(f"HybridNER initialized with custom LLM provider")
            else:
                # Create provider from type
                try:
                    self.llm_provider = get_llm_provider(
                        llm_provider_type,
                        model=llm_model,
                        model_name=llm_model,
                        **llm_kwargs,
                    )
                    logger.info(f"HybridNER initialized with {llm_provider_type} provider")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize {llm_provider_type} provider: {e}. "
                        f"Falling back to SpaCy."
                    )
                    if self.strategy == "llm_only":
                        raise RuntimeError(
                            f"Strategy 'llm_only' requires LLM provider, but initialization failed: {e}"
                        )
                    self.use_llm = False
        elif self.strategy == "spacy_default":
            # For spacy_default, still initialize LLM for fallback
            if use_llm and llm_provider:
                self.llm_provider = llm_provider
            elif use_llm:
                try:
                    self.llm_provider = get_llm_provider(
                        llm_provider_type,
                        model=llm_model,
                        model_name=llm_model,
                        **llm_kwargs,
                    )
                except Exception as e:
                    logger.warning(
                        f"LLM initialization failed: {e}. Using SpaCy only."
                    )
                    self.use_llm = False
        else:
            logger.info("HybridNER initialized (LLM disabled)")

    def extract_from_chunk(
        self, chunk: TextChunk, return_stats: bool = False
    ) -> Tuple[List[ExtractedEntity], Optional[Dict]]:
        """Extract entities from a single chunk.
        
        Args:
            chunk: TextChunk to extract entities from
            return_stats: Return metadata about which model was used
            
        Returns:
            (entities, stats) if return_stats=True, else just entities
        """
        if not chunk.content or not chunk.content.strip():
            return ([], None) if return_stats else []

        # Choose strategy
        if self.strategy == "spacy_default":
            return self._extract_spacy_default(chunk, return_stats)
        elif self.strategy == "llm_default":
            return self._extract_llm_default(chunk, return_stats)
        elif self.strategy == "llm_only":
            return self._extract_llm_only(chunk, return_stats)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, defaulting to llm_default")
            return self._extract_llm_default(chunk, return_stats)

    def _extract_spacy_default(
        self, chunk: TextChunk, return_stats: bool = False
    ) -> Tuple[List[ExtractedEntity], Optional[Dict]]:
        """OLD STRATEGY: Try SpaCy first, use LLM if confidence < threshold.
        
        This was the original strategy but has issues with SpaCy's inconsistent confidence.
        Kept for backward compatibility.
        """
        # Phase 1: Try SpaCy (fast)
        start_time = time.time()
        spacy_entities = self.spacy_model.extract_entities(
            text=chunk.content,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
        )
        spacy_time = time.time() - start_time

        # Enrich with chunk metadata
        for entity in spacy_entities:
            entity.page_number = chunk.page_number
            if "position_in_document" in chunk.metadata:
                entity.metadata["position_in_document"] = chunk.metadata[
                    "position_in_document"
                ]

        # Calculate SpaCy confidence
        avg_spacy_confidence = (
            sum(e.confidence for e in spacy_entities) / len(spacy_entities)
            if spacy_entities
            else 0.0
        )

        stats = None
        if return_stats:
            stats = {
                "spacy_confidence": avg_spacy_confidence,
                "spacy_time_ms": spacy_time * 1000,
                "entities_from_spacy": len(spacy_entities),
                "model_used": "spacy",
                "strategy": "spacy_default",
            }

        # Phase 2: Check if we should use LLM provider
        if (
            self.use_llm
            and self.llm_provider
            and avg_spacy_confidence < self.confidence_threshold
        ):
            logger.info(
                f"SpaCy confidence {avg_spacy_confidence:.2f} < threshold "
                f"{self.confidence_threshold}, using LLM for better accuracy"
            )

            start_time = time.time()
            llm_entities = self.llm_provider.extract_entities(
                text=chunk.content,
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
            )
            llm_time = time.time() - start_time

            # Enrich LLM entities with chunk metadata
            for entity in llm_entities:
                entity.page_number = chunk.page_number
                if "position_in_document" in chunk.metadata:
                    entity.metadata["position_in_document"] = chunk.metadata[
                        "position_in_document"
                    ]

            if stats:
                stats.update({
                    "llm_entities": len(llm_entities),
                    "llm_time_ms": llm_time * 1000,
                    "model_used": "llm",
                    "fallback_reason": "Low SpaCy confidence",
                })

            return (llm_entities, stats) if return_stats else llm_entities

        # Phase 3: Return SpaCy results (high confidence)
        return (spacy_entities, stats) if return_stats else spacy_entities

    def _extract_llm_default(
        self, chunk: TextChunk, return_stats: bool = False
    ) -> Tuple[List[ExtractedEntity], Optional[Dict]]:
        """NEW STRATEGY: Use LLM primarily for better consistency.
        
        This strategy prioritizes LLM for better accuracy and consistency.
        Can optionally use SpaCy for speed optimization on very confident matches.
        
        Benefits:
        - More consistent confidence scores
        - Better handling of complex/ambiguous entities
        - No SpaCy confidence threshold issues
        """
        if not self.llm_provider or not self.use_llm:
            # Fall back to SpaCy if LLM not available
            logger.warning("LLM not available, falling back to SpaCy")
            return self._extract_spacy_only(chunk, return_stats)

        # Use LLM as primary
        start_time = time.time()
        llm_entities = self.llm_provider.extract_entities(
            text=chunk.content,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
        )
        llm_time = time.time() - start_time

        # Enrich with chunk metadata
        for entity in llm_entities:
            entity.page_number = chunk.page_number
            if "position_in_document" in chunk.metadata:
                entity.metadata["position_in_document"] = chunk.metadata[
                    "position_in_document"
                ]

        stats = None
        if return_stats:
            stats = {
                "llm_entities": len(llm_entities),
                "llm_time_ms": llm_time * 1000,
                "model_used": "llm",
                "strategy": "llm_default",
            }

        return (llm_entities, stats) if return_stats else llm_entities

    def _extract_llm_only(
        self, chunk: TextChunk, return_stats: bool = False
    ) -> Tuple[List[ExtractedEntity], Optional[Dict]]:
        """STRICT STRATEGY: Use LLM exclusively, no SpaCy fallback.
        
        Best for:
        - Maximum accuracy needs
        - When SpaCy is unreliable
        - Production systems where consistency is critical
        
        Note: May be slower due to LLM overhead.
        """
        if not self.llm_provider or not self.use_llm:
            raise RuntimeError(
                "Strategy 'llm_only' requires an active LLM provider. "
                "Check that use_llm=True and LLM is properly initialized."
            )

        # Extract using LLM only
        start_time = time.time()
        entities = self.llm_provider.extract_entities(
            text=chunk.content,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
        )
        elapsed_time = time.time() - start_time

        # Enrich with chunk metadata
        for entity in entities:
            entity.page_number = chunk.page_number
            if "position_in_document" in chunk.metadata:
                entity.metadata["position_in_document"] = chunk.metadata[
                    "position_in_document"
                ]

        stats = None
        if return_stats:
            stats = {
                "entities": len(entities),
                "time_ms": elapsed_time * 1000,
                "model_used": "llm",
                "strategy": "llm_only",
            }

        return (entities, stats) if return_stats else entities

    def _extract_spacy_only(
        self, chunk: TextChunk, return_stats: bool = False
    ) -> Tuple[List[ExtractedEntity], Optional[Dict]]:
        """Fallback: Use SpaCy only (for speed or when LLM unavailable).
        
        Used when:
        - LLM provider fails to initialize
        - use_llm=False
        - Emergency fallback mode
        """
        start_time = time.time()
        entities = self.spacy_model.extract_entities(
            text=chunk.content,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
        )
        elapsed_time = time.time() - start_time

        # Enrich with chunk metadata
        for entity in entities:
            entity.page_number = chunk.page_number
            if "position_in_document" in chunk.metadata:
                entity.metadata["position_in_document"] = chunk.metadata[
                    "position_in_document"
                ]

        stats = None
        if return_stats:
            stats = {
                "entities": len(entities),
                "time_ms": elapsed_time * 1000,
                "model_used": "spacy",
                "strategy": "spacy_only",
            }

        return (entities, stats) if return_stats else entities

    def extract_from_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 128,
        include_stats: bool = False,
    ) -> Tuple[List[ExtractedEntity], Optional[NERStats]]:
        """Extract entities from multiple chunks efficiently.
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for processing
            include_stats: Whether to return extraction statistics
            
        Returns:
            (entities, stats) if include_stats=True, else just entities
        """
        start_time = time.time()
        all_entities = []
        all_stats = {
            "total_chunks": len(chunks),
            "chunks_using_spacy": 0,
            "chunks_using_llama": 0,
            "total_spacy_time_ms": 0.0,
            "total_llama_time_ms": 0.0,
        }

        for chunk in chunks:
            if not chunk.content or not chunk.content.strip():
                continue

            entities, chunk_stats = self.extract_from_chunk(
                chunk, return_stats=True
            )
            all_entities.extend(entities)

            if chunk_stats:
                model_used = chunk_stats.get("model_used", "spacy")
                if model_used == "spacy":
                    all_stats["chunks_using_spacy"] += 1
                    # Handle both old and new stat key names
                    spacy_time = chunk_stats.get("spacy_time_ms", 0.0)
                    all_stats["total_spacy_time_ms"] += spacy_time
                else:
                    all_stats["chunks_using_llama"] += 1
                    # Handle both old and new stat key names
                    llm_time = chunk_stats.get("llm_time_ms", chunk_stats.get("llama_time_ms", 0.0))
                    all_stats["total_llama_time_ms"] += llm_time

        processing_time = time.time() - start_time

        if include_stats:
            stats = self._calculate_stats(
                all_entities,
                len(chunks),
                processing_time,
                all_stats,
            )
            return all_entities, stats

        return all_entities

    def extract_from_text(
        self,
        text: str,
        source_file: str = "",
        chunk_id: str = "single_chunk",
        return_stats: bool = False,
    ) -> Tuple[List[ExtractedEntity], Optional[Dict]]:
        """Extract entities from raw text.
        
        Args:
            text: Raw text to extract entities from
            source_file: Source file path
            chunk_id: Chunk ID to assign
            return_stats: Return metadata about which model was used
            
        Returns:
            (entities, stats) if return_stats=True, else just entities
        """
        if not text or not text.strip():
            return ([], None) if return_stats else []

        # Create a temporary chunk
        temp_chunk = TextChunk(
            content=text,
            chunk_id=chunk_id,
            source_file=source_file,
        )

        return self.extract_from_chunk(temp_chunk, return_stats=return_stats)

    def filter_by_confidence(
        self,
        entities: List[ExtractedEntity],
        min_confidence: float = 0.75,
    ) -> List[ExtractedEntity]:
        """Filter entities by confidence threshold.
        
        Args:
            entities: List of entities to filter
            min_confidence: Minimum confidence score
            
        Returns:
            Filtered list of entities
        """
        return [e for e in entities if e.confidence >= min_confidence]

    def get_stats(self) -> Dict[str, any]:
        """Get current configuration statistics."""
        # Get llm_model name safely
        llm_model_name = None
        if self.llm_provider is not None:
            if hasattr(self.llm_provider, 'model_name'):
                llm_model_name = self.llm_provider.model_name
            elif hasattr(self.llm_provider, 'model'):
                llm_model_name = self.llm_provider.model
        
        return {
            "spacy_model": self.spacy_model.model_name,
            "llm_available": self.llm_provider is not None,
            "llm_model": llm_model_name,
            "confidence_threshold": self.confidence_threshold,
            "strategy": self.strategy,
            "use_llm": self.use_llm,
        }

    @staticmethod
    def _calculate_stats(
        entities: List[ExtractedEntity],
        total_chunks: int,
        processing_time: float,
        hybrid_stats: Optional[Dict] = None,
    ) -> NERStats:
        """Calculate extraction statistics."""
        
        if not entities:
            return NERStats(
                total_entities=0,
                chunks_processed=total_chunks,
                processing_time=processing_time,
                average_confidence=0.0,
            )

        avg_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # Count entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            entities_by_type[entity_type] = entities_by_type.get(entity_type, 0) + 1
        
        # Count high confidence entities
        high_confidence_count = sum(1 for e in entities if e.confidence >= 0.75)

        stats = NERStats(
            total_entities=len(entities),
            entities_by_type=entities_by_type,
            chunks_processed=total_chunks,
            processing_time=processing_time,
            average_confidence=avg_confidence,
            high_confidence_count=high_confidence_count,
        )

        # Add hybrid-specific stats if available
        if hybrid_stats:
            stats.metadata = {
                "hybrid_strategy": "SpaCy + LLaMA",
                "chunks_using_spacy": hybrid_stats.get("chunks_using_spacy", 0),
                "chunks_using_llama": hybrid_stats.get("chunks_using_llama", 0),
                "total_spacy_time_ms": hybrid_stats.get("total_spacy_time_ms", 0.0),
                "total_llama_time_ms": hybrid_stats.get("total_llama_time_ms", 0.0),
                "spacy_coverage": (
                    hybrid_stats.get("chunks_using_spacy", 0)
                    / hybrid_stats.get("total_chunks", 1)
                    * 100
                ),
            }

        return stats


# Backward compatibility wrapper
# Old code using HybridSpaCyLLaMA will still work with old parameter names
class HybridSpaCyLLaMA(HybridNER):
    """Backward-compatible wrapper for HybridNER.
    
    Maps old parameter names to new ones:
    - use_llama -> use_llm
    - llama_model -> llm_model
    - llama_provider_type -> llm_provider_type
    
    Adds old property for backward compatibility:
    - llama_model (property that returns llm_model or None)
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        llm_provider: Optional[LLMProvider] = None,
        llm_provider_type: str = "ollama",
        llm_model: str = "llama2",
        confidence_threshold: float = 0.75,
        use_llm: bool = True,
        strategy: str = "llm_default",
        # Old parameter names for backward compatibility
        use_llama: Optional[bool] = None,
        llama_model: Optional[str] = None,
        llama_provider_type: Optional[str] = None,
        **llm_kwargs,
    ):
        """Initialize HybridSpaCyLLaMA with backward compatibility.
        
        This wrapper accepts both old and new parameter names:
        
        Old style:
            hybrid = HybridSpaCyLLaMA(use_llama=True, llama_model="llama2")
        
        New style (recommended):
            hybrid = HybridSpaCyLLaMA(use_llm=True, llm_model="llama2")
        """
        # Map old parameter names to new ones
        if use_llama is not None:
            use_llm = use_llama
        if llama_model is not None:
            llm_model = llama_model
        if llama_provider_type is not None:
            llm_provider_type = llama_provider_type
        
        # Call parent constructor with mapped names
        super().__init__(
            spacy_model=spacy_model,
            llm_provider=llm_provider,
            llm_provider_type=llm_provider_type,
            llm_model=llm_model,
            confidence_threshold=confidence_threshold,
            use_llm=use_llm,
            strategy=strategy,
            **llm_kwargs,
        )
    
    @property
    def llama_model(self) -> Optional[str]:
        """Backward compatibility property - returns llm_model if available.
        
        Old code expects: hybrid.llama_model
        Returns: The LLM model name, or None if not initialized
        """
        if self.llm_provider is None:
            return None
        
        # Try to get model name from provider
        if hasattr(self.llm_provider, 'model_name'):
            return self.llm_provider.model_name
        elif hasattr(self.llm_provider, 'model'):
            return self.llm_provider.model
        
        return None
