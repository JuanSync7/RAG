# @summary
# Public API for the knowledge graph subsystem: config-driven backend dispatcher
# and convenience functions. Re-exports common schemas for callers that need them.
# Exports: get_graph_backend, get_query_expander, export_obsidian,
#          GraphStorageBackend, GraphQueryExpander, KGConfig,
#          Entity, Triple, ExtractionResult, EntityDescription,
#          CommunitySummary, CommunityDiff
# Deps: config.settings, src.knowledge_graph.backend,
#       src.knowledge_graph.backends.*, src.knowledge_graph.query.*,
#       src.knowledge_graph.community.*, src.knowledge_graph.export.obsidian
# Retrieval settings (REQ-KG-1200..1206) loaded from config.settings when present.
# Retrieval config validated at startup via common.validation (REQ-KG-1208).
# get_query_expander passes full KGConfig to GraphQueryExpander (REQ-KG-762) so
# typed traversal dispatch is available.
# @end-summary
"""Public API for the knowledge graph subsystem.

The retrieval and ingestion pipelines import only from this module.
Backend selection is controlled by configuration — changing the config
is all that is needed to swap graph storage implementations.

Dispatcher pattern:
    ``get_graph_backend()`` is a lazy singleton that constructs the
    configured backend on first call.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.common import (
    Entity,
    EntityDescription,
    ExtractionResult,
    Triple,
)
from src.knowledge_graph.common import KGConfig

logger = logging.getLogger("rag.knowledge_graph")

# Process-wide singletons
_graph_backend: Optional[GraphStorageBackend] = None
_kg_config: Optional[KGConfig] = None


def _build_kg_config() -> KGConfig:
    """Build KG config from environment variables."""
    global _kg_config
    if _kg_config is not None:
        return _kg_config
    try:
        from config.settings import (
            RAG_KG_RUNTIME_PHASE,
            RAG_KG_MAX_EXPANSION_DEPTH,
            RAG_KG_MAX_EXPANSION_TERMS,
            RAG_KG_DESCRIPTION_TOKEN_BUDGET,
            RAG_KG_GRAPH_PATH,
            RAG_KG_BACKEND,
        )
        _kg_config = KGConfig(
            backend=RAG_KG_BACKEND,
            runtime_phase=RAG_KG_RUNTIME_PHASE,
            max_expansion_depth=RAG_KG_MAX_EXPANSION_DEPTH,
            max_expansion_terms=RAG_KG_MAX_EXPANSION_TERMS,
            entity_description_token_budget=RAG_KG_DESCRIPTION_TOKEN_BUDGET,
            graph_path=str(RAG_KG_GRAPH_PATH),
        )
        # Load Phase 2 settings if available
        try:
            from config.settings import (
                RAG_KG_ENABLE_GLOBAL_RETRIEVAL,
                RAG_KG_COMMUNITY_RESOLUTION,
                RAG_KG_COMMUNITY_MIN_SIZE,
                RAG_KG_COMMUNITY_SUMMARY_INPUT_MAX_TOKENS,
                RAG_KG_COMMUNITY_SUMMARY_OUTPUT_MAX_TOKENS,
                RAG_KG_COMMUNITY_SUMMARY_TEMPERATURE,
                RAG_KG_COMMUNITY_SUMMARY_MAX_WORKERS,
                RAG_KG_NEO4J_URI,
                RAG_KG_NEO4J_AUTH_USER,
                RAG_KG_NEO4J_AUTH_PASSWORD,
                RAG_KG_NEO4J_DATABASE,
            )
            _kg_config.enable_global_retrieval = RAG_KG_ENABLE_GLOBAL_RETRIEVAL
            _kg_config.community_resolution = RAG_KG_COMMUNITY_RESOLUTION
            _kg_config.community_min_size = RAG_KG_COMMUNITY_MIN_SIZE
            _kg_config.community_summary_input_max_tokens = RAG_KG_COMMUNITY_SUMMARY_INPUT_MAX_TOKENS
            _kg_config.community_summary_output_max_tokens = RAG_KG_COMMUNITY_SUMMARY_OUTPUT_MAX_TOKENS
            _kg_config.community_summary_temperature = RAG_KG_COMMUNITY_SUMMARY_TEMPERATURE
            _kg_config.community_summary_max_workers = RAG_KG_COMMUNITY_SUMMARY_MAX_WORKERS
            _kg_config.neo4j_uri = RAG_KG_NEO4J_URI
            _kg_config.neo4j_auth_user = RAG_KG_NEO4J_AUTH_USER
            _kg_config.neo4j_auth_password = RAG_KG_NEO4J_AUTH_PASSWORD
            _kg_config.neo4j_database = RAG_KG_NEO4J_DATABASE
        except ImportError:
            pass  # Phase 2 settings not yet in config
        try:
            from config.settings import (
                RAG_KG_ENABLE_PYTHON_PARSER,
                RAG_KG_ENABLE_BASH_PARSER,
            )
            _kg_config.enable_python_parser = RAG_KG_ENABLE_PYTHON_PARSER
            _kg_config.enable_bash_parser = RAG_KG_ENABLE_BASH_PARSER
        except ImportError:
            pass
        # Load Phase 3 settings if available
        try:
            from config.settings import (
                RAG_KG_SV_FILELIST,
                RAG_KG_SV_TOP_MODULE,
                RAG_KG_ENABLE_ENTITY_RESOLUTION,
                RAG_KG_ENTITY_RESOLUTION_THRESHOLD,
                RAG_KG_ENTITY_RESOLUTION_ALIAS_PATH,
                RAG_KG_COMMUNITY_MAX_LEVELS,
            )
            _kg_config.sv_filelist = RAG_KG_SV_FILELIST
            _kg_config.sv_top_module = RAG_KG_SV_TOP_MODULE
            _kg_config.enable_entity_resolution = RAG_KG_ENABLE_ENTITY_RESOLUTION
            _kg_config.entity_resolution_threshold = RAG_KG_ENTITY_RESOLUTION_THRESHOLD
            _kg_config.entity_resolution_alias_path = RAG_KG_ENTITY_RESOLUTION_ALIAS_PATH
            _kg_config.community_max_levels = RAG_KG_COMMUNITY_MAX_LEVELS
        except ImportError:
            pass  # Phase 3 settings not yet in config
        # Load Retrieval settings if available
        try:
            from config.settings import (
                RAG_KG_RETRIEVAL_EDGE_TYPES,
                RAG_KG_RETRIEVAL_PATH_PATTERNS,
                RAG_KG_GRAPH_CONTEXT_TOKEN_BUDGET,
                RAG_KG_ENABLE_GRAPH_CONTEXT_INJECTION,
                RAG_KG_COMMUNITY_CONTEXT_TOKEN_BUDGET,
                RAG_KG_GRAPH_CONTEXT_MARKER_STYLE,
                RAG_KG_MAX_HOP_FANOUT,
            )
            _kg_config.retrieval_edge_types = RAG_KG_RETRIEVAL_EDGE_TYPES
            _kg_config.retrieval_path_patterns = RAG_KG_RETRIEVAL_PATH_PATTERNS
            _kg_config.graph_context_token_budget = RAG_KG_GRAPH_CONTEXT_TOKEN_BUDGET
            _kg_config.enable_graph_context_injection = RAG_KG_ENABLE_GRAPH_CONTEXT_INJECTION
            _kg_config.community_context_token_budget = RAG_KG_COMMUNITY_CONTEXT_TOKEN_BUDGET
            _kg_config.graph_context_marker_style = RAG_KG_GRAPH_CONTEXT_MARKER_STYLE
            _kg_config.max_hop_fanout = RAG_KG_MAX_HOP_FANOUT
        except ImportError:
            pass  # Retrieval settings not yet in config
        # Validate retrieval config at startup (REQ-KG-1208)
        if _kg_config.enable_graph_context_injection:
            try:
                from src.knowledge_graph.common.validation import (
                    validate_edge_types,
                    validate_path_patterns,
                )
                if _kg_config.retrieval_edge_types:
                    validate_edge_types(
                        _kg_config.retrieval_edge_types,
                        _kg_config.schema_path,
                    )
                if _kg_config.retrieval_path_patterns:
                    warnings = validate_path_patterns(
                        _kg_config.retrieval_path_patterns,
                        _kg_config.schema_path,
                    )
                    for w in warnings:
                        logger.warning(
                            "Path pattern validation: %s", w.message
                        )
            except ImportError:
                pass  # Validation module not available
    except ImportError:
        _kg_config = KGConfig()
    return _kg_config


def _load_schema(config: KGConfig):
    """Load and validate the YAML schema."""
    from src.knowledge_graph.common import load_schema

    return load_schema(config.schema_path)


def get_graph_backend(config: Optional[KGConfig] = None) -> GraphStorageBackend:
    """Return the process-wide graph backend singleton.

    Constructs the backend on first call based on configuration.
    Loads the graph from disk if the graph file exists.

    Args:
        config: Optional explicit config. If ``None``, builds from env.

    Returns:
        The active ``GraphStorageBackend`` instance.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    global _graph_backend
    if _graph_backend is not None:
        return _graph_backend

    if config is None:
        config = _build_kg_config()

    backend_name = config.backend

    if backend_name == "networkx":
        from src.knowledge_graph.backends import NetworkXBackend

        _graph_backend = NetworkXBackend()
    elif backend_name == "neo4j":
        from src.knowledge_graph.backends import Neo4jBackend

        _graph_backend = Neo4jBackend(config=config)
    else:
        raise ValueError(
            f"Unknown KG backend: {backend_name!r}. "
            "Valid values: 'networkx', 'neo4j'."
        )

    # Load existing graph from disk if available
    graph_path = config.graph_path
    if graph_path and Path(graph_path).exists():
        try:
            _graph_backend.load(Path(graph_path))
            stats = _graph_backend.stats()
            logger.info(
                "Loaded KG from %s: %d nodes, %d edges",
                graph_path,
                stats.get("nodes", 0),
                stats.get("edges", 0),
            )
        except Exception as exc:
            logger.warning("Failed to load KG from %s: %s", graph_path, exc)

    return _graph_backend


def get_query_expander(
    backend: Optional[GraphStorageBackend] = None,
    config: Optional[KGConfig] = None,
):
    """Build a query expander from the given (or default) backend.

    Args:
        backend: Explicit backend. If ``None``, uses ``get_graph_backend()``.
        config: Explicit config for depth/term limits.

    Returns:
        A :class:`GraphQueryExpander` instance.
    """
    from src.knowledge_graph.query import GraphQueryExpander

    if backend is None:
        backend = get_graph_backend()
    if config is None:
        config = _build_kg_config()

    # Phase 2: Community-aware expansion
    community_detector = None
    if config.enable_global_retrieval:
        try:
            from src.knowledge_graph.community import CommunityDetector
            from src.knowledge_graph.community import CommunitySummarizer

            detector = CommunityDetector(
                backend=backend,
                config=config,
                graph_path=getattr(config, "graph_path", None),
            )
            # Run detection + summarization if not already loaded from sidecar
            if not detector.is_ready:
                communities = detector.detect()
                if communities:
                    summarizer = CommunitySummarizer(config=config)
                    summaries = summarizer.summarize_all(communities, backend)
                    detector.summaries = summaries
                    detector.save_sidecar()
            community_detector = detector
        except Exception as exc:
            logger.warning("Failed to initialize community detector: %s", exc)

    return GraphQueryExpander(
        backend=backend,
        max_depth=config.max_expansion_depth,
        max_terms=config.max_expansion_terms,
        community_detector=community_detector,
        enable_global_retrieval=config.enable_global_retrieval,
        config=config,
    )


def run_post_ingestion_steps(
    backend: Optional[GraphStorageBackend] = None,
    config: Optional[KGConfig] = None,
    update_mode: bool = False,
) -> None:
    """Execute post-ingestion batch steps.

    Ordering:
        1. SV connectivity batch (if sv_filelist configured)
        2. Entity resolution (if enabled)
        3. Community detection (hierarchical Leiden)

    Args:
        backend: Populated graph backend. If None, uses singleton.
        config: KG runtime configuration. If None, builds from env.
        update_mode: Whether this is an incremental update run.
    """
    if backend is None:
        backend = get_graph_backend()
    if config is None:
        config = _build_kg_config()

    _pipeline_t0 = time.monotonic()
    logger.info("KG post-ingestion steps starting (update_mode=%s)", update_mode)

    # Step 1: SV connectivity batch
    if config.sv_filelist:
        try:
            from src.knowledge_graph.extraction import (
                SVConnectivityAnalyzer,
                SV_CONNECTIVITY_SOURCE,
            )

            filelist_path = Path(config.sv_filelist)
            if not filelist_path.is_file():
                logger.warning(
                    "sv_filelist configured but file not found: %s — skipping",
                    config.sv_filelist,
                )
            else:
                if update_mode:
                    stats = backend.remove_by_source(SV_CONNECTIVITY_SOURCE)
                    logger.info("Removed previous SV connectivity: %s", stats)
                analyzer = SVConnectivityAnalyzer(
                    filelist_path=config.sv_filelist,
                    backend=backend,
                    top_module=config.sv_top_module or None,
                )
                triples = analyzer.analyze()
                if triples:
                    backend.upsert_triples(triples)
                    logger.info(
                        "SV connectivity: upserted %d connects_to triples",
                        len(triples),
                    )
        except Exception as exc:
            logger.warning("SV connectivity batch failed: %s", exc)

    # Step 2: Entity resolution
    if config.enable_entity_resolution:
        _step_t0 = time.monotonic()
        logger.info("KG step 2/3: entity resolution starting")
        try:
            from src.knowledge_graph.resolution import EntityResolver

            resolver = EntityResolver(backend=backend, config=config)
            report = resolver.resolve()
            logger.info(
                "KG step 2/3: entity resolution complete — merged=%d elapsed=%.1fs",
                report.total_merged, time.monotonic() - _step_t0,
            )
        except Exception as exc:
            logger.warning("Entity resolution failed: %s", exc)

    # Step 3: Community detection (hierarchical if max_levels > 1)
    if config.enable_global_retrieval:
        _step_t0 = time.monotonic()
        logger.info("KG step 3/3: community detection starting")
        try:
            from src.knowledge_graph.community import CommunityDetector
            from src.knowledge_graph.community import CommunitySummarizer

            detector = CommunityDetector(
                backend=backend, config=config, graph_path=config.graph_path,
            )
            if config.community_max_levels > 1:
                hierarchy = detector.detect_hierarchical()
            else:
                detector.detect()
            if detector._detection_complete:
                summarizer = CommunitySummarizer(config=config)
                communities = detector._communities
                if communities:
                    summaries = summarizer.summarize_all(communities, backend)
                    detector.summaries = summaries
                    detector.save_sidecar()
            logger.info(
                "KG step 3/3: community detection complete — elapsed=%.1fs",
                time.monotonic() - _step_t0,
            )
        except Exception as exc:
            logger.warning("Community detection failed: %s", exc)

    logger.info(
        "KG post-ingestion steps complete — total elapsed=%.1fs",
        time.monotonic() - _pipeline_t0,
    )


def reset_singletons() -> None:
    """Reset cached singletons. Used in tests."""
    global _graph_backend, _kg_config
    _graph_backend = None
    _kg_config = None


# Re-export for convenience
from src.knowledge_graph.export import export_obsidian
from src.knowledge_graph.export import export_html
from src.knowledge_graph.query import GraphQueryExpander
from src.knowledge_graph.community import (
    CommunityDiff,
    CommunitySummary,
)

__all__ = [
    # Dispatcher functions
    "get_graph_backend",
    "get_query_expander",
    "run_post_ingestion_steps",
    "reset_singletons",
    # Re-exported types
    "GraphStorageBackend",
    "GraphQueryExpander",
    "KGConfig",
    "Entity",
    "Triple",
    "ExtractionResult",
    "EntityDescription",
    "CommunitySummary",
    "CommunityDiff",
    # Utilities
    "export_obsidian",
    "export_html",
]
