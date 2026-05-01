# @summary
# Configuration types and YAML schema loader for the KG subsystem.
# Exports: NodeTypeDefinition, EdgeTypeDefinition, SchemaDefinition, KGConfig, load_schema
# Deps: dataclasses, typing, yaml, logging
# New retrieval fields on KGConfig: retrieval_edge_types, retrieval_path_patterns,
#   graph_context_token_budget, enable_graph_context_injection (REQ-KG-1200..1206),
#   strict_path_validation (REQ-KG-778)
# @end-summary
"""Configuration and schema types for the KG subsystem.

Provides typed dataclasses for KG configuration (``KGConfig``) and the parsed
YAML schema (``SchemaDefinition``), plus the ``load_schema()`` loader that
validates and deserialises ``config/kg_schema.yaml``.
"""

from __future__ import annotations

import logging
import os

import yaml

from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "NodeTypeDefinition",
    "EdgeTypeDefinition",
    "SchemaDefinition",
    "KGConfig",
    "load_schema",
]

_schema_logger = logging.getLogger("rag.knowledge_graph.schema")

VALID_CATEGORIES = {"structural", "semantic"}
VALID_PHASES = {"phase_1", "phase_1b", "phase_2"}


@dataclass
class NodeTypeDefinition:
    """Schema definition for a single node type.

    Attributes:
        name: Unique identifier for this node type (e.g. ``RTL_Module``).
        description: Human-readable description of what this type represents.
        category: Either ``"structural"`` (parser-extracted) or ``"semantic"`` (LLM-extracted).
        phase: Earliest runtime phase at which this type is active.
        gliner_label: Optional label string passed to GLiNER NER model.
        extraction_hints: Optional free-text hints for LLM extraction prompts.
    """

    name: str
    description: str
    category: str       # "structural" | "semantic"
    phase: str          # "phase_1" | "phase_1b" | "phase_2"
    gliner_label: Optional[str] = None
    extraction_hints: Optional[str] = None


@dataclass
class EdgeTypeDefinition:
    """Schema definition for a single edge (relationship) type.

    Attributes:
        name: Unique identifier for this edge type (e.g. ``instantiates``).
        description: Human-readable description of the relationship.
        category: Either ``"structural"`` or ``"semantic"``.
        phase: Earliest runtime phase at which this edge type is active.
        source_types: Node types that may appear as the subject.
        target_types: Node types that may appear as the object.
    """

    name: str
    description: str
    category: str
    phase: str
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)


@dataclass
class SchemaDefinition:
    """Parsed and validated KG schema from ``config/kg_schema.yaml``.

    Attributes:
        version: Schema version string (e.g. ``"1.0"``).
        description: Free-text description of the schema.
        node_types: Ordered list of node type definitions.
        edge_types: Ordered list of edge type definitions.
    """

    version: str
    description: str
    node_types: List[NodeTypeDefinition] = field(default_factory=list)
    edge_types: List[EdgeTypeDefinition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._node_index: Dict[str, NodeTypeDefinition] = {n.name: n for n in self.node_types}
        self._edge_index: Dict[str, EdgeTypeDefinition] = {e.name: e for e in self.edge_types}

    def active_node_types(self, runtime_phase: str) -> List[NodeTypeDefinition]:
        """Return node types active for *runtime_phase* (phase-ordered subset)."""
        from src.knowledge_graph.common.utils import is_phase_active
        return [n for n in self.node_types if is_phase_active(n.phase, runtime_phase)]

    def active_edge_types(self, runtime_phase: str) -> List[EdgeTypeDefinition]:
        """Return edge types active for *runtime_phase* (phase-ordered subset)."""
        from src.knowledge_graph.common.utils import is_phase_active
        return [e for e in self.edge_types if is_phase_active(e.phase, runtime_phase)]

    def is_valid_node_type(self, type_name: str, runtime_phase: str) -> bool:
        """Return True if *type_name* exists in the schema and is active for the runtime phase."""
        if type_name not in self._node_index:
            return False
        from src.knowledge_graph.common.utils import is_phase_active
        return is_phase_active(self._node_index[type_name].phase, runtime_phase)

    def is_valid_edge_type(self, type_name: str, runtime_phase: str) -> bool:
        """Return True if *type_name* exists in the schema and is active for the runtime phase."""
        if type_name not in self._edge_index:
            return False
        from src.knowledge_graph.common.utils import is_phase_active
        return is_phase_active(self._edge_index[type_name].phase, runtime_phase)


@dataclass
class KGConfig:
    """Runtime configuration for the knowledge graph subsystem.

    All fields have sane defaults suitable for a Phase 1, NetworkX-backed,
    regex-only extraction run.

    Attributes:
        backend: Storage backend name — ``"networkx"`` (default) or ``"neo4j"``.
        schema_path: Path to the KG schema YAML file.
        enable_regex_extractor: Enable lightweight regex-based extraction.
        enable_gliner_extractor: Enable GLiNER NER-based extraction.
        enable_llm_extractor: Enable LLM structured-output extraction.
        enable_sv_parser: Enable tree-sitter-verilog parser extraction.
        entity_description_token_budget: Max tokens for accumulated entity descriptions.
        entity_description_top_k_mentions: Max raw mentions kept per entity.
        max_expansion_depth: Hop depth for KG-based query expansion.
        max_expansion_terms: Maximum additional terms injected by expansion.
        enable_llm_query_fallback: Enable LLM fallback for entity matching.
        llm_fallback_timeout_ms: Wall-clock budget for the LLM fallback in ms.
        enable_global_retrieval: Enable community-summary global retrieval (Phase 2).
        runtime_phase: Active schema phase — ``"phase_1"``, ``"phase_1b"``, or ``"phase_2"``.
        regex_fallback_type: Node type assigned to regex-extracted entities lacking a known type.
        extractor_priority: Ordered list of extractor names (first = highest priority).
    """

    backend: str = "networkx"
    schema_path: str = "config/kg_schema.yaml"
    enable_regex_extractor: bool = True
    enable_gliner_extractor: bool = False
    enable_llm_extractor: bool = False
    enable_sv_parser: bool = False
    entity_description_token_budget: int = 512
    entity_description_top_k_mentions: int = 5
    max_expansion_depth: int = 1
    max_expansion_terms: int = 3
    enable_llm_query_fallback: bool = False
    llm_fallback_timeout_ms: int = 1000
    enable_global_retrieval: bool = False
    runtime_phase: str = "phase_1"
    regex_fallback_type: str = "concept"
    extractor_priority: List[str] = field(
        default_factory=lambda: ["sv_parser", "llm", "gliner", "regex"]
    )
    # Phase 1b: LLM extractor settings
    llm_extraction_model: str = "default"  # LLMProvider model alias
    llm_extraction_prompt_template: Optional[str] = None  # file path override
    llm_extraction_max_retries: int = 1
    llm_extraction_temperature: float = 0.1  # low temp for structured output

    # Phase 2: Community detection
    community_resolution: float = 1.0
    """Leiden resolution parameter. Higher = more, smaller communities."""
    community_min_size: int = 3
    """Minimum entities per community; smaller clusters merge to community_id=-1."""

    # Phase 2: Community summarization
    community_summary_input_max_tokens: int = 4096
    """Max token budget for concatenated entity descriptions in LLM prompt."""
    community_summary_output_max_tokens: int = 512
    """max_tokens passed to LLM call for summary generation."""
    community_summary_temperature: float = 0.2
    """LLM temperature for community summarization calls."""
    community_summary_max_workers: int = 4
    """ThreadPoolExecutor worker count for parallel summarization."""

    # Phase 2: Neo4j backend — defaults consult env so dataclass instances
    # built without explicit overrides still honour deployment configuration.
    neo4j_uri: str = field(
        default_factory=lambda: os.environ.get("RAG_KG_NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_auth_user: str = field(
        default_factory=lambda: os.environ.get("RAG_KG_NEO4J_AUTH_USER", "neo4j")
    )
    neo4j_auth_password: str = field(
        default_factory=lambda: os.environ.get("RAG_KG_NEO4J_AUTH_PASSWORD", ""),
        repr=False,
    )
    neo4j_database: str = field(
        default_factory=lambda: os.environ.get("RAG_KG_NEO4J_DATABASE", "neo4j")
    )

    # Phase 2: Optional parsers
    enable_python_parser: bool = False
    enable_bash_parser: bool = False

    # Phase 2: Graph path (used by sidecar persistence)
    graph_path: Optional[str] = None

    # Phase 3: SV port connectivity
    sv_filelist: str = ""
    """Newline-separated list of .sv/.v file paths for dataflow analysis."""
    sv_top_module: str = ""
    """Top-level module name passed to pyverilog DataflowAnalyzer."""

    # Phase 3: Entity resolution
    enable_entity_resolution: bool = False
    """Enable embedding-based entity deduplication and alias merging."""
    entity_resolution_threshold: float = 0.85
    """Cosine similarity threshold above which two entities are merged."""
    entity_resolution_alias_path: str = "config/kg_aliases.yaml"
    """Path to a YAML file containing manual alias mappings."""

    # Phase 3: Hierarchical Leiden
    community_max_levels: int = 3
    """Maximum recursion depth for hierarchical Leiden community detection."""

    # Retrieval enhancements
    retrieval_edge_types: List[str] = field(default_factory=list)
    """REQ-KG-1200: Edge type whitelist for typed traversal. Empty = untyped."""

    retrieval_path_patterns: List[List[str]] = field(default_factory=list)
    """REQ-KG-1202: Ordered edge type sequences for path pattern matching."""

    graph_context_token_budget: int = 500
    """REQ-KG-1204: Max tokens for graph context block in generation prompt."""

    enable_graph_context_injection: bool = False
    """REQ-KG-1206: Master toggle. False = skip all retrieval enhancements."""

    strict_path_validation: bool = False
    """REQ-KG-778: When True, PatternWarning promoted to KGConfigValidationError."""

    # Phase 2 Retrieval: Community context + operator configurability (REQ-KG-1320..1326)
    community_context_token_budget: int = 200
    """REQ-KG-1320: Independent token budget for community context section. 0 = disabled."""

    graph_context_marker_style: str = "markdown"
    """REQ-KG-1322: Section marker style — 'markdown', 'xml', or 'plain'."""

    max_hop_fanout: int = 50
    """REQ-KG-1324: Max entities explored per hop in path pattern evaluation."""

    def __post_init__(self) -> None:
        if self.community_min_size < 1:
            raise ValueError(f"community_min_size must be >= 1, got {self.community_min_size}")
        if self.community_resolution <= 0:
            raise ValueError(f"community_resolution must be > 0, got {self.community_resolution}")
        if not (0.0 < self.entity_resolution_threshold <= 1.0):
            raise ValueError(
                f"entity_resolution_threshold must be in (0.0, 1.0], "
                f"got {self.entity_resolution_threshold}"
            )
        if self.community_max_levels < 1:
            raise ValueError(
                f"community_max_levels must be >= 1, got {self.community_max_levels}"
            )
        if self.graph_context_token_budget < 0:
            raise ValueError("graph_context_token_budget must be >= 0")
        if self.community_context_token_budget < 0:
            raise ValueError(
                f"community_context_token_budget must be >= 0, got {self.community_context_token_budget}"
            )
        if self.graph_context_marker_style not in {"markdown", "xml", "plain"}:
            raise ValueError(
                f"graph_context_marker_style must be 'markdown', 'xml', or 'plain', "
                f"got '{self.graph_context_marker_style}'"
            )
        if self.max_hop_fanout < 1:
            raise ValueError(f"max_hop_fanout must be >= 1, got {self.max_hop_fanout}")


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------


def load_schema(path: str) -> SchemaDefinition:
    """Load and validate ``config/kg_schema.yaml`` from *path*.

    Algorithm
    ---------
    1. Open the YAML file with ``yaml.safe_load()``. Raise ``FileNotFoundError``
       if the file is missing.
    2. Parse ``node_types``: construct a ``NodeTypeDefinition`` for each entry.
       Required fields: ``name``, ``description``, ``category``, ``phase``.
       Optional: ``gliner_label``, ``extraction_hints``.
    3. Parse ``edge_types``: construct an ``EdgeTypeDefinition`` for each entry.
       Required fields: ``name``, ``description``, ``category``, ``phase``.
       Optional: ``source_types``, ``target_types``.
    4. Raise ``ValueError`` for any of:
       - Duplicate ``name`` values within ``node_types`` or ``edge_types``.
       - ``category`` not in ``{"structural", "semantic"}``.
       - ``phase`` not in ``{"phase_1", "phase_1b", "phase_2"}``.
       - Duplicate ``gliner_label`` values (among non-None labels).
    5. Log warnings for ``gliner_label`` values that collide with another
       type's ``name`` (ambiguous but not an error).
    6. Return ``SchemaDefinition``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML fails any validation check.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"KG schema file not found: {path}")

    version = str(raw.get("version", "1.0"))
    description = str(raw.get("description", ""))

    # --- flatten nested YAML structure if needed ---
    # The YAML may use either:
    #   A) flat list:   node_types: [{name: X, ...}, ...]
    #   B) nested dict: node_types: {structural: {X: {...}, ...}, semantic: {...}}
    # Normalize to flat list form.
    def _flatten_typed_section(section) -> List[dict]:
        if isinstance(section, list):
            return section
        if isinstance(section, dict):
            flat: List[dict] = []
            for category_or_name, value in section.items():
                if isinstance(value, dict) and "description" in value:
                    # Direct entry: {TypeName: {description: ..., ...}}
                    flat.append({"name": category_or_name, **value})
                elif isinstance(value, dict):
                    # Category grouping: {structural: {TypeName: {...}, ...}}
                    for type_name, type_def in value.items():
                        if isinstance(type_def, dict):
                            flat.append({"name": type_name, **type_def})
            return flat
        return []

    # --- node types ---------------------------------------------------------
    node_names_seen: set[str] = set()
    gliner_labels_seen: set[str] = set()
    node_types: List[NodeTypeDefinition] = []

    for entry in _flatten_typed_section(raw.get("node_types", [])):
        name = entry["name"]
        if name in node_names_seen:
            raise ValueError(f"Duplicate node type name: '{name}'")
        node_names_seen.add(name)

        category = entry["category"]
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Node type '{name}' has invalid category '{category}'. "
                f"Must be one of {VALID_CATEGORIES}."
            )

        phase = entry["phase"]
        if phase not in VALID_PHASES:
            raise ValueError(
                f"Node type '{name}' has invalid phase '{phase}'. "
                f"Must be one of {VALID_PHASES}."
            )

        gliner_label: Optional[str] = entry.get("gliner_label")
        if gliner_label is not None:
            if gliner_label in gliner_labels_seen:
                raise ValueError(f"Duplicate gliner_label '{gliner_label}' in node types.")
            gliner_labels_seen.add(gliner_label)

        node_types.append(
            NodeTypeDefinition(
                name=name,
                description=entry["description"],
                category=category,
                phase=phase,
                gliner_label=gliner_label,
                extraction_hints=entry.get("extraction_hints"),
            )
        )

    # --- edge types ---------------------------------------------------------
    edge_names_seen: set[str] = set()
    edge_types: List[EdgeTypeDefinition] = []

    for entry in _flatten_typed_section(raw.get("edge_types", [])):
        name = entry["name"]
        if name in edge_names_seen:
            raise ValueError(f"Duplicate edge type name: '{name}'")
        edge_names_seen.add(name)

        category = entry["category"]
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Edge type '{name}' has invalid category '{category}'. "
                f"Must be one of {VALID_CATEGORIES}."
            )

        phase = entry["phase"]
        if phase not in VALID_PHASES:
            raise ValueError(
                f"Edge type '{name}' has invalid phase '{phase}'. "
                f"Must be one of {VALID_PHASES}."
            )

        edge_types.append(
            EdgeTypeDefinition(
                name=name,
                description=entry["description"],
                category=category,
                phase=phase,
                source_types=entry.get("source_types", []),
                target_types=entry.get("target_types", []),
            )
        )

    # --- cross-warnings: gliner_label collides with another type's name ----
    all_node_names = {n.name for n in node_types}
    for node in node_types:
        if (
            node.gliner_label is not None
            and node.gliner_label != node.name
            and node.gliner_label in all_node_names
        ):
            _schema_logger.warning(
                "Node type '%s' has gliner_label '%s' which collides with "
                "another type's name — this is ambiguous.",
                node.name,
                node.gliner_label,
            )

    return SchemaDefinition(
        version=version,
        description=description,
        node_types=node_types,
        edge_types=edge_types,
    )
