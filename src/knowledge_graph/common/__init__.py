"""Common contracts, configuration types, and shared helpers for the KG subsystem."""

# --- Auto-generated re-exports (fix_encapsulation.py) ---
from src.knowledge_graph.common.schemas import (
    Entity,
    EntityDescription,
    ExtractionResult,
    Triple,
)
from src.knowledge_graph.common.types import (
    KGConfig,
    SchemaDefinition,
    load_schema,
)
from src.knowledge_graph.common.utils import derive_gliner_labels
from src.knowledge_graph.common.validation import (
    KGConfigValidationError,
    PatternWarning,
    validate_edge_types,
    validate_path_patterns,
)
