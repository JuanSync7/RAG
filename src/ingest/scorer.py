#!/usr/bin/env python3
"""Static-analysis scorer for ingestion pipeline quality.

Checks 17 pipeline node files + pipeline-wide criteria across three
dimensions: robustness (logging), speed, and code quality.

Score = passed_criteria / total_criteria.  Higher is better.

Correctness guard: runs ``pytest tests/ingest/ -x -q --tb=short`` after
the static checks.  If any test fails, the iteration is a crash regardless
of the static score.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent  # src/ingest/
PROJECT_ROOT = ROOT.parent.parent  # RagWeave/
SUPPORT_DIR = ROOT / "support"

# ── Node files to check ────────────────────────────────────────────────
PHASE1_NODES = [
    ROOT / "doc_processing/nodes/document_ingestion.py",
    ROOT / "doc_processing/nodes/structure_detection.py",
    ROOT / "doc_processing/nodes/multimodal_processing.py",
    ROOT / "doc_processing/nodes/text_cleaning.py",
    ROOT / "doc_processing/nodes/document_refactoring.py",
]

PHASE2_NODES = [
    ROOT / "embedding/nodes/document_storage_node.py",
    ROOT / "embedding/nodes/chunking.py",
    ROOT / "embedding/nodes/vlm_enrichment.py",
    ROOT / "embedding/nodes/chunk_enrichment.py",
    ROOT / "embedding/nodes/metadata_generation.py",
    ROOT / "embedding/nodes/cross_reference_extraction.py",
    ROOT / "embedding/nodes/knowledge_graph_extraction.py",
    ROOT / "embedding/nodes/quality_validation.py",
    ROOT / "embedding/nodes/cross_document_dedup.py",
    ROOT / "embedding/nodes/embedding_storage.py",
    ROOT / "embedding/nodes/visual_embedding.py",
    ROOT / "embedding/nodes/knowledge_graph_storage.py",
]

ALL_NODES = PHASE1_NODES + PHASE2_NODES

# ── Pipeline-wide files ────────────────────────────────────────────────
LLM_HELPER = ROOT / "support/llm.py"
UTILS_FILE = ROOT / "common/utils.py"
SHARED_FILE = ROOT / "common/shared.py"
IMPL_FILE = ROOT / "impl.py"
DOC_INGESTION = ROOT / "doc_processing/nodes/document_ingestion.py"
EMBEDDING_IMPL = ROOT / "embedding/impl.py"
METADATA_GEN = ROOT / "embedding/nodes/metadata_generation.py"
EMBEDDING_STORAGE = ROOT / "embedding/nodes/embedding_storage.py"
KG_EXTRACTION = ROOT / "embedding/nodes/knowledge_graph_extraction.py"
KG_STORAGE = ROOT / "embedding/nodes/knowledge_graph_storage.py"
DOC_STORAGE = ROOT / "embedding/nodes/document_storage_node.py"
CHUNKING = ROOT / "embedding/nodes/chunking.py"
QUALITY_VALIDATION = ROOT / "embedding/nodes/quality_validation.py"
VISUAL_EMBEDDING = ROOT / "embedding/nodes/visual_embedding.py"
MULTIMODAL = ROOT / "doc_processing/nodes/multimodal_processing.py"
DOCUMENT_PY = SUPPORT_DIR / "document.py"
DEDUP_UTILS = ROOT / "embedding/common/dedup_utils.py"
TYPES_FILE = ROOT / "common/types.py"


class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Per-node checks ───────────────────────────────────────────────────

def check_logger_naming(path: Path) -> CheckResult:
    """Node uses logging.getLogger('rag.ingest...') — not __name__."""
    src = _read(path)
    node_name = path.stem
    # Pass: has getLogger("rag.ingest.  (with a rag.ingest prefix)
    has_named = bool(re.search(r'getLogger\(\s*["\']rag\.ingest\.', src))
    has_dunder = bool(re.search(r'getLogger\(\s*__name__\s*\)', src))
    if has_named and not has_dunder:
        return CheckResult(f"{node_name}:logger_naming", True, "uses rag.ingest.* hierarchy")
    if has_dunder:
        return CheckResult(f"{node_name}:logger_naming", False, "uses __name__ — not filterable under rag.ingest.*")
    return CheckResult(f"{node_name}:logger_naming", False, "no logger defined")


def check_info_logging(path: Path) -> CheckResult:
    """Node has at least one logger.info() call for stage visibility."""
    src = _read(path)
    node_name = path.stem
    if re.search(r'logger\.info\(', src):
        return CheckResult(f"{node_name}:info_logging", True, "has INFO-level logging")
    return CheckResult(f"{node_name}:info_logging", False, "no logger.info() — stage is silent at INFO level")


def check_debug_timing(path: Path) -> CheckResult:
    """Node measures duration via time.monotonic() and logs at DEBUG."""
    src = _read(path)
    node_name = path.stem
    has_monotonic = "time.monotonic()" in src or "time.time()" in src
    has_debug = bool(re.search(r'logger\.debug\(', src))
    if has_monotonic and has_debug:
        return CheckResult(f"{node_name}:debug_timing", True, "has timing + DEBUG logging")
    parts = []
    if not has_monotonic:
        parts.append("no time.monotonic()")
    if not has_debug:
        parts.append("no logger.debug()")
    return CheckResult(f"{node_name}:debug_timing", False, " and ".join(parts))


# ── Pipeline-wide checks ─────────────────────────────────────────────

def check_llm_failure_logging() -> CheckResult:
    """LLM helper logs failures at WARNING or higher (not just DEBUG)."""
    src = _read(LLM_HELPER)
    # Check logger naming first
    has_named = bool(re.search(r'getLogger\(\s*["\']rag\.ingest\.', src))
    # Check that the except block has WARNING+ logging
    has_warning_in_except = bool(re.search(r'logger\.(?:warning|error|exception)\(', src))
    if has_named and has_warning_in_except:
        return CheckResult("llm:failure_logging", True, "uses rag.ingest.* logger and WARNING+ for failures")
    parts = []
    if not has_named:
        parts.append("logger uses __name__ not rag.ingest.*")
    if not has_warning_in_except:
        parts.append("failures logged at DEBUG only — invisible at INFO level")
    return CheckResult("llm:failure_logging", False, "; ".join(parts))


def check_streaming_hash() -> CheckResult:
    """sha256_path uses streaming (chunked) reads instead of read_bytes()."""
    src = _read(UTILS_FILE)
    # Look for .update( pattern inside sha256_path function
    # Bad: hashlib.sha256(path.read_bytes())
    # Good: h = hashlib.sha256(); for chunk in ...: h.update(chunk)
    has_read_bytes = bool(re.search(r'sha256\(.*\.read_bytes\(\)', src))
    has_update = bool(re.search(r'\.update\(', src))
    if has_update and not has_read_bytes:
        return CheckResult("utils:streaming_hash", True, "sha256_path uses streaming reads")
    return CheckResult("utils:streaming_hash", False, "sha256_path reads entire file into memory")


def check_unique_logger_names() -> CheckResult:
    """shared.py and utils.py use distinct logger names."""
    shared_src = _read(SHARED_FILE)
    utils_src = _read(UTILS_FILE)
    shared_names = re.findall(r'getLogger\(\s*["\']([^"\']+)', shared_src)
    utils_names = re.findall(r'getLogger\(\s*["\']([^"\']+)', utils_src)
    overlap = set(shared_names) & set(utils_names)
    if not overlap:
        return CheckResult("common:unique_loggers", True, "shared.py and utils.py have distinct logger names")
    return CheckResult("common:unique_loggers", False, f"shared logger name collision: {overlap}")


# ── Speed checks ─────────────────────────────────────────────────────

def check_single_file_read() -> CheckResult:
    """document_ingestion_node reads source file only once (no double read)."""
    src = _read(DOC_INGESTION)
    # Bad: calls both read_text_with_fallbacks AND sha256_path (two file reads)
    has_read_text = "read_text_with_fallbacks" in src
    has_sha256_path = "sha256_path" in src
    if has_read_text and has_sha256_path:
        return CheckResult("speed:single_file_read", False,
                           "reads file twice via read_text_with_fallbacks + sha256_path")
    return CheckResult("speed:single_file_read", True, "file read in single pass")


def check_retry_delay() -> CheckResult:
    """Embedding retry delay is <=0.5s to limit event-loop blocking."""
    src = _read(EMBEDDING_STORAGE)
    match = re.search(r'_BATCH_RETRY_DELAY\s*=\s*([\d.]+)', src)
    if match:
        delay = float(match.group(1))
        if delay <= 0.5:
            return CheckResult("speed:retry_delay", True, f"retry delay={delay}s (<=0.5s)")
        return CheckResult("speed:retry_delay", False, f"retry delay={delay}s blocks event loop")
    return CheckResult("speed:retry_delay", False, "could not find _BATCH_RETRY_DELAY")


def check_paragraph_early_exit() -> CheckResult:
    """_best_paragraph_span has early exit for high-confidence matches."""
    src = _read(SHARED_FILE)
    # Look for a break statement inside _best_paragraph_span
    in_func = False
    for line in src.split("\n"):
        if "def _best_paragraph_span" in line:
            in_func = True
        elif in_func and line and not line[0].isspace() and "def " in line:
            in_func = False
        if in_func and "break" in line:
            return CheckResult("speed:paragraph_early_exit", True,
                               "early exit on high-confidence match")
    return CheckResult("speed:paragraph_early_exit", False,
                       "no early exit — scans all paragraphs even after high-confidence match")


def check_manifest_write_frequency() -> CheckResult:
    """Manifest is not written after every skipped document."""
    src = _read(IMPL_FILE)
    # Count save_manifest calls in the ingest_directory function
    # Bad: save after skip AND save after success AND save at end = 3 calls
    # Acceptable: save after success + save at end, or periodic + end
    count = src.count("save_manifest(manifest)")
    if count <= 2:
        return CheckResult("speed:manifest_writes", True,
                           f"{count} save_manifest calls (reduced from 3)")
    return CheckResult("speed:manifest_writes", False,
                       f"{count} save_manifest calls — per-doc writes cause unnecessary I/O")


def check_kg_extractor_reuse() -> CheckResult:
    """KG extraction reuses EntityExtractor instead of creating per-document."""
    src = _read(KG_EXTRACTION)
    # Bad: EntityExtractor() called inside the node function
    # Good: pulled from state["runtime"] or passed in
    if "EntityExtractor()" in src:
        # Check if it's inside the function (not at module level)
        in_func = False
        for line in src.split("\n"):
            if "def knowledge_graph_extraction_node" in line:
                in_func = True
            if in_func and "EntityExtractor()" in line:
                return CheckResult("speed:kg_extractor_reuse", False,
                                   "creates new EntityExtractor() per document invocation")
        return CheckResult("speed:kg_extractor_reuse", True,
                           "EntityExtractor at module level")
    return CheckResult("speed:kg_extractor_reuse", True,
                       "no per-invocation EntityExtractor construction")


# ── Code quality checks ──────────────────────────────────────────────

def check_metadata_state_contract() -> CheckResult:
    """metadata_generation_node returns chunks in state update dict."""
    src = _read(METADATA_GEN)
    # Check that the return dict includes "chunks"
    # Look for "chunks": in a return statement context
    if re.search(r'"chunks"\s*:', src):
        return CheckResult("quality:metadata_state_contract", True,
                           "returns chunks in state update (explicit contract)")
    return CheckResult("quality:metadata_state_contract", False,
                       "mutates chunks via side-effect without returning them")


def check_no_dead_aliases() -> CheckResult:
    """No dead backward-compat aliases in llm.py."""
    src = _read(LLM_HELPER)
    if "_ollama_json" in src:
        return CheckResult("quality:no_dead_aliases", False,
                           "_ollama_json alias is dead code")
    return CheckResult("quality:no_dead_aliases", True, "no dead aliases")


def check_deprecated_params() -> CheckResult:
    """Dead parameters have deprecation warnings."""
    src = _read(EMBEDDING_IMPL)
    has_dead_param = "docling_document" in src
    has_warning = "DeprecationWarning" in src or "warnings.warn" in src
    if not has_dead_param:
        return CheckResult("quality:deprecated_params", True,
                           "no dead parameters")
    if has_dead_param and has_warning:
        return CheckResult("quality:deprecated_params", True,
                           "dead parameter has deprecation warning")
    return CheckResult("quality:deprecated_params", False,
                       "docling_document param is dead but has no deprecation warning")


# ── Robustness Phase 2 checks ────────────────────────────────────────

# Nodes that have except blocks returning errors to state without logging
_ERROR_PATH_NODES = [
    (KG_EXTRACTION, "knowledge_graph_extraction"),
    (KG_STORAGE, "knowledge_graph_storage"),
    (DOC_STORAGE, "document_storage"),
    (CHUNKING, "chunking"),
    (EMBEDDING_STORAGE, "embedding_storage"),
]


def check_error_path_logging(path: Path, node_name: str) -> CheckResult:
    """Node logs errors (logger.error/exception) in except blocks, not just state."""
    src = _read(path)
    # Check if there's an except block AND a logger.error/exception call
    has_except = "except Exception" in src or "except (" in src
    has_error_log = bool(re.search(r'logger\.(?:error|exception)\(', src))
    if not has_except:
        return CheckResult(f"robust:{node_name}_error_log", True, "no except block (N/A)")
    if has_error_log:
        return CheckResult(f"robust:{node_name}_error_log", True,
                           "logs errors in except blocks")
    return CheckResult(f"robust:{node_name}_error_log", False,
                       "except block returns error to state without logging — traceback lost")


def check_visual_monotonic() -> CheckResult:
    """visual_embedding_node uses time.monotonic() not time.time() for elapsed."""
    src = _read(VISUAL_EMBEDDING)
    uses_time_time = bool(re.search(r'time\.time\(\)', src))
    uses_monotonic = "time.monotonic()" in src
    if uses_monotonic and not uses_time_time:
        return CheckResult("robust:visual_monotonic", True, "uses time.monotonic()")
    return CheckResult("robust:visual_monotonic", False,
                       "uses time.time() — affected by clock adjustments")


def check_quality_score_warning() -> CheckResult:
    """quality_validation logs score failures at WARNING (not DEBUG)."""
    src = _read(QUALITY_VALIDATION)
    # Look for logger.debug in the quality score except block
    if re.search(r'logger\.debug\("Quality score computation failed', src):
        return CheckResult("robust:quality_score_level", False,
                           "quality score failures logged at DEBUG — chunks silently dropped")
    if re.search(r'logger\.warning\("Quality score computation failed', src):
        return CheckResult("robust:quality_score_level", True,
                           "quality score failures logged at WARNING")
    # If neither, check generically
    return CheckResult("robust:quality_score_level", True, "no quality score debug suppression")


def check_multimodal_error_logging() -> CheckResult:
    """multimodal_processing_node logs non-strict vision failures."""
    src = _read(MULTIMODAL)
    # In non-strict mode, vision failures should be logged at WARNING
    has_except = "except Exception" in src or "except (" in src
    if not has_except:
        return CheckResult("robust:multimodal_error_log", True, "no except block")
    has_warning = bool(re.search(r'logger\.warning\(', src))
    if has_warning:
        return CheckResult("robust:multimodal_error_log", True,
                           "logs vision failures at WARNING")
    return CheckResult("robust:multimodal_error_log", False,
                       "non-strict vision failures not logged — silently continues")


# ── Speed Phase 2 checks ─────────────────────────────────────────────

def check_str_translate() -> CheckResult:
    """document.py uses str.translate instead of multiple str.replace passes."""
    src = _read(DOCUMENT_PY)
    has_translate = "str.maketrans" in src or ".translate(" in src
    # Count str.replace calls in a loop pattern
    replace_count = len(re.findall(r'text\s*=\s*text\.replace\(', src))
    if has_translate or replace_count <= 2:
        return CheckResult("speed:str_translate", True,
                           "uses str.translate or minimal str.replace")
    return CheckResult("speed:str_translate", False,
                       f"{replace_count} str.replace passes — use str.translate for single-pass")


def check_ensure_collection_not_per_doc() -> CheckResult:
    """embedding_storage_node does not call ensure_collection per document."""
    src = _read(EMBEDDING_STORAGE)
    # Check if ensure_collection is called inside the node function
    in_func = False
    for line in src.split("\n"):
        if "def embedding_storage_node" in line:
            in_func = True
        elif in_func and line and not line[0].isspace() and line.strip().startswith("def "):
            in_func = False
        if in_func and "ensure_collection(" in line:
            return CheckResult("speed:ensure_collection", False,
                               "ensure_collection called per-document — redundant API call")
    return CheckResult("speed:ensure_collection", True,
                       "ensure_collection not called per-document")


def check_ensure_bucket_not_per_doc() -> CheckResult:
    """document_storage_node does not call ensure_bucket per document."""
    src = _read(DOC_STORAGE)
    in_func = False
    for line in src.split("\n"):
        if "def document_storage_node" in line:
            in_func = True
        elif in_func and line and not line[0].isspace() and line.strip().startswith("def "):
            in_func = False
        if in_func and "ensure_bucket(" in line:
            return CheckResult("speed:ensure_bucket", False,
                               "ensure_bucket called per-document — redundant MinIO call")
    return CheckResult("speed:ensure_bucket", True,
                       "ensure_bucket not called per-document")


# ── Code Quality Phase 2 checks ─────────────────────────────────────

def check_no_hardcoded_collection() -> CheckResult:
    """dedup_utils.py uses config for collection name, not hardcoded 'Chunk'."""
    src = _read(DEDUP_UTILS)
    hardcoded = src.count('"Chunk"')
    if hardcoded == 0:
        return CheckResult("quality:no_hardcoded_collection", True,
                           "no hardcoded collection name")
    return CheckResult("quality:no_hardcoded_collection", False,
                       f"{hardcoded} hardcoded 'Chunk' collection references — should use config")


def check_dedup_override_typed() -> CheckResult:
    """dedup_override_sources is typed as list[str] not bare list."""
    src = _read(TYPES_FILE)
    match = re.search(r'dedup_override_sources:\s*(.*?)=', src)
    if not match:
        return CheckResult("quality:dedup_override_typed", True, "field not found (N/A)")
    type_decl = match.group(1).strip()
    if "list[str]" in type_decl or "List[str]" in type_decl:
        return CheckResult("quality:dedup_override_typed", True, "typed as list[str]")
    return CheckResult("quality:dedup_override_typed", False,
                       f"typed as bare '{type_decl}' — should be list[str]")


def check_error_return_consistency() -> CheckResult:
    """Embedding nodes use consistent error return (no {**state} spread on error)."""
    # Check that no embedding node uses {**state, "errors": ...} pattern
    # The correct pattern is {"errors": ..., "processing_log": ...}
    bad_nodes = []
    for path in PHASE2_NODES:
        src = _read(path)
        if re.search(r'return\s*\{\s*\*\*state', src):
            bad_nodes.append(path.stem)
    if not bad_nodes:
        return CheckResult("quality:error_return_consistency", True,
                           "no {**state} spread on error returns")
    return CheckResult("quality:error_return_consistency", False,
                       f"{len(bad_nodes)} nodes use {{**state}} on error: {', '.join(bad_nodes)}")


def check_encoding_dedup() -> CheckResult:
    """document_ingestion_node doesn't duplicate encoding fallback list."""
    src = _read(DOC_INGESTION)
    # Check if the encoding list is duplicated or imported from utils
    encoding_lists = re.findall(r'\("utf-8",\s*"latin-1",\s*"cp1252"\)', src)
    if len(encoding_lists) == 0:
        return CheckResult("quality:encoding_dedup", True,
                           "no inline encoding fallback list")
    if len(encoding_lists) >= 1:
        # Check if read_text_with_fallbacks is imported and used
        if "read_text_with_fallbacks" in src:
            return CheckResult("quality:encoding_dedup", True,
                               "uses read_text_with_fallbacks (canonical)")
        # Has inline encoding list but no import — duplicated
        return CheckResult("quality:encoding_dedup", False,
                           "inline encoding list duplicates read_text_with_fallbacks logic")
    return CheckResult("quality:encoding_dedup", True, "OK")


# ── Correctness guard ─────────────────────────────────────────────────

def run_correctness_guard() -> tuple[bool, str]:
    """Run ingest test suite as correctness guard."""
    project_root = ROOT.parent.parent  # RagWeave/
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/ingest/", "-x", "-q", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=300,
        )
        last_lines = result.stdout.strip().split("\n")[-5:]
        summary = "\n".join(last_lines)
        if result.returncode == 0:
            return True, f"PASS: {summary}"
        return False, f"FAIL (rc={result.returncode}):\n{summary}\n{result.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return False, "FAIL: test suite timed out (300s)"
    except Exception as exc:
        return False, f"FAIL: could not run tests: {exc}"


# ── Main ──────────────────────────────────────────────────────────────

def score() -> tuple[int, int, list[CheckResult]]:
    """Run all checks and return (passed, total, details)."""
    results: list[CheckResult] = []

    # Per-node checks (17 nodes × 3 criteria = 51)
    for node_path in ALL_NODES:
        results.append(check_logger_naming(node_path))
        results.append(check_info_logging(node_path))
        results.append(check_debug_timing(node_path))

    # Pipeline-wide robustness checks (3)
    results.append(check_llm_failure_logging())
    results.append(check_streaming_hash())
    results.append(check_unique_logger_names())

    # Speed checks (5)
    results.append(check_single_file_read())
    results.append(check_retry_delay())
    results.append(check_paragraph_early_exit())
    results.append(check_manifest_write_frequency())
    results.append(check_kg_extractor_reuse())

    # Code quality checks (3)
    results.append(check_metadata_state_contract())
    results.append(check_no_dead_aliases())
    results.append(check_deprecated_params())

    # Robustness Phase 2 (8)
    for path, name in _ERROR_PATH_NODES:
        results.append(check_error_path_logging(path, name))
    results.append(check_visual_monotonic())
    results.append(check_quality_score_warning())
    results.append(check_multimodal_error_logging())

    # Speed Phase 2 (3)
    results.append(check_str_translate())
    results.append(check_ensure_collection_not_per_doc())
    results.append(check_ensure_bucket_not_per_doc())

    # Code Quality Phase 2 (4)
    results.append(check_no_hardcoded_collection())
    results.append(check_dedup_override_typed())
    results.append(check_error_return_consistency())
    results.append(check_encoding_dedup())

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    return passed, total, results


def main() -> None:
    passed, total, results = score()

    print("=" * 60)
    print(f"INGESTION PIPELINE QUALITY SCORE: {passed}/{total}")
    print("=" * 60)

    # Group by status
    failing = [r for r in results if not r.passed]
    passing = [r for r in results if r.passed]

    if failing:
        print(f"\n--- FAILING ({len(failing)}) ---")
        for r in failing:
            print(f"  FAIL  {r.name}: {r.detail}")

    if passing:
        print(f"\n--- PASSING ({len(passing)}) ---")
        for r in passing:
            print(f"  PASS  {r.name}: {r.detail}")

    # Correctness guard
    print("\n--- CORRECTNESS GUARD ---")
    guard_ok, guard_msg = run_correctness_guard()
    print(f"  {'PASS' if guard_ok else 'FAIL'}: {guard_msg}")

    print(f"\nFINAL: {passed}/{total} | guard={'pass' if guard_ok else 'CRASH'}")

    if not guard_ok:
        sys.exit(1)  # Signal crash to orchestrator


if __name__ == "__main__":
    main()
