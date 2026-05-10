"""Mock-based tests for src/ingest/embedding/impl.py.

Covers the run_embedding_pipeline() function body including:
- Deprecation warning on docling_document (lines 65-71)
- initial_state construction (lines 72-96)
- _GRAPH.invoke success path (lines 97-98)
- Exception handler → partial state with error (lines 99-100)
- Return (line 101)

All test functions that rely on mocks are named test_mock_*.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime():
    """Build a minimal mock Runtime."""
    runtime = MagicMock()
    runtime.config = MagicMock()
    return runtime


# ---------------------------------------------------------------------------
# test_mock_run_embedding_pipeline_success
# ---------------------------------------------------------------------------


def test_mock_run_embedding_pipeline_success():
    """Verify the happy path: _GRAPH.invoke() is called with a valid initial_state
    and its return value is returned to the caller."""
    from src.ingest.embedding.impl import run_embedding_pipeline

    runtime = _make_runtime()
    expected_state = {
        "chunks": [MagicMock()],
        "stored_count": 3,
        "errors": [],
    }

    with patch("src.ingest.embedding.impl._GRAPH") as mock_graph:
        mock_graph.invoke.return_value = expected_state

        result = run_embedding_pipeline(
            runtime=runtime,
            source_key="docs/readme.md",
            source_name="readme.md",
            source_uri="file://docs/readme.md",
            source_id="abc123",
            connector="local",
            source_version="v1",
            clean_text="# Hello\nWorld",
            clean_hash="deadbeef",
        )

    assert result is expected_state
    mock_graph.invoke.assert_called_once()

    # Verify key fields of the initial_state passed to invoke
    invoke_arg = mock_graph.invoke.call_args[0][0]
    assert invoke_arg["source_key"] == "docs/readme.md"
    assert invoke_arg["source_name"] == "readme.md"
    assert invoke_arg["source_uri"] == "file://docs/readme.md"
    assert invoke_arg["source_id"] == "abc123"
    assert invoke_arg["connector"] == "local"
    assert invoke_arg["source_version"] == "v1"
    assert invoke_arg["raw_text"] == "# Hello\nWorld"
    assert invoke_arg["cleaned_text"] == "# Hello\nWorld"
    assert invoke_arg["clean_hash"] == "deadbeef"
    assert invoke_arg["chunks"] == []
    assert invoke_arg["errors"] == []
    assert invoke_arg["trace_id"] == ""
    assert invoke_arg["batch_id"] == ""


# ---------------------------------------------------------------------------
# test_mock_run_embedding_pipeline_success_with_trace_and_batch
# ---------------------------------------------------------------------------


def test_mock_run_embedding_pipeline_success_with_trace_and_batch():
    """Verify trace_id and batch_id are forwarded into initial_state."""
    from src.ingest.embedding.impl import run_embedding_pipeline

    runtime = _make_runtime()

    with patch("src.ingest.embedding.impl._GRAPH") as mock_graph:
        mock_graph.invoke.return_value = {"errors": [], "stored_count": 0}

        run_embedding_pipeline(
            runtime=runtime,
            source_key="k",
            source_name="n",
            source_uri="u",
            source_id="i",
            connector="local",
            source_version="v",
            clean_text="txt",
            clean_hash="h",
            trace_id="trace-uuid-1",
            batch_id="batch-001",
        )

    invoke_arg = mock_graph.invoke.call_args[0][0]
    assert invoke_arg["trace_id"] == "trace-uuid-1"
    assert invoke_arg["batch_id"] == "batch-001"


# ---------------------------------------------------------------------------
# test_mock_run_embedding_pipeline_error
# ---------------------------------------------------------------------------


def test_mock_run_embedding_pipeline_error():
    """When _GRAPH.invoke raises, the function returns a partial state with
    the error encoded in the 'errors' key and stored_count=0."""
    from src.ingest.embedding.impl import run_embedding_pipeline

    runtime = _make_runtime()

    with patch("src.ingest.embedding.impl._GRAPH") as mock_graph:
        mock_graph.invoke.side_effect = RuntimeError("graph exploded")

        result = run_embedding_pipeline(
            runtime=runtime,
            source_key="k",
            source_name="n",
            source_uri="u",
            source_id="i",
            connector="c",
            source_version="v",
            clean_text="text",
            clean_hash="h",
        )

    assert result["stored_count"] == 0
    assert len(result["errors"]) == 1
    assert "embedding_graph:" in result["errors"][0]
    assert "graph exploded" in result["errors"][0]

    # The partial state should also carry the original fields
    assert result["source_key"] == "k"
    assert result["clean_hash"] == "h"


# ---------------------------------------------------------------------------
# test_mock_run_embedding_pipeline_deprecated_docling
# ---------------------------------------------------------------------------


def test_mock_run_embedding_pipeline_deprecated_docling():
    """Passing docling_document triggers a DeprecationWarning."""
    from src.ingest.embedding.impl import run_embedding_pipeline

    runtime = _make_runtime()

    with patch("src.ingest.embedding.impl._GRAPH") as mock_graph:
        mock_graph.invoke.return_value = {"errors": [], "stored_count": 0}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_embedding_pipeline(
                runtime=runtime,
                source_key="k",
                source_name="n",
                source_uri="u",
                source_id="i",
                connector="c",
                source_version="v",
                clean_text="text",
                clean_hash="h",
                docling_document="some-doc-object",
            )

    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# test_mock_run_embedding_pipeline_no_warning_without_docling
# ---------------------------------------------------------------------------


def test_mock_run_embedding_pipeline_no_warning_without_docling():
    """Not passing docling_document must not trigger any DeprecationWarning."""
    from src.ingest.embedding.impl import run_embedding_pipeline

    runtime = _make_runtime()

    with patch("src.ingest.embedding.impl._GRAPH") as mock_graph:
        mock_graph.invoke.return_value = {"errors": [], "stored_count": 0}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_embedding_pipeline(
                runtime=runtime,
                source_key="k",
                source_name="n",
                source_uri="u",
                source_id="i",
                connector="c",
                source_version="v",
                clean_text="text",
                clean_hash="h",
            )

    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0


# ---------------------------------------------------------------------------
# test_mock_run_embedding_pipeline_refactored_text
# ---------------------------------------------------------------------------


def test_mock_run_embedding_pipeline_refactored_text():
    """Verify refactored_text is forwarded into initial_state."""
    from src.ingest.embedding.impl import run_embedding_pipeline

    runtime = _make_runtime()

    with patch("src.ingest.embedding.impl._GRAPH") as mock_graph:
        mock_graph.invoke.return_value = {"errors": [], "stored_count": 0}

        run_embedding_pipeline(
            runtime=runtime,
            source_key="k",
            source_name="n",
            source_uri="u",
            source_id="i",
            connector="c",
            source_version="v",
            clean_text="original",
            clean_hash="h",
            refactored_text="refactored",
        )

    invoke_arg = mock_graph.invoke.call_args[0][0]
    assert invoke_arg["refactored_text"] == "refactored"
