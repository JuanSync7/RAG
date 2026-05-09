"""Fixtures for the deep-research A/B eval suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from evals.conftest import load_json_fixture

_FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _load_or_skip(name: str) -> Dict[str, Any]:
    path = _FIXTURES_ROOT / "asic" / name
    if not path.exists():
        pytest.skip(f"Deep-research fixture not populated: {path}")
    data = load_json_fixture(path)
    if not data.get("queries"):
        pytest.skip(f"Deep-research fixture {name} is empty")
    return data


@pytest.fixture(scope="session")
def golden_queries_deep_research_asic() -> Dict[str, Any]:
    """Load the ASIC golden query set for deep-research A/B evals."""
    return _load_or_skip("golden_queries.json")


@pytest.fixture(scope="session")
def messy_queries_deep_research_asic() -> Dict[str, Any]:
    """Misspelled / terse / ungrammatical query variants."""
    return _load_or_skip("messy_queries.json")


@pytest.fixture(scope="session")
def efficiency_queries_deep_research_asic() -> Dict[str, Any]:
    """Single-aspect queries used to bound DR's LLM-call cost."""
    return _load_or_skip("efficiency_queries.json")


@pytest.fixture(scope="session")
def disjoint_queries_deep_research_asic() -> Dict[str, Any]:
    """Disjoint multi-topic queries used by the per-topic rerank A/B harness."""
    return _load_or_skip("disjoint_queries.json")


@pytest.fixture(scope="session")
def disjoint_queries_deep_research_ragweave() -> Dict[str, Any]:
    """Disjoint multi-topic queries against the live RagWeave engineering-doc
    corpus (latency-only A/B; recall is best-effort prefix match)."""
    path = _FIXTURES_ROOT / "ragweave" / "disjoint_queries.json"
    if not path.exists():
        pytest.skip(f"RagWeave disjoint fixture not populated: {path}")
    data = load_json_fixture(path)
    if not data.get("queries"):
        pytest.skip("RagWeave disjoint fixture is empty")
    return data


@pytest.fixture(scope="session")
def adversarial_queries_deep_research_asic() -> Dict[str, Any]:
    """Prompt-injection variants used to assert DR doesn't honor injected
    instructions. Each query carries ``legitimate_topics`` (the real intent)
    and ``injected_topics`` (the attacker's substitution targets)."""
    return _load_or_skip("adversarial_queries.json")


@pytest.fixture(scope="session")
def rag_chain():
    """Session-scoped RAGChain shared across every eval test module.

    Module-scoped fixtures rebuild RAGChain (and its embedded Weaviate /
    model handles) per test file, which routinely exceeds the per-test
    timeout. Sharing one chain keeps eval runs deterministic.
    """
    try:
        from src.retrieval.pipeline import RAGChain
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RAGChain import failed: {exc}")
    try:
        chain = RAGChain()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RAGChain init failed (likely missing Weaviate/models): {exc}")
    yield chain
    try:
        chain.close()
    except Exception:
        pass
