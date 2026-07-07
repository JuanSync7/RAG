"""Integration tests: verify all .co files parse and flows register correctly."""
import pytest
from pathlib import Path


@pytest.fixture
def guardrails_config_dir():
    return str(Path(__file__).resolve().parents[2] / "config" / "guardrails")


@pytest.mark.skip(
    reason="needs real langchain_core; suite-wide langchain stub shadows it "
    "(see tests/conftest.py). Re-enable via a real-langchain boundary. "
    "Tracked in TEST_COVERAGE_PLAN.md (CI-gating slice)."
)
def test_all_co_files_parse(guardrails_config_dir):
    """All .co files must parse without SyntaxError."""
    from nemoguardrails import RailsConfig
    config = RailsConfig.from_path(guardrails_config_dir)
    assert config is not None


def test_co_files_exist(guardrails_config_dir):
    """All expected .co files must exist."""
    expected = [
        "input_rails.co",
        "conversation.co",
        "output_rails.co",
        "safety.co",
        "dialog_patterns.co",
    ]
    config_path = Path(guardrails_config_dir)
    for filename in expected:
        assert (config_path / filename).exists(), f"Missing: {filename}"


def test_actions_py_exists(guardrails_config_dir):
    """actions.py must exist for NeMo auto-discovery."""
    assert (Path(guardrails_config_dir) / "actions.py").exists()
