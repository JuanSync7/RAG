"""Validate Colang 2.0 syntax against the installed nemoguardrails parser.

Colang 2.0 (version "2.x") uses a different syntax from Colang 1.0:
- Flows are defined with `flow <name>` (no `define` keyword)
- Actions are invoked with `await` (not `execute`)
- Variables are prefixed with `$`
- `bot say $text` is a valid flow call (from core library)
- `abort` is a valid statement

Colang 1.0 uses:
- `define user <intent>`, `define flow <name>`
- No `$` variable syntax
"""
import tempfile
from pathlib import Path

import pytest

# Quarantined for CI gating: these tests import ``nemoguardrails``, which requires
# the REAL ``langchain_core`` package. The suite-wide langchain stub installed by
# ``tests/conftest.py`` makes ``langchain_core`` a non-package module, so the
# nemoguardrails import raises ``ModuleNotFoundError: ... 'langchain_core' is not
# a package``. Re-enable once tests/guardrails gets a real-langchain collection
# boundary like tests/llm. Tracked in TEST_COVERAGE_PLAN.md (CI-gating slice).
pytestmark = pytest.mark.skip(
    reason="needs real langchain_core; suite-wide langchain stub shadows it "
    "(see tests/conftest.py). Re-enable via a real-langchain boundary."
)


# ---------------------------------------------------------------------------
# Colang 2.0 sample
# Key differences vs. 1.0:
#   - `flow` keyword (no `define`)
#   - `await` for action/flow calls (not `execute`)
#   - variables prefixed with `$`
#   - `bot say $var` calls the core `bot say` flow
#   - `abort` is a valid statement
# ---------------------------------------------------------------------------
COLANG_2_SAMPLE = """\
flow input rails $input_text
  $result = await check_query_length(query=$input_text)
  if $result.valid == False
    await bot say $result.reason
    abort
"""

CONFIG_YML_V2 = """\
colang_version: "2.x"

models:
  - type: main
    engine: ollama
    model: test
    parameters:
      base_url: http://localhost:11434
"""


def test_colang_2_syntax_parses():
    """Verify our Colang 2.0 flow syntax is accepted by the installed parser."""
    from nemoguardrails import RailsConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "flows.co").write_text(COLANG_2_SAMPLE)
        (p / "config.yml").write_text(CONFIG_YML_V2)

        # Should not raise — the parser validates syntax, not runtime availability.
        config = RailsConfig.from_path(str(p))
        assert config is not None


# ---------------------------------------------------------------------------
# Colang 1.0 sample — backward-compatibility check
# ---------------------------------------------------------------------------
COLANG_1_SAMPLE = """\
define user express greeting
  "hello"
  "hi there"

define bot express greeting
  "Hello! How can I help you today?"

define flow
  user express greeting
  bot express greeting
"""

CONFIG_YML_V1 = """\
models:
  - type: main
    engine: ollama
    model: test
    parameters:
      base_url: http://localhost:11434
"""


def test_colang_1_syntax_still_parses():
    """Sanity check: Colang 1.0 syntax should still parse (backward compat)."""
    from nemoguardrails import RailsConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "intents.co").write_text(COLANG_1_SAMPLE)
        (p / "config.yml").write_text(CONFIG_YML_V1)

        config = RailsConfig.from_path(str(p))
        assert config is not None
