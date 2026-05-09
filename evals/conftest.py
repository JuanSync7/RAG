"""Shared pytest helpers used across all eval sub-packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_fixture(path: Path) -> Any:
    """Load a JSON fixture file. Raises FileNotFoundError if missing."""
    return json.loads(Path(path).read_text())
