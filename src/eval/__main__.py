# @summary
# Module entry point for `python -m src.eval`. Delegates to src.eval.cli.main.
# Exports: (none — script entry)
# Deps: src.eval.cli
# @end-summary
"""Entry point for `python -m src.eval`."""
from __future__ import annotations

import sys

from src.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
