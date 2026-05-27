# @summary
# Single exception type for eval_pack validation failures.
# Exports: PackValidationError
# Deps: stdlib only.
# @end-summary
"""Eval_pack validation errors.

All validator and loader failures surface as :class:`PackValidationError`.
The message must name (a) the file/field path inside the pack and
(b) the reason — see ``validate.py`` for the canonical formatter.
"""
from __future__ import annotations


class PackValidationError(Exception):
    """Raised when an eval_pack fails structural or schema validation."""
