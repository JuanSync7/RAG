# @summary
# Public surface of the eval_pack format (P0.5 keystone).
# Exports: EvalPack, load_pack, validate_pack, PackValidationError,
#          ThresholdOverride.
# Deps: pydantic v2, pyyaml.
# @end-summary
"""Eval_pack format — typed pack loader and validator.

This is the stable import surface; downstream code should never import
from the submodules directly.
"""
from __future__ import annotations

from .errors import PackValidationError
from .loader import load_pack
from .schema import (
    EvalPack,
    Golden,
    JudgeConfig,
    ManifestEntry,
    PackMeta,
    ThresholdOverride,
    Thresholds,
)
from .validate import validate_pack

__all__ = [
    "EvalPack",
    "Golden",
    "JudgeConfig",
    "ManifestEntry",
    "PackMeta",
    "PackValidationError",
    "ThresholdOverride",
    "Thresholds",
    "load_pack",
    "validate_pack",
]
