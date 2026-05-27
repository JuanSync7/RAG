# @summary
# Pydantic v2 models for the eval_pack format (P0.5 keystone).
# Exports: PackMeta, JudgeConfig, ManifestEntry, Golden, Thresholds, EvalPack.
# Deps: pydantic v2.
# @end-summary
"""Typed contracts for the eval_pack format.

Only the ``factoid`` qtype has strict per-field requirements at the
keystone; other qtypes parse loosely with extra fields preserved on
``Golden.raw`` so P2 can extend the schema without re-authoring this
module.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

KNOWN_PROFILES: frozenset[str] = frozenset({"asic_riscv_soc", "eda_command_reference", "generic"})


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    tier1_model: str
    tier1_prompt_version: str
    temperature: float
    samples_per_claim: int


class PackMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    version: int
    profile: str
    corpus_pin: str
    description: str | None = None
    judge: JudgeConfig
    collection_name_template: str


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    sha256: str


class Golden(BaseModel):
    """A single golden query.

    ``factoid`` enforces ``qid``, ``qtype``, ``query``, ``expected_answer_span``.
    Other qtypes accept the same base fields but make span optional, and
    keep the raw line under ``raw`` for downstream consumers.
    """

    model_config = ConfigDict(extra="allow")

    qid: str
    qtype: str
    query: str
    expected_answer_span: str | None = None
    expected_source_docs: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_factoid_required(self) -> Golden:
        if self.qtype == "factoid" and not self.expected_answer_span:
            raise ValueError(
                f"golden qid={self.qid!r} qtype=factoid is missing required field 'expected_answer_span'"
            )
        return self


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="allow")

    profile: str
    defaults: dict[str, dict[str, float]] = Field(default_factory=dict)
    overrides: list[dict[str, Any]] = Field(default_factory=list)


class EvalPack(BaseModel):
    """Aggregate pack: meta + manifest + goldens + thresholds.

    ``collection_name`` is computed from ``meta.collection_name_template``
    with ``{name}`` and ``{corpus_pin_short}`` (first 8 hex chars) substituted.
    """

    model_config = ConfigDict(extra="forbid")

    meta: PackMeta
    manifest: list[ManifestEntry]
    goldens: dict[str, list[Golden]]
    thresholds: Thresholds

    @property
    def collection_name(self) -> str:
        short = self.meta.corpus_pin[:8]
        return self.meta.collection_name_template.format(
            name=self.meta.name,
            corpus_pin_short=short,
        )
