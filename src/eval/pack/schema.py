# @summary
# Pydantic v2 models for the eval_pack format (P0.5 keystone).
# Exports: PackMeta, JudgeConfig, ManifestEntry, Golden, ThresholdOverride,
#          Thresholds, EvalPack.
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
    max_parallel_judges: int = 1


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


class ThresholdOverride(BaseModel):
    """A single threshold override entry (P7f).

    An override targets EXACTLY ONE of:

    - a ``qtype`` — its metric floors REPLACE (per metric, last-wins) the
      matching entries in ``Thresholds.defaults`` and may introduce qtypes
      or metrics absent from defaults;
    - a ``qid`` — its metric floors are checked against that single query's
      individual scores.

    The metric floors arrive as *extra* fields (e.g. ``mrr``, ``recall_at_5``,
    ``faithfulness``) and are surfaced via :meth:`floors`. Validation is
    fail-fast at construction so a malformed entry rejects at pack load
    (``loader.py`` and ``validate.py`` both build :class:`Thresholds`).

    Contract:

    - exactly one of ``qtype`` / ``qid`` must be set (XOR);
    - at least one metric floor must be present and every floor value must be
      a real number (``int``/``float``, but NOT ``bool`` and NOT a string).
    """

    model_config = ConfigDict(extra="allow")

    qtype: str | None = None
    qid: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> ThresholdOverride:
        has_qtype = self.qtype is not None
        has_qid = self.qid is not None
        if has_qtype == has_qid:
            raise ValueError(
                "threshold override must set EXACTLY ONE of 'qtype' or 'qid' "
                f"(got qtype={self.qtype!r}, qid={self.qid!r})"
            )
        extras = self.model_extra or {}
        if not extras:
            raise ValueError(
                "threshold override must declare at least one metric floor "
                f"(target={'qtype=' + self.qtype if has_qtype else 'qid=' + self.qid!r})"
            )
        for key, value in extras.items():
            # bool is an int subclass — exclude it explicitly.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"threshold override metric {key!r} must be a real number, "
                    f"got {value!r} ({type(value).__name__})"
                )
        return self

    def floors(self) -> dict[str, float]:
        """Return the metric floors as a ``{metric: float}`` mapping."""
        return {k: float(v) for k, v in (self.model_extra or {}).items()}


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="allow")

    profile: str
    defaults: dict[str, dict[str, float]] = Field(default_factory=dict)
    overrides: list[ThresholdOverride] = Field(default_factory=list)
    min_goldens_per_qtype: dict[str, int] = Field(default_factory=dict)


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
