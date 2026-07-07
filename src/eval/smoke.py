# @summary
# P10 — offline CI-gate smoke for the eval loop. Drives the FULL loop
# (run_nightly → run_eval → persist → gate → alert) against a tiny synthetic
# eval_pack built on disk, using the run_eval/run_nightly DI seams to inject
# deterministic fakes for ingest, retrieval, and judging — NO live Weaviate /
# embeddings / LLM and NO monkeypatch. main() runs TWO scenarios and exits 0
# iff BOTH behave correctly: clean config → rc==0, injected regression → rc==1
# (the loop CORRECTLY catches the regression).
# Exports: build_smoke_pack, make_smoke_retrieve, run_smoke_loop, main,
#          _smoke_execute, _smoke_judge_factory, _SmokeJudge
# Deps: stdlib (hashlib, json, sys, tempfile, datetime, pathlib), yaml;
#       src.eval.orchestrator.run_nightly; src.eval.runner.report.IngestReport;
#       src.eval.runner.retrieve (QueryRetrievalResult, RetrievalResults);
#       src.eval.runner.judge.JudgmentScore.
# @end-summary
"""P10 — offline eval-loop smoke (the PR-gate sanity check).

This module is a first-class, importable, fully OFFLINE smoke that exercises
the entire eval loop end-to-end::

    run_nightly → run_eval → persist → gate → alert

against a tiny synthetic eval_pack written to a temp dir. It uses the
``execute_fn`` / ``retrieve_fn`` / ``judge_client_factory`` dependency-injection
seams on :func:`src.eval.cli.run_eval` (threaded through
:func:`src.eval.orchestrator.run_nightly`) to swap in deterministic in-process
fakes — so it needs NO live Weaviate, embeddings, or LLM, and uses NO
monkeypatching.

``python -m src.eval.smoke`` runs TWO scenarios and exits ``0`` iff BOTH behave
correctly:

* **clean** — good retrieval + a low recall floor → the loop returns ``0``.
* **regression** — wrong retrieved doc + a high recall floor → the loop returns
  ``1`` (the loop CORRECTLY catches the regression).

If either scenario returns the wrong code the loop itself is broken, and the
smoke exits non-zero. This is the dedicated offline PR gate wired into
``ci.yml``; the live judge-calibrated gate lives in the
``eval-gate.yml`` workflow_dispatch job instead.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Imported at module level so tests can monkeypatch ``src.eval.smoke.run_nightly``.
from src.eval.orchestrator import run_nightly

# A fixed, timezone-aware timestamp so persisted report filenames are
# deterministic across smoke runs.
_FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

_DOC_TEXT = "alpha beta gamma\n"


# ---------------------------------------------------------------------------
# Synthetic pack builder
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_smoke_pack(dest_dir: str | Path, *, recall_floor: float) -> Path:
    """Write a minimal VALID eval_pack under ``dest_dir`` and return its dir.

    Mirrors the shape :func:`src.eval.pack.load_pack` accepts: a single-doc
    corpus with a manifest, a ``pack.yaml`` (generic profile, fake judge), one
    factoid golden, and a ``thresholds.yaml`` whose ``factoid.recall_at_5`` floor
    is ``recall_floor`` (the smoke raises this to force a regression). The
    structure intentionally matches the in-test ``_make_synthetic_pack`` but is
    re-authored here so the smoke imports NOTHING from ``tests/``.
    """
    import yaml

    pack_dir = Path(dest_dir) / "synthetic_pack"
    pack_dir.mkdir(parents=True)
    (pack_dir / "corpus").mkdir()
    (pack_dir / "goldens").mkdir()

    doc_path = pack_dir / "corpus" / "doc1.md"
    doc_path.write_text(_DOC_TEXT)
    doc_sha = _sha256_text(_DOC_TEXT)

    manifest = {"entries": [{"path": "doc1.md", "sha256": doc_sha}]}
    (pack_dir / "corpus" / "manifest.json").write_text(json.dumps(manifest))

    # corpus_pin = sha256 of "<path>:<sha256>" (single entry here).
    corpus_pin = hashlib.sha256(
        f"doc1.md:{doc_sha}".encode("utf-8")
    ).hexdigest()

    pack_yaml = {
        "name": "synpack",
        "version": 1,
        "profile": "generic",
        "corpus_pin": corpus_pin,
        "description": "P10 offline smoke pack",
        "judge": {
            "tier1_model": "fake-model",
            "tier1_prompt_version": "v1",
            "temperature": 0.0,
            "samples_per_claim": 1,
        },
        "collection_name_template": "test_{name}_{corpus_pin_short}",
    }
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(pack_yaml))

    golden = {
        "qid": "g1",
        "qtype": "factoid",
        "query": "what is alpha?",
        "expected_answer_span": "alpha",
        "expected_source_docs": ["doc1.md"],
    }
    (pack_dir / "goldens" / "factoid.jsonl").write_text(
        json.dumps(golden) + "\n"
    )

    thresholds = {
        "profile": "generic",
        "defaults": {
            "factoid": {"recall_at_5": recall_floor, "faithfulness": 0.5}
        },
        "min_goldens_per_qtype": {"factoid": 1},
    }
    (pack_dir / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    return pack_dir


# ---------------------------------------------------------------------------
# Deterministic in-process fakes (the DI-seam payloads)
# ---------------------------------------------------------------------------


def _smoke_execute(plan, *, fresh: bool = True):
    """Fake ``execute_fn``: return a synthetic single-doc IngestReport.

    No Weaviate, no embeddings — just enough shape for the downstream
    retrieve/judge/report stages.
    """
    from src.eval.runner.report import IngestReport

    return IngestReport(
        collection_name=plan.collection_name,
        processed=1,
        skipped=0,
        failed=0,
        stored_chunks=1,
        document_count=1,
        plan=plan,
        errors=(),
    )


def make_smoke_retrieve(retrieved_source: str) -> Callable[..., Any]:
    """Build a fake ``retrieve_fn`` that returns ``retrieved_source`` for all.

    The returned callable produces a :class:`RetrievalResults` with one
    :class:`QueryRetrievalResult` per golden (across every qtype), each
    retrieving the single ``retrieved_source`` doc. Passing the EXPECTED source
    (``doc1.md``) yields recall 1.0; passing a WRONG source (``other.md``)
    yields recall 0.0 — the regression lever.
    """
    from src.eval.runner.retrieve import QueryRetrievalResult, RetrievalResults

    def _retrieve_fn(collection_name: str, goldens, k: int = 5):
        per_query: dict[str, QueryRetrievalResult] = {}
        for qtype, golden_list in goldens.items():
            for golden in golden_list:
                per_query[golden.qid] = QueryRetrievalResult(
                    qid=golden.qid,
                    qtype=qtype,
                    query=golden.query,
                    retrieved_sources=(retrieved_source,),
                    k=k,
                    retrieved_chunks=("alpha is the first",),
                )
        return RetrievalResults(
            collection_name=collection_name, k=k, per_query=per_query
        )

    return _retrieve_fn


class _SmokeJudge:
    """Duck-typed judge client: always scores 0.9 (above the 0.5 floor)."""

    def score(self, question):
        from src.eval.runner.judge import JudgmentScore

        return JudgmentScore(score=0.9, reasoning="smoke")


def _smoke_judge_factory(
    pack, *, chat_model_factory: Callable[..., Any] | None = None, pack_dir=None
) -> _SmokeJudge:
    """Fake ``judge_client_factory``: hand back a deterministic judge."""
    return _SmokeJudge()


# ---------------------------------------------------------------------------
# The smoke loop
# ---------------------------------------------------------------------------


def run_smoke_loop(
    *,
    inject_regression: bool,
    output_dir: str | Path | None = None,
    now: datetime | None = None,
) -> int:
    """Run the full eval loop once, offline, and return its exit code.

    Builds a synthetic pack in a temp dir and drives
    :func:`~src.eval.orchestrator.run_nightly` with the in-process fakes wired
    through the run_eval DI seams.

    Both scenarios use the SAME ``recall_at_5`` floor (0.5); the SINGLE
    load-bearing lever is the retrieved doc, so the regression has exactly one
    unambiguous cause — a metric dropping below a fixed gate floor (the
    canonical "gate catches a sub-threshold metric" check):

    * ``inject_regression=False`` — retrieve ``doc1.md`` (recall 1.0 ≥ 0.5
      floor; faithfulness 0.9 ≥ 0.5 floor) → the gate passes → rc ``0``.
    * ``inject_regression=True`` — retrieve the WRONG doc ``other.md`` (recall
      0.0 < 0.5 floor) → the gate fails → rc ``1`` (the loop CAUGHT it).

    When ``output_dir`` is None a :class:`tempfile.TemporaryDirectory` holds
    both the pack and the persisted reports.
    """
    if now is None:
        now = _FIXED_NOW

    # ONE lever: the wrong retrieved doc drops recall below the fixed floor.
    # (No second floor lever — a redundant lever has no independent teeth.)
    recall_floor = 0.5
    retrieved_source = "other.md" if inject_regression else "doc1.md"

    def _drive(base_dir: Path) -> int:
        pack_dir = build_smoke_pack(base_dir, recall_floor=recall_floor)
        reports_dir = base_dir / "reports"
        reports_dir.mkdir()
        alerts: list = []
        return run_nightly(
            [pack_dir],
            output_dir=reports_dir,
            now=now,
            alert_fn=alerts.append,
            execute_fn=_smoke_execute,
            retrieve_fn=make_smoke_retrieve(retrieved_source),
            judge_client_factory=_smoke_judge_factory,
        )

    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return _drive(Path(tmp))
    return _drive(Path(output_dir))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run BOTH smoke scenarios; exit 0 iff the loop behaves correctly.

    The healthy contract is: clean scenario → ``0`` AND regression scenario →
    ``1``. Any deviation (e.g. a broken loop that always passes) makes the
    smoke fail. Prints a one-line PASS/FAIL summary naming each scenario's rc.
    """
    clean_rc = run_smoke_loop(inject_regression=False)
    regression_rc = run_smoke_loop(inject_regression=True)
    ok = clean_rc == 0 and regression_rc == 1
    verdict = "PASS" if ok else "FAIL"
    print(
        f"eval-loop smoke {verdict}: "
        f"clean_rc={clean_rc} (expected 0), "
        f"regression_rc={regression_rc} (expected 1)"
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
