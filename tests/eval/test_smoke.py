# @summary
# P10 — offline CI smoke tests. Drives the full eval loop (run_nightly →
# run_eval → persist → gate → alert) against a tiny synthetic pack via the
# new run_eval/run_nightly DI seams (NO monkeypatch of infra, NO live
# Weaviate/embeddings/LLM). Covers: clean scenario rc==0, injected-regression
# scenario rc==1 (loop CORRECTLY catches it), main() == 0 on a healthy loop,
# main() != 0 when the loop is broken (always-pass run_nightly), and that
# run_eval actually invokes the injected execute/retrieve/judge seams.
# Exports: test_smoke_clean_scenario_returns_zero,
#          test_smoke_regression_scenario_returns_one,
#          test_smoke_main_returns_zero_on_healthy_loop,
#          test_smoke_main_detects_broken_loop,
#          test_run_eval_uses_injected_execute_seam,
#          test_run_eval_uses_injected_retrieve_seam,
#          test_run_eval_uses_injected_judge_seam
# Deps: src.eval.smoke, src.eval.cli
# @end-summary
"""P10 — offline CI smoke tests (TDD red-first)."""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Scenario-level: the smoke loop end-to-end
# ---------------------------------------------------------------------------


def test_smoke_clean_scenario_returns_zero() -> None:
    """Clean config (good retrieval, low floor) → loop returns rc==0."""
    from src.eval.smoke import run_smoke_loop

    assert run_smoke_loop(inject_regression=False) == 0


def test_smoke_regression_scenario_returns_one() -> None:
    """Injected regression (wrong doc + high floor) → loop returns rc==1.

    META-ASSERTION: the eval loop CORRECTLY catches the injected regression.
    """
    from src.eval.smoke import run_smoke_loop

    assert run_smoke_loop(inject_regression=True) == 1


def test_smoke_main_returns_zero_on_healthy_loop() -> None:
    """main([]) runs BOTH scenarios and exits 0 on a healthy loop."""
    from src.eval.smoke import main

    assert main([]) == 0


def test_smoke_main_detects_broken_loop(monkeypatch) -> None:
    """TEETH: if the loop ALWAYS passes (rc==0), main must report it broken.

    A run_nightly that always returns 0 makes the regression scenario
    wrongly return 0, so main([]) must be non-zero — proving main genuinely
    verifies BOTH expected exit codes rather than just smoke-running.
    """
    import src.eval.smoke as smoke

    monkeypatch.setattr(smoke, "run_nightly", lambda *a, **k: 0)

    assert smoke.main([]) != 0


# ---------------------------------------------------------------------------
# Seam-level: run_eval actually uses the injected DI seams
# ---------------------------------------------------------------------------


def _build_pack(tmp_path: Path, *, recall_floor: float = 0.5) -> Path:
    from src.eval.smoke import build_smoke_pack

    return build_smoke_pack(tmp_path, recall_floor=recall_floor)


def test_run_eval_uses_injected_execute_seam(tmp_path) -> None:
    """run_eval invokes the injected execute_fn (not live execute_plan)."""
    from src.eval import smoke
    from src.eval.cli import EvalRunResult, run_eval

    pack_dir = _build_pack(tmp_path)
    calls: list = []

    def recording_execute(plan, *, fresh: bool = True):
        calls.append(("execute", fresh))
        return smoke._smoke_execute(plan, fresh=fresh)

    result = run_eval(
        str(pack_dir),
        execute_fn=recording_execute,
        retrieve_fn=smoke.make_smoke_retrieve("doc1.md"),
        judge_client_factory=smoke._smoke_judge_factory,
    )

    assert calls, "execute_fn seam was not invoked"
    assert isinstance(result, EvalRunResult)


def test_run_eval_uses_injected_retrieve_seam(tmp_path) -> None:
    """run_eval invokes the injected retrieve_fn (not live retrieve)."""
    from src.eval import smoke
    from src.eval.cli import EvalRunResult, run_eval

    pack_dir = _build_pack(tmp_path)
    calls: list = []
    real_retrieve = smoke.make_smoke_retrieve("doc1.md")

    def recording_retrieve(collection_name, goldens, k=5):
        calls.append(("retrieve", collection_name, k))
        return real_retrieve(collection_name, goldens, k=k)

    result = run_eval(
        str(pack_dir),
        execute_fn=smoke._smoke_execute,
        retrieve_fn=recording_retrieve,
        judge_client_factory=smoke._smoke_judge_factory,
    )

    assert calls, "retrieve_fn seam was not invoked"
    assert isinstance(result, EvalRunResult)


def test_run_eval_uses_injected_judge_seam(tmp_path) -> None:
    """run_eval invokes the injected judge_client_factory (not live build)."""
    from src.eval import smoke
    from src.eval.cli import EvalRunResult, run_eval

    pack_dir = _build_pack(tmp_path)
    calls: list = []

    def recording_judge_factory(pack, *, chat_model_factory=None, pack_dir=None):
        calls.append(("judge", pack.meta.name))
        return smoke._smoke_judge_factory(
            pack, chat_model_factory=chat_model_factory, pack_dir=pack_dir
        )

    result = run_eval(
        str(pack_dir),
        execute_fn=smoke._smoke_execute,
        retrieve_fn=smoke.make_smoke_retrieve("doc1.md"),
        judge_client_factory=recording_judge_factory,
    )

    assert calls, "judge_client_factory seam was not invoked"
    assert isinstance(result, EvalRunResult)
