# @summary
# Locks in the production default for enable_kg_phase2b. The Temporal handoff
# (KG runs in the kgweave worker fleet) is the production path; the in-process
# legacy path stays available but only as opt-in for offline CLI / bench runs.
# Exports: (test module)
# Deps: pytest, src.ingest.common.types
# @end-summary
"""Default-flip regression test for enable_kg_phase2b."""

from __future__ import annotations

from src.ingest.common.types import IngestionConfig


def test_enable_kg_phase2b_defaults_to_true() -> None:
    """Production default: KG runs via Temporal handoff to the KGWeave worker."""
    config = IngestionConfig()
    assert config.enable_kg_phase2b is True


def test_enable_kg_phase2b_can_be_overridden_to_false() -> None:
    """Explicit False keeps the legacy in-process path for offline CLI/bench."""
    config = IngestionConfig(enable_kg_phase2b=False)
    assert config.enable_kg_phase2b is False


def test_default_is_wired_to_settings_env_var() -> None:
    """The dataclass default tracks RAG_INGESTION_ENABLE_KG_PHASE2B from settings.

    Reloading the module under a patched env would pollute session state for
    subsequent tests, so we verify the wire-up structurally: the default value
    of the field equals the resolved settings constant.
    """
    from config.settings import RAG_INGESTION_ENABLE_KG_PHASE2B

    field = next(f for f in IngestionConfig.__dataclass_fields__.values()
                 if f.name == "enable_kg_phase2b")
    assert field.default == RAG_INGESTION_ENABLE_KG_PHASE2B
