# @summary
# Unit tests for the routing glossary, lexical comparison-intent detector,
# term canonicalization, and the typed routing contracts (schemas).
# Covers the design's comparison examples (§1, §6.5), the informal->canonical
# alias maps ("hub-based"->CHI, "extension-based"->ACE), and the conservative
# no-false-positive requirement for plain single-topic queries.
# Deps: pytest, src.retrieval.routing
# @end-summary
"""Unit tests for ``src.retrieval.routing`` (S0c slice).

Three concerns are exercised independently:

1.  ``detect_comparison_intent`` — must flag comparison/juxtaposition phrasing
    (the design's examples) and must NOT flag plain single-topic queries.
2.  ``canonicalize_terms`` — resolves glossary aliases/entities to the
    de-duplicated list of canonical entities present in a query.
3.  ``PROTOCOL_GLOSSARY`` shape + the typed contracts (``DocCard``,
    ``RoutingResult``, ``DecompositionResult``).

Canonical-form convention under test (documented here so the tests pin it):
*the most specific named entity wins*. A versioned token like ``"AXI4"``
canonicalizes to ``"AXI4"`` (NOT collapsed to the ``"AXI"`` umbrella); the bare
word ``"AXI"`` canonicalizes to ``"AXI"``. Informal phrases map to their
canonical entity (``"hub-based"`` -> ``"CHI"``, ``"extension-based"`` -> ``"ACE"``).
"""
from __future__ import annotations

import pytest

from src.retrieval.routing import (
    PROTOCOL_GLOSSARY,
    DecompositionResult,
    DocCard,
    RoutingResult,
    canonicalize_terms,
    detect_comparison_intent,
)


# ─── detect_comparison_intent — TRUE cases (design examples) ────────────────


@pytest.mark.parametrize(
    "query",
    [
        "Compare AMBA AXI4 / AXI5 / CHI",
        "AXI4 vs CHI",
        "AXI4 vs. CHI",
        "AXI4 versus CHI",
        "the hub-based approach or the extension-based one?",
        "difference between AHB and APB",
        "differences between AHB and APB",
        "How does AXI compare to CHI?",
        "AHB compared to APB",
        "What is the comparison between AXI and ACE",
    ],
)
def test_detect_comparison_intent_true(query):
    """Comparison/juxtaposition phrasing must be detected (case-insensitive)."""
    assert detect_comparison_intent(query) is True
    # Case-insensitivity sanity check.
    assert detect_comparison_intent(query.upper()) is True


# ─── detect_comparison_intent — FALSE cases (plain single-topic) ────────────


@pytest.mark.parametrize(
    "query",
    [
        "reset value of Filters_N_Status at 0x0064",
        "what is the reset value of Filters_N_Status",
        "explain the MBIST flow",
        "steps before inserting MBIST for a block",
        "describe the AXI4 write channel handshake",
        "list the AHB signals",
        "",
    ],
)
def test_detect_comparison_intent_false(query):
    """Plain single-topic queries must NOT be flagged as comparisons.

    Note: "before" is sequencing, not comparison; bare "and" inside a single
    topic (e.g. "AHB signals and timing") must not by itself trigger.
    """
    assert detect_comparison_intent(query) is False


def test_detect_comparison_intent_bare_and_is_not_comparison():
    """A bare 'and' joining aspects of one topic is not a comparison."""
    assert detect_comparison_intent("the AHB signals and their timing") is False


def test_detect_comparison_intent_slash_list_of_entities():
    """A slash-list of known entities is a comparison."""
    assert detect_comparison_intent("AXI4 / AXI5 / CHI") is True


# ─── canonicalize_terms ─────────────────────────────────────────────────────


def test_canonicalize_hub_based_to_chi():
    assert "CHI" in canonicalize_terms("the hub-based approach")


def test_canonicalize_extension_based_to_ace():
    assert "ACE" in canonicalize_terms("the extension-based one")


def test_canonicalize_implicit_comparison_pair():
    """The design's implicit-comparison example resolves to [CHI, ACE]."""
    out = canonicalize_terms(
        "the hub-based approach vs the extension-based one"
    )
    assert set(out) == {"CHI", "ACE"}


def test_canonicalize_axi4_vs_chi():
    """Versioned token wins: AXI4 stays AXI4 (not collapsed to AXI)."""
    out = canonicalize_terms("AXI4 vs CHI")
    assert set(out) == {"AXI4", "CHI"}


def test_canonicalize_bare_axi():
    """The bare umbrella term resolves to AXI."""
    assert canonicalize_terms("describe the AXI protocol") == ["AXI"]


def test_canonicalize_long_form_aliases():
    assert "CHI" in canonicalize_terms("the coherent hub interface")
    assert "AXI" in canonicalize_terms("the advanced extensible interface")


def test_canonicalize_deduplicates_and_is_order_stable():
    """Repeated mentions de-duplicate; result order is deterministic."""
    out = canonicalize_terms("CHI vs CHI and the hub-based approach")
    assert out == ["CHI"]


def test_canonicalize_plain_text_empty():
    assert canonicalize_terms("reset value of Filters_N_Status at 0x0064") == []
    assert canonicalize_terms("") == []


def test_canonicalize_case_insensitive():
    assert set(canonicalize_terms("axi4 VS chi")) == {"AXI4", "CHI"}


def test_canonicalize_does_not_match_substrings():
    """'AXIS' or 'TAXI' must not match the 'AXI' entity (word-boundary)."""
    assert canonicalize_terms("the data axis labels") == []
    assert canonicalize_terms("call a taxi") == []


# ─── PROTOCOL_GLOSSARY structure ────────────────────────────────────────────


REQUIRED_CANONICAL_KEYS = {
    "AMBA",
    "AXI",
    "AXI3",
    "AXI4",
    "AXI5",
    "ACE",
    "ACE-Lite",
    "AHB",
    "APB",
    "CHI",
}


def test_glossary_required_keys_present():
    assert REQUIRED_CANONICAL_KEYS.issubset(set(PROTOCOL_GLOSSARY))


def test_glossary_entry_shape():
    """Every entry is a dict with aliases:list, spec_ids:list, description:str."""
    for canonical, entry in PROTOCOL_GLOSSARY.items():
        assert isinstance(canonical, str)
        assert isinstance(entry, dict)
        assert set(entry) >= {"aliases", "spec_ids", "description"}
        assert isinstance(entry["aliases"], list)
        assert all(isinstance(a, str) for a in entry["aliases"])
        assert isinstance(entry["spec_ids"], list)
        assert all(isinstance(s, str) for s in entry["spec_ids"])
        assert isinstance(entry["description"], str)
        assert entry["description"]  # non-empty


def test_glossary_informal_aliases_present():
    """The design's informal->canonical aliases are encoded."""
    assert "hub-based" in PROTOCOL_GLOSSARY["CHI"]["aliases"]
    assert "extension-based" in PROTOCOL_GLOSSARY["ACE"]["aliases"]
    assert "coherent hub interface" in PROTOCOL_GLOSSARY["CHI"]["aliases"]
    assert "advanced extensible interface" in PROTOCOL_GLOSSARY["AXI"]["aliases"]


def test_glossary_named_spec_ids_present():
    """Only the two spec IDs named in the design are asserted."""
    assert "IHI0022E" in PROTOCOL_GLOSSARY["AXI"]["spec_ids"]
    assert "IHI0098" in PROTOCOL_GLOSSARY["CHI"]["spec_ids"]


def test_glossary_aliases_resolve_via_canonicalize():
    """Every declared alias resolves back to its canonical entity."""
    for canonical, entry in PROTOCOL_GLOSSARY.items():
        for alias in entry["aliases"]:
            assert canonical in canonicalize_terms(alias), (
                f"alias {alias!r} did not resolve to {canonical!r}"
            )


# ─── typed contracts (schemas) ──────────────────────────────────────────────


def test_doccard_defaults():
    card = DocCard(
        document_id="doc-1",
        title="AXI Protocol Spec",
        section_headings=["Overview", "Channels"],
        card_text="AXI Protocol Spec\nOverview\nChannels",
    )
    assert card.document_id == "doc-1"
    assert card.summary == ""
    assert card.num_chunks == 0


def test_routing_result_defaults():
    rr = RoutingResult(doc_ids=["a", "b"])
    assert rr.scores == {}
    assert rr.used is False
    # Distinct instances do not share the mutable default.
    rr2 = RoutingResult(doc_ids=[])
    rr.scores["a"] = 1.0
    assert rr2.scores == {}


def test_decomposition_result_defaults():
    dr = DecompositionResult(subqueries=["q"])
    assert dr.method == "identity"
    assert dr.decomposed is False
