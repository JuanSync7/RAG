"""Caption-label parsing for Figure/Fig prefixes.

Mirrors the Table caption-label tests in spirit: cover positive forms
("Figure 4-1", "Fig. 2", "Figure 10.3") and exclude false positives like
"figured out" or "configure" that share the prefix.
"""

from __future__ import annotations

import pytest

from src.ingest.support.docling import _extract_figure_caption_label


@pytest.mark.parametrize(
    "caption, expected",
    [
        ("Figure 4-1: SoC block diagram", "Figure 4-1"),
        ("Figure 4-1", "Figure 4-1"),
        ("Fig. 2 — pin layout", "Figure 2"),
        ("Fig 2", "Figure 2"),
        ("FIGURE 10.3 power tree", "Figure 10.3"),
        ("figure 7: clock", "Figure 7"),
        ("  Figure 3-22  Reset behaviour", "Figure 3-22"),
    ],
)
def test_extract_figure_caption_label_positive(caption, expected):
    assert _extract_figure_caption_label(caption) == expected


@pytest.mark.parametrize(
    "caption",
    [
        "",
        "figured out the wiring",
        "Configure the GPIO matrix",
        "Refigure the layout",
        "Block diagram of the chip",
        "Table 4-1: GPIO mux",
    ],
)
def test_extract_figure_caption_label_negative(caption):
    assert _extract_figure_caption_label(caption) == ""
