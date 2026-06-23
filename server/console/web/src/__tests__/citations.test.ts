// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import { buildCitationsHtml } from "../citations";
import type { ChunkResult } from "../user-types";

// buildCitationsHtml shows the chunks the answer actually CITED (parsed from the
// `[N]` markers in the answer text) first, each tagged with its original [N], and
// tucks the remaining retrieved-but-uncited chunks under a collapsed expander.
// When no citations are detectable it falls back to listing every retrieved chunk.

function chunk(source: string, text = "body text"): ChunkResult {
    return { text, score: 0.9, metadata: { source } };
}

describe("buildCitationsHtml — cited-only filtering", () => {
    it("returns empty string when there are no results", () => {
        expect(buildCitationsHtml([])).toBe("");
    });

    it("shows only cited chunks first and collapses the rest under 'show more'", () => {
        const results = [chunk("a.pdf"), chunk("b.pdf"), chunk("c.pdf")];
        const html = buildCitationsHtml(results, "See [1] and [3] for details.");
        // Label reflects the CITED count, not the retrieved count.
        expect(html).toContain("2 sources cited");
        // Cited cards carry their original [N] badge.
        expect(html).toContain('class="citation-num">[1]');
        expect(html).toContain('class="citation-num">[3]');
        // The uncited chunk is collapsed under the expander.
        expect(html).toContain("citations-more");
        expect(html).toContain("Show 1 more retrieved source");
        expect(html).toContain('class="citation-num">[2]');
        // Cited cards render before the expander.
        expect(html.indexOf("[1]")).toBeLessThan(html.indexOf("citations-more"));
        expect(html.indexOf("[3]")).toBeLessThan(html.indexOf("citations-more"));
    });

    it("singularizes the label and omits the expander when every chunk is cited", () => {
        const html = buildCitationsHtml([chunk("only.pdf")], "Per [1] this is so.");
        expect(html).toContain("1 source cited");
        expect(html).not.toContain("citations-more");
    });

    it("falls back to listing all chunks as 'retrieved' with no answer text", () => {
        const results = [chunk("a.pdf"), chunk("b.pdf")];
        const html = buildCitationsHtml(results);
        expect(html).toContain("2 sources retrieved");
        expect(html).not.toContain("citations-more");
        expect(html).toContain('class="citation-num">[1]');
        expect(html).toContain('class="citation-num">[2]');
    });

    it("ignores out-of-range / zero citation markers and falls back to retrieved", () => {
        const results = [chunk("a.pdf"), chunk("b.pdf")];
        // [9] is beyond the result set; [0] is not a valid 1-based index.
        const html = buildCitationsHtml(results, "Nonsense [9] and [0].");
        expect(html).toContain("2 sources retrieved");
        expect(html).not.toContain("citations-more");
    });

    it("deduplicates repeated citations of the same chunk", () => {
        const results = [chunk("a.pdf"), chunk("b.pdf")];
        const html = buildCitationsHtml(results, "[1] says X. [1] also says Y.");
        expect(html).toContain("1 source cited");
        expect(html).toContain("Show 1 more retrieved source");
    });
});
