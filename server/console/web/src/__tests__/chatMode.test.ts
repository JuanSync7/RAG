// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import { docKeyFromMeta, groupSources, isSourcesTurn } from "../chatMode";
import type { SourceRef } from "../user-types";

// docKeyFromMeta resolves a stable doc identifier from a citation metadata blob,
// preferring document_id > source_key > source_uri > source.
describe("docKeyFromMeta", () => {
    it("returns empty string for null", () => {
        expect(docKeyFromMeta(null)).toBe("");
    });

    it("returns empty string for undefined", () => {
        expect(docKeyFromMeta(undefined)).toBe("");
    });

    it("prefers document_id over all other keys", () => {
        expect(
            docKeyFromMeta({ document_id: "doc1", source_key: "sk", source: "s" }),
        ).toBe("doc1");
    });

    it("falls back to source_key when document_id is absent", () => {
        expect(docKeyFromMeta({ source_key: "sk", source_uri: "uri" })).toBe("sk");
    });

    it("falls back to source_uri before source", () => {
        expect(docKeyFromMeta({ source_uri: "uri", source: "s" })).toBe("uri");
    });

    it("falls back to source last", () => {
        expect(docKeyFromMeta({ source: "s" })).toBe("s");
    });

    it("coerces a non-string value to string", () => {
        expect(docKeyFromMeta({ document_id: 123 })).toBe("123");
    });

    it("trims surrounding whitespace", () => {
        expect(docKeyFromMeta({ source: "  s  " })).toBe("s");
    });
});

// isSourcesTurn flags a stored history turn that should render as cards: empty
// content AND at least one source.
describe("isSourcesTurn", () => {
    it("is true when content is blank and sources exist", () => {
        expect(isSourcesTurn({ content: "   ", sources: [{ source: "a" }] })).toBe(true);
    });

    it("is false when content is non-empty", () => {
        expect(isSourcesTurn({ content: "hi", sources: [{ source: "a" }] })).toBe(false);
    });

    it("is false when there are no sources", () => {
        expect(isSourcesTurn({ content: "", sources: [] })).toBe(false);
    });

    it("is false when sources is undefined", () => {
        expect(isSourcesTurn({ content: "" })).toBe(false);
    });
});

// groupSources collapses SourceRefs sharing a doc key, tracks the best score /
// excerpt, and returns groups sorted by best score descending.
describe("groupSources", () => {
    it("returns an empty array for no sources", () => {
        expect(groupSources([])).toEqual([]);
    });

    it("groups refs that share a doc key into one group", () => {
        const refs: SourceRef[] = [
            { document_id: "d1", score: 0.4, text: "lo" },
            { document_id: "d1", score: 0.9, text: "hi" },
        ];
        const groups = groupSources(refs);
        expect(groups).toHaveLength(1);
        expect(groups[0].refs).toHaveLength(2);
    });

    it("tracks the highest score and its excerpt as bestScore/excerpt", () => {
        const refs: SourceRef[] = [
            { document_id: "d1", score: 0.4, text: "lo" },
            { document_id: "d1", score: 0.9, text: "hi" },
        ];
        const groups = groupSources(refs);
        expect(groups[0].bestScore).toBe(0.9);
        expect(groups[0].excerpt).toBe("hi");
    });

    it("sorts multiple groups by best score descending", () => {
        const refs: SourceRef[] = [
            { document_id: "low", score: 0.2 },
            { document_id: "high", score: 0.95 },
            { document_id: "mid", score: 0.5 },
        ];
        const groups = groupSources(refs);
        expect(groups.map((g) => g.docKey)).toEqual(["high", "mid", "low"]);
    });

    it("derives the display name from the basename of the source path", () => {
        const groups = groupSources([{ source: "a/b/report.pdf", score: 1 }]);
        expect(groups[0].name).toBe("report.pdf");
    });

    it("assigns synthetic keys to keyless refs so they are not merged", () => {
        const groups = groupSources([{ score: 0.1 }, { score: 0.2 }]);
        expect(groups).toHaveLength(2);
        expect(groups.every((g) => g.docKey.startsWith("__synth_"))).toBe(true);
    });
});
