// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import { normalizeMarkdown, formatApiPayload } from "../markdown";

// normalizeMarkdown splits inline list items onto their own lines and injects a
// synthetic GFM header separator for orphaned pipe-table fragments, while leaving
// fenced code blocks untouched.
describe("normalizeMarkdown", () => {
    it("splits an inline numbered list onto separate lines", () => {
        expect(normalizeMarkdown("1. Foo 2. Bar")).toBe("1. Foo\n2. Bar");
    });

    it("splits an inline bullet list when the line starts with a bullet", () => {
        expect(normalizeMarkdown("- A - B")).toBe("- A\n- B");
    });

    it("does not split standalone dashes in prose (not bullet-led)", () => {
        expect(normalizeMarkdown("cost-benefit - tradeoff here")).toBe(
            "cost-benefit - tradeoff here",
        );
    });

    it("leaves a short line (under 3 words) unchanged", () => {
        expect(normalizeMarkdown("1. Foo")).toBe("1. Foo");
    });

    it("does not modify content inside fenced code blocks", () => {
        const src = "```\n1. Foo 2. Bar\n```";
        expect(normalizeMarkdown(src)).toBe(src);
    });

    it("injects a separator row for an orphan pipe table", () => {
        const src = "| A | B |\n| 1 | 2 |";
        expect(normalizeMarkdown(src)).toBe("| A | B |\n| --- | --- |\n| 1 | 2 |");
    });

    it("does not double-inject a separator when one already exists", () => {
        const src = "| A | B |\n| --- | --- |\n| 1 | 2 |";
        expect(normalizeMarkdown(src)).toBe(src);
    });
});

// formatApiPayload coerces arbitrary API payloads into a markdown-friendly string.
describe("formatApiPayload", () => {
    it("returns empty string for null", () => {
        expect(formatApiPayload(null)).toBe("");
    });

    it("returns empty string for undefined", () => {
        expect(formatApiPayload(undefined)).toBe("");
    });

    it("passes strings through unchanged", () => {
        expect(formatApiPayload("**already markdown**")).toBe("**already markdown**");
    });

    it("stringifies numbers", () => {
        expect(formatApiPayload(42)).toBe("42");
    });

    it("stringifies booleans", () => {
        expect(formatApiPayload(false)).toBe("false");
    });

    it("renders an object as a humanized bullet list", () => {
        expect(formatApiPayload({ doc_count: 3 })).toBe("- **Doc Count**: 3");
    });

    it("renders a primitive array as a bullet list", () => {
        expect(formatApiPayload(["a", "b"])).toBe("\n- a\n- b");
    });

    it("renders an empty list marker for an empty array", () => {
        expect(formatApiPayload([])).toBe("_(empty list)_");
    });

    it("renders the empty-string placeholder for blank string values", () => {
        expect(formatApiPayload({ note: "  " })).toBe("- **Note**: _(empty)_");
    });

    it("renders the dash placeholder for null values", () => {
        expect(formatApiPayload({ note: null })).toBe("- **Note**: _—_");
    });
});
