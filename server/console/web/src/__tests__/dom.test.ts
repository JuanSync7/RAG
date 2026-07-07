// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { byId, escHtml, fmtRelative } from "../dom";

// escHtml escapes the six HTML-significant characters used by the console
// when interpolating untrusted text into markup strings.
describe("escHtml", () => {
    it("escapes ampersand", () => {
        expect(escHtml("a&b")).toBe("a&amp;b");
    });

    it("escapes less-than and greater-than", () => {
        expect(escHtml("<b>")).toBe("&lt;b&gt;");
    });

    it("escapes double quotes", () => {
        expect(escHtml('say "hi"')).toBe("say &quot;hi&quot;");
    });

    it("escapes single quotes", () => {
        expect(escHtml("it's")).toBe("it&#39;s");
    });

    it("escapes forward slash", () => {
        expect(escHtml("a/b")).toBe("a&#x2F;b");
    });

    it("escapes ampersand first so entities are not double-escaped", () => {
        expect(escHtml("<")).toBe("&lt;");
        expect(escHtml("&lt;")).toBe("&amp;lt;");
    });

    it("leaves plain text untouched", () => {
        expect(escHtml("hello world")).toBe("hello world");
    });

    it("escapes a full XSS payload", () => {
        expect(escHtml('<img src=x onerror="alert(1)">')).toBe(
            "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;",
        );
    });
});

// byId resolves an element by id and throws a clear error when it is missing.
describe("byId", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
    });

    it("returns the element when present", () => {
        const div = document.createElement("div");
        div.id = "target";
        document.body.appendChild(div);
        expect(byId("target")).toBe(div);
    });

    it("throws a descriptive error naming the id when missing", () => {
        expect(() => byId("nope")).toThrow("Missing required element #nope");
    });
});

// fmtRelative buckets a timestamp into Today / Yesterday / a short date.
describe("fmtRelative", () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it("returns 'Today' for a timestamp under 24h old", () => {
        const now = new Date("2026-06-03T12:00:00Z").getTime();
        vi.useFakeTimers();
        vi.setSystemTime(now);
        expect(fmtRelative(now - 3600_000)).toBe("Today");
    });

    it("returns 'Yesterday' for a timestamp 24-48h old", () => {
        const now = new Date("2026-06-03T12:00:00Z").getTime();
        vi.useFakeTimers();
        vi.setSystemTime(now);
        expect(fmtRelative(now - 90000_000)).toBe("Yesterday");
    });

    it("returns a short date for timestamps older than 48h", () => {
        const now = new Date("2026-06-03T12:00:00Z").getTime();
        vi.useFakeTimers();
        vi.setSystemTime(now);
        const result = fmtRelative(now - 5 * 86400_000);
        expect(result).not.toBe("Today");
        expect(result).not.toBe("Yesterday");
    });
});
