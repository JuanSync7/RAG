// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest";
import {
    TURN_LOOP_EVENT_NAMES,
    createActivityLog,
    isTurnLoopEvent,
    renderClarifyChips,
} from "../activityLog";

// isTurnLoopEvent gates the SSE switch: exactly the nine design-§8 names are
// turn-loop events; the pre-existing stream vocabulary must NOT match.
describe("isTurnLoopEvent", () => {
    it("accepts all nine turn-loop event names", () => {
        for (const name of [
            "turn_action",
            "hyde_query",
            "retrieve_result",
            "judge_verdict",
            "deep_study",
            "llm_call",
            "draft",
            "gate",
            "clarify",
        ]) {
            expect(isTurnLoopEvent(name)).toBe(true);
        }
        expect(TURN_LOOP_EVENT_NAMES.size).toBe(9);
    });

    it("rejects the pre-existing stream event names", () => {
        for (const name of ["token", "reasoning", "retrieval", "error", "done", ""]) {
            expect(isTurnLoopEvent(name)).toBe(false);
        }
    });
});

describe("createActivityLog", () => {
    it("renders a turn_action as a compact '#N ACTION — reason' line", () => {
        const log = createActivityLog();
        log.handle("turn_action", { index: 2, action: "RETRIEVE", reason: "need more evidence" });
        const line = log.root.querySelector(".activity-action");
        expect(line?.textContent).toContain("#2 RETRIEVE");
        expect(line?.textContent).toContain("need more evidence");
    });

    it("writes event text via textContent so markup in payloads cannot inject", () => {
        const log = createActivityLog();
        const hostile = '<img src=x onerror="window.__pwned=1"><script>bad()</script>';
        log.handle("turn_action", { index: 1, action: "ANSWER", reason: hostile });
        // The hostile string must appear as inert text, never as elements.
        expect(log.root.querySelector("img")).toBeNull();
        expect(log.root.querySelector("script")).toBeNull();
        expect(log.root.textContent).toContain(hostile);
    });

    it("shows the HyDE hypothetical verbatim, open when short", () => {
        const log = createActivityLog();
        const hypo = "The verification env is created via the flow tool `setup_env`.";
        log.handle("hyde_query", { round: 1, hypothetical_answer: hypo, search_terms: ["setup_env"] });
        const block = log.root.querySelector<HTMLDetailsElement>(".activity-hyde");
        expect(block?.open).toBe(true);
        expect(block?.querySelector(".activity-hyde-text")?.textContent).toBe(hypo);
        expect(block?.textContent).toContain("search terms: setup_env");
    });

    it("collapses (but keeps) a long HyDE hypothetical", () => {
        const log = createActivityLog();
        const hypo = "x".repeat(500);
        log.handle("hyde_query", { round: 2, hypothetical_answer: hypo });
        const block = log.root.querySelector<HTMLDetailsElement>(".activity-hyde");
        expect(block?.open).toBe(false);
        expect(block?.querySelector(".activity-hyde-text")?.textContent).toBe(hypo);
    });

    it("renders retrieve_result as '+N new (pool P): doc › heading'", () => {
        const log = createActivityLog();
        log.handle("retrieve_result", {
            round: 1,
            added: 5,
            dup: 2,
            pool_size: 12,
            top: [{ doc: "verif_guide.md", heading: "Setup", score: 0.91 }],
        });
        const line = log.root.querySelector(".activity-retrieve");
        expect(line?.textContent).toContain("+5 new (pool 12), 2 dup");
        expect(line?.textContent).toContain("verif_guide.md › Setup");
    });

    it("renders judge_verdict with kept/confidence/missing information", () => {
        const log = createActivityLog();
        log.handle("judge_verdict", {
            round: 1,
            kept: 4,
            sufficient: false,
            confidence: 0.55,
            missing_information: "tool versions; license server",
        });
        expect(log.root.textContent).toContain("kept 4");
        expect(log.root.textContent).toContain("confidence 0.55");
        expect(log.root.textContent).toContain("insufficient");
        expect(log.root.textContent).toContain("missing: tool versions; license server");
    });

    it("renders deep_study as 'reading <title> — window w/of'", () => {
        const log = createActivityLog();
        log.handle("deep_study", {
            document_id: "d1",
            title: "Verification guide",
            window: 2,
            of_windows: 8,
            notes_preview: "found the env bootstrap section",
        });
        expect(log.root.textContent).toContain("reading Verification guide — window 2/8");
        expect(log.root.textContent).toContain("found the env bootstrap section");
    });

    it("renders llm_call as a dim one-liner with seconds and token total", () => {
        const log = createActivityLog();
        log.handle("llm_call", {
            alias: "controller",
            purpose: "decide",
            ms: 1234,
            prompt_tokens: 800,
            completion_tokens: 50,
        });
        const line = log.root.querySelector(".activity-llm");
        expect(line?.classList.contains("dim")).toBe(true);
        expect(line?.textContent).toContain("controller");
        expect(line?.textContent).toContain("1.2s");
        expect(line?.textContent).toContain("850tok");
    });

    it("streams draft deltas and REPLACES the draft on a new attempt", () => {
        const log = createActivityLog();
        log.handle("draft", { attempt: 1, text_delta: "First " });
        log.handle("draft", { attempt: 1, text_delta: "draft." });
        const draftEl = log.root.querySelector(".activity-draft-text");
        expect(draftEl?.textContent).toBe("First draft.");
        log.handle("draft", { attempt: 2, text_delta: "Second draft." });
        expect(log.root.querySelectorAll(".activity-draft-text")).toHaveLength(1);
        expect(draftEl?.textContent).toBe("Second draft.");
        expect(log.root.querySelector(".activity-draft-label")?.textContent).toContain("attempt 2");
    });

    it("renders gate pass/fail with score vs threshold and weakest component", () => {
        const log = createActivityLog();
        log.handle("gate", { attempt: 1, score: 0.62, threshold: 0.75, passed: false, weakest: "citation" });
        const fail = log.root.querySelector(".activity-gate.fail");
        expect(fail?.textContent).toContain("FAIL");
        expect(fail?.textContent).toContain("0.62 vs threshold 0.75");
        expect(fail?.textContent).toContain("weakest: citation");
        log.handle("gate", { attempt: 2, score: 0.81, threshold: 0.75, passed: true, weakest: "self" });
        expect(log.root.querySelector(".activity-gate.pass")?.textContent).toContain("PASS");
    });

    it("starts open, then finalize(true) collapses and relabels with step count", () => {
        const log = createActivityLog();
        expect(log.root.open).toBe(true);
        log.handle("turn_action", { index: 1, action: "RETRIEVE", reason: "r" });
        log.handle("gate", { attempt: 1, passed: true });
        log.finalize(true);
        expect(log.root.open).toBe(false);
        expect(log.root.querySelector(".activity-summary")?.textContent).toContain("2 steps");
        expect(log.eventCount()).toBe(2);
    });

    it("finalize(false) relabels but leaves the log expanded (clarify turns)", () => {
        const log = createActivityLog();
        log.handle("clarify", { question: "Which project?" });
        log.finalize(false);
        expect(log.root.open).toBe(true);
        expect(log.root.textContent).toContain("clarify: Which project?");
    });

    it("ignores non-turn-loop event names without rendering or counting", () => {
        const log = createActivityLog();
        log.handle("token", { token: "hi" });
        expect(log.eventCount()).toBe(0);
        expect(log.root.querySelectorAll(".activity-line")).toHaveLength(0);
    });
});

describe("renderClarifyChips", () => {
    it("renders the question plus one chip per hint and scoping question", () => {
        const el = renderClarifyChips(
            {
                question: "Which flow do you mean?",
                hints: ["Digital top", "Analog macro"],
                scoping_questions: ["How do I set up the digital top env?"],
            },
            () => undefined,
        );
        expect(el).not.toBeNull();
        expect(el?.querySelector(".clarify-question")?.textContent).toBe("Which flow do you mean?");
        expect(el?.querySelectorAll(".clarify-chip")).toHaveLength(3);
        expect(el?.querySelectorAll(".clarify-chip-scope")).toHaveLength(1);
    });

    it("resubmits the chip's own text on click", () => {
        const onResubmit = vi.fn();
        const el = renderClarifyChips(
            { question: "Which one?", hints: ["Option A", "Option B"] },
            onResubmit,
        )!;
        const chips = el.querySelectorAll<HTMLButtonElement>(".clarify-chip");
        chips[1].click();
        expect(onResubmit).toHaveBeenCalledTimes(1);
        expect(onResubmit).toHaveBeenCalledWith("Option B");
    });

    it("writes chip text via textContent so LLM-authored hints cannot inject", () => {
        const el = renderClarifyChips(
            { question: "<b>q</b>", hints: ['<img src=x onerror="1">'] },
            () => undefined,
        )!;
        expect(el.querySelector("img")).toBeNull();
        expect(el.querySelector("b")).toBeNull();
        expect(el.querySelector(".clarify-chip")?.textContent).toBe('<img src=x onerror="1">');
    });

    it("returns null when the payload has nothing renderable", () => {
        expect(renderClarifyChips({}, () => undefined)).toBeNull();
        expect(renderClarifyChips({ question: "  ", hints: [] }, () => undefined)).toBeNull();
    });
});
