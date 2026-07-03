// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from "vitest";
import { renderQueryProcessing } from "./queryProcessing";
import type { StreamEventData } from "./user-types";

// Mirror thread.ts appendPendingAssistant + streaming.ts reasoningBody: a
// bubble-wrap whose children are [typing, reasoning-block, bubble, ...].
function makeBubbleWrap(withReasoning: boolean): { wrap: HTMLElement; bubble: HTMLElement } {
    const wrap = document.createElement("div");
    wrap.className = "bubble-wrap";
    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    wrap.appendChild(typing);
    if (withReasoning) {
        const reasoning = document.createElement("details");
        reasoning.className = "reasoning-block";
        wrap.appendChild(reasoning);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    wrap.appendChild(bubble);
    return { wrap, bubble };
}

// The exact shape a live retrieval SSE event carries on this branch.
const LIVE: StreamEventData = {
    processed_query: "What are the steps before inserting MBIST for a block?",
    query_confidence: 0.5,
    metadata: {
        agentic_retrieval: {
            rounds_run: 1,
            hyde_variants_tried: 1,
            hyde_failures: 0,
            kept_count: 4,
            ranker: "judge",
            stop_reason: "single_round",
            elapsed_ms: 2790,
            hyde_rounds: [
                {
                    round: 1,
                    hypothetical_answer: "Before inserting MBIST for a block, you need to define the test vectors and fault models.",
                    search_terms: ["test vectors", "fault models", "MBIST integration"],
                    target_aspect: "Defining test vectors and fault models before integrating MBIST",
                    fell_back: false,
                },
            ],
        },
    },
};

describe("renderQueryProcessing", () => {
    let wrap: HTMLElement;
    beforeEach(() => {
        wrap = makeBubbleWrap(true).wrap;
    });

    it("inserts the panel ABOVE the reasoning block", () => {
        renderQueryProcessing(wrap, LIVE, "steps before MBIST?");
        const panel = wrap.querySelector(".query-processing-block");
        const reasoning = wrap.querySelector(".reasoning-block");
        expect(panel).toBeTruthy();
        // panel must come before reasoning in document order
        const pos = panel!.compareDocumentPosition(reasoning!);
        expect(pos & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it("renders the HyDE hypothetical, target aspect and search terms as text", () => {
        renderQueryProcessing(wrap, LIVE, "steps before MBIST?");
        const txt = wrap.querySelector(".query-processing-block")!.textContent || "";
        expect(txt).toContain("HyDE retrieval");
        expect(txt).toContain("Defining test vectors and fault models");
        expect(txt).toContain("test vectors, fault models, MBIST integration");
        expect(txt).toContain("Before inserting MBIST for a block");
        expect(txt).toContain("stop: single_round");
    });

    it("summarizes HyDE round count in the <summary>", () => {
        renderQueryProcessing(wrap, LIVE, "q");
        const s = wrap.querySelector(".qp-summary")!.textContent || "";
        expect(s).toContain("Query processing");
        expect(s).toContain("HyDE");
        expect(s).toContain("1 round");
    });

    it("renders nothing on the linear path (empty metadata, query unchanged)", () => {
        renderQueryProcessing(wrap, { processed_query: "q", metadata: {} }, "q");
        expect(wrap.querySelector(".query-processing-block")).toBeNull();
    });

    it("is idempotent — a second render replaces, never stacks", () => {
        renderQueryProcessing(wrap, LIVE, "q");
        renderQueryProcessing(wrap, LIVE, "q");
        expect(wrap.querySelectorAll(".query-processing-block").length).toBe(1);
    });

    it("flags a literal-fallback round with a warning badge", () => {
        const fb: StreamEventData = {
            metadata: {
                agentic_retrieval: {
                    rounds_run: 1,
                    hyde_failures: 1,
                    hyde_rounds: [{ round: 1, hypothetical_answer: "q", fell_back: true }],
                },
            },
        };
        renderQueryProcessing(wrap, fb, "q");
        const txt = wrap.querySelector(".query-processing-block")!.textContent || "";
        expect(txt).toContain("literal fallback");
        expect(txt).toContain("fell back");
        expect(wrap.querySelector(".qp-badge-warn")).toBeTruthy();
    });

    it("does not use innerHTML for model text (no injection)", () => {
        const evil: StreamEventData = {
            metadata: {
                agentic_retrieval: {
                    rounds_run: 1,
                    hyde_rounds: [{ round: 1, hypothetical_answer: "<img src=x onerror=alert(1)>" }],
                },
            },
        };
        renderQueryProcessing(wrap, evil, "q");
        // The <img> must be inert text, not a real element.
        expect(wrap.querySelector(".query-processing-block img")).toBeNull();
        expect(wrap.querySelector(".query-processing-block")!.textContent).toContain("<img src=x");
    });

    it("shows the rewrite section when processed_query differs from the raw query", () => {
        renderQueryProcessing(wrap, { processed_query: "rewritten form", metadata: {} }, "original");
        const txt = wrap.querySelector(".query-processing-block")?.textContent || "";
        expect(txt).toContain("Query rewrite");
        expect(txt).toContain("rewritten form");
    });

    it("stays above the thinking block in the REAL sequence (retrieval before reasoning)", () => {
        // Real order: the retrieval SSE event fires before any reasoning, so the
        // panel is inserted while no .reasoning-block exists yet (anchors before
        // .bubble); the reasoning block is then inserted before .bubble too. The
        // panel must remain first.
        const { wrap: w, bubble } = makeBubbleWrap(false); // no reasoning yet
        renderQueryProcessing(w, LIVE, "q");
        // Now simulate streaming.ts reasoningBody(): insert reasoning before bubble.
        const reasoning = document.createElement("details");
        reasoning.className = "reasoning-block";
        w.insertBefore(reasoning, bubble);
        const kids = Array.from(w.children).map((c) => c.className.split(" ")[0]);
        expect(kids.indexOf("query-processing-block")).toBeLessThan(kids.indexOf("reasoning-block"));
        expect(kids.indexOf("reasoning-block")).toBeLessThan(kids.indexOf("bubble"));
    });
});
