// @summary
// Query execution: SSE streaming + non-stream fallback. Owns sendQuery as the
// canonical entry point for any plain-text user message; slash-command flows
// reuse streamQuery/nonStreamQuery directly.
// @end-summary

import { byId, escHtml, fmtTime } from "./dom";
import { api, apiBase, authHeaders, getSettings } from "./api";
import { parseMarkdown } from "./markdown";
import { state, setActiveConversation } from "./state";
import { refs } from "./refs";
import { appendUserMsg, appendErrorMsg, appendPendingAssistant } from "./thread";
import { scrollToBottom } from "./scrollFab";
import { buildCitationsHtml, buildSourcesLineHtml, revealCitations } from "./citations";
import { updateContextIndicator, clearLastTurnStats } from "./contextWindow";
import { attachFeedback } from "./feedback";
import { loadConversations, updateConvTitle } from "./conversations";
import { getChatMode, getDeepResearch, getRetrievalSubMode, getSourcesTopK, appendSourcesTurn, applyDocState, cacheDocsFromSources, wireCitationActions, showDrSuggestionChip, hideDrSuggestionChip, registerDrSuggestionResubmit, renderFollowUps } from "./chatMode";
import { renderQueryProcessing } from "./queryProcessing";
import type { ChunkResult, QueryMetadata, SourceRef, StreamEventData, TokenBudget } from "./user-types";

function buildQueryBody(queryText: string): Record<string, unknown> {
    const s = getSettings();
    const body: Record<string, unknown> = {
        query: queryText,
        search_limit: parseInt(String(s.searchLimit ?? "10"), 10),
        rerank_top_k: parseInt(String(s.rerankTopK ?? "5"), 10),
        memory_enabled: s.memory_enabled !== false,
        conversation_id: state.activeConversationId ?? undefined,
    };
    // Tree retrieval per-request override (TREE_RETRIEVAL_DESIGN.md §6).
    // Only set on body when user has explicitly toggled it; absent ⇒ server uses config default.
    if (s.tree_retrieval !== undefined) {
        body.tree_retrieval = Boolean(s.tree_retrieval);
    }
    if (getChatMode() === "sources") {
        body.mode = "retrieval";
        body.retrieval_sub_mode = getRetrievalSubMode();
        const topK = getSourcesTopK();
        body.rerank_top_k = topK;
        body.search_limit = Math.max(parseInt(String(s.searchLimit ?? "10"), 10), topK * 2);
    }
    if (getDeepResearch()) {
        body.deep_research = true;
    }
    return body;
}

function chunkToSourceRef(c: ChunkResult): SourceRef {
    const m = c.metadata || {};
    return {
        source: String(m.source ?? ""),
        source_uri: String(m.source_uri ?? ""),
        source_key: String(m.source_key ?? ""),
        document_id: String(m.document_id ?? ""),
        section: String(m.section ?? m.heading ?? ""),
        score: c.score,
        text: c.text,
        original_char_start: typeof m.original_char_start === "number" ? (m.original_char_start as number) : undefined,
        original_char_end: typeof m.original_char_end === "number" ? (m.original_char_end as number) : undefined,
    };
}

async function sourcesOnlyQuery(queryText: string): Promise<void> {
    appendUserMsg(queryText);
    try {
        const data = await api<{
            results?: ChunkResult[];
            conversation_id?: string;
            relevant_doc_ids?: string[];
            ignored_doc_ids?: string[];
            token_budget?: TokenBudget;
            dr_suggestion?: DrSuggestionPayload;
        }>("POST", "/console/query", buildQueryBody(queryText));
        maybeShowDrChip(data.dr_suggestion, queryText);
        const cid = String(data.conversation_id ?? "").trim();
        if (cid) setActiveConversation(cid);
        const sources = (data.results ?? []).map(chunkToSourceRef);
        appendSourcesTurn(refs.thread, sources);
        if (data.relevant_doc_ids || data.ignored_doc_ids) {
            applyDocState(data.relevant_doc_ids ?? [], data.ignored_doc_ids ?? []);
        }
        if (data.token_budget) {
            updateContextIndicator(data.token_budget);
        }
        scrollToBottom();
        await loadConversations();
        updateConvTitle();
    } catch (err) {
        appendErrorMsg("Sources query failed: " + String(err));
    }
}

export async function streamQuery(queryText: string): Promise<void> {
    if (state.isStreaming) {
        state.streamAbortCtrl?.abort();
    }
    state.isStreaming = true;
    state.streamAbortCtrl = new AbortController();
    state.pendingQueryText = queryText;

    state.pendingUserGroup = appendUserMsg(queryText);
    const pending = appendPendingAssistant();
    const { bubbleEl, typingEl, citationsEl, actionsEl, metaEl } = pending;
    state.pendingAssistantGroup = pending.group;
    attachFeedback(pending.fbUpBtn, pending.fbDownBtn, {
        conversationId: state.activeConversationId,
        query: queryText,
        answer: () => bubbleEl.innerText,
    });
    setSendButtonStop(true);

    const url = apiBase() + "/query/stream";
    let response: Response;
    try {
        response = await fetch(url, {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify(buildQueryBody(queryText)),
            signal: state.streamAbortCtrl.signal,
        });
    } catch (err) {
        typingEl.remove();
        bubbleEl.innerHTML = "&#9888; Network error: " + escHtml(String(err));
        bubbleEl.classList.add("error-bubble");
        bubbleEl.style.display = "block";
        state.isStreaming = false;
        state.pendingUserGroup = null;
        state.pendingAssistantGroup = null;
        state.pendingQueryText = "";
        setSendButtonStop(false);
        return;
    }

    if (!response.ok || !response.body) {
        typingEl.remove();
        let detail = "";
        try {
            const body = await response.text();
            const parsed = JSON.parse(body) as { detail?: unknown };
            const d = parsed.detail;
            if (Array.isArray(d)) {
                detail = d
                    .map((e: { loc?: unknown[]; msg?: string }) =>
                        `${(e.loc ?? []).slice(1).join(".")}: ${e.msg ?? ""}`,
                    )
                    .join("; ");
            } else if (typeof d === "string") {
                detail = d;
            }
        } catch {
            /* ignore — fall through to status-only */
        }
        bubbleEl.innerHTML =
            `&#9888; Stream error (HTTP ${response.status})` +
            (detail ? ` — ${escHtml(detail)}` : "");
        bubbleEl.classList.add("error-bubble");
        bubbleEl.style.display = "block";
        state.isStreaming = false;
        state.pendingUserGroup = null;
        state.pendingAssistantGroup = null;
        state.pendingQueryText = "";
        setSendButtonStop(false);
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let started = false;
    let errorShown = false;
    let pendingClarification = "";
    let firstTokenAt = 0;
    let lastTokenAt = 0;
    let tokenEventCount = 0;
    let lastBudget: import("./user-types").TokenBudget | null = null;
    // Retrieved chunks arrive in the `retrieval` event (before any answer token);
    // stashed here so the `done` handler can render citations with the COMPLETE
    // answer and thus collapse them to the chunks actually cited ([N]).
    let lastResults: ChunkResult[] = [];

    // Live reasoning ("thinking") — a collapsible block rendered above the answer
    // bubble, filled from `reasoning` SSE events emitted by reasoning models.
    let reasoningText = "";
    let reasoningStartAt = 0;

    clearLastTurnStats();

    let renderRaf = 0;
    const flushRender = () => {
        renderRaf = 0;
        bubbleEl.innerHTML = parseMarkdown(answer);
    };
    const scheduleRender = () => {
        if (renderRaf) return;
        renderRaf = requestAnimationFrame(flushRender);
    };
    const cancelRender = () => {
        if (renderRaf) {
            cancelAnimationFrame(renderRaf);
            renderRaf = 0;
        }
    };

    // Returns this turn's reasoning-body element, lazily creating the collapsible
    // block above the answer bubble on first use. Scoped to this bubble-wrap, so
    // it never collides with reasoning blocks from earlier turns. (Querying the
    // DOM rather than caching in a closure-mutated var sidesteps a TS narrowing
    // pitfall where such vars get inferred as `never` at read sites.)
    const reasoningBody = (): HTMLElement => {
        const wrap = bubbleEl.parentElement;
        let el = wrap?.querySelector<HTMLDetailsElement>(".reasoning-block") ?? null;
        if (!el) {
            el = document.createElement("details");
            el.className = "reasoning-block";
            el.open = true;
            el.innerHTML =
                `<summary class="reasoning-summary">&#128173; Thinking&hellip;</summary>` +
                `<div class="reasoning-body"></div>`;
            wrap?.insertBefore(el, bubbleEl);
            reasoningStartAt = performance.now();
        }
        return el.querySelector<HTMLElement>(".reasoning-body")!;
    };
    // Relabel the summary with the elapsed thinking time; collapse once the
    // answer begins (when there is no answer, leave it open so the user can read
    // what the model worked through).
    const finalizeReasoning = (collapse: boolean) => {
        const el = bubbleEl.parentElement?.querySelector<HTMLDetailsElement>(".reasoning-block");
        if (!el) return;
        const summaryEl = el.querySelector(".reasoning-summary");
        if (summaryEl) {
            const secs = reasoningStartAt ? (performance.now() - reasoningStartAt) / 1000 : 0;
            summaryEl.innerHTML =
                secs > 0 ? `&#128173; Thought for ${secs.toFixed(1)}s` : "&#128173; Thought process";
        }
        if (collapse) el.open = false;
    };

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop() || "";

            for (const evt of events) {
                const lines = evt.split("\n");
                const evtType = (lines.find((l) => l.startsWith("event: ")) ?? "").slice(7);
                const dataRaw = (lines.find((l) => l.startsWith("data: ")) ?? "data: {}").slice(6);
                let data: StreamEventData;
                try {
                    data = JSON.parse(dataRaw) as StreamEventData;
                } catch {
                    data = {};
                }

                if (evtType === "token") {
                    if (!started) {
                        typingEl.style.display = "none";
                        bubbleEl.style.display = "block";
                        bubbleEl.classList.add("streaming");
                        started = true;
                        firstTokenAt = performance.now();
                        finalizeReasoning(true);
                    }
                    lastTokenAt = performance.now();
                    tokenEventCount++;
                    answer += data.token || "";
                    scheduleRender();
                    scrollToBottom();
                } else if (evtType === "reasoning") {
                    // Live chain-of-thought: dimmed, collapsible, above the answer.
                    // Raw model text → textContent (never innerHTML) so it can't inject markup.
                    typingEl.style.display = "none";
                    reasoningText += String(data.text ?? "");
                    reasoningBody().textContent = reasoningText;
                    scrollToBottom();
                } else if (evtType === "retrieval") {
                    const cid = String(data.conversation_id ?? "").trim();
                    if (cid) setActiveConversation(cid);
                    const drSugg = (data as { dr_suggestion?: DrSuggestionPayload }).dr_suggestion;
                    maybeShowDrChip(drSugg, queryText);

                    const clar = String(data.clarification_message ?? "").trim();
                    if (clar) {
                        // Prefix the clarification with a typed reason badge
                        // when the chain surfaces ``ask_user_reason`` (parity
                        // with the CLI). Reason -> short human label.
                        const reason = String(data.ask_user_reason ?? "").trim();
                        const reasonLabels: Record<string, string> = {
                            sanitizer_reject: "Empty or invalid query",
                            injection_blocked: "Query blocked by safety rails",
                            vague_query: "Query too vague to retrieve",
                            budget_exhausted: "Retrieval timeout",
                            no_results: "No matching documents",
                        };
                        const label = reasonLabels[reason];
                        pendingClarification = label ? `[${label}] ${clar}` : clar;
                    }

                    if (data.token_budget) {
                        lastBudget = data.token_budget;
                        updateContextIndicator(lastBudget);
                    }

                    const results = (data.results ?? []) as ChunkResult[];
                    if (results.length) {
                        lastResults = results;
                        const sourceRefs = results.map(chunkToSourceRef);
                        cacheDocsFromSources(sourceRefs);
                    }
                    // Citations are NOT rendered here: this event fires before any
                    // answer token, so no [N] markers exist yet and we could only
                    // show "all retrieved". They are rendered in the `done` handler
                    // below, once `answer` is complete, so the panel goes straight
                    // to the cited-only view (no all-12 flash, no click needed).
                    if (data.relevant_doc_ids || data.ignored_doc_ids) {
                        applyDocState(
                            (data.relevant_doc_ids ?? []) as string[],
                            (data.ignored_doc_ids ?? []) as string[],
                        );
                    }
                    // Query-processing panel (HyDE/rewrite/DR) — rendered ABOVE the
                    // thinking block. The retrieval event carries the full metadata
                    // and fires before any reasoning/token, so the panel lands first.
                    renderQueryProcessing(bubbleEl.parentElement, data, queryText);
                } else if (evtType === "error") {
                    errorShown = true;
                    cancelRender();
                    finalizeReasoning(false);
                    bubbleEl.classList.remove("streaming");
                    typingEl.style.display = "none";
                    bubbleEl.innerHTML = "&#9888; " + escHtml(String(data.message ?? "Unknown error"));
                    bubbleEl.classList.add("error-bubble");
                    bubbleEl.style.display = "block";
                    scrollToBottom();
                } else if (evtType === "done") {
                    const cid = String(data.conversation_id ?? "").trim();
                    if (cid) setActiveConversation(cid);
                    // No answer followed the reasoning — leave it expanded so the
                    // user can still see what the model worked through.
                    if (!started) finalizeReasoning(false);

                    if (data.token_budget) lastBudget = data.token_budget;
                    const completionTokens =
                        Number(lastBudget?.actual_completion_tokens) || tokenEventCount;
                    const promptTokens =
                        Number(lastBudget?.actual_prompt_tokens) ||
                        Number(lastBudget?.input_tokens) || 0;
                    const elapsedMs = lastTokenAt > firstTokenAt ? lastTokenAt - firstTokenAt : 0;
                    const tokensPerSecond = elapsedMs > 0 && completionTokens > 0
                        ? completionTokens / (elapsedMs / 1000)
                        : 0;
                    updateContextIndicator(lastBudget, {
                        promptTokens,
                        completionTokens,
                        tokensPerSecond,
                        costUsd: Number(lastBudget?.cost_usd ?? 0),
                    });

                    cancelRender();
                    bubbleEl.classList.remove("streaming");
                    typingEl.style.display = "none";
                    if (!errorShown) {
                        if (!started) {
                            const msg =
                                pendingClarification ||
                                "I couldn't find relevant information for that query. " +
                                    "Could you rephrase your question or provide more details?";
                            bubbleEl.innerHTML = parseMarkdown(msg);
                            bubbleEl.style.display = "block";
                        } else {
                            bubbleEl.innerHTML = parseMarkdown(answer) + buildSourcesLineHtml(lastResults, answer);
                            bubbleEl.style.display = "block";
                        }
                    }

                    const showCitations = byId<HTMLInputElement>("citationsToggle").checked;
                    if (showCitations && lastResults.length) {
                        // Render now, with the COMPLETE answer, so the panel shows
                        // the cited chunks first and collapses the uncited rest.
                        citationsEl.innerHTML = buildCitationsHtml(lastResults, answer);
                        wireCitationActions(citationsEl);
                    }
                    if (showCitations && citationsEl.innerHTML) {
                        revealCitations(citationsEl);
                    }
                    actionsEl.style.display = "flex";
                    metaEl.textContent = fmtTime(Date.now());
                    metaEl.style.display = "block";
                    // Suggested "you might also ask…" block at the tail of this
                    // answer's bubble (arrives in the done event).
                    if (!errorShown && started) {
                        const sq = (data as { suggested_questions?: string[] }).suggested_questions;
                        renderFollowUps(bubbleEl.parentElement, Array.isArray(sq) ? sq : []);
                    }
                    scrollToBottom();

                    await loadConversations();
                    updateConvTitle();
                }
            }
        }
    } catch (err) {
        if ((err as Error).name !== "AbortError") {
            appendErrorMsg("Stream interrupted: " + String(err));
        }
    } finally {
        cancelRender();
        bubbleEl.classList.remove("streaming");
    }

    state.isStreaming = false;
    state.pendingUserGroup = null;
    state.pendingAssistantGroup = null;
    state.pendingQueryText = "";
    setSendButtonStop(false);
}

export function setSendButtonStop(isStop: boolean): void {
    const btn = byId<HTMLButtonElement>("sendBtn");
    if (isStop) {
        btn.classList.add("stop-mode");
        btn.title = "Stop generation (cancel)";
        btn.innerHTML = "&#9632;";
    } else {
        btn.classList.remove("stop-mode");
        btn.title = "Send";
        btn.innerHTML = "&#9650;";
    }
}

/** Cancel the in-flight stream, remove the pending bubbles, restore the textarea. */
export function cancelStream(): void {
    if (!state.isStreaming) return;
    state.streamAbortCtrl?.abort();

    state.pendingUserGroup?.remove();
    state.pendingAssistantGroup?.remove();
    state.pendingUserGroup = null;
    state.pendingAssistantGroup = null;

    if (state.pendingQueryText && !refs.ta.value.trim()) {
        refs.ta.value = state.pendingQueryText;
        refs.ta.style.height = "auto";
        refs.ta.style.height = Math.min(refs.ta.scrollHeight, 220) + "px";
        refs.ta.focus();
    }
    state.pendingQueryText = "";
    state.isStreaming = false;
    setSendButtonStop(false);
}

export async function nonStreamQuery(queryText: string): Promise<void> {
    appendUserMsg(queryText);
    const handles = appendPendingAssistant();
    const { bubbleEl, typingEl, citationsEl, actionsEl, metaEl } = handles;
    try {
        const data = await api<{
            generated_answer?: string;
            clarification_message?: string;
            ask_user_reason?: string;
            degraded?: boolean;
            results?: ChunkResult[];
            conversation_id?: string;
            token_budget?: TokenBudget;
            dr_suggestion?: DrSuggestionPayload;
            suggested_questions?: string[];
            processed_query?: string;
            query_confidence?: number;
            kg_expanded_terms?: string[] | null;
            metadata?: QueryMetadata;
        }>("POST", "/console/query", buildQueryBody(queryText));
        maybeShowDrChip(data.dr_suggestion, queryText);

        const cid = String(data.conversation_id ?? "").trim();
        if (cid) setActiveConversation(cid);

        attachFeedback(handles.fbUpBtn, handles.fbDownBtn, {
            conversationId: state.activeConversationId,
            query: queryText,
            answer: () => bubbleEl.innerText,
        });

        typingEl.style.display = "none";
        const answer = data.generated_answer ?? data.clarification_message ?? "No response.";
        bubbleEl.innerHTML = parseMarkdown(answer) + buildSourcesLineHtml(data.results ?? [], answer);
        bubbleEl.style.display = "block";
        // Query-processing panel (HyDE/rewrite/DR) above the answer (no reasoning
        // block on the non-stream path, so it anchors before the bubble).
        renderQueryProcessing(bubbleEl.parentElement, data, queryText);

        const tb = data.token_budget;
        if (tb) {
            updateContextIndicator(tb, {
                promptTokens: Number(tb.actual_prompt_tokens) || Number(tb.input_tokens) || 0,
                completionTokens: Number(tb.actual_completion_tokens) || 0,
                tokensPerSecond: 0,
                costUsd: Number(tb.cost_usd) || 0,
            });
        }

        const showCitations = byId<HTMLInputElement>("citationsToggle").checked;
        const results = data.results ?? [];
        if (results.length) {
            cacheDocsFromSources(results.map(chunkToSourceRef));
        }
        if (showCitations && results.length) {
            citationsEl.innerHTML = buildCitationsHtml(results, answer);
            wireCitationActions(citationsEl);
            revealCitations(citationsEl);
        }

        actionsEl.style.display = "flex";
        metaEl.textContent = fmtTime(Date.now());
        metaEl.style.display = "block";
        // Suggested "you might also ask…" block at the tail of this answer's
        // bubble — only for a real generated answer (parity with the streaming
        // path, which guards on `started`); never on a clarification/no-answer turn.
        if (data.generated_answer) {
            renderFollowUps(
                bubbleEl.parentElement,
                Array.isArray(data.suggested_questions) ? data.suggested_questions : [],
            );
        }
        scrollToBottom();
        await loadConversations();
        updateConvTitle();
    } catch (err) {
        typingEl.style.display = "none";
        bubbleEl.innerHTML = "&#9888; " + escHtml(String(err));
        bubbleEl.classList.add("error-bubble");
        bubbleEl.style.display = "block";
    }
}

export async function sendQuery(text: string): Promise<void> {
    // Hide any prior deep-research suggestion — a new query supersedes it. The
    // follow-up block is per-turn (lives in its own answer bubble), so there is
    // no global one to clear here.
    hideDrSuggestionChip();
    if (getChatMode() === "sources") {
        await sourcesOnlyQuery(text);
        return;
    }
    const s = getSettings();
    const useStreaming = s.streaming !== false;
    if (useStreaming) await streamQuery(text);
    else await nonStreamQuery(text);
}

registerDrSuggestionResubmit(sendQuery);

interface DrSuggestionPayload {
    suggest?: boolean;
    reason?: string | null;
}

function maybeShowDrChip(payload: DrSuggestionPayload | undefined, query: string): void {
    if (!payload || !payload.suggest) return;
    if (getDeepResearch()) return;
    showDrSuggestionChip(query);
}
