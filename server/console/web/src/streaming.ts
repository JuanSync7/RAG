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
import { buildCitationsHtml, revealCitations } from "./citations";
import { updateContextIndicator, clearLastTurnStats } from "./contextWindow";
import { attachFeedback } from "./feedback";
import { loadConversations, updateConvTitle } from "./conversations";
import { getChatMode, getRetrievalSubMode, getSourcesTopK, appendSourcesTurn, applyDocState, cacheDocsFromSources, wireCitationActions } from "./chatMode";
import type { ChunkResult, SourceRef, StreamEventData, TokenBudget } from "./user-types";

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
        }>("POST", "/console/query", buildQueryBody(queryText));
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
                    }
                    lastTokenAt = performance.now();
                    tokenEventCount++;
                    answer += data.token || "";
                    scheduleRender();
                    scrollToBottom();
                } else if (evtType === "retrieval") {
                    const cid = String(data.conversation_id ?? "").trim();
                    if (cid) setActiveConversation(cid);

                    const clar = String(data.clarification_message ?? "").trim();
                    if (clar) pendingClarification = clar;

                    if (data.token_budget) {
                        lastBudget = data.token_budget;
                        updateContextIndicator(lastBudget);
                    }

                    const results = (data.results ?? []) as ChunkResult[];
                    if (results.length) {
                        const sourceRefs = results.map(chunkToSourceRef);
                        cacheDocsFromSources(sourceRefs);
                    }
                    const showCitations = byId<HTMLInputElement>("citationsToggle").checked;
                    if (showCitations && results.length) {
                        citationsEl.innerHTML = buildCitationsHtml(results);
                        wireCitationActions(citationsEl);
                    }
                    if (data.relevant_doc_ids || data.ignored_doc_ids) {
                        applyDocState(
                            (data.relevant_doc_ids ?? []) as string[],
                            (data.ignored_doc_ids ?? []) as string[],
                        );
                    }
                } else if (evtType === "error") {
                    errorShown = true;
                    cancelRender();
                    bubbleEl.classList.remove("streaming");
                    typingEl.style.display = "none";
                    bubbleEl.innerHTML = "&#9888; " + escHtml(String(data.message ?? "Unknown error"));
                    bubbleEl.classList.add("error-bubble");
                    bubbleEl.style.display = "block";
                    scrollToBottom();
                } else if (evtType === "done") {
                    const cid = String(data.conversation_id ?? "").trim();
                    if (cid) setActiveConversation(cid);

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
                            bubbleEl.innerHTML = parseMarkdown(answer);
                            bubbleEl.style.display = "block";
                        }
                    }

                    const showCitations = byId<HTMLInputElement>("citationsToggle").checked;
                    if (showCitations && citationsEl.innerHTML) {
                        revealCitations(citationsEl);
                    }
                    actionsEl.style.display = "flex";
                    metaEl.textContent = fmtTime(Date.now());
                    metaEl.style.display = "block";
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
            results?: ChunkResult[];
            conversation_id?: string;
            token_budget?: TokenBudget;
        }>("POST", "/console/query", buildQueryBody(queryText));

        const cid = String(data.conversation_id ?? "").trim();
        if (cid) setActiveConversation(cid);

        attachFeedback(handles.fbUpBtn, handles.fbDownBtn, {
            conversationId: state.activeConversationId,
            query: queryText,
            answer: () => bubbleEl.innerText,
        });

        typingEl.style.display = "none";
        const answer = data.generated_answer ?? data.clarification_message ?? "No response.";
        bubbleEl.innerHTML = parseMarkdown(answer);
        bubbleEl.style.display = "block";

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
            citationsEl.innerHTML = buildCitationsHtml(results);
            wireCitationActions(citationsEl);
            revealCitations(citationsEl);
        }

        actionsEl.style.display = "flex";
        metaEl.textContent = fmtTime(Date.now());
        metaEl.style.display = "block";
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
    if (getChatMode() === "sources") {
        await sourcesOnlyQuery(text);
        return;
    }
    const s = getSettings();
    const useStreaming = s.streaming !== false;
    if (useStreaming) await streamQuery(text);
    else await nonStreamQuery(text);
}
