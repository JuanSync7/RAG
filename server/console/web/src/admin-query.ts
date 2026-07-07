/**
 * @summary
 * Query tab handlers, SSE stream consumer, and slash-command dispatcher for the
 * admin/operator console. Renders the same turn-loop ACTIVITY LOG and clarify
 * chips as the user console (parity, TURN_LOOP_DESIGN.md §8) via the shared
 * `activityLog.ts` renderer; chips resubmit by refilling the query box and
 * re-running the stream.
 * Exports: queryStatusSnapshot, bindQuery
 * Deps: admin-types, admin-state, admin-api, admin-render, admin-conversations,
 *   admin-health, activityLog
 * @end-summary
 */

import type {
    ConsoleCommandSpec,
    ConversationMeta,
    JsonObject,
    QueryResult,
    StreamEventData,
    TokenBudgetPayload,
} from "./admin-types.js";
import { getActiveConversationId } from "./admin-state.js";
import {
    api,
    authHeaders,
    byId,
    executeConsoleCommand,
    parseSlash,
} from "./admin-api.js";
import {
    asOptionalNumber,
    commandSummary,
    escapeHtml,
    renderMarkdown,
    renderRerankedOriginalDocs,
    renderTiming,
} from "./admin-render.js";
import {
    createConversation,
    loadConversationHistory,
    loadConversations,
    setActiveConversation,
} from "./admin-conversations.js";
import { refreshHealth } from "./admin-health.js";
import { createActivityLog, isTurnLoopEvent, renderClarifyChips } from "./activityLog.js";
import type { ActivityLog } from "./activityLog.js";

export function queryStatusSnapshot(): JsonObject {
    return {
        query: byId<HTMLTextAreaElement>("queryText").value,
        search_limit: Number(byId<HTMLInputElement>("searchLimit").value || 10),
        rerank_top_k: Number(byId<HTMLInputElement>("rerankTopK").value || 5),
        conversation_id: getActiveConversationId(),
        memory_enabled: byId<HTMLInputElement>("memoryEnabled").checked,
    };
}

export function bindQuery(): void {
    byId("runQueryBtn").addEventListener("click", async () => {
        try {
            const payload = {
                query: byId<HTMLTextAreaElement>("queryText").value,
                search_limit: Number(byId<HTMLInputElement>("searchLimit").value || 10),
                rerank_top_k: Number(byId<HTMLInputElement>("rerankTopK").value || 5),
                stream: false,
                conversation_id: getActiveConversationId(),
                memory_enabled: byId<HTMLInputElement>("memoryEnabled").checked,
            };
            const out = await api("POST", "/console/query", payload);
            const data = (out.data as JsonObject | undefined) || out;
            if (typeof data.conversation_id === "string" && data.conversation_id) {
                setActiveConversation(data.conversation_id);
            }
            renderMarkdown("queryMarkdown", String((data.generated_answer as string | undefined) || ""));
            await renderRerankedOriginalDocs((data.results as QueryResult[] | undefined) || []);
            renderTiming({
                latency_ms: asOptionalNumber(data.latency_ms),
                stage_timings: (data.stage_timings as Array<Record<string, unknown>> | undefined) || [],
                timing_totals: (data.timing_totals as Record<string, unknown> | undefined) || {},
            });
            await loadConversations();
            await loadConversationHistory();
        } catch (err) {
            renderMarkdown("queryMarkdown", `**Query error:** ${String(err)}`);
            byId("rerankDocsOut").innerHTML = '<div class="muted">No reranked results.</div>';
            renderTiming(null);
        }
    });

    byId("runStreamBtn").addEventListener("click", async () => {
        byId("rerankDocsOut").innerHTML = "";
        // Drop any reasoning block / activity log / clarify chips left over
        // from a previous run.
        const mdHost = byId("queryMarkdown").parentElement;
        mdHost?.querySelector(".reasoning-block")?.remove();
        mdHost?.querySelector(".activity-log")?.remove();
        mdHost?.querySelector(".clarify-chips")?.remove();
        renderTiming(null);
        const body = {
            query: byId<HTMLTextAreaElement>("queryText").value,
            search_limit: Number(byId<HTMLInputElement>("searchLimit").value || 10),
            rerank_top_k: Number(byId<HTMLInputElement>("rerankTopK").value || 5),
            conversation_id: getActiveConversationId(),
            memory_enabled: byId<HTMLInputElement>("memoryEnabled").checked,
        };
        const response = await fetch("/query/stream", {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify(body),
        });
        if (!response.ok || !response.body) {
            renderMarkdown("queryMarkdown", `**Stream error:** Failed to open stream (HTTP ${response.status}).`);
            return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let chunkBuffer = "";
        let answer = "";
        let reasoningText = "";
        let sawToken = false;
        // Turn-loop activity log (shared renderer; parity with the user
        // console). Created lazily on the first turn-loop event; collapsed
        // when the accepted answer starts streaming. Ref object (not a bare
        // closure-mutated `let`) to sidestep the TS `never`-narrowing pitfall.
        const activity: { current: ActivityLog | null } = { current: null };
        const activityLog = (): ActivityLog => {
            if (!activity.current) {
                activity.current = createActivityLog();
                const md = byId("queryMarkdown");
                const anchor =
                    md.parentElement?.querySelector<HTMLElement>(".reasoning-block") ?? md;
                md.parentElement?.insertBefore(activity.current.root, anchor);
            }
            return activity.current;
        };
        // Clarify chips resubmit by refilling the query box and re-running
        // the stream — the admin-console analogue of the user console's
        // registered sendQuery resubmit sink.
        const resubmit = (text: string): void => {
            byId<HTMLTextAreaElement>("queryText").value = text;
            byId("runStreamBtn").click();
        };
        // Returns the reasoning-body element, lazily creating the collapsible block
        // above #queryMarkdown. DOM-queried (not a closure-cached var) to avoid a TS
        // `never`-narrowing pitfall. Static markup only; the model's raw reasoning
        // text is written via textContent (never innerHTML) so it cannot inject markup.
        const reasoningBody = (): HTMLElement => {
            const md = byId("queryMarkdown");
            let el = md.parentElement?.querySelector<HTMLDetailsElement>(".reasoning-block") ?? null;
            if (!el) {
                el = document.createElement("details");
                el.className = "reasoning-block";
                el.open = true;
                el.innerHTML =
                    `<summary class="reasoning-summary">&#128173; Thinking&hellip;</summary>` +
                    `<div class="reasoning-body"></div>`;
                md.parentElement?.insertBefore(el, md);
            }
            return el.querySelector<HTMLElement>(".reasoning-body")!;
        };
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            chunkBuffer += decoder.decode(value, { stream: true });
            const events = chunkBuffer.split("\n\n");
            chunkBuffer = events.pop() || "";
            for (const evt of events) {
                const lines = evt.split("\n");
                const eventLine = lines.find((line) => line.startsWith("event: "));
                const dataLine = lines.find((line) => line.startsWith("data: "));
                const eventType = eventLine ? eventLine.slice(7) : "";
                const dataRaw = dataLine ? dataLine.slice(6) : "{}";
                let data: StreamEventData;
                try {
                    data = JSON.parse(dataRaw) as StreamEventData;
                } catch {
                    data = { raw: dataRaw };
                }

                if (eventType === "token") {
                    if (!sawToken) {
                        sawToken = true;
                        // Accepted-answer replay begins — collapse the log.
                        activity.current?.finalize(true);
                    }
                    answer += data.token || "";
                    renderMarkdown("queryMarkdown", answer);
                } else if (eventType === "reasoning") {
                    reasoningText += String(data.text ?? "");
                    reasoningBody().textContent = reasoningText;
                } else if (isTurnLoopEvent(eventType)) {
                    // Typed turn-loop activity events (TURN_LOOP_DESIGN.md §8);
                    // textContent-only rendering inside the shared module.
                    activityLog().handle(eventType, data);
                    if (eventType === "clarify") {
                        const chips = renderClarifyChips(data, resubmit);
                        // After the markdown pane so chips survive the log
                        // collapsing (parity with the user console).
                        if (chips) byId("queryMarkdown").insertAdjacentElement("afterend", chips);
                    }
                } else if (eventType === "retrieval") {
                    const cid = typeof data.conversation_id === "string" ? data.conversation_id : "";
                    if (cid) {
                        setActiveConversation(cid);
                    }
                    await renderRerankedOriginalDocs(data.results || []);
                } else if (eventType === "error") {
                    renderMarkdown("queryMarkdown", `**Stream error:** ${String(data.message || JSON.stringify(data))}`);
                } else if (eventType === "done") {
                    const cid = typeof data.conversation_id === "string" ? data.conversation_id : "";
                    if (cid) {
                        setActiveConversation(cid);
                    }
                    const rb = byId("queryMarkdown").parentElement?.querySelector<HTMLDetailsElement>(".reasoning-block");
                    if (rb) rb.open = false;
                    // No answer replay (e.g. terminal clarify): leave the
                    // activity log open, relabeled with its step count.
                    if (!sawToken) activity.current?.finalize(false);
                    renderMarkdown("queryMarkdown", answer);
                    renderTiming({
                        latency_ms: asOptionalNumber(data.latency_ms),
                        retrieval_ms: asOptionalNumber(data.retrieval_ms),
                        generation_ms: asOptionalNumber(data.generation_ms),
                        token_count: asOptionalNumber(data.token_count),
                        stage_timings: (data.stage_timings as Array<Record<string, unknown>> | undefined) || [],
                        timing_totals: (data.timing_totals as Record<string, unknown> | undefined) || {},
                        token_budget: (data.token_budget as TokenBudgetPayload | undefined) || undefined,
                    });
                    await loadConversations();
                    await loadConversationHistory();
                }
            }
        }
    });

    byId("querySlashRunBtn").addEventListener("click", async () => {
        const input = byId<HTMLInputElement>("querySlashInput");
        const { name, arg } = parseSlash(input.value);
        if (!name) {
            return;
        }
        try {
            const result = await executeConsoleCommand("query", name, arg, queryStatusSnapshot());
            const action = String(result.action || "noop");
            if (action === "run_stream_query") {
                byId("runStreamBtn").click();
            } else if (action === "run_non_stream_query") {
                byId("runQueryBtn").click();
            } else if (action === "list_conversations") {
                await loadConversations();
                await loadConversationHistory();
            } else if (action === "new_conversation") {
                const conversation = (result.data?.conversation as ConversationMeta | undefined) || null;
                if (conversation?.conversation_id) {
                    setActiveConversation(conversation.conversation_id);
                } else {
                    await createConversation("New conversation");
                }
                await loadConversations();
                await loadConversationHistory();
            } else if (action === "switch_conversation") {
                const cid = String(result.data?.conversation_id || arg || "").trim();
                if (cid) {
                    setActiveConversation(cid);
                    await loadConversationHistory();
                }
            } else if (action === "show_history") {
                await loadConversationHistory();
            } else if (action === "compact_conversation") {
                const summary = String(result.data?.summary ?? "").trim();
                if (summary) {
                    renderMarkdown("queryMarkdown", `### Compacted summary\n\n${summary}`);
                }
                await loadConversations();
                await loadConversationHistory();
            } else if (action === "delete_conversation") {
                const deleted = Boolean(result.data?.deleted);
                const cid = String(result.data?.conversation_id ?? "").trim();
                if (deleted && cid && getActiveConversationId() === cid) {
                    setActiveConversation(null);
                }
                await loadConversations();
                if (deleted) {
                    renderMarkdown("queryMarkdown", `**Deleted** conversation \`${escapeHtml(cid)}\`.`);
                } else {
                    renderMarkdown("queryMarkdown", "**/delete:** Conversation not found (may already be deleted).");
                }
            } else if (action === "clear_view") {
                renderMarkdown("queryMarkdown", "");
                byId("rerankDocsOut").innerHTML = "";
                renderTiming(null);
            } else if (action === "refresh_health") {
                await refreshHealth();
                renderMarkdown(
                    "queryMarkdown",
                    `\`\`\`json\n${JSON.stringify(queryStatusSnapshot(), null, 2)}\n\`\`\``,
                );
            } else if (action === "render_help") {
                const cmds = Array.isArray(result.data?.commands)
                    ? (result.data?.commands as ConsoleCommandSpec[])
                    : [];
                renderMarkdown(
                    "queryMarkdown",
                    `### Query Slash Commands\n\n\`\`\`\n${commandSummary(cmds)}\n\`\`\``,
                );
            } else if (action === "render_status") {
                const state = (result.data?.state as JsonObject | undefined) || queryStatusSnapshot();
                renderMarkdown("queryMarkdown", `\`\`\`json\n${JSON.stringify(state, null, 2)}\n\`\`\``);
            } else {
                renderMarkdown("queryMarkdown", result.message || `No action mapped for /${name}`);
            }
        } catch (err) {
            renderMarkdown("queryMarkdown", `**Command error:** ${String(err)}`);
        }
        input.value = "";
    });

    byId("querySlashInput").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            byId("querySlashRunBtn").click();
        }
    });
}
