// @summary
// Context-window usage indicator (chip + tooltip) and compact button.
// Driven by the `token_budget` payload that streaming.ts forwards from
// retrieval/done events. Adds tokens/sec computed on the frontend across
// the streamed `token` events.
// @end-summary

import { byId } from "./dom";
import { api } from "./api";
import { state } from "./state";
import { showToast } from "./toast";
import type { LastTurnStats, TokenBudget } from "./user-types";

function fmtTokens(n: number): string {
    if (!n) return "0";
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(n >= 10_000 ? 0 : 1) + "k";
    return String(n);
}

function fmtCost(usd: number): string {
    if (usd <= 0) return "$0";
    if (usd < 0.01) return "$" + usd.toFixed(4);
    return "$" + usd.toFixed(3);
}

/**
 * Update the chip + tooltip from a backend token_budget snapshot.
 * Shows context-window usage as `used / total` and updates "last turn" stats
 * (prompt/completion tokens, tokens/sec, cost) when post-generation numbers
 * are present.
 */
export function updateContextIndicator(tb: TokenBudget | null, stats?: LastTurnStats): void {
    const chip = byId("ctxChip");
    const used = Number(tb?.input_tokens ?? 0);
    const total = Number(tb?.context_length ?? 0);
    const reserved = Number(tb?.output_reservation ?? 0);
    const pctRaw = Number(tb?.usage_percent ?? 0);
    const pct = pctRaw > 1.5 ? pctRaw : pctRaw * 100; // backend may emit 0..1 or 0..100

    byId("ctxBarFill").style.width = Math.min(pct, 100) + "%";
    if (total > 0) {
        byId("ctxPct").textContent = `${fmtTokens(used)} / ${fmtTokens(total)}`;
    } else {
        byId("ctxPct").textContent = pct > 0 ? "~" + Math.round(pct) + "%" : "—";
    }

    chip.classList.remove("warn", "crit");
    if (pct >= 85) chip.classList.add("crit");
    else if (pct >= 60) chip.classList.add("warn");

    byId("ttModel").textContent = String(tb?.model_name || "—");
    byId("ttUsed").textContent = total > 0 ? `${fmtTokens(used)} / ${fmtTokens(total)} tok` : `${fmtTokens(used)} tok`;
    byId("ttReserved").textContent = reserved > 0 ? `${fmtTokens(reserved)} tok` : "—";

    if (stats) {
        byId("ttPrompt").textContent = stats.promptTokens > 0 ? `${fmtTokens(stats.promptTokens)} tok` : "—";
        byId("ttCompletion").textContent = stats.completionTokens > 0 ? `${fmtTokens(stats.completionTokens)} tok` : "—";
        byId("ttSpeed").textContent = stats.tokensPerSecond > 0 ? `${stats.tokensPerSecond.toFixed(1)} tok/s` : "—";
        const costRow = byId("ttCostRow");
        if (stats.costUsd > 0) {
            costRow.style.display = "";
            byId("ttCost").textContent = fmtCost(stats.costUsd);
        } else {
            costRow.style.display = "none";
        }
    }

    byId("ctxCompactBtn").style.display = pct >= 60 ? "block" : "none";
}

/** Reset post-turn stats (prompt/completion/speed/cost) without touching budget. */
export function clearLastTurnStats(): void {
    byId("ttPrompt").textContent = "—";
    byId("ttCompletion").textContent = "—";
    byId("ttSpeed").textContent = "—";
    byId("ttCostRow").style.display = "none";
}

export function initContextIndicator(): void {
    byId("ctxCompactBtn").addEventListener("click", async () => {
        if (!state.activeConversationId) return;
        try {
            await api("POST", `/console/conversations/${state.activeConversationId}/compact`);
            showToast("Conversation compacted");
            updateContextIndicator(null);
        } catch (err) {
            showToast("Compact failed: " + String(err));
        }
    });

    byId("ctxChip").addEventListener("click", () => {
        byId("ctxChip").classList.toggle("tooltip-open");
    });
}
