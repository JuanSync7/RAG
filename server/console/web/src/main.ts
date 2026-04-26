/**
 * @summary
 * Operator console orchestrator. Wires per-feature modules (query, ingest,
 * conversations, health, admin) into the page on startup.
 * Exports: none (module side effects initialize UI)
 * Deps: admin-api, admin-render, admin-conversations, admin-query, admin-ingest,
 *   admin-health, admin-admin
 * @end-summary
 */

import { byId } from "./admin-api.js";
import { initSlashCommandHints, renderMarkdown, renderTiming } from "./admin-render.js";
import {
    bindConversations,
    loadConversationHistory,
    loadConversations,
} from "./admin-conversations.js";
import { bindQuery } from "./admin-query.js";
import { bindIngest } from "./admin-ingest.js";
import { bindHealth, refreshHealth } from "./admin-health.js";
import { bindAdmin } from "./admin-admin.js";

const tabMap: Record<string, string> = {
    query: "tab-query",
    ingest: "tab-ingest",
    health: "tab-health",
    admin: "tab-admin",
};

function initTabs(): void {
    const tabs = document.querySelectorAll<HTMLButtonElement>(".tabs button");
    tabs.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabs.forEach((t) => t.classList.remove("active"));
            btn.classList.add("active");
            Object.values(tabMap).forEach((id) => byId(id).classList.add("hidden"));
            const tabName = btn.dataset.tab || "";
            const targetId = tabMap[tabName];
            if (targetId) {
                byId(targetId).classList.remove("hidden");
            }
        });
    });
}

function initializeConsoleUi(): void {
    initTabs();
    bindQuery();
    bindConversations();
    bindIngest();
    bindHealth();
    bindAdmin();
    renderMarkdown("queryMarkdown", "");
    renderTiming(null);
    void initSlashCommandHints();
    void loadConversations().then(loadConversationHistory).catch(() => undefined);
    void refreshHealth();
}

initializeConsoleUi();
