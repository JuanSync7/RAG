// @summary
// User Console orchestrator. Resolves DOM refs, wires every feature module,
// and kicks off the initial data loads. All UI behavior lives in feature
// modules (sidebar, settings, conversations, streaming, slash, attachments,
// input, citations, contextWindow, scrollFab, thread, toast).
// @end-summary

import { populateRefs, refs } from "./refs";
import { state } from "./state";
import { initToast } from "./toast";
import { initCitations } from "./citations";
import { initContextIndicator } from "./contextWindow";
import { initScrollFab } from "./scrollFab";
import { initSidebar } from "./sidebar";
import { initSettings, loadSettings } from "./settings";
import { initConversations, loadConversationHistory, loadConversations } from "./conversations";
import { loadCommands } from "./slash";
import { loadModelInfo } from "./modelBadge";
import { initAttachments } from "./attachments";
import { initInput } from "./input";
import { initIngestView } from "./ingest";
import { initRetrievalView } from "./retrieval";

document.addEventListener("DOMContentLoaded", () => {
    populateRefs();

    // Order: toast first (others may showToast at init), then layout features,
    // then input (which depends on slash + attachments closeCmdPicker bridge).
    initToast();
    initCitations();
    initContextIndicator();
    initScrollFab();
    initSidebar();
    initSettings();
    initConversations();
    initAttachments();
    initInput();
    initIngestView();
    initRetrievalView();

    // Restore persisted UI state.
    loadSettings();

    // Kick off parallel data loads.
    void loadModelInfo();
    void Promise.all([loadCommands(), loadConversations()]).then(() => {
        if (state.activeConversationId) {
            void loadConversationHistory(state.activeConversationId);
        }
    });

    setTimeout(() => {
        refs.thread.scrollTop = refs.thread.scrollHeight;
    }, 100);
});
