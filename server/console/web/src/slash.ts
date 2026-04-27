// @summary
// Slash-command surface: dynamic-command load, autocomplete dropdown, full
// command picker, and `submitSlashCommand` (the dispatcher invoked by input.ts
// when the user sends a leading-slash message).
// @end-summary

import { byId, escHtml } from "./dom";
import { api } from "./api";
import { formatApiPayload } from "./markdown";
import { refs } from "./refs";
import { state, setActiveConversation } from "./state";
import {
    appendUserMsg,
    appendErrorMsg,
    appendSystemMsg,
    setEmptyState,
} from "./thread";
import {
    createNewConversation,
    deleteConversation,
    loadConversationHistory,
    loadConversations,
    selectConversation,
} from "./conversations";
import { streamQuery, nonStreamQuery } from "./streaming";
import type { CommandResult, ConversationMeta, SlashCommand } from "./user-types";

// ── Rendering ──

function renderSlashDropdown(cmds: SlashCommand[]): void {
    const container = byId("slashItems");
    container.innerHTML = cmds
        .map((c) =>
            `<div class="slash-item" data-cmd="/${escHtml(c.name)}">` +
            `<span class="slash-cmd">/${escHtml(c.name)}</span>` +
            `<span class="slash-desc">${escHtml(c.description)}</span>` +
            `</div>`
        )
        .join("");
    state.allSlashItems = Array.from(container.querySelectorAll<HTMLElement>(".slash-item"));
    state.allSlashItems.forEach((item) =>
        item.addEventListener("click", () => executeCmd(item.dataset.cmd || ""))
    );
}

function renderCmdPicker(cmds: SlashCommand[]): void {
    const container = byId("cmdPickerBody");
    const grouped: Record<string, SlashCommand[]> = {};
    cmds.forEach((c) => {
        const cat = c.category || "General";
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push(c);
    });
    let html = "";
    for (const [cat, items] of Object.entries(grouped)) {
        html += `<div class="cmd-group-label">${escHtml(cat)}</div>`;
        items.forEach((c) => {
            html +=
                `<div class="cmd-picker-item" data-cmd="/${escHtml(c.name)}">` +
                `<span class="cmd-picker-icon">&#47;</span>` +
                `<span class="cmd-picker-name">/${escHtml(c.name)}</span>` +
                `<span class="cmd-picker-desc">${escHtml(c.description)}</span>` +
                `</div>`;
        });
    }
    container.innerHTML = html;
    state.allPickerItems = Array.from(container.querySelectorAll<HTMLElement>(".cmd-picker-item"));
    state.allPickerItems.forEach((item) =>
        item.addEventListener("click", () => executePicker(item))
    );
}

export async function loadCommands(): Promise<void> {
    try {
        const data = await api<{ commands: SlashCommand[] }>("GET", "/console/commands?mode=query");
        state.dynamicCmds = data.commands || [];
        renderSlashDropdown(state.dynamicCmds);
        renderCmdPicker(state.dynamicCmds);
    } catch {
        // Non-fatal
    }
}

// ── Autocomplete dropdown ──

export function closeDropdown(): void {
    refs.dropdown.classList.remove("open");
}

export function setSelected(i: number): void {
    const vis = state.allSlashItems.filter((x) => x.style.display !== "none");
    if (!vis.length) return;
    state.slashSelIdx = (i + vis.length) % vis.length;
    vis.forEach((el, j) => el.classList.toggle("selected", j === state.slashSelIdx));
}

export function executeCmd(cmd: string): void {
    refs.ta.value = cmd + " ";
    refs.ta.focus();
    refs.ta.style.height = "auto";
    closeDropdown();
}

export function handleSlashInput(): void {
    const val = refs.ta.value;
    if (!val.startsWith("/")) {
        closeDropdown();
        return;
    }
    const q = val.slice(1).toLowerCase();
    let vis = 0;
    state.allSlashItems.forEach((item) => {
        const cmdAttr = (item.dataset.cmd || "").toLowerCase();
        const descEl = item.querySelector<HTMLElement>(".slash-desc");
        const descText = descEl ? descEl.textContent?.toLowerCase() || "" : "";
        const match = cmdAttr.includes(q) || descText.includes(q);
        item.style.display = match ? "" : "none";
        if (match) vis++;
    });
    if (!vis) {
        closeDropdown();
        return;
    }
    refs.dropdown.classList.add("open");
    setSelected(0);
}

// ── Command picker ──

export function setPickerSelected(idx: number): void {
    if (!state.allPickerItems.length) return;
    state.pickerIdx = (idx + state.allPickerItems.length) % state.allPickerItems.length;
    state.allPickerItems.forEach((el, i) => el.classList.toggle("selected", i === state.pickerIdx));
    state.allPickerItems[state.pickerIdx]?.scrollIntoView({ block: "nearest" });
}

export function executePicker(item: HTMLElement): void {
    const cmd = item.dataset.cmd || "";
    closeCmdPicker();
    refs.ta.value = cmd + " ";
    refs.ta.focus();
    refs.ta.style.height = "auto";
    closeDropdown();
}

export function closeCmdPicker(): void {
    refs.cmdPicker.classList.remove("open");
    refs.cmdBtn.classList.remove("active");
}

// ── Submission ──

export async function submitSlashCommand(text: string): Promise<void> {
    const trimmed = text.trim();
    const spaceIdx = trimmed.indexOf(" ");
    const commandName = (spaceIdx === -1 ? trimmed.slice(1) : trimmed.slice(1, spaceIdx)).toLowerCase();
    const arg = spaceIdx === -1 ? "" : trimmed.slice(spaceIdx + 1).trim();

    appendUserMsg(trimmed);

    try {
        const result = await api<CommandResult>("POST", "/console/command", {
            mode: "query",
            command: commandName,
            arg: arg || undefined,
            state: { conversation_id: state.activeConversationId ?? undefined },
        });
        const action = String(result.action ?? "noop");

        if (action === "run_stream_query") {
            await streamQuery(arg || commandName);
        } else if (action === "run_non_stream_query") {
            await nonStreamQuery(arg || commandName);
        } else if (action === "new_conversation") {
            const conv = result.data?.conversation as ConversationMeta | undefined;
            if (conv?.conversation_id) setActiveConversation(conv.conversation_id);
            else createNewConversation();
            await loadConversations();
        } else if (action === "switch_conversation") {
            const cid = String(result.data?.conversation_id ?? arg).trim();
            if (cid) await selectConversation(cid);
        } else if (action === "list_conversations") {
            await loadConversations();
            appendSystemMsg("Conversation list refreshed.");
        } else if (action === "show_history") {
            await loadConversationHistory();
        } else if (action === "compact_conversation") {
            const summary = String(result.data?.summary ?? "").trim();
            appendSystemMsg(summary ? `Compacted. Summary:\n\n${summary}` : "Conversation compacted.");
            await loadConversations();
        } else if (action === "delete_conversation") {
            const cid = String(result.data?.conversation_id ?? "").trim();
            if (cid) await deleteConversation(cid);
        } else if (action === "clear_view") {
            refs.thread.innerHTML = "";
            setEmptyState(true);
        } else if (action === "refresh_health") {
            const h = result.data?.health;
            appendSystemMsg("**Health**\n\n" + (h ? formatApiPayload(h) : "_unavailable_"));
        } else if (action === "render_help") {
            const cmds = result.data?.commands as SlashCommand[] | undefined;
            if (cmds?.length) {
                const lines = cmds.map((c) => `- **/${c.name}** — ${c.description}`).join("\n");
                appendSystemMsg("**Available commands**\n\n" + lines);
            }
        } else if (result.message) {
            appendSystemMsg(String(result.message));
        } else if (result.data && Object.keys(result.data).length) {
            appendSystemMsg(formatApiPayload(result.data));
        } else {
            appendSystemMsg("Command executed.");
        }
    } catch (err) {
        appendErrorMsg("Command failed: " + String(err));
    }
}
