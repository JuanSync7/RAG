// @summary
// Textarea input + send button + keyboard handling for the slash dropdown
// and command picker. Bridges user keystrokes to slash.ts and streaming.ts.
// @end-summary

import { byId } from "./dom";
import { refs } from "./refs";
import { state } from "./state";
import { sendQuery, cancelStream } from "./streaming";
import {
    closeCmdPicker,
    closeDropdown,
    executeCmd,
    executePicker,
    handleSlashInput,
    setPickerSelected,
    setSelected,
    submitSlashCommand,
} from "./slash";
import { closeAllPopovers, setCloseCmdPicker } from "./attachments";

function toggleCmdPicker(): void {
    const isOpen = refs.cmdPicker.classList.contains("open");
    closeAllPopovers();
    if (isOpen) {
        closeCmdPicker();
    } else {
        refs.cmdPicker.classList.add("open");
        refs.cmdBtn.classList.add("active");
        setPickerSelected(0);
    }
}

function triggerSend(): void {
    if (state.isStreaming) {
        cancelStream();
        return;
    }
    const text = refs.ta.value.trim();
    if (!text) return;
    closeDropdown();
    refs.ta.value = "";
    refs.ta.style.height = "auto";
    if (text.startsWith("/")) {
        void submitSlashCommand(text);
    } else {
        void sendQuery(text);
    }
}

export function initInput(): void {
    setCloseCmdPicker(closeCmdPicker);

    refs.ta.addEventListener("input", () => {
        refs.ta.style.height = "auto";
        refs.ta.style.height = Math.min(refs.ta.scrollHeight, 220) + "px";
        handleSlashInput();
    });

    let lastEscAt = 0;
    refs.ta.addEventListener("keydown", (e: KeyboardEvent) => {
        if (e.key === "Escape") {
            const now = Date.now();
            if (refs.dropdown.classList.contains("open")) {
                closeDropdown();
                lastEscAt = 0;
                return;
            }
            if (now - lastEscAt < 400 && refs.ta.value) {
                refs.ta.value = "";
                refs.ta.style.height = "auto";
                lastEscAt = 0;
            } else {
                lastEscAt = now;
            }
            return;
        }
        if (!refs.dropdown.classList.contains("open")) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                triggerSend();
            }
            return;
        }
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setSelected(state.slashSelIdx + 1);
        }
        if (e.key === "ArrowUp") {
            e.preventDefault();
            setSelected(state.slashSelIdx - 1);
        }
        if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
            e.preventDefault();
            const vis = state.allSlashItems.filter((x) => x.style.display !== "none");
            if (vis[state.slashSelIdx]) executeCmd(vis[state.slashSelIdx].dataset.cmd || "");
        }
    });

    byId("sendBtn").addEventListener("click", triggerSend);

    refs.cmdBtn.addEventListener("click", toggleCmdPicker);
    document.getElementById("cmdPickerClose")?.addEventListener("click", closeCmdPicker);

    document.addEventListener("keydown", (e: KeyboardEvent) => {
        if (!refs.cmdPicker.classList.contains("open")) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setPickerSelected(state.pickerIdx + 1);
        }
        if (e.key === "ArrowUp") {
            e.preventDefault();
            setPickerSelected(state.pickerIdx - 1);
        }
        if (e.key === "Enter") {
            e.preventDefault();
            executePicker(state.allPickerItems[state.pickerIdx]);
        }
        if (e.key === "Escape") closeCmdPicker();
    });

    // Outside-click handler — closes any popover/picker/dropdown when the user
    // clicks outside the input bar.
    document.addEventListener("click", (e: MouseEvent) => {
        const target = e.target as HTMLElement;
        if (!target.closest(".input-bar")) {
            closeAllPopovers();
            closeCmdPicker();
            closeDropdown();
            document.getElementById("ctxChip")?.classList.remove("tooltip-open");
        }
    });
}
