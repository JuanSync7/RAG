// @summary
// Attachment toolbar: file/web/KB context attachments + chip rendering, plus
// the popover open/close coordination shared with the command picker (so only
// one can be open at a time).
// @end-summary

import { byId, escHtml } from "./dom";
import { refs } from "./refs";
import { state } from "./state";
import { showToast } from "./toast";

function renderChips(): void {
    const container = byId("attachChips");
    container.innerHTML = "";
    state.attachments.forEach((a) => {
        const chip = document.createElement("div");
        chip.className = "attach-chip";
        chip.innerHTML =
            `<span class="attach-chip-icon">${a.icon}</span>` +
            `<span class="attach-chip-label">${escHtml(a.label)}</span>` +
            `<button class="attach-chip-remove" title="Remove">&#215;</button>`;
        chip.querySelector<HTMLButtonElement>(".attach-chip-remove")?.addEventListener(
            "click",
            () => removeChip(a.id)
        );
        container.appendChild(chip);
    });
}

function addChip(icon: string, label: string, id: string): void {
    if (state.attachments.find((a) => a.id === id)) return;
    state.attachments.push({ id, icon, label });
    renderChips();
}

function removeChip(id: string): void {
    state.attachments = state.attachments.filter((a) => a.id !== id);
    renderChips();
}

export function closeAllPopovers(): void {
    refs.attachPopover.classList.remove("open");
    refs.webInputPanel.classList.remove("open");
    refs.kbPanel.classList.remove("open");
    refs.attachBtn.classList.remove("active");
}

function toggleAttachPopover(): void {
    const isOpen = refs.attachPopover.classList.contains("open");
    closeAllPopovers();
    // Close cmd picker too — coordinator imports closeCmdPicker.
    closeCmdPickerExternal();
    if (!isOpen) {
        refs.attachPopover.classList.add("open");
        refs.attachBtn.classList.add("active");
    }
}

// Indirection so we don't create an import cycle with slash.ts.
let closeCmdPickerExternal: () => void = () => {};
export function setCloseCmdPicker(fn: () => void): void {
    closeCmdPickerExternal = fn;
}

function openWebInput(): void {
    closeAllPopovers();
    refs.webInputPanel.classList.add("open");
    setTimeout(() => document.getElementById("webUrlInput")?.focus(), 50);
}

function openKBSelect(): void {
    closeAllPopovers();
    refs.kbPanel.classList.add("open");
}

function triggerFileUpload(): void {
    closeAllPopovers();
    byId<HTMLInputElement>("fileInput").click();
}

function handleFileSelect(input: HTMLInputElement): void {
    if (!input.files) return;
    Array.from(input.files).forEach((file) => addChip("&#128196;", file.name, "file:" + file.name));
    input.value = "";
    showToast("File added to context");
}

function attachWebUrl(): void {
    const input = byId<HTMLInputElement>("webUrlInput");
    const url = input.value.trim();
    if (!url) return;
    try {
        new URL(url);
    } catch {
        showToast("Invalid URL");
        return;
    }
    addChip("&#127760;", new URL(url).hostname.replace("www.", ""), "web:" + url);
    input.value = "";
    refs.webInputPanel.classList.remove("open");
    showToast("Web page added to context");
}

function filterKB(q: string): void {
    document.querySelectorAll<HTMLElement>(".kb-item").forEach((el) => {
        const name = el.querySelector<HTMLElement>(".kb-item-name")?.textContent?.toLowerCase() || "";
        el.style.display = name.includes(q.toLowerCase()) ? "" : "none";
    });
}

function attachKBDocs(): void {
    document.querySelectorAll<HTMLInputElement>("#kbList input[type=checkbox]:checked").forEach((cb) => {
        addChip("&#128218;", cb.value, "kb:" + cb.value);
        cb.checked = false;
    });
    refs.kbPanel.classList.remove("open");
    showToast("Documents added to context");
}

export function initAttachments(): void {
    refs.attachBtn.addEventListener("click", toggleAttachPopover);
    document.getElementById("attachOptFile")?.addEventListener("click", triggerFileUpload);
    document.getElementById("attachOptWeb")?.addEventListener("click", openWebInput);
    document.getElementById("attachOptKB")?.addEventListener("click", openKBSelect);

    document.getElementById("fileInput")?.addEventListener("change", (e: Event) => {
        handleFileSelect(e.target as HTMLInputElement);
    });

    document.getElementById("webUrlInput")?.addEventListener("keydown", (e: KeyboardEvent) => {
        if (e.key === "Enter") attachWebUrl();
        if (e.key === "Escape") refs.webInputPanel.classList.remove("open");
    });
    document.getElementById("webAddBtn")?.addEventListener("click", attachWebUrl);

    document.getElementById("kbSearch")?.addEventListener("input", (e: Event) =>
        filterKB((e.target as HTMLInputElement).value)
    );
    document.getElementById("kbAddBtn")?.addEventListener("click", attachKBDocs);
    document.getElementById("kbPanelClose")?.addEventListener("click", closeAllPopovers);

    const win = window as unknown as Record<string, unknown>;
    win["removeChip"] = removeChip;
    win["openWebInput"] = openWebInput;
    win["openKBSelect"] = openKBSelect;
    win["triggerFileUpload"] = triggerFileUpload;
    win["filterKB"] = filterKB;
    win["attachKBDocs"] = attachKBDocs;
    win["attachWebUrl"] = attachWebUrl;
    win["handleFileSelect"] = handleFileSelect;
}
