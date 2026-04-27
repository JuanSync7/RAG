/** @summary
 * Ingest tab orchestrator. Wires the top-level Chat/Ingest view tabs and
 * the Files/URL/Directory sub-mode tabs to the focused submodules:
 *   - ingest-modes: file/URL/directory submission flows
 *   - ingest-stream: SSE attach + REST refresh of jobs
 *   - ingest-jobs: job-list rendering helpers and shared state
 * The public surface is `initIngestView` (consumed by `user-console.ts`).
 * Exports: initIngestView
 * Deps: dom, ingest-modes, ingest-stream
 * @end-summary
 */

import { byId } from "./dom";
import {
    addFiles,
    checkDirectory,
    clearSelected,
    startDirectoryIngestion,
    startFileIngestion,
    startUrlIngestion,
} from "./ingest-modes";
import { refreshJobsList } from "./ingest-stream";

function switchView(view: "chat" | "ingest"): void {
    document.querySelectorAll<HTMLElement>(".view-tab").forEach((b) => {
        b.classList.toggle("active", b.dataset.view === view);
    });
    document.querySelectorAll<HTMLElement>(".view-pane").forEach((p) => {
        p.classList.toggle("active", p.id === `view-${view}`);
    });
    if (view === "ingest") void refreshJobsList();
}

function switchMode(mode: "files" | "url" | "directory"): void {
    document.querySelectorAll<HTMLElement>(".ingest-mode-tab").forEach((b) => {
        b.classList.toggle("active", b.dataset.mode === mode);
    });
    document.querySelectorAll<HTMLElement>(".ingest-mode-pane").forEach((p) => {
        p.classList.toggle("active", p.dataset.modePane === mode);
    });
}

export function initIngestView(): void {
    // Top-level Chat / Ingest tabs.
    document.querySelectorAll<HTMLElement>(".view-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            const view = btn.dataset.view as "chat" | "ingest";
            if (view) switchView(view);
        });
    });

    // Ingest sub-mode tabs.
    document.querySelectorAll<HTMLElement>(".ingest-mode-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            const mode = btn.dataset.mode as "files" | "url" | "directory";
            if (mode) switchMode(mode);
        });
    });

    // Files mode: dropzone.
    const dz = byId("ingestDropzone");
    const fi = byId<HTMLInputElement>("ingestFileInput");
    dz.addEventListener("click", () => fi.click());
    fi.addEventListener("change", () => {
        if (fi.files) addFiles(fi.files);
        fi.value = "";
    });
    dz.addEventListener("dragover", (e) => {
        e.preventDefault();
        dz.classList.add("drag");
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
    dz.addEventListener("drop", (e) => {
        e.preventDefault();
        dz.classList.remove("drag");
        const dt = (e as DragEvent).dataTransfer;
        if (dt?.files) addFiles(dt.files);
    });
    byId("ingestClearBtn").addEventListener("click", () => clearSelected());
    byId("ingestSubmitBtn").addEventListener("click", () => { void startFileIngestion(); });

    // URL mode.
    const urlInput = byId<HTMLInputElement>("ingestUrlInput");
    const urlBtn = byId<HTMLButtonElement>("ingestUrlBtn");
    urlInput.addEventListener("input", () => {
        urlBtn.disabled = urlInput.value.trim().length === 0;
    });
    urlInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !urlBtn.disabled) void startUrlIngestion();
    });
    urlBtn.addEventListener("click", () => { void startUrlIngestion(); });

    // Directory mode.
    const dirInput = byId<HTMLInputElement>("ingestDirInput");
    const dirCheck = byId<HTMLButtonElement>("ingestDirCheckBtn");
    const dirBtn = byId<HTMLButtonElement>("ingestDirBtn");
    dirInput.addEventListener("input", () => {
        // Force re-check when path changes.
        dirBtn.disabled = true;
    });
    dirInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") void checkDirectory();
    });
    dirCheck.addEventListener("click", () => { void checkDirectory(); });
    dirBtn.addEventListener("click", () => { void startDirectoryIngestion(); });
}
