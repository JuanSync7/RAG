/** @summary
 * Ingest submission modes: file upload (with drag/drop selection state),
 * URL fetch, and server-side directory ingestion (with reachability check).
 * Each `start*Ingestion` helper kicks off jobs and hands them to the stream
 * module for live updates.
 * Exports: PathCheckResult, addFiles, renderSelected, clearSelected,
 *          startFileIngestion, startUrlIngestion, checkDirectory,
 *          startDirectoryIngestion
 * Deps: api, dom, format, toast, ingest-jobs, ingest-stream
 * @end-summary
 */

import { byId } from "./dom";
import { apiBase, authHeaders } from "./api";
import { fmtSize } from "./format";
import { showToast } from "./toast";
import { JobSummary, escapeHtml, jobMeta, renderJob } from "./ingest-jobs";
import { attachStream } from "./ingest-stream";

export interface PathCheckResult {
    path: string;
    reachable: boolean;
    is_dir?: boolean;
    is_file?: boolean;
    file_count?: number;
    files?: string[];
    truncated?: boolean;
    reason?: string;
    size_bytes?: number;
}

const _selected: File[] = [];

export function renderSelected(): void {
    const list = byId("ingestSelectedList");
    const wrap = byId("ingestSelected");
    const submit = byId<HTMLButtonElement>("ingestSubmitBtn");
    if (_selected.length === 0) {
        wrap.style.display = "none";
        submit.disabled = true;
        return;
    }
    wrap.style.display = "block";
    submit.disabled = false;
    byId("ingestSelectedCount").textContent =
        `${_selected.length} file${_selected.length > 1 ? "s" : ""} selected`;
    list.innerHTML = _selected
        .map(
            (f, i) =>
                `<div class="ingest-selected-item">
                    <span class="name">${escapeHtml(f.name)}</span>
                    <span class="size">${fmtSize(f.size)}</span>
                    <button class="remove" data-idx="${i}" title="Remove">&times;</button>
                </div>`,
        )
        .join("");
    list.querySelectorAll<HTMLButtonElement>(".remove").forEach((btn) => {
        btn.addEventListener("click", () => {
            const idx = Number(btn.dataset.idx);
            _selected.splice(idx, 1);
            renderSelected();
        });
    });
}

export function addFiles(files: FileList | File[]): void {
    for (const f of Array.from(files)) {
        if (f.size === 0) continue;
        if (_selected.some((s) => s.name === f.name && s.size === f.size)) continue;
        _selected.push(f);
    }
    renderSelected();
}

export function clearSelected(): void {
    _selected.length = 0;
    renderSelected();
}

async function uploadOne(file: File): Promise<JobSummary | null> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("options", "{}");
    const headers = authHeaders();
    delete (headers as Record<string, string>)["Content-Type"];
    try {
        const res = await fetch(apiBase() + "/api/v1/ingest/upload", {
            method: "POST",
            headers,
            body: fd,
        });
        if (!res.ok) {
            const txt = await res.text();
            showToast(`Upload failed (${res.status}): ${txt.slice(0, 120)}`);
            return null;
        }
        return (await res.json()) as JobSummary;
    } catch (err) {
        showToast(`Network error: ${String(err)}`);
        return null;
    }
}

export async function startFileIngestion(): Promise<void> {
    if (_selected.length === 0) return;
    const submit = byId<HTMLButtonElement>("ingestSubmitBtn");
    const statusEl = byId("ingestSubmitStatus");
    submit.disabled = true;
    statusEl.textContent = `Uploading 0/${_selected.length}…`;

    const jobs: JobSummary[] = [];
    let i = 0;
    for (const file of _selected) {
        statusEl.textContent = `Uploading ${++i}/${_selected.length}: ${file.name}`;
        const job = await uploadOne(file);
        if (job) {
            jobs.push(job);
            jobMeta.set(job.job_id, job);
            renderJob(job);
            attachStream(job.job_id);
        }
    }
    _selected.length = 0;
    renderSelected();
    statusEl.textContent = jobs.length > 0 ? `Started ${jobs.length} job${jobs.length > 1 ? "s" : ""}` : "";
    submit.disabled = _selected.length === 0;
    setTimeout(() => { statusEl.textContent = ""; }, 4000);
}

export async function startUrlIngestion(): Promise<void> {
    const input = byId<HTMLInputElement>("ingestUrlInput");
    const btn = byId<HTMLButtonElement>("ingestUrlBtn");
    const statusEl = byId("ingestUrlStatus");
    const url = input.value.trim();
    if (!url) return;
    btn.disabled = true;
    statusEl.textContent = "Fetching…";
    try {
        const res = await fetch(apiBase() + "/api/v1/ingest/url", {
            method: "POST",
            headers: { ...authHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });
        const data = await res.json();
        if (!res.ok) {
            statusEl.textContent = `Error: ${data.detail ?? res.status}`;
            return;
        }
        const job = data as JobSummary;
        jobMeta.set(job.job_id, job);
        renderJob(job);
        attachStream(job.job_id);
        input.value = "";
        statusEl.textContent = `Job started: ${job.filename}`;
        setTimeout(() => { statusEl.textContent = ""; }, 4000);
    } catch (err) {
        statusEl.textContent = `Network error: ${String(err)}`;
    } finally {
        btn.disabled = input.value.trim().length === 0;
    }
}

export async function checkDirectory(): Promise<void> {
    const input = byId<HTMLInputElement>("ingestDirInput");
    const result = byId("ingestDirResult");
    const ingestBtn = byId<HTMLButtonElement>("ingestDirBtn");
    const path = input.value.trim();
    if (!path) return;
    result.style.display = "none";
    result.className = "ingest-dir-result";
    try {
        const res = await fetch(apiBase() + "/api/v1/ingest/check-path", {
            method: "POST",
            headers: { ...authHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
        });
        const data = (await res.json()) as PathCheckResult;
        result.style.display = "block";
        if (!data.reachable) {
            result.classList.add("bad");
            result.innerHTML = `<strong>Unreachable.</strong> ${escapeHtml(data.reason ?? "unknown")}<br>The ingestion host could not access <code>${escapeHtml(path)}</code>.`;
            ingestBtn.disabled = true;
            return;
        }
        result.classList.add("ok");
        if (data.is_file) {
            result.innerHTML = `<strong>Reachable file.</strong> <code>${escapeHtml(data.path)}</code> · ${fmtSize(data.size_bytes ?? 0)}<br>Tip: use the Files tab for single-file uploads.`;
            ingestBtn.disabled = true;
            return;
        }
        const files = data.files ?? [];
        const truncNote = data.truncated ? ` (showing first ${files.length})` : "";
        const fileList = files.length > 0
            ? `<div class="ingest-dir-files">${files.map(escapeHtml).join("<br>")}</div>`
            : "";
        result.innerHTML =
            `<strong>Reachable directory.</strong> <code>${escapeHtml(data.path)}</code><br>`
            + `${data.file_count ?? 0} supported file${(data.file_count ?? 0) === 1 ? "" : "s"} found${truncNote}.`
            + fileList;
        ingestBtn.disabled = (data.file_count ?? 0) === 0;
    } catch (err) {
        result.style.display = "block";
        result.classList.add("bad");
        result.textContent = `Network error: ${String(err)}`;
    }
}

export async function startDirectoryIngestion(): Promise<void> {
    const input = byId<HTMLInputElement>("ingestDirInput");
    const btn = byId<HTMLButtonElement>("ingestDirBtn");
    const statusEl = byId("ingestDirStatus");
    const path = input.value.trim();
    if (!path) return;
    btn.disabled = true;
    statusEl.textContent = "Submitting…";
    try {
        const res = await fetch(apiBase() + "/api/v1/ingest/directory", {
            method: "POST",
            headers: { ...authHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
        });
        const data = await res.json();
        if (!res.ok) {
            statusEl.textContent = `Error: ${data.detail ?? res.status}`;
            return;
        }
        const jobs: JobSummary[] = data.jobs ?? [];
        for (const job of jobs) {
            jobMeta.set(job.job_id, job);
            renderJob(job);
            attachStream(job.job_id);
        }
        statusEl.textContent = `Submitted ${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
        setTimeout(() => { statusEl.textContent = ""; }, 4000);
    } catch (err) {
        statusEl.textContent = `Network error: ${String(err)}`;
    } finally {
        btn.disabled = false;
    }
}
