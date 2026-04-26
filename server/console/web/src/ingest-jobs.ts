/** @summary
 * Ingest job-list rendering: job card HTML, formatting helpers, status pills,
 * and the in-memory stores (`_jobLog`, `_jobMeta`) shared with the stream module.
 * Exports: JobSummary, fmtSize, escapeHtml, renderJob, cancelJob,
 *          jobLog, jobMeta
 * Deps: api, dom, toast
 * @end-summary
 */

import { byId } from "./dom";
import { apiBase, authHeaders } from "./api";
import { showToast } from "./toast";
import { refreshJob } from "./ingest-stream";

export interface JobSummary {
    job_id: string;
    filename: string;
    size_bytes: number;
    status: string;
    created_at: number;
    started_at: number | null;
    finished_at: number | null;
    stored_chunks: number;
    error: string | null;
}

// Shared job state — consumed by both rendering (this module) and the
// SSE stream module (`ingest-stream.ts`). Kept here because it is the
// data backing the job list.
export const jobLog = new Map<string, string[]>();
export const jobMeta = new Map<string, JobSummary>();

export function fmtSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export function escapeHtml(s: string): string {
    return s.replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string),
    );
}

function jobCardHtml(job: JobSummary): string {
    const log = (jobLog.get(job.job_id) ?? []).slice(-30).join("\n") || "Queued…";
    const isTerminal = ["done", "failed", "cancelled"].includes(job.status);
    const meta: string[] = [fmtSize(job.size_bytes)];
    if (job.stored_chunks) meta.push(`${job.stored_chunks} chunks`);
    if (job.started_at && job.finished_at) {
        meta.push(`${(job.finished_at - job.started_at).toFixed(1)}s`);
    } else if (job.started_at && !isTerminal) {
        meta.push(`running ${((Date.now() / 1000 - job.started_at)).toFixed(0)}s`);
    }
    const progressClass = job.error ? "ingest-job-progress ingest-job-error" : "ingest-job-progress";
    const progressText = job.error ? job.error : log;
    const cancelBtn = !isTerminal
        ? `<button class="ingest-job-action" data-cancel="${job.job_id}">Cancel</button>`
        : "";
    return `
        <div class="ingest-job ${job.status}" data-job="${job.job_id}">
            <div class="ingest-job-head">
                <span class="ingest-job-name">${escapeHtml(job.filename)}</span>
                <span class="ingest-job-status ${job.status}">${job.status}</span>
            </div>
            <div class="ingest-job-meta">${meta.join(" · ")}</div>
            <div class="${progressClass}">${escapeHtml(progressText)}</div>
            <div class="ingest-job-actions">${cancelBtn}</div>
        </div>`;
}

export function renderJob(job: JobSummary): void {
    jobMeta.set(job.job_id, job);
    const list = byId("ingestJobsList");
    const empty = list.querySelector(".ingest-jobs-empty");
    if (empty) empty.remove();
    let card = list.querySelector<HTMLElement>(`[data-job="${job.job_id}"]`);
    if (!card) {
        const wrap = document.createElement("div");
        wrap.innerHTML = jobCardHtml(job);
        const el = wrap.firstElementChild as HTMLElement;
        list.prepend(el);
        card = el;
    } else {
        card.outerHTML = jobCardHtml(job);
    }
    list.querySelectorAll<HTMLButtonElement>("[data-cancel]").forEach((btn) => {
        const id = btn.dataset.cancel!;
        btn.onclick = () => cancelJob(id);
    });
}

export async function cancelJob(jobId: string): Promise<void> {
    const meta = jobMeta.get(jobId);
    if (meta) {
        renderJob({ ...meta, status: "cancelling" });
    }
    showToast("Cancelling job…");
    try {
        const res = await fetch(apiBase() + `/api/v1/ingest/jobs/${jobId}/cancel`, {
            method: "POST",
            headers: authHeaders(),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        await refreshJob(jobId);
        showToast("Cancel sent");
    } catch (err) {
        showToast(`Cancel failed: ${String(err)}`);
        if (meta) renderJob(meta);
    }
}
