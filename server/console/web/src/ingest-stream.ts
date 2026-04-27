/** @summary
 * Ingest job SSE stream attachment plus REST polling fallbacks
 * (`refreshJob`, `refreshJobsList`). Mutates the shared `jobLog`/`jobMeta`
 * stores from `ingest-jobs` and re-renders cards as events arrive.
 * Exports: attachStream, refreshJob, refreshJobsList
 * Deps: api, dom, ingest-jobs
 * @end-summary
 */

import { byId } from "./dom";
import { apiBase, authHeaders } from "./api";
import { JobSummary, jobLog, jobMeta, renderJob } from "./ingest-jobs";

const _activeStreams = new Map<string, EventSource>();

export function attachStream(jobId: string): void {
    if (_activeStreams.has(jobId)) return;
    const url = apiBase() + `/api/v1/ingest/jobs/${jobId}/stream`;
    const es = new EventSource(url);
    _activeStreams.set(jobId, es);
    const finish = () => {
        es.close();
        _activeStreams.delete(jobId);
        refreshJob(jobId);
    };
    const onEvt = (e: MessageEvent) => {
        try {
            const data = JSON.parse(e.data) as {
                kind: string;
                message: string;
                detail?: Record<string, unknown>;
                status: string;
            };
            const buf = jobLog.get(jobId) ?? [];
            buf.push(data.message);
            jobLog.set(jobId, buf);
            const meta = jobMeta.get(jobId);
            if (meta) {
                meta.status = data.status;
                if (data.kind === "done" && data.detail?.stored_chunks) {
                    meta.stored_chunks = Number(data.detail.stored_chunks);
                    meta.finished_at = Date.now() / 1000;
                }
                if (data.kind === "error") {
                    meta.error = data.message;
                    meta.finished_at = Date.now() / 1000;
                }
                renderJob(meta);
            }
            if (["done", "error"].includes(data.kind)) finish();
        } catch {
            /* ignore */
        }
    };
    es.addEventListener("stage", onEvt as EventListener);
    es.addEventListener("done", onEvt as EventListener);
    es.addEventListener("error", (e) => {
        const ev = e as MessageEvent;
        if (ev.data) onEvt(ev);
        else finish();
    });
}

export async function refreshJob(jobId: string): Promise<void> {
    try {
        const res = await fetch(apiBase() + `/api/v1/ingest/jobs/${jobId}`, { headers: authHeaders() });
        if (!res.ok) return;
        const job = (await res.json()) as JobSummary;
        jobMeta.set(job.job_id, job);
        renderJob(job);
    } catch {
        /* ignore */
    }
}

export async function refreshJobsList(): Promise<void> {
    try {
        const res = await fetch(apiBase() + "/api/v1/ingest/jobs", { headers: authHeaders() });
        if (!res.ok) return;
        const data = (await res.json()) as { jobs: JobSummary[] };
        const list = byId("ingestJobsList");
        if (data.jobs.length === 0) {
            list.innerHTML = '<div class="ingest-jobs-empty">No jobs yet.</div>';
            return;
        }
        list.innerHTML = "";
        for (const j of data.jobs) {
            jobMeta.set(j.job_id, j);
            renderJob(j);
            if (["pending", "running"].includes(j.status)) attachStream(j.job_id);
        }
    } catch {
        /* ignore */
    }
}
