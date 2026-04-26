// @summary
// JSON API layer for the user console. The console is served from the same
// origin as the API, so requests are relative and unauthenticated by default
// (backend dev mode falls back to anonymous when no API keys/JWT are required).
// Exports: getSettings, authHeaders, apiBase, api
// Deps: (none)
// @end-summary

export function getSettings(): Record<string, unknown> {
    const raw = localStorage.getItem("nc_settings");
    return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
}

export function authHeaders(): Record<string, string> {
    return { "Content-Type": "application/json" };
}

export function apiBase(): string {
    return "";
}

export async function api<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = apiBase() + path;
    const opts: RequestInit = { method, headers: authHeaders() };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    const json = (await res.json()) as { ok: boolean; data?: T; error?: { message: string } };
    if (!res.ok || !json.ok) {
        throw new Error(json.error?.message || `HTTP ${res.status}`);
    }
    return json.data as T;
}
