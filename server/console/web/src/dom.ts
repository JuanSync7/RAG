// @summary
// DOM helper utilities shared across the user console.
// Exports: byId, escHtml, fmtTime, fmtRelative
// Deps: (none)
// @end-summary

export const byId = <T extends HTMLElement = HTMLElement>(id: string): T => {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing required element #${id}`);
    return el as T;
};

export function escHtml(s: string): string {
    return s
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
        .replace(/\//g, "&#x2F;");
}

export function fmtTime(ms: number): string {
    return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function fmtRelative(ms: number): string {
    const now = Date.now();
    const diff = now - ms;
    if (diff < 86400000) return "Today";
    if (diff < 172800000) return "Yesterday";
    return new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" });
}
