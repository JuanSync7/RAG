/**
 * @summary
 * Display formatting helpers for primitive values rendered in the console UI.
 * Exports: fmtSize, pct
 * Deps: (none)
 * @end-summary
 */

/**
 * Render a unit-interval relevance score (0..1) as an integer percentage,
 * clamped to [0, 100]. Guards the display class "a value shown as a %": any
 * score that violates the [0,1] relevance contract — e.g. an un-normalized
 * ranker ordinal — can never render as "700%" or overflow a progress-bar
 * width. Non-finite / null / undefined inputs render as 0.
 */
export function pct(score: number | null | undefined): number {
    const v = typeof score === "number" && Number.isFinite(score) ? score : 0;
    return Math.max(0, Math.min(100, Math.round(v * 100)));
}

/**
 * Humanize a byte count into B / KB / MB / GB using base-1024 units.
 * Negative or non-finite inputs return "0 B".
 */
export function fmtSize(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
