/**
 * @summary
 * Display formatting helpers for primitive values rendered in the console UI.
 * Exports: fmtSize
 * Deps: (none)
 * @end-summary
 */

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
