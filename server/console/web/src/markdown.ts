// @summary
// Markdown rendering pipeline for streamed assistant output.
// Configures marked with a custom code-block renderer (copy-button UI),
// normalizes inline-list markers, and sanitizes the final HTML with DOMPurify.
// Exports: parseMarkdown, normalizeMarkdown
// Deps: marked, dompurify
// @end-summary

import { marked } from "marked";
import DOMPurify from "dompurify";
import { escHtml } from "./dom";

// Custom code-block renderer: wraps <pre><code> in a copy-button UI.
// Clicks are handled via event delegation on #thread — no inline onclick needed,
// which also means DOMPurify does not need to allow event attributes.
marked.use({
    gfm: true,
    breaks: false,
    renderer: {
        code({ text, lang }: { text: string; lang?: string }): string {
            const langLabel = escHtml(lang ?? "code");
            const escaped = text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");
            return [
                `<div class="code-block-wrap">`,
                `<div class="code-block-header">`,
                `<span>${langLabel}</span>`,
                `<button class="copy-code-btn">&#128203; Copy</button>`,
                `</div>`,
                `<div class="code-block">${escaped}</div>`,
                `</div>`,
            ].join("");
        },
    },
});

/** Returns true if a word looks like a numbered list marker: "1.", "2.", "10.", etc. */
function isListMarker(word: string): boolean {
    if (!word.endsWith(".")) return false;
    const n = Number(word.slice(0, -1));
    return Number.isInteger(n) && n > 0;
}

/**
 * If a line packs multiple list items inline (e.g. "1. Foo 2. Bar" or "- A - B"),
 * splits each item onto its own line. Bullet splitting is guarded: only fires when
 * the line already starts with "- "/"* " so standalone dashes in prose are unaffected.
 */
function splitInlineList(line: string): string {
    const words = line.split(" ");
    if (words.length < 3) return line;

    const lineStartsWithBullet = words[0] === "-" || words[0] === "*";
    const subLines: string[] = [];
    let current: string[] = [];

    for (let i = 0; i < words.length; i++) {
        const w = words[i];
        const isBulletCont = lineStartsWithBullet && i > 0 && (w === "-" || w === "*");
        const isNumberedCont = i > 0 && isListMarker(w);
        if (current.length > 0 && (isBulletCont || isNumberedCont)) {
            subLines.push(current.join(" "));
            current = [w];
        } else {
            current.push(w);
        }
    }
    if (current.length) subLines.push(current.join(" "));
    return subLines.length > 1 ? subLines.join("\n") : line;
}

const PIPE_ROW_RE = /^\s*\|.*\|\s*$/;
const PIPE_SEP_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/;

function countPipeColumns(line: string): number {
    const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    return trimmed.split("|").length;
}

/**
 * Inject a synthetic header separator when a chunk fragment contains a block
 * of pipe-rows without the GFM `|---|---|` separator. Without this, retrieval
 * chunks that begin mid-table render as one collapsed paragraph.
 */
function fixOrphanPipeTables(text: string): string {
    const lines = text.split("\n");
    const out: string[] = [];
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        if (PIPE_ROW_RE.test(line)) {
            let j = i;
            while (j < lines.length && PIPE_ROW_RE.test(lines[j])) j++;
            const block = lines.slice(i, j);
            const hasSep = block.length >= 2 && PIPE_SEP_RE.test(block[1]);
            if (!hasSep && block.length >= 2) {
                const cols = countPipeColumns(block[0]);
                const sep = "|" + " --- |".repeat(cols);
                out.push(block[0]);
                out.push(sep);
                for (let k = 1; k < block.length; k++) out.push(block[k]);
            } else {
                out.push(...block);
            }
            i = j;
            continue;
        }
        out.push(line);
        i++;
    }
    return out.join("\n");
}

/**
 * Pre-process LLM output so list items always start on their own line.
 * Code fences are split out first so their content is never modified.
 */
export function normalizeMarkdown(raw: string): string {
    const segments = raw.split(/(```[\s\S]*?```)/);
    return segments.map((seg, i) => {
        if (i % 2 === 1) return seg;
        const listFixed = seg.split("\n").map(splitInlineList).join("\n");
        return fixOrphanPipeTables(listFixed);
    }).join("");
}

/**
 * Render markdown to sanitized HTML.
 * marked.parse is synchronous when no async hooks are configured.
 */
export function parseMarkdown(raw: string): string {
    return DOMPurify.sanitize(marked.parse(normalizeMarkdown(raw)) as string);
}

function humanizeKey(key: string): string {
    return key
        .replace(/[_-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/\b\w/g, (c) => c.toUpperCase());
}

function isPrimitive(v: unknown): boolean {
    return v == null || typeof v === "string" || typeof v === "number" || typeof v === "boolean";
}

function formatPrimitive(v: unknown): string {
    if (v == null) return "_—_";
    if (typeof v === "string") return v.trim() || "_(empty)_";
    return String(v);
}

function formatValue(value: unknown, depth: number): string {
    if (depth > 4) return "_…_";
    if (isPrimitive(value)) return formatPrimitive(value);

    if (Array.isArray(value)) {
        if (!value.length) return "_(empty list)_";
        if (value.every(isPrimitive)) {
            return "\n" + value.map((v) => `- ${formatPrimitive(v)}`).join("\n");
        }
        return "\n" + value.map((v) => {
            const inner = formatValue(v, depth + 1).replace(/\n/g, "\n  ");
            return `- ${inner.startsWith("\n") ? inner.trimStart() : inner}`;
        }).join("\n");
    }

    if (typeof value === "object") {
        const entries = Object.entries(value as Record<string, unknown>);
        if (!entries.length) return "_(empty)_";
        const lines = entries.map(([k, v]) => {
            const label = `**${humanizeKey(k)}**`;
            if (isPrimitive(v)) return `- ${label}: ${formatPrimitive(v)}`;
            const inner = formatValue(v, depth + 1).replace(/\n/g, "\n  ");
            return `- ${label}:${inner.startsWith("\n") ? inner : "\n  " + inner}`;
        });
        return (depth === 0 ? "" : "\n") + lines.join("\n");
    }

    return String(value);
}

/**
 * Coerce arbitrary API payloads into a markdown-friendly string for display.
 * Strings pass through (assumed to be markdown). Objects/arrays render as
 * humanized bullet lists. null/undefined become empty.
 */
export function formatApiPayload(value: unknown): string {
    if (value == null) return "";
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    return formatValue(value, 0);
}
