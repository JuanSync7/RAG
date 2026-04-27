/**
 * @summary
 * HTTP + slash-command client helpers for the admin/operator console.
 * Wraps fetch with auth headers, JSON parsing, and error normalization. Also
 * provides slash-command cache lookups and parsing.
 * Exports: byId, authHeaders, api, fetchConsoleCommands, executeConsoleCommand, parseSlash
 * Deps: admin-types, admin-state
 * @end-summary
 */

import type { ConsoleCommandSpec, CommandExecution, JsonObject } from "./admin-types.js";
import { getCommandCache } from "./admin-state.js";

export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) {
        throw new Error(`Missing required element #${id}`);
    }
    return el as T;
}

export function authHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const apiKey = byId<HTMLInputElement>("apiKey").value.trim();
    const bearer = byId<HTMLInputElement>("bearerToken").value.trim();
    if (apiKey) {
        headers["x-api-key"] = apiKey;
    }
    if (bearer) {
        headers["Authorization"] = `Bearer ${bearer}`;
    }
    return headers;
}

export async function api(method: string, path: string, body?: JsonObject): Promise<JsonObject> {
    const response = await fetch(path, {
        method,
        headers: authHeaders(),
        body: body ? JSON.stringify(body) : undefined,
    });
    const text = await response.text();
    let payload: JsonObject;
    try {
        payload = text ? (JSON.parse(text) as JsonObject) : {};
    } catch {
        payload = { raw: text };
    }
    if (!response.ok) {
        const err = payload.error as JsonObject | undefined;
        const rawDetail = (payload as JsonObject).detail;
        const detailStr = typeof rawDetail === "string" ? rawDetail : undefined;
        const message: string =
            typeof err?.message === "string"
                ? err.message
                : detailStr ?? `HTTP ${response.status}`;
        throw new Error(message);
    }
    return payload;
}

export async function fetchConsoleCommands(mode: "query" | "ingest"): Promise<ConsoleCommandSpec[]> {
    const cache = getCommandCache();
    if (cache[mode]) {
        return cache[mode];
    }
    const payload = await api("GET", `/console/commands?mode=${mode}`);
    const data = (payload.data as JsonObject | undefined) || {};
    const commands = Array.isArray(data.commands) ? (data.commands as ConsoleCommandSpec[]) : [];
    cache[mode] = commands;
    return commands;
}

export async function executeConsoleCommand(
    mode: "query" | "ingest",
    command: string,
    arg: string,
    state: JsonObject,
): Promise<CommandExecution> {
    const payload = await api("POST", "/console/command", {
        mode,
        command,
        arg,
        state,
    });
    const data = (payload.data as JsonObject | undefined) || {};
    return data as CommandExecution;
}

export function parseSlash(raw: string): { name: string; arg: string } {
    const text = raw.trim();
    const normalized = (text.startsWith("/") ? text.slice(1) : text).trim();
    const [name, ...rest] = normalized.split(/\s+/);
    return {
        name: (name || "").toLowerCase(),
        arg: rest.join(" ").trim(),
    };
}
