/**
 * @summary
 * Health check + log refresh handlers for the admin/operator console.
 * Exports: refreshHealth, bindHealth
 * Deps: admin-types, admin-api, admin-render
 * @end-summary
 */

import type { JsonObject } from "./admin-types.js";
import { api, byId } from "./admin-api.js";
import { write } from "./admin-render.js";

export async function refreshHealth(): Promise<void> {
    try {
        const data = await api("GET", "/console/health");
        write("healthOut", data);
        const status = String(((data.data as JsonObject | undefined)?.status as string | undefined) || "unknown");
        const pill = byId("healthPill");
        pill.textContent = status;
        pill.className = `pill ${status === "healthy" ? "ok" : "warn"}`;
    } catch (err) {
        const pill = byId("healthPill");
        pill.textContent = "error";
        pill.className = "pill err";
        write("healthOut", String(err));
    }
}

export function bindHealth(): void {
    byId("refreshHealthBtn").addEventListener("click", refreshHealth);
    byId("pingBtn").addEventListener("click", refreshHealth);
    byId("refreshLogsBtn").addEventListener("click", async () => {
        try {
            const out = await api("GET", "/console/logs?lines=200");
            write("logsOut", out);
        } catch (err) {
            write("logsOut", String(err));
        }
    });
}
