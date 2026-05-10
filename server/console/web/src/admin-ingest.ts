/**
 * @summary
 * Ingestion tab handlers + slash command dispatcher for the admin/operator console.
 * Exports: ingestStatusSnapshot, bindIngest
 * Deps: admin-types, admin-api, admin-render
 * @end-summary
 */

import type { ConsoleCommandSpec, JsonObject } from "./admin-types.js";
import { api, byId, executeConsoleCommand, parseSlash } from "./admin-api.js";
import { commandSummary, write } from "./admin-render.js";

export function ingestStatusSnapshot(): JsonObject {
    return {
        mode: byId<HTMLSelectElement>("ingestMode").value,
        target_path: byId<HTMLInputElement>("targetPath").value.trim() || null,
        update_mode: byId<HTMLInputElement>("updateMode").checked,
        build_kg: byId<HTMLInputElement>("buildKg").checked,
        verbose_stages: byId<HTMLInputElement>("verboseStages").checked,
        docling_enabled: byId<HTMLInputElement>("doclingEnabled").checked,
        docling_model: byId<HTMLInputElement>("doclingModel").value.trim() || null,
        vision_enabled: byId<HTMLInputElement>("visionEnabled").checked,
        vision_provider: byId<HTMLSelectElement>("visionProvider").value,
        vision_model: byId<HTMLInputElement>("visionModel").value.trim() || null,
    };
}

export function bindIngest(): void {
    byId("runIngestBtn").addEventListener("click", async () => {
        try {
            const payload = {
                mode: byId<HTMLSelectElement>("ingestMode").value,
                target_path: byId<HTMLInputElement>("targetPath").value.trim() || null,
                update_mode: byId<HTMLInputElement>("updateMode").checked,
                build_kg: byId<HTMLInputElement>("buildKg").checked,
                verbose_stages: byId<HTMLInputElement>("verboseStages").checked,
                docling_enabled: byId<HTMLInputElement>("doclingEnabled").checked,
                docling_model: byId<HTMLInputElement>("doclingModel").value.trim() || null,
                docling_artifacts_path: byId<HTMLInputElement>("doclingArtifactsPath").value.trim() || null,
                docling_strict: byId<HTMLInputElement>("doclingStrict").checked,
                docling_auto_download: byId<HTMLInputElement>("doclingAutoDownload").checked,
                vision_enabled: byId<HTMLInputElement>("visionEnabled").checked,
                vision_provider: byId<HTMLSelectElement>("visionProvider").value,
                vision_model: byId<HTMLInputElement>("visionModel").value.trim() || null,
                vision_api_base_url: byId<HTMLInputElement>("visionApiBaseUrl").value.trim() || null,
                vision_max_figures: Number(byId<HTMLInputElement>("visionMaxFigures").value || 4),
                vision_timeout_seconds: Number(byId<HTMLInputElement>("visionTimeoutSeconds").value || 60),
                vision_auto_pull: byId<HTMLInputElement>("visionAutoPull").checked,
                vision_strict: byId<HTMLInputElement>("visionStrict").checked,
            };
            const out = await api("POST", "/console/ingest", payload);
            write("ingestOut", out);
        } catch (err) {
            write("ingestOut", String(err));
        }
    });

    byId("ingestSlashRunBtn").addEventListener("click", async () => {
        const input = byId<HTMLInputElement>("ingestSlashInput");
        const { name, arg } = parseSlash(input.value);
        if (!name) {
            return;
        }
        try {
            const result = await executeConsoleCommand("ingest", name, arg, ingestStatusSnapshot());
            const action = String(result.action || "noop");
            if (action === "run_ingest") {
                byId("runIngestBtn").click();
            } else if (action === "clear_view") {
                write("ingestOut", "");
            } else if (action === "render_status") {
                write("ingestOut", (result.data?.state as JsonObject | undefined) || ingestStatusSnapshot());
            } else if (action === "render_help") {
                const cmds = Array.isArray(result.data?.commands)
                    ? (result.data?.commands as ConsoleCommandSpec[])
                    : [];
                write("ingestOut", commandSummary(cmds));
            } else {
                write("ingestOut", result.message || `No action mapped for /${name}`);
            }
        } catch (err) {
            write("ingestOut", `Command error: ${String(err)}`);
        }
        input.value = "";
    });

    byId("ingestSlashInput").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            byId("ingestSlashRunBtn").click();
        }
    });
}
