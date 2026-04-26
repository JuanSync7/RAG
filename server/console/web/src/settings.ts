// @summary
// Settings panel: theme + retrieval presets + persisted preferences.
// `loadSettings` is also exported because the orchestrator runs it once at
// startup so the UI controls reflect saved values before the panel is opened.
// @end-summary

import { byId } from "./dom";
import { getSettings } from "./api";
import { refs } from "./refs";
import { showToast } from "./toast";
import type { PresetConfig, ThemeValue } from "./user-types";

const mq: MediaQueryList = window.matchMedia("(prefers-color-scheme: light)");

function applyThemeToDOM(val: ThemeValue | string): void {
    const resolved = val === "system" ? (mq.matches ? "light" : "dark") : val;
    document.documentElement.dataset.theme = resolved;
    document.querySelectorAll<HTMLElement>(".theme-opt").forEach((el) => {
        el.classList.toggle("active", el.dataset.themeVal === val);
    });
}

function setTheme(val: ThemeValue | string): void {
    applyThemeToDOM(val);
    localStorage.setItem("nc_theme", val);
}

const PRESETS: Record<string, PresetConfig> = {
    balanced: { searchLimit: 10, rerankTopK: 5 },
    precise: { searchLimit: 8, rerankTopK: 3 },
    broad: { searchLimit: 25, rerankTopK: 10 },
    fast: { searchLimit: 5, rerankTopK: 2 },
};

function applyPreset(name: string): void {
    const p = PRESETS[name];
    if (!p) return;
    byId<HTMLInputElement>("searchLimit").value = String(p.searchLimit);
    byId("searchLimitVal").textContent = String(p.searchLimit);
    byId<HTMLInputElement>("rerankTopK").value = String(p.rerankTopK);
    byId("rerankVal").textContent = String(p.rerankTopK);
}

function openSettings(): void {
    refs.settingsOverlay.classList.add("open");
    refs.settingsPanel.classList.add("open");
    loadSettings();
}

function closeSettings(): void {
    refs.settingsOverlay.classList.remove("open");
    refs.settingsPanel.classList.remove("open");
}

function saveSettings(): void {
    const s = {
        theme: localStorage.getItem("nc_theme") || "dark",
        preset: byId<HTMLSelectElement>("presetSelect").value,
        searchLimit: byId<HTMLInputElement>("searchLimit").value,
        rerankTopK: byId<HTMLInputElement>("rerankTopK").value,
        streaming: byId<HTMLInputElement>("streamingToggle").checked,
        memory_enabled: byId<HTMLInputElement>("memoryToggle").checked,
        citations: byId<HTMLInputElement>("citationsToggle").checked,
    };
    localStorage.setItem("nc_settings", JSON.stringify(s));
    closeSettings();
    showToast("Settings saved");
}

export function loadSettings(): void {
    const s = getSettings();
    const theme = localStorage.getItem("nc_theme") || (s.theme as string) || "dark";
    applyThemeToDOM(theme);
    if (s.preset) byId<HTMLSelectElement>("presetSelect").value = s.preset as string;
    if (s.searchLimit) {
        byId<HTMLInputElement>("searchLimit").value = s.searchLimit as string;
        byId("searchLimitVal").textContent = s.searchLimit as string;
    }
    if (s.rerankTopK) {
        byId<HTMLInputElement>("rerankTopK").value = s.rerankTopK as string;
        byId("rerankVal").textContent = s.rerankTopK as string;
    }
    if (s.streaming !== undefined) byId<HTMLInputElement>("streamingToggle").checked = s.streaming as boolean;
    if (s.memory_enabled !== undefined) byId<HTMLInputElement>("memoryToggle").checked = s.memory_enabled as boolean;
    if (s.citations !== undefined) byId<HTMLInputElement>("citationsToggle").checked = s.citations as boolean;
}

function resetSettings(): void {
    localStorage.removeItem("nc_settings");
    localStorage.removeItem("nc_theme");
    applyThemeToDOM("dark");
    byId<HTMLSelectElement>("presetSelect").value = "balanced";
    applyPreset("balanced");
    byId<HTMLInputElement>("streamingToggle").checked = true;
    byId<HTMLInputElement>("memoryToggle").checked = true;
    byId<HTMLInputElement>("citationsToggle").checked = true;
    showToast("Settings reset to defaults");
}

export function initSettings(): void {
    mq.addEventListener("change", () => {
        if (localStorage.getItem("nc_theme") === "system") applyThemeToDOM("system");
    });

    document.querySelectorAll<HTMLElement>(".theme-opt").forEach((el) => {
        el.addEventListener("click", () => setTheme(el.dataset.themeVal || "dark"));
    });

    document.getElementById("presetSelect")?.addEventListener("change", (e: Event) => {
        applyPreset((e.target as HTMLSelectElement).value);
    });

    document.getElementById("searchLimit")?.addEventListener("input", (e: Event) => {
        byId("searchLimitVal").textContent = (e.target as HTMLInputElement).value;
    });
    document.getElementById("rerankTopK")?.addEventListener("input", (e: Event) => {
        byId("rerankVal").textContent = (e.target as HTMLInputElement).value;
    });

    document.getElementById("settingsBtn")?.addEventListener("click", openSettings);
    document.getElementById("customizeOpenSettings")?.addEventListener("click", openSettings);
    refs.settingsOverlay.addEventListener("click", closeSettings);
    document.getElementById("settingsClose")?.addEventListener("click", closeSettings);
    document.getElementById("settingsSaveBtn")?.addEventListener("click", saveSettings);
    document.getElementById("settingsResetBtn")?.addEventListener("click", resetSettings);

    applyThemeToDOM(localStorage.getItem("nc_theme") || "dark");
}
