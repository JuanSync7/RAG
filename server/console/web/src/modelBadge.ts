// @summary
// Header model badge: queries /console/model-info on init and renders the
// active provider/model. Hides the dot and shows "Generation off" when the
// backend has generation disabled (retrieval-only mode).
// @end-summary

import { api } from "./api";

interface ModelInfo {
    generation_enabled: boolean;
    provider: string;
    model: string;
    display: string;
}

const PROVIDER_LABELS: Record<string, string> = {
    ollama: "Ollama",
    openai: "OpenAI",
    anthropic: "Anthropic",
    openrouter: "OpenRouter",
    azure: "Azure",
    bedrock: "Bedrock",
    gemini: "Gemini",
    groq: "Groq",
    mistral: "Mistral",
};

function formatLabel(info: ModelInfo): string {
    if (!info.generation_enabled) return "Generation off";
    const provider = info.provider ? PROVIDER_LABELS[info.provider.toLowerCase()] ?? info.provider : "";
    const model = info.display || info.model || "unknown";
    return provider ? `${provider} · ${model}` : model;
}

export async function loadModelInfo(): Promise<void> {
    const badge = document.getElementById("modelBadge");
    const text = document.getElementById("modelBadgeText");
    if (!badge || !text) return;
    try {
        const info = await api<ModelInfo>("GET", "/console/model-info");
        text.textContent = formatLabel(info);
        badge.classList.toggle("disabled", !info.generation_enabled);
        badge.title = info.generation_enabled
            ? `${info.provider || "model"}: ${info.display}`
            : "Backend is in retrieval-only mode (RAG_GENERATION_ENABLED=false)";
    } catch {
        text.textContent = "Model unknown";
        badge.classList.add("disabled");
    }
}
