// @summary
// Thumbs-up / thumbs-down feedback wiring. Click opens a modal asking for an
// optional comment; submit POSTs {rating, comment, transcript[]} to
// /console/feedback. The transcript is the full visible thread snapshot.
// @end-summary

import { byId } from "./dom";
import { api } from "./api";
import { showToast } from "./toast";
import { refs } from "./refs";

export interface FeedbackContext {
    conversationId: string | null;
    query: string;
    answer: () => string;
}

interface PendingFeedback {
    rating: "up" | "down";
    ctx: FeedbackContext;
    upBtn: HTMLButtonElement;
    downBtn: HTMLButtonElement;
    submitted: { value: "up" | "down" | null };
}

let pending: PendingFeedback | null = null;
let modalInited = false;

function captureTranscript(): { role: string; text: string }[] {
    const turns: { role: string; text: string }[] = [];
    const rows = refs.thread.querySelectorAll<HTMLElement>(".msg-row");
    rows.forEach((row) => {
        const role = row.classList.contains("user") ? "user" : "assistant";
        const bubble = row.querySelector<HTMLElement>(".bubble");
        const text = (bubble?.innerText ?? "").trim();
        if (text) turns.push({ role, text });
    });
    return turns;
}

function openModal(rating: "up" | "down"): void {
    initModal();
    const overlay = byId("feedbackModal");
    const title = byId("feedbackModalTitle");
    const input = byId<HTMLTextAreaElement>("feedbackInput");
    title.textContent = rating === "up" ? "Give positive feedback" : "Give negative feedback";
    input.value = "";
    overlay.classList.add("open");
    overlay.setAttribute("aria-hidden", "false");
    setTimeout(() => input.focus(), 50);
}

function closeModal(): void {
    const overlay = byId("feedbackModal");
    overlay.classList.remove("open");
    overlay.setAttribute("aria-hidden", "true");
    pending = null;
}

async function submitFeedback(): Promise<void> {
    if (!pending) return closeModal();
    const { rating, ctx, upBtn, downBtn, submitted } = pending;
    const comment = byId<HTMLTextAreaElement>("feedbackInput").value.trim();
    const winner = rating === "up" ? upBtn : downBtn;
    const loser = rating === "up" ? downBtn : upBtn;
    closeModal();
    submitted.value = rating;
    winner.classList.add("active");
    loser.disabled = true;
    try {
        await api("POST", "/console/feedback", {
            rating,
            conversation_id: ctx.conversationId,
            query: ctx.query,
            answer: ctx.answer(),
            comment: comment || null,
            transcript: captureTranscript(),
        });
        showToast(rating === "up" ? "Thanks for the feedback!" : "Got it, we'll improve.");
    } catch (err) {
        submitted.value = null;
        winner.classList.remove("active");
        loser.disabled = false;
        showToast("Feedback failed: " + String(err));
    }
}

function initModal(): void {
    if (modalInited) return;
    modalInited = true;
    byId("feedbackCancel").addEventListener("click", closeModal);
    byId("feedbackSubmit").addEventListener("click", () => void submitFeedback());
    byId("feedbackModal").addEventListener("click", (e) => {
        if ((e.target as HTMLElement).id === "feedbackModal") closeModal();
    });
    document.addEventListener("keydown", (e) => {
        const overlay = byId("feedbackModal");
        if (!overlay.classList.contains("open")) return;
        if (e.key === "Escape") {
            e.preventDefault();
            closeModal();
        } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            void submitFeedback();
        }
    });
}

export function attachFeedback(
    upBtn: HTMLButtonElement,
    downBtn: HTMLButtonElement,
    ctx: FeedbackContext,
): void {
    const submitted: { value: "up" | "down" | null } = { value: null };
    const onClick = (rating: "up" | "down") => () => {
        if (submitted.value) return;
        pending = { rating, ctx, upBtn, downBtn, submitted };
        openModal(rating);
    };
    upBtn.addEventListener("click", onClick("up"));
    downBtn.addEventListener("click", onClick("down"));
}
