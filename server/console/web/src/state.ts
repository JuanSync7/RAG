// @summary
// Mutable singleton state shared across user-console feature modules.
// Centralizes everything that was previously closure-scoped in the original IIFE
// so feature modules can read/write without prop-drilling. Object property
// updates are visible across module boundaries (single shared reference).
// @end-summary

import type { SlashCommand } from "./user-types";

export interface AttachmentChip {
    id: string;
    icon: string;
    label: string;
}

export interface ConsoleState {
    activeConversationId: string | null;
    isStreaming: boolean;
    dynamicCmds: SlashCommand[];
    allSlashItems: HTMLElement[];
    slashSelIdx: number;
    allPickerItems: HTMLElement[];
    pickerIdx: number;
    attachments: AttachmentChip[];
    userScrolledUp: boolean;
    streamAbortCtrl: AbortController | null;
    pendingUserGroup: HTMLElement | null;
    pendingAssistantGroup: HTMLElement | null;
    pendingQueryText: string;
    convMenuTargetId: string | null;
    convMenuTargetTitle: string;
    renameTargetId: string | null;
}

export const state: ConsoleState = {
    activeConversationId: localStorage.getItem("nc_active_conv") || null,
    isStreaming: false,
    dynamicCmds: [],
    allSlashItems: [],
    slashSelIdx: 0,
    allPickerItems: [],
    pickerIdx: 0,
    attachments: [],
    userScrolledUp: false,
    streamAbortCtrl: null,
    pendingUserGroup: null,
    pendingAssistantGroup: null,
    pendingQueryText: "",
    convMenuTargetId: null,
    convMenuTargetTitle: "",
    renameTargetId: null,
};

export function setActiveConversation(id: string | null): void {
    state.activeConversationId = id;
    if (id) localStorage.setItem("nc_active_conv", id);
    else localStorage.removeItem("nc_active_conv");
}
