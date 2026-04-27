// @summary
// Lazily-populated singleton of DOM references for the user console.
// `populateRefs()` is invoked once during DOMContentLoaded to bind all required
// elements; feature modules then import `refs` and dereference fields directly.
// Centralizing the lookups here means a missing element triggers exactly one
// clear error at startup instead of fanning out across feature modules.
// @end-summary

import { byId } from "./dom";

export interface ConsoleRefs {
    sidebar: HTMLElement;
    backdrop: HTMLElement;
    settingsOverlay: HTMLElement;
    settingsPanel: HTMLElement;
    thread: HTMLElement;
    fab: HTMLElement;
    dropdown: HTMLElement;
    ta: HTMLTextAreaElement;
    attachPopover: HTMLElement;
    webInputPanel: HTMLElement;
    kbPanel: HTMLElement;
    cmdPicker: HTMLElement;
    attachBtn: HTMLElement;
    cmdBtn: HTMLElement;
}

const _refs: Partial<ConsoleRefs> = {};

export const refs = _refs as ConsoleRefs;

export function populateRefs(): void {
    _refs.sidebar = byId("sidebar");
    _refs.backdrop = byId("sidebarBackdrop");
    _refs.settingsOverlay = byId("settingsOverlay");
    _refs.settingsPanel = byId("settingsPanel");
    _refs.thread = byId("thread");
    _refs.fab = byId("scrollFab");
    _refs.dropdown = byId("slashDropdown");
    _refs.ta = byId<HTMLTextAreaElement>("inputArea");
    _refs.attachPopover = byId("attachPopover");
    _refs.webInputPanel = byId("webInputPanel");
    _refs.kbPanel = byId("kbPanel");
    _refs.cmdPicker = byId("cmdPicker");
    _refs.attachBtn = byId("attachBtn");
    _refs.cmdBtn = byId("cmdBtn");
}
