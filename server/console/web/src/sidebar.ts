// @summary
// Sidebar navigation: panel switching, collapse/expand, mobile open/close,
// and touch swipe-to-open. Exports openSidebar/closeSidebar for cross-feature
// callers (e.g. nav clicks need to close the sidebar on mobile).
// @end-summary

import { byId } from "./dom";
import { refs } from "./refs";

function isDesktop(): boolean {
    return window.innerWidth > 1024;
}

function showPanel(panelId: string | undefined): void {
    document.querySelectorAll<HTMLElement>(".sidebar-panel").forEach((p) =>
        p.classList.remove("active")
    );
    if (panelId) document.getElementById(panelId)?.classList.add("active");
}

function setNavActive(el: HTMLElement): void {
    document.querySelectorAll<HTMLElement>(".sidebar-nav-item").forEach((n) =>
        n.classList.remove("active")
    );
    el.classList.add("active");
    if (!refs.sidebar.classList.contains("collapsed")) showPanel(el.dataset.panel);
    if (!isDesktop()) closeSidebar();
}

function toggleSidebarCollapse(): void {
    refs.sidebar.classList.toggle("collapsed");
    byId("sidebarCollapseBtn").innerHTML = refs.sidebar.classList.contains("collapsed")
        ? "&#8250;"
        : "&#8249;";
    if (refs.sidebar.classList.contains("collapsed")) {
        document.querySelectorAll<HTMLElement>(".sidebar-panel").forEach((p) =>
            p.classList.remove("active")
        );
    } else {
        const activeNav = refs.sidebar.querySelector<HTMLElement>(".sidebar-nav-item.active");
        if (activeNav) showPanel(activeNav.dataset.panel);
    }
}

export function openSidebar(): void {
    if (isDesktop()) {
        refs.sidebar.classList.remove("collapsed");
        byId("sidebarCollapseBtn").innerHTML = "&#8249;";
        const activeNav = refs.sidebar.querySelector<HTMLElement>(".sidebar-nav-item.active");
        if (activeNav) showPanel(activeNav.dataset.panel);
    } else {
        refs.sidebar.classList.add("open");
        refs.backdrop.classList.add("active");
    }
}

export function closeSidebar(): void {
    if (isDesktop()) {
        refs.sidebar.classList.add("collapsed");
        byId("sidebarCollapseBtn").innerHTML = "&#8250;";
    } else {
        refs.sidebar.classList.remove("open");
        refs.backdrop.classList.remove("active");
    }
}

export function initSidebar(): void {
    byId("toggleBtn").addEventListener("click", () =>
        refs.sidebar.classList.contains("open") ? closeSidebar() : openSidebar()
    );
    document.querySelectorAll<HTMLElement>(".sidebar-nav-item").forEach((item) =>
        item.addEventListener("click", () => setNavActive(item))
    );
    document.getElementById("sidebarCollapseBtn")?.addEventListener("click", toggleSidebarCollapse);

    window.addEventListener("resize", () => {
        if (isDesktop()) {
            refs.sidebar.classList.remove("open");
            refs.backdrop.classList.remove("active");
        }
    });

    let touchStartX = 0;
    document.addEventListener(
        "touchstart",
        (e: TouchEvent) => {
            touchStartX = e.touches[0].clientX;
        },
        { passive: true }
    );
    document.addEventListener(
        "touchend",
        (e: TouchEvent) => {
            const dx = e.changedTouches[0].clientX - touchStartX;
            if (!isDesktop()) {
                if (dx > 60 && touchStartX < 30) openSidebar();
                if (dx < -60 && refs.sidebar.classList.contains("open")) closeSidebar();
            }
        },
        { passive: true }
    );
}
