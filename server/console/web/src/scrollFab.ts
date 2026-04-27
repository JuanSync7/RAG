// @summary
// "Scroll to bottom" floating action button + scroll-tracking for the message
// thread. `scrollToBottom` is the shared API used by thread + streaming.
// @end-summary

import { refs } from "./refs";
import { state } from "./state";

export function scrollToBottom(): void {
    if (!state.userScrolledUp) refs.thread.scrollTop = refs.thread.scrollHeight;
}

export function initScrollFab(): void {
    refs.thread.addEventListener("scroll", () => {
        const atBottom =
            refs.thread.scrollHeight - refs.thread.scrollTop - refs.thread.clientHeight < 80;
        state.userScrolledUp = !atBottom;
        refs.fab.classList.toggle("visible", state.userScrolledUp);
    });

    refs.fab.addEventListener("click", () => {
        refs.thread.scrollTop = refs.thread.scrollHeight;
    });
}
