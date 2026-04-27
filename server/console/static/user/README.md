<!-- @summary
Served HTML shell and stylesheet for the user-facing chat console. The runtime bundle (`user-console.js`) is loaded from the parent `static/` directory; it is produced by esbuild from `../../web/src/user-console.ts` (a thin orchestrator) plus 20+ sibling feature modules under `../../web/src/`.
@end-summary -->

# server/console/static/user

Static assets for the user-facing RagWeave chat UI, served at `/console`. The page is a single-page application: `index.html` defines the DOM and `styles.css` supplies the visual design (themes, layout, components).

The runtime JavaScript is **not** in this directory. `index.html` loads `/console/static/user-console.js` from the sibling `static/` directory — that bundle is emitted by esbuild from `../../web/src/user-console.ts`. The `.ts` entry point is a ~50-line orchestrator: the actual logic lives in 20+ sibling `.ts` modules under `../../web/src/` (for example `streaming.ts`, `sidebar.ts`, `slash.ts`, `attachments.ts`, `citations.ts`, `conversations.ts`, `settings.ts`, `thread.ts`, `input.ts`, `markdown.ts`, `state.ts`, `dom.ts`, `api.ts`, `toast.ts`, `refs.ts`, `scrollFab.ts`, `contextWindow.ts`, `user-types.ts`, …). esbuild bundles all of them into a single ES module before serving.

`marked` and `dompurify` are not bundled — they are declared in the `<script type="importmap">` block at the bottom of `index.html` and loaded directly from a CDN by the browser.

## Contents

| Path | Purpose |
| --- | --- |
| `index.html` | SPA shell — sidebar navigation, message thread, input bar with attachment toolbar, settings panel overlay, slash-command dropdown, command picker, and the importmap + `<script type="module">` tag that boots `user-console.js`. |
| `styles.css` | Full UI stylesheet — CSS custom properties for theming (dark/light/system), sidebar layout, message bubbles, citation cards, context window indicator, and responsive breakpoints. |
