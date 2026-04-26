<!-- @summary
Web console module for dual-console UX: User Console at /console, Admin Console at /console/admin.
Routes, service helpers, and UI asset handling for both surfaces.
@end-summary -->

# server/console

## Overview

This package owns the dual web console backend:

- `routes.py`: all `/console/*` FastAPI endpoints.
- `services.py`: shared console helpers (health probe, logs tail, source rendering, static UI path resolution).
- `static/user/index.html`: User Console (modern chat interface) served at `/console`.
- `static/console.html`: Admin Console (tabbed debug/ops interface) served at `/console/admin`.
- `web/src/user-console.ts`: TypeScript source for the User Console.
- `web/src/main.ts`: TypeScript source for the Admin Console.
- `web/build.mjs` + `web/package.json` + `web/tsconfig.json`: esbuild bundler driver and TypeScript config.

## Console URLs

| URL | Interface | Purpose |
|-----|-----------|---------|
| `http://localhost:8000/console` | **User Console** | Modern chat interface for end users |
| `http://localhost:8000/console/admin` | **Admin Console** | Tabbed debug/ops interface for operators |

## Build TypeScript UI

```bash
npm --prefix server/console/web install
npm --prefix server/console/web run build
```

The build emits two ES-module bundles into `server/console/static/`:

- `main.js` — Admin Console (entry: `web/src/main.ts`)
- `user-console.js` — User Console (entry: `web/src/user-console.ts`)

Both are served via `GET /console/static/{asset_path}` (mounted in `routes.py`).

Equivalent root shortcuts:

```bash
make console-install && make console-build
# or
npm run console:install && npm run console:build
```

## CLI/UI Parity Rule

Console UI changes should track the same shared product contract as `cli.py` and
`server/cli_client.py`. New user-facing options should be added through shared
schemas/metadata first, then surfaced in both CLI and UI adapters.

The shared slash-command contract now lives in `src/platform/command_catalog.py`
and is served to the web UI through `GET /console/commands?mode=query|ingest`.
This keeps `/` command names/descriptions consistent across:

- terminal `cli.py`,
- terminal `server/cli_client.py`,
- web console slash-command helper inputs.

Command intent dispatch for web console now uses:

- `POST /console/command` with `{mode, command, arg, state}`
- response includes normalized `{intent, action, data, message}` payload

This allows the frontend to remain a renderer/adapter while command semantics
stay centralized in backend logic.

## Conversation UX

The Query tab includes a left chat pane with:

- conversation list (`GET /console/conversations`),
- create new chat (`POST /console/conversations/new`),
- turn history (`GET /console/conversations/{id}/history`),
- manual compact (`POST /console/conversations/{id}/compact`).

Query requests pass `conversation_id` and `memory_enabled` so the backend can
apply tenant-persistent memory and return `conversation_id` on each response.

## Architecture Overview

The console exposes two HTML surfaces, each backed by its own TypeScript bundle:

| Surface | HTML | Bundle | TS entry |
| --- | --- | --- | --- |
| Admin / operator | `static/console.html` | `static/main.js` | `web/src/main.ts` |
| End user | `static/user/index.html` | `static/user-console.js` | `web/src/user-console.ts` |

Both bundles are produced by the esbuild driver at `web/build.mjs`, which emits
native ES modules (`format: "esm"`, `target: "es2021"`) with sourcemaps directly
into `static/`. The TypeScript migration is complete; `tsc` is retained only for
type-checking (`npm run check`).

Two libraries — `marked` (Markdown rendering) and `dompurify` (HTML sanitization)
— are kept **external** to the bundles. The browser resolves their bare specifiers
at runtime:

- `static/user/index.html` declares an `<script type="importmap">` that maps
  `marked` and `dompurify` to jsdelivr CDN ESM builds.
- `static/console.html` loads them via classic `<script src="...">` tags from
  jsdelivr before the module entry point.

This keeps bundle size small and avoids vendoring third-party Markdown code.

FastAPI wiring lives in `routes.py`:

- `GET /console` → serves `static/user/index.html`
- `GET /console/admin` → serves `static/console.html`
- `GET /console/static/{asset_path}` → serves any compiled bundle / asset under `static/`

See `web/README.md` for the build workspace and `static/README.md` for the
emitted asset layout.
