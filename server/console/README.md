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

## Turn-Loop Activity Log & Clarify Chips

Both consoles render the turn-level agentic conversation loop's stream
observability (see `docs/retrieval/TURN_LOOP_DESIGN.md` §8) through one shared
renderer, `web/src/activityLog.ts`:

- **Request flag.** The user console's `buildQueryBody` (`web/src/streaming.ts`)
  sends `turn_loop: true` only while the "Turn loop" toolbar toggle
  (`#chatTurnLoop`, owned by `web/src/chatMode.ts`, persisted in
  `localStorage["rw_turn_loop"]`) is active — absent means the server config
  default (`RAG_TURN_LOOP_ENABLED`) applies. The toggle is mutually exclusive
  with the Deep research toggle, matching the request-schema validator
  (design §5).
- **Activity log.** A lazy collapsible `<details class="activity-log">` block
  is inserted into the assistant bubble wrap **before** the reasoning block on
  the first turn-loop SSE event, and auto-collapses when the first answer
  token arrives (expandable afterwards). It renders the nine typed events —
  `turn_action`, `hyde_query` (the hypothetical answer verbatim, collapsed
  when long), `retrieve_result`, `judge_verdict`, `deep_study`, `llm_call`
  (dim telemetry one-liner), `draft` (live draft text, replaced per attempt),
  `gate`, `clarify` — as compact lines.
- **Clarify chips.** A terminal `clarify` event renders the question plus one
  clickable chip per `hints[]` / `scoping_questions[]` entry OUTSIDE the
  collapsible log; clicking a chip resubmits the chip text as the next user
  query through the shared resubmit sink (`registerQueryResubmit` /
  `resubmitQuery` in `chatMode.ts` — the same mechanism as the deep-research
  suggestion chip). The admin console resubmits by refilling the query box and
  re-running the stream.
- **XSS discipline.** Every event-derived string (LLM/document content) is
  written via `textContent`, never `innerHTML`.
- **Contracts.** The nine payload shapes live in `web/src/shared-types.ts`
  (re-exported by `user-types.ts` / `admin-types.ts`), mirroring
  `TurnEventType` in `src/retrieval/pipeline/turn_loop/schemas.py`; SSE
  `event:` names must match 1:1.

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
