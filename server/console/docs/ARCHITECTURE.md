<!-- @summary
High-level architecture and dependency map for the RagWeave web console.
Covers the two surfaces (User Console + Admin/Operator Console), the shared
backend route/service layer, the esbuild bundling pipeline, the per-feature
TypeScript module graph, and known pending refactors.
@end-summary -->

# Console Architecture

## Elevator Pitch

The **RagWeave Console** is a browser UI surface served from the same FastAPI
process as the rest of the platform. It exposes two distinct front-ends from a
single Python feature package (`server/console/`):

- **User Console** (`/console`) — modern chat interface for end users:
  streaming retrieval-augmented answers, conversation history, slash commands,
  attachments, citations, settings.
- **Admin / Operator Console** (`/console/admin`) — tabbed debug/ops surface
  for operators: query inspection, ingest control, rerank diagnostics, timing
  breakdowns, API-key/quota management.

Both surfaces are TypeScript-authored, bundled by esbuild into ES modules, and
served as static assets by the same `/console/static/*` route. They share a
single backend route module (`routes.py`) and a single service helper module
(`services.py`).

---

## Loading Flow

End-to-end flow from URL hit to a fully-wired UI:

```
Browser GET /console
        |
        v
+--------------------------------+
| FastAPI router (routes.py)     |
|  GET /console        --------> serves static/user/index.html
|  GET /console/admin  --------> serves static/console.html
|  GET /console/static/{path}  -> serves static/<path>
+--------------------------------+
        |
        v  (HTML response)
+--------------------------------+
| Browser parses HTML            |
|  <script type="importmap">     |   maps "marked" + "dompurify"
|    marked    -> CDN ESM        |   to CDN ESM URLs (User Console)
|    dompurify -> CDN ESM        |
|  </script>                     |
|  <script type="module"         |   triggers ESM load of bundle
|    src="/console/static/       |
|         user-console.js">      |
+--------------------------------+
        |
        v
+--------------------------------+
| esbuild bundle loads as ESM    |
|  - imports resolved within     |
|    bundle (one file)           |
|  - "marked" + "dompurify"      |
|    kept external -> resolved   |
|    by browser via importmap    |
+--------------------------------+
        |
        v  (DOMContentLoaded)
+--------------------------------+
| Orchestrator init() runs       |
|  populateRefs()                |   bind DOM nodes once
|  initToast() / initSidebar()   |   wire feature modules
|  initSettings() ...            |
|  loadSettings()                |   restore persisted prefs
|  Promise.all([                 |   parallel initial fetches
|    loadCommands(),             |
|    loadConversations()])       |
+--------------------------------+
        |
        v
   UI is interactive
```

The Admin Console differs only in two places:

- It loads **`marked` + `dompurify` via classic `<script>` CDN tags** (not an
  importmap), because the operator console's bundled code references the
  globals rather than ESM imports.
- Its bundle entry is `static/main.js` (built from `src/main.ts`).

Build pipeline (`server/console/web/build.mjs`):

```
src/user-console.ts  --(esbuild, ESM, sourcemap)-->  static/user-console.js
src/main.ts          --(esbuild, ESM, sourcemap)-->  static/main.js

externals: ["marked", "dompurify"]   (kept as bare specifiers; resolved at
                                      runtime by importmap or CDN globals)
```

Run via `npm --prefix server/console/web run build` (or
`make console-build`). `npm run watch` runs esbuild in incremental mode.

---

## Two Entry Points

### User Console — `web/src/user-console.ts`

Post-modularization (PR 2/N), the User Console orchestrator is a thin file
that imports and wires every feature module.

```
user-console.ts (orchestrator, ~50 lines)
|
+-- refs.ts          (DOM references singleton)
+-- state.ts         (mutable cross-module state)
|
+-- toast.ts         <-- dom, refs
+-- citations.ts     <-- dom, user-types
+-- contextWindow.ts <-- dom, api, state, toast, user-types
+-- scrollFab.ts     <-- refs, state
+-- sidebar.ts       <-- dom, refs
+-- settings.ts      <-- dom, api, refs, toast, user-types
|
+-- conversations.ts <-- dom, api, markdown, refs, state, toast,
|                        thread, citations, user-types
+-- slash.ts         <-- dom, api, refs, state, thread,
|                        conversations, streaming, user-types
+-- attachments.ts   <-- dom, refs, state, toast
+-- input.ts         <-- dom, refs, state, streaming, slash, attachments
|
+-- streaming.ts     <-- dom, api, markdown, state, thread,
                         scrollFab, citations, contextWindow,
                         conversations, user-types

Leaves (no internal deps): dom.ts, api.ts, markdown.ts (marked + dompurify),
                           user-types.ts
```

Module dependency layering (top depends on bottom):

```
                 user-console.ts (orchestrator)
                          |
        +------+----------+----------+----------+
        |      |          |          |          |
     input  streaming  settings  conversations  ...
        |      |          |          |
        +------+----+-----+----+-----+
                    |          |
              thread, slash, attachments, contextWindow,
              scrollFab, citations, sidebar, toast
                          |
                  +-------+-------+
                  | refs   state  |
                  +-------+-------+
                          |
              dom    api    markdown    user-types
                          |
                  marked, dompurify (external CDN ESM)
```

### Admin / Operator Console — `web/src/main.ts`

```
main.ts (1022 lines, monolith — pending split)
  |
  +-- types.ts (shared envelope/query/conversation types)

  Internal sections (all in one file, no module split yet):
    - byId/escape DOM helpers
    - initTabs (tab switching)
    - authHeaders / fetch helpers
    - markdown render (uses marked + dompurify globals)
    - command cache + slash parsing
    - conversation list rendering
    - timing display
    - rerank table
    - bindQueryActions / bindIngestActions / bindAdminActions
    - DOMContentLoaded bootstrap
```

The Admin Console has not yet been modularized — see "Known pending refactors"
below. It is the next major refactor target after the User Console split.

---

## Module Roles Table

All paths relative to `server/console/web/src/`. Source-of-truth role text is
the `@summary` block at the top of each file.

### Shared

| File | Role | Key exports |
| --- | --- | --- |
| `types.ts` | Operator-console shared types: API envelope, conversation, query/stream, slash command, settings, health summary | `ConsoleEnvelope`, `MessageRole`, `ConversationMeta`, `QueryParams`, `QueryResult`, `StreamEventData`, `SlashCommand`, `UserSettings`, `HealthSummary` |
| `vendor.d.ts` | Ambient module declarations for `marked` + `dompurify`; fallback stubs before `npm install` | (ambient only) |

### User Console — orchestrator + leaves

| File | Role | Key exports |
| --- | --- | --- |
| `user-console.ts` | Orchestrator: resolves DOM refs, wires feature modules, kicks off initial loads | (side effects) |
| `user-types.ts` | User-console-specific types (kept separate from `types.ts` so each console can evolve independently) | `ThemeValue`, `SlashCommand`, `PresetConfig`, `ContextBreakdown`, `ChunkResult`, `ConversationMeta`, `SourceRef`, `sourceRefToChunkResult` |
| `dom.ts` | DOM helper utilities | `byId`, `escHtml`, `fmtTime`, `fmtRelative` |
| `api.ts` | Auth + JSON API layer; reads localStorage settings to construct authenticated fetch | `getSettings`, `authHeaders`, `apiBase`, `api` |
| `markdown.ts` | Markdown render pipeline; configures `marked` with custom code-block renderer + `DOMPurify` sanitization | `parseMarkdown`, `normalizeMarkdown` |

### User Console — feature modules

| File | Role | Key exports |
| --- | --- | --- |
| `refs.ts` | Lazily-populated singleton of DOM references; missing element produces one clear startup error | `populateRefs`, `refs`, `ConsoleRefs` |
| `state.ts` | Mutable singleton state shared across feature modules (replaces closure-scoped IIFE state) | `state`, `setActiveConversation`, `ConsoleState`, `AttachmentChip` |
| `toast.ts` | Toast notifications + clipboard-copy helpers | `initToast`, `showToast`, `copyMsg`, `copyBubble` |
| `sidebar.ts` | Sidebar nav: panel switching, collapse, mobile open/close, swipe-to-open | `initSidebar`, `openSidebar`, `closeSidebar` |
| `settings.ts` | Settings panel: theme + retrieval presets + persisted preferences | `initSettings`, `loadSettings` |
| `conversations.ts` | Conversation list, history loading, context menu, rename, delete | `initConversations`, `loadConversations`, `loadConversationHistory`, `selectConversation`, `createNewConversation`, `deleteConversation`, `updateConvTitle`, `renderConversationList` |
| `slash.ts` | Slash-command surface: dynamic load, autocomplete dropdown, full picker, dispatcher | `loadCommands`, `submitSlashCommand`, `handleSlashInput`, `executeCmd`, `executePicker`, `closeDropdown`, `closeCmdPicker`, `setSelected`, `setPickerSelected` |
| `attachments.ts` | Attachment toolbar (file/web/KB) + chip rendering + popover open/close coordination | `initAttachments`, `closeAllPopovers`, `setCloseCmdPicker` |
| `input.ts` | Textarea + send button + keyboard handling for slash dropdown and command picker | `initInput` |
| `streaming.ts` | Query execution: SSE streaming + non-stream fallback; canonical `sendQuery` entry | `sendQuery`, `streamQuery`, `nonStreamQuery` |
| `thread.ts` | Message-thread rendering: append user/system/error bubbles, build pending assistant bubble, toggle empty state | `setEmptyState`, `appendUserMsg`, `appendSystemMsg`, `appendErrorMsg`, `appendPendingAssistant` |
| `citations.ts` | Citation rendering + expand/collapse handlers, used by both streaming and history-replay paths | `initCitations`, `buildCitationsHtml`, `revealCitations` |
| `contextWindow.ts` | Context-window usage indicator (chip + tooltip + breakdown) and compact button | `initContextIndicator`, `updateContextIndicator` |
| `scrollFab.ts` | "Scroll to bottom" FAB + scroll-tracking for the message thread | `initScrollFab`, `scrollToBottom` |

### Admin / Operator Console

| File | Role | Key exports |
| --- | --- | --- |
| `main.ts` | Operator console — currently a monolith covering tabs, auth, fetch, conversation mgmt, markdown render, rerank table, timing display, slash commands, admin actions. Pending split. | (side effects) |

---

## Backend Integration

The frontend talks to a single FastAPI router built by `create_console_router`
in `server/console/routes.py` and exported via `server/console/__init__.py`.
All endpoints return the standard `ConsoleEnvelope` `{ok, request_id, data,
error}` shape (HTML/SSE endpoints excepted).

### Endpoint surface

| Method + path | Purpose | Frontend caller |
| --- | --- | --- |
| `GET /console` | Serve User Console HTML | (browser nav) |
| `GET /console/admin` | Serve Admin Console HTML | (browser nav) |
| `GET /console/static/{path}` | Serve admin static assets (incl. `main.js`) | `<script type="module">` |
| `GET /console/static/user/{path}` | Serve user static assets (incl. `user-console.js`, `styles.css`) | `<link>`, `<script>` |
| `GET /console/health` | Health probe (Ollama + log tail summary) | admin console health tab |
| `GET /console/logs?lines=N` | Tail recent server log lines | admin console logs tab |
| `GET /console/commands?mode=query|ingest` | Shared slash-command catalog | `slash.ts -> loadCommands()` |
| `POST /console/command` | Slash-command dispatch (returns normalized `{intent, action, data, message}`) | `slash.ts -> submitSlashCommand()` |
| `POST /console/query` | One-shot RAG query (non-stream) | `streaming.ts -> nonStreamQuery()` |
| `GET /console/source-document` | Source preview JSON payload | `citations.ts` |
| `GET /console/source-document/view` | Source preview rendered as HTML page | citation links |
| `GET /console/conversations` | List conversations for current principal | `conversations.ts -> loadConversations()` |
| `POST /console/conversations/new` | Create / ensure conversation | `conversations.ts -> createNewConversation()` |
| `GET /console/conversations/{id}/history` | Load turn history | `conversations.ts -> loadConversationHistory()` |
| `POST /console/conversations/{id}/compact` | Force compact a conversation | `conversations.ts` (compact button) |
| `DELETE /console/conversations/{id}` | Delete conversation | `conversations.ts -> deleteConversation()` |
| `POST /console/ingest` | Trigger ingest (admin) | `main.ts` ingest tab |
| `GET /console/admin/api-keys` | List API keys | `main.ts` admin tab |
| `POST /console/admin/api-keys` | Create API key | `main.ts` admin tab |
| `GET /console/admin/quotas` | Quota inspection | `main.ts` admin tab |

Streaming queries do **not** go through `/console/query`. The streaming path
calls `POST <api_endpoint>/query/stream` directly (server-sent events), where
`<api_endpoint>` is the configurable backend base URL stored in localStorage
("nc_settings.api_endpoint") — see `streaming.ts -> streamQuery()` and
`api.ts -> apiBase()`. The console router still owns the non-stream
`POST /console/query` path, which reuses the same backend service layer.

### `services.py` — shared backend helpers

Pure helpers consumed by `routes.py`:

- `resolve_console_html_path()` / `resolve_user_console_html_path()` — locate
  the HTML entry files (with backward-compat fallback for legacy checkouts).
- `resolve_console_static_asset()` / `resolve_user_console_static_asset()` —
  validate + resolve a static asset path (path-traversal guard).
- `is_ollama_reachable()` — health probe used by `/console/health`.
- `tail_log_lines(lines)` — read the last N log lines for `/console/logs`.
- `resolve_console_source_path()` / `build_source_preview_payload()` /
  `render_source_document_html()` — back the source-document preview
  endpoints.

Route handlers stay thin; business logic lives either in `services.py` or in
deeper platform modules (memory provider, command catalog, RAG chain).

### Auth + envelope

Every non-static endpoint depends on `authenticate_request`, which produces a
`Principal` (tenant + subject + project + roles). The frontend reads
`auth_token` from `localStorage` ("nc_settings") and sends it as both
`Authorization: Bearer …` and `x-api-key: …` (see `api.ts -> authHeaders()`).
This dual-header strategy lets the same UI work against either header style.

---

## Extension Points

### Add a new tab in the Admin Console

1. **HTML** — add a `<button>` to the tab bar in `static/console.html` and a
   matching `<section>` panel.
2. **TS** — extend the `tabMap` in `web/src/main.ts` to map the new tab id to
   its panel id; `initTabs()` will pick it up automatically.
3. **Wire actions** — bind your handlers in `main.ts` (similar to
   `bindQueryActions`, `bindIngestActions`, `bindAdminActions`).
4. **Backend** — add the corresponding `@router.<verb>(...)` handler in
   `server/console/routes.py`. Use `ConsoleEnvelope`, `console_ok`, and
   `standard_error_responses` to stay consistent.
5. **Build** — run `make console-build`. The bundle is rewritten into
   `static/main.js`.

> Note: once the admin split lands (see pending refactors), step 2-3 will
> instead mean creating a new `web/src/admin/<tab>.ts` feature module and
> registering it from a thin `main.ts` orchestrator.

### Add a new ingest mode

1. **Catalog** — add the mode metadata to `src/platform/command_catalog.py`
   (the shared CLI/UI source of truth). The new entry will surface
   automatically via `GET /console/commands?mode=ingest`.
2. **Backend dispatch** — handle the mode in the `/console/ingest` handler
   (`routes.py`) or in the underlying ingest service it calls.
3. **CLI parity** — verify the same mode works through `cli.py` /
   `server/cli_client.py` — the catalog is shared. The CLI/UI Parity Rule
   (see `server/console/README.md`) is non-optional.
4. **Frontend** — typically zero changes; the slash-command UI and admin
   ingest tab pull modes from `loadCommands()`.

### Add a new slash command

1. **Catalog** — register the command in `src/platform/command_catalog.py`
   with name, description, args hint, and category (`query` or `ingest`).
2. **Backend dispatch** — extend the `POST /console/command` handler in
   `routes.py` to translate the new command into an intent/action payload
   `{intent, action, data, message}`. The frontend treats this payload
   generically.
3. **Frontend** — usually nothing to do. `slash.ts -> submitSlashCommand()`
   POSTs to `/console/command` and renders the response. If a new intent
   needs custom UI rendering, branch on `intent` in `submitSlashCommand`.
4. **CLI parity** — confirm the command also runs via terminal `cli.py` and
   `server/cli_client.py`. Both pull from the same catalog.

---

## Known Pending Refactors

These are in flight as part of the `frontend/console-modularize-pr*` batch.
Cross-references reflect parallel PRs from this same batch.

1. **Admin Console split (`main.ts` -> per-feature modules).**
   `web/src/main.ts` is still ~1000 lines and owns tabs, auth, fetch,
   conversation mgmt, markdown render, rerank table, timing, slash, and
   admin actions. The plan mirrors the User Console split (PR 2/N): pull
   each cohesive concern (`tabs.ts`, `auth.ts`, `query.ts`, `ingest.ts`,
   `rerank.ts`, `timing.ts`, `admin.ts`, …) into its own module and reduce
   `main.ts` to an orchestrator.

2. **`web/src/` subdirectory restructure.**
   The flat `src/` layout is fine at the current module count but will
   become hard to scan once the admin split adds another ~10 modules. The
   intended layout groups by surface plus a shared layer:
   ```
   web/src/
     shared/      (dom, api, markdown, types, vendor.d.ts)
     user/        (User Console feature modules + user-console.ts)
     admin/       (Admin Console feature modules + main.ts)
   ```
   `build.mjs` entry points update accordingly.

3. **`console.html` CSS extraction.**
   The Admin Console HTML still embeds ~400 lines of `<style>` inline.
   These should move to `static/console.css` (matching the User Console's
   `static/user/styles.css` pattern) so the HTML file becomes a thin shell
   and the CSS becomes diff-able / cacheable independently.

4. **Util consolidation (`dom.ts`, fetch helpers, markdown).**
   Both consoles have their own copy of `byId`, `escHtml`, `authHeaders`,
   and a markdown-render configuration. After the admin split, these
   should consolidate into `web/src/shared/` so both surfaces import the
   same primitives. Keep aliases in feature modules during the migration
   to preserve a stable internal import surface.

5. **`web/README.md` is stale.**
   It still describes a `tsc`-only build with no bundler. After PR 1/N
   introduced esbuild, that doc should be patched to describe the bundler
   pipeline (entry points, externals, sourcemaps, watch mode).

---

## See Also

- `server/console/README.md` — package overview, URL table, CLI/UI parity rule.
- `server/console/web/README.md` — build commands.
- `server/console/web/src/README.md` — per-file purpose summary.
- `src/platform/command_catalog.py` — shared CLI/UI command catalog.
