<!-- @summary
TypeScript source for the operator console (`main.ts`) and the user console (modular). User-console code is split into entry, shared utilities, and feature modules; bundled by esbuild and emitted into `../static/`.
@end-summary -->

# server/console/web/src

Source TypeScript for both browser console surfaces. The operator console is a single self-initializing file (`main.ts`); the user console (PR 2/N) is a thin orchestrator (`user-console.ts`) plus per-feature modules sharing a singleton DOM-refs/state pair. Bundling is handled by esbuild (PR 1/N); see `../package.json` for the build script.

## Entry points

| File | Purpose | Key exports |
| --- | --- | --- |
| `main.ts` | Operator console — tabs, auth headers, fetch helpers, conversation management, markdown rendering, rerank table, timing display, slash commands, admin actions. | (side effects only) |
| `user-console.ts` | User-console orchestrator — calls `populateRefs()`, then each feature's `init()`, then kicks off initial data loads. | (side effects only) |

## Shared utilities (user console)

| File | Purpose | Key exports |
| --- | --- | --- |
| `api.ts` | Auth + JSON API layer; reads saved settings from localStorage to construct authenticated fetch calls. | `getSettings`, `authHeaders`, `apiBase`, `api` |
| `dom.ts` | Tiny DOM helpers shared across feature modules. | `byId`, `escHtml`, `fmtTime`, `fmtRelative` |
| `state.ts` | Mutable singleton state; everything previously closure-scoped in the IIFE so feature modules read/write without prop-drilling. | `AttachmentChip`, `ConsoleState`, `state`, `setActiveConversation` |
| `refs.ts` | Lazily-populated singleton of DOM references; one `populateRefs()` call at startup means missing elements fail loudly in one place. | `ConsoleRefs`, `refs`, `populateRefs` |
| `toast.ts` | Toast notifications + clipboard-copy helpers (bubbles + code blocks). Installs `copyMsg`/`copyBubble`/`showToast` on `window` for legacy inline `onclick` handlers. | `showToast`, `copyMsg`, `copyBubble`, `initToast` |
| `markdown.ts` | Markdown render pipeline: `marked` with a custom code-block renderer (copy-button UI), inline-list normalization, and DOMPurify sanitization. | `parseMarkdown`, `normalizeMarkdown` |

## User-console features

Each feature module owns its DOM event wiring; the orchestrator only needs to call its `init*()` once.

| File | Purpose | Key exports |
| --- | --- | --- |
| `conversations.ts` | List, history loading, context menu, rename modal, delete; also owns `updateConvTitle` and `createNewConversation` for cross-feature use. | `renderConversationList`, `loadConversations`, `selectConversation`, `loadConversationHistory`, `createNewConversation`, `deleteConversation`, `updateConvTitle`, `initConversations` |
| `streaming.ts` | Query execution: SSE streaming + non-stream fallback. `sendQuery` is the canonical entry point for plain-text user messages; slash flows reuse `streamQuery`/`nonStreamQuery`. | `streamQuery`, `nonStreamQuery`, `sendQuery` |
| `slash.ts` | Slash-command surface: dynamic command load, autocomplete dropdown, full command picker, and the dispatcher invoked when a leading-slash message is sent. | `loadCommands`, `closeDropdown`, `setSelected`, `executeCmd`, `handleSlashInput`, `setPickerSelected`, `executePicker`, `closeCmdPicker`, `submitSlashCommand` |
| `sidebar.ts` | Sidebar navigation: panel switching, collapse/expand, mobile open/close, touch swipe-to-open. | `openSidebar`, `closeSidebar`, `initSidebar` |
| `settings.ts` | Settings panel: theme + retrieval presets + persisted preferences. `loadSettings` is exported so the orchestrator can apply saved values at startup. | `loadSettings`, `initSettings` |
| `input.ts` | Textarea input + send button + keyboard handling for the slash dropdown and command picker. | `initInput` |
| `attachments.ts` | Attachment toolbar: file/web/KB chips + popovers, plus open/close coordination shared with the command picker. | `closeAllPopovers`, `setCloseCmdPicker`, `initAttachments` |
| `citations.ts` | Citation card rendering + expand/collapse handlers. `buildCitationsHtml` is consumed by streaming, non-stream, and history-replay paths. | `buildCitationsHtml`, `revealCitations`, `initCitations` |
| `contextWindow.ts` | Context-window usage indicator (chip + tooltip + breakdown) and compact button. | `updateContextIndicator`, `initContextIndicator` |
| `thread.ts` | Message-thread rendering: append user/system/error bubbles, build the pending assistant bubble used by both query paths, toggle empty-state placeholder. | `setEmptyState`, `appendUserMsg`, `PendingAssistantHandles`, `appendPendingAssistant`, `appendSystemMsg`, `appendErrorMsg` |
| `scrollFab.ts` | "Scroll to bottom" floating action button + scroll-tracking. `scrollToBottom` is the shared helper used by `thread` and `streaming`. | `scrollToBottom`, `initScrollFab` |

## Ingest

No dedicated ingest modules live in `web/src/` yet. The operator console drives document ingestion inline from `main.ts` (`bindIngestionActions`, ingest-mode slash commands); the user console exposes attachments via `attachments.ts`. A future PR may extract `ingest.ts` once the surface area justifies it.

## Types

| File | Purpose | Key exports |
| --- | --- | --- |
| `types.ts` | Operator-console shared type definitions. | `ConsoleEnvelope`, `MessageRole`, `CitationSource`, `ChatMessage`, `ConversationMeta`, `ConversationTurn`, `QueryParams`, `QueryResult`, `StreamEventData`, `ContextBreakdown`, `SlashCommand`, `CommandResult`, `ContextAttachment`, `UserSettings`, `HealthSummary`, `SidebarNavItem` |
| `user-types.ts` | User-console-specific type definitions; deliberately separate from `types.ts` so each console can evolve its own contract. | `ThemeValue`, `SlashCommand`, `PresetConfig`, `ContextBreakdown`, `ConversationMeta`, `ChunkResult`, `SourceRef`, `TokenBudget`, `StreamEventData`, `CommandResult`, `sourceRefToChunkResult` |

## Vendor

| File | Purpose |
| --- | --- |
| `vendor.d.ts` | Ambient module declarations for `marked` and `dompurify` — fallback stubs used before `npm install`; superseded by the real package types once `node_modules/` exists. |

## Pending refactors

- **Admin-console split.** `main.ts` is still a single large file; planned to split along the same per-feature pattern as the user console (tabs, rerank table, timing display, admin actions, ingest).
- **Directory restructure.** Once both consoles are modularized, source will likely move into `user/` and `admin/` subdirectories with `shared/` for the cross-console utilities. Feature modules currently coexist flat for diff readability during the migration.
- **CSS extraction.** Component styles still live in static HTML/CSS files under `../static/`; they should follow modules into per-feature stylesheets so a feature lives in one place.

See [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) (forthcoming) for the target end-state and migration timeline.
