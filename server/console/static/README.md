<!-- @summary
Browser-ready assets for the dual web console: HTML entry points plus the bundled JavaScript outputs produced by esbuild from `../web/src/`. The admin console (`console.html`) loads `main.js`; the user console (`user/index.html`) loads `user-console.js`.
@end-summary -->

# server/console/static

Static assets served by the console server. The TypeScript sources live in `../web/src/` and are bundled into this directory by esbuild — see `../web/build.mjs`. Each entry point is bundled (with all its sibling modules inlined) into a single ES module per surface; the per-feature `.ts` files under `../web/src/` are not emitted as individual `.js` files here.

## Bundle wiring

| HTML | Bundle | Surface |
| --- | --- | --- |
| `console.html` | `main.js` | Admin / operator console (tabbed debug & ops UI) |
| `user/index.html` | `user-console.js` | User chat console |

Both bundles are emitted as native ES modules (`format: "esm"`, `target: "es2021"`). `marked` and `dompurify` are marked external in `build.mjs` and provided by the page itself: `user/index.html` declares them in a `<script type="importmap">`, and `console.html` loads them as plain `<script src="…cdn…">` tags before the module bundle.

## Contents

| Path | Purpose |
| --- | --- |
| `console.html` | Admin console HTML entry point. Loads `main.js` as `<script type="module">`. |
| `user/` | User console HTML shell + stylesheet (see `user/README.md`). |
| `main.js` | esbuild bundle of `../web/src/main.ts` and its imports — admin console runtime. |
| `user-console.js` | esbuild bundle of `../web/src/user-console.ts` and its 20+ sibling feature modules — user console runtime. |
| `user-console.js.map` | Sourcemap for `user-console.js` (emitted by esbuild via `sourcemap: true`). A matching `main.js.map` is produced on every build. |
| `types.js` | Empty/legacy artifact left over from the previous `tsc`-based per-file emit. Not loaded by either HTML page; can be removed once nothing references it. |

## Rebuilding

From the repo root:

```bash
npm --prefix server/console/web install
npm --prefix server/console/web run build      # one-shot bundle via build.mjs
npm --prefix server/console/web run watch      # rebuild on change
```

Or via the make targets: `make console-install && make console-build`.
