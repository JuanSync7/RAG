<!-- @summary
TypeScript build workspace for the console browser UIs. Compiles sources from `src/` into `../static/` using the TypeScript compiler directly (no bundler).
@end-summary -->

# server/console/web

Node.js workspace that compiles the console TypeScript sources. `tsc` reads `tsconfig.json` and emits ES-module JavaScript into `../static/`. No bundler is used — output files are loaded directly by the browser via native ES module imports.

## Contents

| Path | Purpose |
| --- | --- |
| `src/` | TypeScript source files for both the operator and user consoles |
| `package.json` | npm workspace with `build`, `check`, and `watch` scripts; devDependencies: `typescript`, `marked`, `dompurify` |
| `tsconfig.json` | Compiler config — ES2021 target, ES2022 modules, `Bundler` resolution, strict mode; `rootDir: src`, `outDir: ../static` |
## Building

```bash
# One-shot compile
cd server/console/web && npm run build

# Type-check without emitting
npm run check

# Incremental watch mode
npm run watch
```
