// @summary
// esbuild driver for the user + operator consoles.
// Bundles each entry point into ../static/ as native ES modules. Bare specifiers
// for `marked` and `dompurify` are kept external because the browser resolves
// them via the <script type="importmap"> declarations in the served HTML pages.
// @end-summary

import { build, context } from "esbuild";
import { resolve } from "node:path";

const watch = process.argv.includes("--watch");
const root = resolve(import.meta.dirname);

const common = {
    bundle: true,
    format: "esm",
    target: "es2021",
    sourcemap: true,
    logLevel: "info",
    external: ["marked", "dompurify"],
    absWorkingDir: root,
};

const entries = [
    { entryPoints: ["src/user-console.ts"], outfile: "../static/user-console.js" },
    { entryPoints: ["src/main.ts"], outfile: "../static/main.js" },
];

if (watch) {
    const ctxs = await Promise.all(entries.map((e) => context({ ...common, ...e })));
    await Promise.all(ctxs.map((c) => c.watch()));
} else {
    await Promise.all(entries.map((e) => build({ ...common, ...e })));
}
