import { defineConfig } from "vitest/config";

// Default to the lightweight node environment. Test files that need a DOM
// (e.g. localStorage, document, window) opt in per-file with the pragma:
//   // @vitest-environment jsdom
// Module resolution mirrors tsconfig.json (Bundler resolution, ESM), so the
// same relative "./dom" style imports the source uses resolve here too.
export default defineConfig({
    test: {
        environment: "node",
        include: ["src/**/*.test.ts", "src/__tests__/**/*.test.ts"],
    },
});
