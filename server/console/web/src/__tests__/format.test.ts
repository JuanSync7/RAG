import { describe, it, expect } from "vitest";
import { fmtSize } from "../format";

// fmtSize humanizes a byte count into B/KB/MB/GB using base-1024 units.
describe("fmtSize", () => {
    it("returns '0 B' for zero", () => {
        expect(fmtSize(0)).toBe("0 B");
    });

    it("returns '0 B' for negative input", () => {
        expect(fmtSize(-5)).toBe("0 B");
    });

    it("returns '0 B' for NaN", () => {
        expect(fmtSize(NaN)).toBe("0 B");
    });

    it("returns '0 B' for Infinity", () => {
        expect(fmtSize(Infinity)).toBe("0 B");
    });

    it("renders raw bytes below 1024 with a 'B' suffix", () => {
        expect(fmtSize(512)).toBe("512 B");
    });

    it("renders 1023 bytes as raw bytes (boundary below 1 KB)", () => {
        expect(fmtSize(1023)).toBe("1023 B");
    });

    it("renders exactly 1024 bytes as '1.0 KB' (KB boundary)", () => {
        expect(fmtSize(1024)).toBe("1.0 KB");
    });

    it("renders 1536 bytes as '1.5 KB' with one decimal", () => {
        expect(fmtSize(1536)).toBe("1.5 KB");
    });

    it("renders exactly 1 MB as '1.0 MB' (MB boundary)", () => {
        expect(fmtSize(1024 * 1024)).toBe("1.0 MB");
    });

    it("renders 1.5 MB as '1.5 MB'", () => {
        expect(fmtSize(1024 * 1024 * 1.5)).toBe("1.5 MB");
    });

    it("renders exactly 1 GB as '1.0 GB' (GB boundary)", () => {
        expect(fmtSize(1024 * 1024 * 1024)).toBe("1.0 GB");
    });

    it("renders multi-GB values in GB", () => {
        expect(fmtSize(1024 * 1024 * 1024 * 3.25)).toBe("3.3 GB");
    });
});
