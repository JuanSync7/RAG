# Ibex Core — Pipeline and ISA Support Summary

Ibex is a small, 2-stage, in-order RISC-V CPU core developed by
lowRISC and used as the secure boot/main processor in OpenTitan. It
is implemented in SystemVerilog and is configurable at elaboration
time for a range of area/performance trade-offs.

## ISA support

Ibex implements the RV32 base integer ISA with optional extensions:

| Extension | Meaning | Notes |
|---|---|---|
| `I`  | RV32I base integer ISA | Always present |
| `E`  | RV32E reduced register file (16 regs) | Configurable alternative to `I` |
| `M`  | Integer multiply/divide | Optional; single- or multi-cycle |
| `C`  | Compressed 16-bit instructions | Optional; reduces code size |
| `B`  | Bit-manipulation (Zba/Zbb/Zbc/Zbs subsets) | Optional |
| `Zicsr` | CSR instructions | Always present in privileged builds |
| `Zifencei` | Instruction-fence | Always present |

The `RV32E` configuration reduces the integer register file from
32 entries to 16, halving register-file area at the cost of ABI
incompatibility with general-purpose RV32I toolchains.

## Pipeline

Ibex uses a 2-stage in-order pipeline:

1. **IF (Instruction Fetch)** — fetches 32 bits per cycle from the
   instruction bus, with a prefetch buffer that absorbs the longer
   latency of memory-mapped flash. Compressed instructions are
   expanded to their 32-bit equivalents before issue.
2. **ID/EX (Decode and Execute)** — decodes the instruction, reads
   the register file, executes the ALU operation, performs the
   load/store address calculation, and writes back to the register
   file. Multi-cycle operations (multiply, divide, CSR access,
   load/store with wait states) stall this stage.

Branches and jumps incur a single-cycle bubble on a taken transfer.

## Privilege and security

Ibex supports M-mode and optionally U-mode (`PMP`-enabled builds only).
A Physical Memory Protection (PMP) unit with up to 16 regions can
enforce per-region read/write/execute permissions, satisfying the
RISC-V Privileged ISA's PMP specification.

The Secure Ibex configuration adds:

- Dual-core lockstep with output comparators
- ECC on register file and instruction-cache RAMs
- Bus integrity checks
- Hardened branch predictor and control-flow integrity hooks
- Data-independent timing (DIT) mode for selected instructions

These features make Ibex suitable as the main processor in
security-sensitive SoCs such as OpenTitan, where it boots, verifies
firmware, and orchestrates the cryptographic accelerators.

## Memory interfaces

Ibex exposes separate instruction and data memory interfaces, each
using a simple valid/ready handshake. The OpenTitan platform wraps
these in TileLink-UL bridges for integration with the wider SoC
fabric.
