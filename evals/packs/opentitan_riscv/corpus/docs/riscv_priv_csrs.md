# RISC-V Privileged ISA — Selected CSRs (mstatus, mtvec, mepc)

The RISC-V Privileged ISA defines a Control and Status Register (CSR)
address space of up to 4096 registers, accessed via the `csrrw`,
`csrrs`, `csrrc`, and their immediate-form siblings. This document
summarises three of the most load-bearing M-mode CSRs.

## `mstatus` — Machine Status Register

`mstatus` carries the global enable bits and bookkeeping fields that
gate interrupts and record the prior privilege mode across traps. Key
fields on RV32:

| Field | Bits | Meaning |
|---|---|---|
| `MIE`  | 3   | Global machine-interrupt enable |
| `MPIE` | 7   | Previous value of `MIE`, restored on `mret` |
| `MPP`  | 12:11 | Previous privilege mode (00=U, 01=S, 11=M) |
| `MPRV` | 17  | Modify memory privilege — loads/stores use `MPP` |
| `SUM`  | 18  | Supervisor user-memory access permit |
| `MXR`  | 19  | Make-executable-readable |
| `TVM`  | 20  | Trap virtual-memory operations |
| `TW`   | 21  | Timeout wait — trap on `wfi` from lower modes |
| `TSR`  | 22  | Trap on `sret` from S-mode |

On a trap into M-mode the hardware copies `MIE` into `MPIE`, clears
`MIE`, sets `MPP` to the prior mode, and clears `MIE`. The `mret`
instruction inverts this transformation.

## `mtvec` — Machine Trap Vector Base Address

`mtvec` holds the base address of the M-mode trap handler. Its low two
bits select the mode:

- `MODE = 0` (Direct): all traps jump to `BASE`.
- `MODE = 1` (Vectored): asynchronous interrupts jump to
  `BASE + 4 * cause`; synchronous exceptions still jump to `BASE`.

`BASE` must be 4-byte aligned and is shifted left by 2 in the CSR
encoding. Implementations that hard-wire `mtvec` (for example
ROM-resident handlers) are permitted.

## `mepc` — Machine Exception Program Counter

`mepc` holds the PC of the instruction that caused a trap (for
synchronous exceptions) or the address to which the hart will return
(for interrupts — the next instruction after the interrupted one).
On `mret`, execution resumes at `mepc`.

Writes to `mepc` are required to be implemented with at least
`IALIGN`-bit alignment: implementations with the C extension (16-bit
instructions) permit any even address; implementations without C
require 32-bit alignment and ignore writes that would set bit 1.

## Reading and writing CSRs

The `csrrw rd, csr, rs1` instruction atomically writes `rs1` into the
CSR and returns the previous value in `rd`. Software must use this
read-modify-write primitive when touching status fields to avoid
losing concurrent hardware updates.
