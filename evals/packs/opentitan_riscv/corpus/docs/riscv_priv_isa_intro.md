# RISC-V Privileged ISA — Privilege Levels Overview

The RISC-V Privileged Architecture defines a small set of privilege modes
that a hardware thread (hart) operates in at any moment. The base
specification lists three modes:

- **Machine mode (M-mode)** — the highest-privilege mode. M-mode is
  mandatory in every RISC-V implementation and is the mode entered on
  reset. It has unrestricted access to all CSRs, physical memory, and
  trap handling.
- **Supervisor mode (S-mode)** — optional. When implemented, S-mode is
  used by operating-system kernels that manage virtual memory through a
  paging MMU.
- **User mode (U-mode)** — optional. When implemented, U-mode is the
  mode in which ordinary application code runs.

Implementations advertise which combinations of modes they support. Three
common profiles are:

| Profile | Modes implemented | Typical use |
|---|---|---|
| M | M | Deeply embedded microcontrollers, e.g. Ibex `lowRISC` core |
| MU | M, U | Embedded with user/kernel separation |
| MSU | M, S, U | Application-class cores running Linux-style OSes |

## Mode transitions

Lower-privilege code requests services from higher-privilege code via
the synchronous `ecall` instruction. Traps (interrupts and exceptions)
are delivered to a privilege mode determined by per-mode delegation
CSRs (`medeleg`, `mideleg`); by default everything traps to M-mode.

On a trap, the hart saves the prior privilege mode in `mstatus.MPP`
(or `sstatus.SPP` for S-mode delegation), records the trap cause in
`mcause`/`scause`, the trapped PC in `mepc`/`sepc`, and the faulting
address (where relevant) in `mtval`/`stval`. Execution resumes in the
target privilege mode at the address stored in `mtvec`/`stvec`.

The `mret`/`sret` instructions return from a trap, restoring the
previous privilege mode from `MPP`/`SPP` and jumping to `mepc`/`sepc`.

## Hart state

Each privilege mode has its own bank of trap-handling CSRs but the
general-purpose integer registers `x0`–`x31` are shared. Software is
responsible for saving and restoring caller-saved registers across
trap boundaries; the privileged architecture provides only the
control-flow primitives.
