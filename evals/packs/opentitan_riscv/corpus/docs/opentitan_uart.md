# OpenTitan UART IP — Register Map Summary

The OpenTitan UART (`hw/ip/uart`) is a configurable asynchronous serial
transmitter and receiver intended for boot-console, debug, and
low-bandwidth host communication. Each UART instance is a peripheral
on the TileLink-UL device bus and exposes a register window plus
single-line TX and RX pins.

## Feature summary

- Programmable baud rate via a 16-bit clock divider (`CTRL.NCO` /
  parity- and frame-format fields).
- 8-N-1 framing with optional even/odd parity selected by
  `CTRL.PARITY_EN` and `CTRL.PARITY_ODD`.
- Independent 32-byte TX and RX FIFOs with programmable watermark
  interrupts.
- Loopback and line-break generation for self-test.
- Interrupt sources for TX/RX watermark, RX timeout, RX parity error,
  RX frame error, RX overflow, and TX empty.

## Selected registers

| Offset | Name | Purpose |
|---|---|---|
| `0x00` | `INTR_STATE` | Interrupt status (read/write-1-to-clear) |
| `0x04` | `INTR_ENABLE` | Per-source interrupt enable mask |
| `0x08` | `INTR_TEST` | Software-driven interrupt assertion |
| `0x0c` | `ALERT_TEST` | Software-driven alert assertion |
| `0x10` | `CTRL` | Master enable, parity, line-loopback, NCO |
| `0x14` | `STATUS` | TX/RX FIFO full/empty/idle flags |
| `0x18` | `RDATA` | Read pop from the RX FIFO |
| `0x1c` | `WDATA` | Write push to the TX FIFO |
| `0x20` | `FIFO_CTRL` | Reset and watermark configuration |
| `0x24` | `FIFO_STATUS` | Live TX/RX FIFO depth |
| `0x28` | `OVRD` | Direct pin override for hardware bring-up |
| `0x2c` | `VAL`  | Direct pin sample for hardware bring-up |
| `0x30` | `TIMEOUT_CTRL` | RX-idle timeout configuration |

## CTRL register fields

`CTRL` is the primary configuration register. Notable fields:

- `TX` (bit 0) — enable transmitter
- `RX` (bit 1) — enable receiver
- `NF` (bit 2) — noise-filter enable on RX
- `SLPBK` (bit 4) — system-level loopback (TX → RX)
- `LLPBK` (bit 5) — line-level loopback
- `PARITY_EN` (bit 6) — enable parity bit generation/check
- `PARITY_ODD` (bit 7) — 0 = even, 1 = odd
- `RXBLVL` (bits 9:8) — RX break detection level (in bit times)
- `NCO` (bits 31:16) — 16-bit numerically-controlled oscillator value
  that sets the baud rate.

## Software bring-up sequence

1. Compute `NCO = (16 * baud * 2^16) / fclk` and program `CTRL`.
2. Set FIFO watermarks via `FIFO_CTRL`.
3. Enable desired interrupt sources via `INTR_ENABLE`.
4. Set `CTRL.TX` and `CTRL.RX` to start operation.
