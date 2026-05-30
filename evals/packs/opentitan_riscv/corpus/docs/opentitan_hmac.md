# OpenTitan HMAC IP — Block Diagram and Key Registers

The OpenTitan HMAC (`hw/ip/hmac`) is a hardware accelerator implementing
the SHA-2 family of hash functions and the keyed HMAC construction
defined by FIPS 198-1. It is exposed to software as a TileLink-UL
peripheral with a memory-mapped data-in window backed by a streaming
FIFO.

## Supported algorithms

- SHA-256 — 256-bit digest, 512-bit block
- SHA-384 — 384-bit digest, 1024-bit block
- SHA-512 — 512-bit digest, 1024-bit block
- HMAC variants of each of the above, with key lengths up to
  1024 bits (longer keys are pre-hashed per RFC 2104).

## Top-level block diagram

```
TL-UL bus ──▶ register file ──▶ FIFO ──▶ message scheduler ──▶ core
                                                                │
                                  ┌─────────────────────────────┘
                                  ▼
                       digest registers (DIGEST_0..15)
                                  │
                                  ▼
                          interrupt + alert logic
```

The register file holds configuration and key material; software pushes
data into the streaming FIFO via the `MSG_FIFO` aperture. The message
scheduler converts the FIFO output into the algorithm's block size and
feeds the compression core. Completed digests appear in the
`DIGEST_0..15` registers.

## Selected registers

| Offset | Name | Purpose |
|---|---|---|
| `0x00` | `INTR_STATE` | Interrupt status (W1C) |
| `0x04` | `INTR_ENABLE` | Per-source interrupt enable |
| `0x10` | `CFG` | Algorithm select, endianness, HMAC enable, SHA enable |
| `0x14` | `CMD` | `hash_start`, `hash_process`, `hash_continue`, `hash_stop` |
| `0x18` | `STATUS` | FIFO empty/full, HMAC/SHA idle |
| `0x1c` | `ERR_CODE` | Last error code |
| `0x20` | `WIPE_SECRET` | Securely wipes key material |
| `0x24` | `KEY_0..31` | 1024-bit key, written before `hash_start` |
| `0x6c` | `DIGEST_0..15` | 512-bit digest, valid after `hash_done` |
| `0xac` | `MSG_LENGTH_LOWER` | Low 32 bits of total message length |
| `0xb0` | `MSG_LENGTH_UPPER` | Upper 32 bits of total message length |
| `0x1000` | `MSG_FIFO` | Write-only data-in aperture |

## CFG register fields

- `HMAC_EN` (bit 0) — enable HMAC mode; if clear, the block performs
  plain SHA-2.
- `SHA_EN` (bit 1) — global enable for the SHA-2 core.
- `ENDIAN_SWAP` (bit 2) — swap byte order on register reads/writes.
- `DIGEST_SWAP` (bit 3) — swap byte order of digest output.
- `KEY_SWAP` (bit 4) — swap byte order of key writes.
- `DIGEST_SIZE` (bits 7:4) — selects SHA-256/384/512.
- `KEY_LENGTH` (bits 13:8) — selects key length for HMAC mode.

## Software flow

1. Configure `CFG` for the desired algorithm and HMAC mode.
2. (HMAC only) Write the key into `KEY_0..31`.
3. Issue `CMD.hash_start`.
4. Stream the message into `MSG_FIFO`. Poll or take an interrupt on
   FIFO empty if pushing more than the FIFO depth.
5. Issue `CMD.hash_process` to finalise.
6. Read the digest from `DIGEST_0..15` once `hash_done` is signalled.
