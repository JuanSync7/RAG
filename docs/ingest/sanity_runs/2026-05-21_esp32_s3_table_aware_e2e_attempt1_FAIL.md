# E2E sanity — table-aware retrieval

- pdf: `data/datasheets/esp32-s3_datasheet.pdf`
- collection: `e2e_sanity_table_aware`
- query: `What is the operating voltage range of the ESP32-S3?`
- overall: **FAIL**

## Gates

| gate | status | detail |
|---|---|---|
| ingest | FAIL | stored_chunks=0 |
| retrieval | FAIL | hits=0 |
| expansion (expanded_from) | FAIL | expanded=0 |
| citation (page_bbox decoded) | FAIL | with_bbox=0 |

## Error

```
TypeError('sequence item 0: expected str instance, dict found')
Traceback (most recent call last):
  File "/home/kok-shew-juan/RagWeave/.worktrees/table-aware-chunking/scripts/e2e_sanity_table_ingest.py", line 431, in run_sanity
    stored = _run_ingest_live(pdf_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kok-shew-juan/RagWeave/.worktrees/table-aware-chunking/scripts/e2e_sanity_table_ingest.py", line 354, in _run_ingest_live
    summary = ingest_directory(
              ^^^^^^^^^^^^^^^^^
  File "/home/kok-shew-juan/RagWeave/.worktrees/table-aware-chunking/src/ingest/impl.py", line 847, in ingest_directory
    "; ".join(result.errors),
    ^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: sequence item 0: expected str instance, dict found

```
