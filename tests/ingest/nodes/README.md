<!-- @summary
Integration-level tests for ingest pipeline nodes that cut across subsystem
boundaries, currently focused on batch-embedding optimisation (FR-1210–FR-1214).
@end-summary -->

# tests/ingest/nodes

Cross-subsystem node tests for the ingest pipeline. Unlike the per-subsystem
test directories, tests here exercise behaviours that span multiple modules or
validate feature requirements that do not belong to a single pipeline stage.

## Contents

| Path | Purpose |
| --- | --- |
| `test_embedding_storage_batching.py` | Batch embedding optimisation — `_form_batches` ordering/partial/empty, config validation, `_embed_batches` retry isolation, observability logs, and `embedding_storage_node` batching integration |
