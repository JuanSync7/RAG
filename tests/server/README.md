<!-- @summary
Contract tests for the server's Pydantic schema models (FR-3060–FR-3068). Validates round-trip serialization, required-field enforcement, and default values for all Document and Collection Management response models exposed by the server API.
@end-summary -->

# tests/server/

Tests for the server layer, currently focused on the Pydantic schema contracts defined in `server/schemas.py`. Each model is exercised for correct field defaults, round-trip serialization, and `ValidationError` on missing required fields.

## Contents

| Path | Purpose |
| --- | --- |
| `test_document_management_schemas.py` | Contract tests for `DocumentSummary`, `DocumentListResponse`, `DocumentDetailResponse`, `DocumentUrlResponse`, `SourceSummary`, `SourceListResponse`, `CollectionItem`, `CollectionStatsResponse`, and `CollectionListResponse` |
