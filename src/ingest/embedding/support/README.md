<!-- @summary
Support libraries for the Embedding Pipeline, currently providing the MinHash
fingerprint engine used by Tier 2 fuzzy cross-document deduplication.
@end-summary -->

# embedding/support

Houses backend support libraries that pipeline nodes depend on but that are not
pipeline stages themselves. Currently contains the MinHash engine for optional
fuzzy deduplication; the `datasketch` dependency is runtime-optional and raises
`ImportError` at call time when not installed.

## Contents

| Path | Purpose |
| --- | --- |
| `minhash_engine.py` | MinHash fingerprint computation, Jaccard similarity estimation, and Weaviate-backed fuzzy chunk lookup (Tier 2 dedup) |
| `__init__.py` | Package marker |
