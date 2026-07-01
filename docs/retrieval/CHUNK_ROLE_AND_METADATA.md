<!-- @summary
Engineering guide for the chunk-ROLE taxonomy (content|navigation|boilerplate) and
the wider chunk-metadata surface that drives retrieval: the single-source-of-truth
vocabulary (RAG_CHUNK_ROLES), the ingest→store→query→backfill mechanism, which
metadata properties are actually used vs dormant, and the roadmap of metadata-driven
retrieval improvements.
@end-summary -->

# Chunk Roles & Retrieval Metadata

This guide documents two related things:

1. The **chunk-role taxonomy** — the `content | navigation | boilerplate` label the
   LLM classifier assigns to every chunk, and how the query side uses it.
2. The **wider metadata surface** every chunk carries, which parts of it actually
   influence retrieval today, and the roadmap for using metadata more.

It is the canonical written-down definition of the taxonomy (previously only the
classifier prompt + code docstrings defined it).

---

## 1. The role taxonomy — why exactly three

`chunk_role` answers **one** question: *is this chunk answer-bearing, or is it
structural cruft that should not reach the answer context?* Three functional roles
cover that axis completely:

| role | what it captures | at query time |
| --- | --- | --- |
| **`content`** | Answer-bearing: facts, specs, values, steps, definitions — **and any data table** (register fields, measurements, status rows, even sparse/`|`-delimited ones). Also the fail-open default. | **kept** |
| **`navigation`** | Table-of-contents / index (section titles → page numbers, dotted leaders), a bare heading list, or a cross-reference pointer ("see chapter X"). Fingerprint = section-names + page-numbers, *not* data. | excluded (default) |
| **`boilerplate`** | Title page, copyright / proprietary / legal / trademark notice, document-metadata front-matter. | excluded (default) |

**There is no `toc`, `figure`, `table`, `front_matter`, or `reference` role, and
there should not be.** Those are structural *kinds*, not retrieval *roles* — a ToC
is `navigation`, a legal notice is `boilerplate`, a data table (even a figure that
holds data) is `content`. The role is judged by **function**, never by vendor,
heading, or phrasing (CLAUDE.md §0). This is why it generalises to an unseen
vendor/language: the classifier asks "is this answer-bearing?", not "does this match
a known ToC template?".

**When you think you need a fourth role, you almost always need a different
*property*, not a fourth role.** Finer axes — "what kind of document is this"
(`doc_type`), "what does this section do" (`section_function`), "how information-dense
is this chunk" (`info_density`) — are *separate metadata dimensions* that live on
their own properties and compose with `chunk_role`. Expanding `chunk_role` past its
one axis would overload it. See §6.

All roles live in **one collection** — tagging does not move chunks anywhere. A
`navigation`/`boilerplate` chunk stays exactly where it is in the vector store; the
query-time filter simply excludes it from the candidate pool. Nothing is deleted at
ingest.

---

## 2. Single source of truth: `RAG_CHUNK_ROLES`

The vocabulary is defined **once**, in `config/settings.py`:

```python
RAG_CHUNK_ROLES: tuple[str, ...] = (...)  # default: content, navigation, boilerplate
```

Everything else references it — there is no second copy of the literal list:

- `src/ingest/common/role_classify.py` → `_resolve_valid_roles()` reads
  `settings.RAG_CHUNK_ROLES` (lazily, to keep the module import-light); the config
  facade `classify_roles_from_config` passes it through explicitly.
- `config/settings.py` → `validate_nav_role_config()` validates
  `RAG_NAV_ROLE_DEFAULT` and `RAG_RETRIEVAL_EXCLUDED_ROLES` against
  `RAG_CHUNK_ROLES` (and fails fast if the vocabulary is empty).
- The classifier **prompt** `prompts/chunk_role_classify.md` carries the
  human-readable *semantics* of each role. The tuple is the machine-enforced
  vocabulary: any label the model emits that is not in `RAG_CHUNK_ROLES` is
  out-of-vocabulary and **fails open** to `RAG_NAV_ROLE_DEFAULT`.

> Changing the vocabulary is a three-part change: edit `RAG_CHUNK_ROLES`, update the
> prompt's role descriptions to match, and update `RAG_RETRIEVAL_EXCLUDED_ROLES` if
> the new role should be filtered. The validator enforces the invariants (default
> role must be in the vocabulary; default role must not be excluded).

---

## 3. End-to-end mechanism

```
INGEST (Temporal ingest worker: rag-worker / rag-worker-dev)
  chunking_node
    └─ _tag_chunk_roles()              (gated by RAG_INGESTION_NAV_CLASSIFY, default on)
         └─ classify_roles_sync → classify_roles_from_config → classify_roles
              └─ LLM router alias "judge"  (RAG_NAV_CLASSIFY_MODEL_ALIAS)
                   → in dev resolves to qwen2.5-7b-judge at ai01:8005 (via rag-vllm-tunnel:18005)
              batched 40/call, first 350 chars/chunk, temp 0, JSON mode
              → writes metadata["chunk_role"]  (NOTHING dropped; fail-open to "content")

STORE (Weaviate)
  chunk_role = filterable TEXT property (index_filterable, not searchable)
  ⚠ see §5 — ingest does NOT currently persist it; the backfill does.

QUERY (retrieval pipeline)
  RAGChain._build_role_exclusion_clauses()
    └─ if RAG_RETRIEVAL_ROLE_FILTER and RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT:
         one `ne` clause per role in RAG_RETRIEVAL_EXCLUDED_ROLES  (fail-open)
       injected once into every search mode (standard / deep-research / agentic)
  Agentic loop: orchestrator drops excluded-role chunks before the judge
    (RAG_AGENTIC_ROLE_BACKSTOP, a cheap no-LLM backstop)

BACKFILL (offline, no re-ingest)
  scripts/backfill_chunk_roles.py → role_backfill.backfill_roles_from_corpus
    re-tags chunks already in a collection using the SAME classifier
    → update_chunk_role() writes chunk_role in place (idempotent, resumable, fail-open)
```

The classifier runs **at ingest**, on the ingest worker, but the *compute* is on
**ai01** (the `judge` alias). Caveat: if `RAG_LLM_JUDGE_MODEL` is unset, `judge`
falls back to the default model (a reasoning model that emits messy JSON) — which is
why the JSON-salvage parser and `RAG_NAV_CLASSIFY_JSON_MODE=false` escape hatch
exist.

---

## 4. Configuration keys

**Ingest / tagging**

| key | default | purpose |
| --- | --- | --- |
| `RAG_CHUNK_ROLES` | `content,navigation,boilerplate` | **Canonical role vocabulary (single source of truth).** |
| `RAG_INGESTION_NAV_CLASSIFY` | `true` | Master toggle for LLM role-tagging at ingest. Off → chunks left untagged (nothing dropped). |
| `RAG_NAV_CLASSIFY_MODEL_ALIAS` | `judge` | LLM router alias used to classify (fast, JSON-reliable instruct model). |
| `RAG_NAV_CLASSIFY_BATCH_SIZE` | `40` | Chunks per classification call. |
| `RAG_NAV_CLASSIFY_PREFIX_CHARS` | `350` | Only the first N chars of each chunk are sent (role is decided by the head). |
| `RAG_NAV_CLASSIFY_TIMEOUT_SECONDS` | `120` | Per-call timeout. |
| `RAG_NAV_CLASSIFY_MAX_OUTPUT_TOKENS` | `1500` | Per-call output cap (must fit one JSON entry per batched chunk). |
| `RAG_NAV_CLASSIFY_JSON_MODE` | `true` | Request guided `json_object`. Set false for a JSON-unreliable reasoning alias. |
| `RAG_NAV_ROLE_DEFAULT` | `content` | Fail-open role on ANY classifier failure/ambiguity. Must be the answer-bearing role; validated ∈ `RAG_CHUNK_ROLES` and ∉ excluded set. |

**Query-time filter**

| key | default | purpose |
| --- | --- | --- |
| `RAG_RETRIEVAL_ROLE_FILTER` | `true` | Feature toggle (gate condition 1). |
| `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT` | `false` | **Migration gate (condition 2).** Only filter when the collection actually has `chunk_role`. Flip true after migrate + backfill. |
| `RAG_RETRIEVAL_EXCLUDED_ROLES` | `navigation,boilerplate` | Roles excluded at query time (one `ne` clause each — fail-open). |
| `RAG_AGENTIC_ROLE_BACKSTOP` | `true` | Agentic-loop metadata fast-path: drop excluded-role chunks before the judge. |

### The two-condition gate

The query-time exclusion is a no-op unless **both**
`RAG_RETRIEVAL_ROLE_FILTER` **and** `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT` are true.
`SCHEMA_PRESENT` defaults **false** so the filter never references `chunk_role` on a
collection that has not yet been migrated/backfilled. The filter uses `ne` per role,
which is deliberately **fail-open**: `chunk_role != "navigation"` still keeps chunks
whose role is `NULL`/legacy or `content`, so an un-backfilled chunk is never dropped.
This mirrors the tree-retrieval gate `RAG_TREE_SCHEMA_PRESENT`.

---

## 5. ⚠ Known gap: ingest does not persist `chunk_role`

**Verified:** `chunking_node` tags `metadata["chunk_role"]` in memory, but
`add_documents` (`src/vector_db/weaviate/store.py`) builds an explicit
`properties`/`optional` dict that has **no `chunk_role` key**, so the tag is dropped
at the store boundary. The **only** code that writes `chunk_role` onto a stored
object is `update_chunk_role`, used exclusively by the backfill.

Consequences:

- The **backfill is currently the only way roles land in the store.** A full
  re-ingest would *not* populate `chunk_role`.
- This is why an un-backfilled collection shows 0 tagged chunks even with tagging on.

**Recommended fix (one line, low risk):** add
`"chunk_role": metadata.get("chunk_role", "")` to the `optional` dict in
`add_documents`, so future ingests self-tag (behind the same schema property). The
backfill remains for existing corpora.

### Enabling the filter on a collection

1. `uv run python scripts/migrate_weaviate_table_schema.py --collection <NAME>` — adds
   the `chunk_role` property (and other `TABLE_AWARE_PROPERTIES`) if missing.
2. `uv run python scripts/backfill_chunk_roles.py --collection <NAME>` — tags every
   chunk in place (idempotent, resumable, `--dry-run` to preview).
3. Set `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT=true` and recreate the API/worker so it
   picks up the env. Verify the CHI/AXI-vs-CHI pools are clean.

> The backfill mutates only the `chunk_role` column of the target collection
> (non-destructive to text/vectors) and does not touch prod. With the gate off it has
> **zero effect on query results** — retrieval ignores `chunk_role` until the gate is
> flipped. Note (from ops history): deleting/altering chunks in Weaviate does not
> clear the ingest manifest.

---

## 6. The wider metadata surface

`chunk_role` is the first *model-driven* metadata signal, but chunks carry ~50
properties. Only ~a dozen actually influence retrieval today; the rest are
provenance / dedup / citation / idempotency data.

### Actively used in retrieval

| property | how it's used |
| --- | --- |
| `text` | dense-embedding + BM25 source; cross-encoder input; text-based nav/thin drop. |
| `tenant_id` | hard equality filter for tenant isolation (when supplied). |
| `document_id` | card-router soft `in`/`not_in` filter; per-doc diversity cap; xref/table expansion scope; dedup key. |
| `node_kind` | leaf-only filter (`= chunk`), on by default via `RAG_TREE_SCHEMA_PRESENT`. |
| `heading_path` | `heading_match_score` term in rerank-fusion (`RAG_RERANK_HEADING_LAMBDA`). |
| `section_path` | xref section-expansion `like`-prefix filter. |
| `chunk_type` + `table_group_id` + `table_row_index` | table-group expansion (stitch a table's rows back together). |
| `xref_targets` + `caption_label` | cross-reference expansion (pull referenced section/table/figure into context). |
| `source` / `heading` | optional opt-in equality filters; deep-research source-diversity + dedup. |

### Shipped but dormant by default

- **`chunk_role`** — the query filter is built and injected, but gated off by
  `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT=false` (see §5).
- **Tree descent** — `node_kind='section'`, `parent_section_id`, `_anchor_rank` only
  fire under `RAG_TREE_RETRIEVAL_ENABLED` (default false).

### Fully dormant (no retrieval reference)

`provenance_confidence`, `original/refactored_char_*`, `chunk_index`, `total_chunks`,
`heading_level`, `tree_level`, `content_hash`, `page_no/label/bbox`, `table_num_*`,
`table_caption`, `table_markdown`, `figure_image_uri`, and the provenance/citation
fields (`source_uri`, `source_key`, `title`, `date`, `tags`, `author`, …). Several
are read only by the **generation**-stage document formatter for citation display —
not by retrieval.

---

## 7. Roadmap: metadata-driven retrieval

Ranked by value/effort. All are config-gated and default-inert (CLAUDE.md §3), and
all must pass the Generic-Fix test (§0) — no per-vendor/per-document literals; every
signal is a general, measurable property that works on unseen documents.

**Quick wins — safe today, no ingest, default weight 0.0 until tuned**

1. **CE-attenuation blend.** The qwen3 cross-encoder is measured net-negative on
   most queries yet is the *dominant* term in `fuse_scores`. Make it a tunable weight
   (`RAG_RERANK_CE_LAMBDA`, default 1.0 = today) and optionally re-inject the pre-CE
   hybrid (dense+BM25) order as its own RRF term (`RAG_RERANK_HYBRID_RRF_LAMBDA`).
   This is the enabler — priors below are swamped while the CE base term dominates.
2. **Info-density prior** from chunk *text* (alnum ratio + type-token ratio + digit
   bonus, data-table-exempt) — a graded generalisation of the binary nav-drop that
   demotes format/template junk. Zero metadata, works on un-backfilled collections.
3. **`chunk_role` soft prior** (demote nav/boilerplate in fusion instead of the
   dormant hard filter) — fail-open, but only *moves numbers* once roles are
   backfilled.

**Needs an ingest / backfill pass**

4. **`chunk_role` hard `in`-allowlist filter** — cheaper, closes a fail-open leak;
   requires 100% backfill first (contains_any drops NULL chunks).
5. **`doc_type` + `section_function` tags** — question-type-aware routing/section
   boost (route how-to queries to procedure docs). Two new ingest classifiers +
   schema + backfill; best after a shared query-intent classifier exists.

**Cross-cutting decision:** proposals for per-query hybrid-alpha, `chunk_type`
scoping, and `doc_type` routing all assume a **query-intent classifier**, which does
not exist yet (`query_shape.py` only does the deep-research suggestion). Building one
shared intent classifier unlocks three of these at once, at the cost of ~one LLM call
per query.

Every non-zero prior weight needs a sweep against the blind-LLM-judge eval set before
it ships — the defaults are inert precisely so the code can land first and be tuned
later.

---

## 8. Troubleshooting

| symptom | likely cause |
| --- | --- |
| Filter has no effect after backfill | `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT` still false, or container not recreated to pick up the env. |
| 0 tagged chunks after a full re-ingest | The ingest-persistence gap (§5) — run the backfill, or apply the `add_documents` fix. |
| All chunks tagged `content` | Classifier failing open — check `RAG_LLM_JUDGE_MODEL`/`_API_BASE` resolve to a reachable JSON-reliable model; inspect logs for provider/parse errors. |
| Config fails fast at startup | `validate_nav_role_config` — `RAG_NAV_ROLE_DEFAULT` not in `RAG_CHUNK_ROLES`, or it appears in `RAG_RETRIEVAL_EXCLUDED_ROLES`, or `RAG_CHUNK_ROLES` is empty. |
| Real content missing from answers after enabling the filter | A content chunk mis-tagged `navigation`/`boilerplate`. Roles are soft/fail-open by design — prefer soft demotion (§7.2/7.3) over the hard filter, or widen the excluded set carefully. |

## Related docs

- `TREE_RETRIEVAL_DESIGN.md` — `node_kind`/tree metadata (a separate axis).
- `DOCUMENT_ROUTING_DESIGN.md` — RAPTOR-lite routing (headings/cards as a navigation
  *layer* — different sense of "navigation").
- `AGENTIC_RETRIEVAL_DESIGN.md` — the agentic judge that subsumes the older regex
  nav-filter.
