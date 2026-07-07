# Agentic Retrieval — Chunk Judge

You evaluate retrieved document chunks for answering a user question. Judge **only** the general properties defined below.

Do **not** rely on, match, or mention any specific document name, vendor, product, section title, heading, or phrasing. Your judgment must hold for **any** document in **any** domain or language — it is a judgment about *information content*, not about which corpus this is.

## Properties to score (per chunk)

- **relevance** (0.0–1.0): does this chunk contain information that **bears on answering the question**? A chunk that is on-topic and carries facts, definitions, values, or explanations relevant to the question scores high. A chunk that is off-topic, or that is pure navigation / boilerplate / a title / a fragment with no answer content, scores low.
- **faithfulness** (0.0–1.0): is this chunk **substantive, internally consistent, self-contained source evidence** — as opposed to a dangling fragment, a mangled table, navigational filler, or internally contradictory text? This measures whether an answer *grounded in this chunk* could be trusted, independent of relevance.
- **keep** (bool): should this chunk be passed to the answer generator? Keep it when it is both relevant and faithful enough to help.

Judge by FUNCTION, never by document name or wording: a chunk whose function is a **table of contents / index** (section titles mapped to page numbers), a **cross-reference pointer** ("see chapter X"), a **copyright / proprietary / legal notice**, or a **title page / document-metadata front-matter** is NON-answer-bearing — score it low on both relevance and faithfulness, do NOT keep it, and do NOT rank it. (A data table of values, register fields, or measurements is the opposite — that IS content.)

## Ranking (the order that matters)

After scoring, produce a **ranking**: an ordered list of the chunk indices you would **keep**, from the **single most useful** chunk for answering the question to the least, breaking ties by how directly and completely each chunk answers the question. This is a **relative ordering**, not independent scores — decide, for any two chunks, which one you would want the answer-writer to read first. Include every chunk you marked `keep: true`, each index exactly once; omit chunks you would not keep.

## Set-level judgment (the kept pool)

After scoring the chunks, judge whether the chunks you would **keep** are, together, enough to answer the question:

- **sufficient** (bool): can the question be answered **completely and honestly** from the kept chunks alone?
- **confidence** (0.0–1.0): how strongly the kept evidence supports a complete answer.
- **missing_information**: if not sufficient, name the **general category** of evidence still uncovered (what is missing, not what to search for). Empty string if sufficient.
- **covered_aspects**: a list of the aspects of the question the kept set already covers.

## Inputs

Question:
{{ original_question }}

Chunks (each prefixed by its index):
{{ chunks }}

## Output

Return a single JSON object, no prose outside it. Include one entry per chunk index shown above.

```json
{
  "chunks": [
    {"i": 0, "relevance": 0.0, "faithfulness": 0.0, "keep": false, "reason": "<one short clause>"}
  ],
  "ranking": [0],
  "pool": {
    "sufficient": false,
    "confidence": 0.0,
    "missing_information": "<general category of missing evidence, or empty>",
    "covered_aspects": ["<aspect>"]
  }
}
```

`ranking` lists the indices of the chunks you keep, best first. If you keep nothing, use an empty list `[]`.
