# Turn Loop — Deep-Study Window Read

You are reading one window of a full source document to answer a focused question. The document is being read window-by-window in order; you see the current window's text and the notes accumulated from the windows already read. Extract only what this window contributes — the notes are the memory that carries across windows, so write them to be useful without the window text.

## Rules

- `notes`: the NEW facts from THIS window that bear on the question — specific values, names, steps, definitions, verbatim identifiers. Do not repeat notes already recorded; do not summarize the window generally; record nothing when the window is irrelevant (empty string is a valid answer).
- Preserve exact terminology: values, signal/register/parameter names, versions, and numbers verbatim — never paraphrase a specific into a generality.
- `answer_found`: `true` only when the accumulated notes (previous + this window) now contain enough to answer the question — not merely something related to it.
- `next_window_hint`: one short sentence on what to look for next (e.g. a section the text references ahead), or an empty string when you have no signal about what follows.
- Judge only from the text given. Never import outside knowledge into the notes.

## Inputs

Question this study must answer:
{{ question }}

Document title:
{{ document_title }}

Window {{ window_index }} of {{ window_count }}:
{{ window_text }}

Notes accumulated from previous windows:
{{ notes_so_far }}

## Output

Return a single JSON object, no prose outside it:

```json
{
  "notes": "<new question-relevant facts from this window, or empty string>",
  "answer_found": false,
  "next_window_hint": "<what to look for in later windows, or empty string>"
}
```
