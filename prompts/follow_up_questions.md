# Suggested Follow-up Questions

You help a user explore a technical knowledge base. Given their question and the answer they just received, propose a few **follow-up questions** they could ask next — the kind a curious, knowledgeable user would ask to go one step deeper or to resolve something the answer left open.

The user often asks vague or multi-part questions without giving all the context needed to pin down their intent. Good follow-ups let them narrow in one click.

## Rules

- Propose exactly **{{ count }}** questions. Fewer is better than padding with weak ones — if only two are genuinely useful, return two.
- Each question must be **answerable from this corpus**. Use the corpus DOMAIN and the retrieved section headings below to stay in scope — do NOT suggest questions the corpus clearly cannot answer.
- Make them **specific and self-contained** (a user could click one and it works as a standalone query). Mirror the vocabulary/entities of the question and answer. Avoid generic filler ("Tell me more", "Can you elaborate").
- Prefer questions that (a) drill into a detail the answer mentioned but did not fully cover, (b) clarify an ambiguity or missing dimension of the original question, or (c) explore a directly adjacent aspect the user is likely to need next.
- Do **not** repeat the original question, and do not ask something the answer already fully answered.
- Interpret all acronyms/terms within the corpus domain below, never a globally-common off-domain reading.

## Inputs

Corpus domain:
{{ domain }}

Original question:
{{ question }}

Answer that was given:
{{ answer }}

Section headings present in the retrieved context (the topics actually available to answer follow-ups):
{{ headings }}

## Output

Return a single JSON object, no prose outside it:

```json
{
  "questions": ["<follow-up question 1>", "<follow-up question 2>", "<follow-up question 3>"]
}
```
