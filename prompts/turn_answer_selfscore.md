# Turn Loop — Answer Self-Score

You audit a drafted answer against the evidence it was drafted from. The draft will only be shown to the user if it clears a confidence gate; your score is one component of that gate, so score the draft's **grounding**, not its style. Be strict: an ungrounded fluent answer is worse than a grounded partial one.

## Rules

- `self_score`: a [0, 1] judgment of how well the draft answers the question **using the evidence** — 1.0 means every claim is supported by the digest and the question is fully addressed; 0.0 means the draft is off-question or unsupported. Calibrate: a correct-but-partial answer with full grounding scores higher than a complete answer with invented specifics.
- `unsupported_claims`: every factual claim in the draft that the evidence digest does not support — quote or closely paraphrase each claim so it can be located in the draft. Empty list when everything is grounded.
- A claim that is common knowledge framing (not a domain fact) does not count as unsupported; a specific value, name, step, or behavior does.
- Judge only against the evidence digest below. Do not use outside knowledge to declare a claim true — if the digest does not support it, it is unsupported here.
- The draft was instructed to synthesize across chunks, draw *clearly-signalled* supported inferences, give partial answers that note what the context does not cover, and surface conflicts / unconfirmed assumptions. **Credit these behaviours, do not punish them:** a signalled inference ("Based on the described architecture…") whose premises ARE in the digest counts as supported; an explicit gap / conflict / unconfirmed-assumption flag is good grounding, not an unsupported claim; a sentence the draft openly marked `[background]` (declared outside the context) is a correct disclosure, not an unsupported claim. Flag as unsupported only a specific value, name, step, or behavior asserted as an in-context fact that the digest does not actually contain.

## Inputs

User question:
{{ user_query }}

Drafted answer:
{{ draft_answer }}

Evidence digest the draft must be grounded in:
{{ evidence_digest }}

## Output

Return a single JSON object, no prose outside it:

```json
{
  "self_score": 0.8,
  "unsupported_claims": ["the timeout defaults to 30 seconds"]
}
```
