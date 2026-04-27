<!-- @summary
Backend-agnostic ML rail modules. Implements the individual guardrail checks (intent, injection, PII, toxicity, topic safety, faithfulness) that are consumed by any guardrail backend.
@end-summary -->

# shared

This package contains the backend-agnostic guardrail rail implementations. Each module performs one specific safety check and is designed to be instantiated with constructor injection — no module calls `GuardrailsRuntime.get()` directly.

Rails are consumed by the `nemo_guardrails` executor classes and can be reused by any future backend.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package init; re-exports all rail classes and constants |
| `faithfulness.py` | `FaithfulnessChecker` — hallucination detection via NeMo self-check-facts prompts, per-claim LLM scoring, and deterministic entity checks |
| `injection.py` | `InjectionDetector` — layered prompt-injection / jailbreak detection (perplexity heuristics, model classifier, regex, LLM fallback) |
| `intent.py` | `IntentClassifier` — canonical intent classification (RAG search vs. greetings/off-topic) using NeMo runtime with keyword fallback |
| `pii.py` | `PIIDetector` — PII detection and redaction using Presidio NLP with regex fallback |
| `gliner_pii.py` | `GLiNERPIIDetector` — supplementary zero-shot NER-based PII detection via GLiNER for entity types Presidio regex may miss |
| `topic_safety.py` | `TopicSafetyChecker` — LLM-based on/off-topic classification adapted from NeMo's built-in topic_safety action |
| `toxicity.py` | `ToxicityFilter` — toxicity detection and output filtering using NeMo self-check prompts with keyword fallback |
