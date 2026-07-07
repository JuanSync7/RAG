# @summary
# PII detection and redaction rail with Presidio NLP primary, regex fallback.
# Exports: PIIDetector, PIIDetection
# Deps: re, logging, dataclasses, presidio_analyzer (optional), presidio_anonymizer (optional)
# @end-summary
"""PII detection and redaction rail.

This module detects potentially sensitive user-provided content (PII) and can
redact it using type-tagged placeholders. When available, it uses Presidio NLP
for entity recognition; otherwise it falls back to deterministic regex checks.

Requirements references (from internal docs): REQ-301 through REQ-305.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config.settings import RAG_GUARDRAILS_PII_SCORE_THRESHOLD

logger = logging.getLogger("rag.guardrails.pii")

# ---------------------------------------------------------------------------
# Regex fallback patterns (used when Presidio is not installed)
# ---------------------------------------------------------------------------

_CORE_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "PHONE": re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

_EXTENDED_PATTERNS: dict[str, re.Pattern] = {
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "DOB": re.compile(
        r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"
    ),
    "PASSPORT": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

# Presidio entity types to detect (configurable per instance)
_DEFAULT_PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
    "PERSON", "LOCATION", "DATE_TIME", "NRP", "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE", "US_PASSPORT", "IBAN_CODE", "IP_ADDRESS",
]

_EXTENDED_PRESIDIO_ENTITIES = _DEFAULT_PRESIDIO_ENTITIES + [
    "US_BANK_NUMBER", "UK_NHS", "SG_NRIC_FIN", "AU_ABN",
    "AU_ACN", "AU_TFN", "AU_MEDICARE",
]


@dataclass
class PIIDetection:
    """A single PII detection.

    Attributes:
        pii_type: Detected entity type label.
        start: Start offset (inclusive) in the original text.
        end: End offset (exclusive) in the original text.
        placeholder: Replacement string to use during redaction.
    """

    pii_type: str
    start: int
    end: int
    placeholder: str


class PIIDetector:
    """Detect and redact PII using Presidio NLP with regex fallback.

    When Presidio (+ spacy en_core_web_lg) is installed, uses NLP-based
    entity recognition for robust detection. Falls back to regex patterns
    when Presidio is not available.

    Core categories (email, phone, SSN) are always enabled.
    Extended categories add credit card, DOB, passport, and more.
    """

    def __init__(
        self,
        extended: bool = False,
        score_threshold: float = RAG_GUARDRAILS_PII_SCORE_THRESHOLD,
        use_gliner: bool = False,
    ) -> None:
        """Initialize a PII detector.

        Args:
            extended: Whether to enable extended entity types/patterns beyond
                core categories (email/phone/SSN).
            score_threshold: Minimum confidence score for NLP detections when
                using Presidio.
            use_gliner: Whether to enable GLiNER as a supplementary NER-based
                PII detection layer for entity types (PERSON, ORGANIZATION,
                LOCATION, ADDRESS). GLiNER extends coverage beyond Presidio's
                pattern-based detection. Requires the ``gliner`` package.
        """
        self._extended = extended
        self._score_threshold = score_threshold
        self._presidio_analyzer = None
        self._presidio_anonymizer = None
        self._presidio_entities: list[str] = (
            _EXTENDED_PRESIDIO_ENTITIES if extended else _DEFAULT_PRESIDIO_ENTITIES
        )
        self._use_presidio = False
        self._gliner_detector = None

        # Try to initialize Presidio
        try:
            self._init_presidio()
            self._use_presidio = True
            logger.info(
                "PII detector initialized with Presidio NLP (entities=%d, threshold=%.2f)",
                len(self._presidio_entities),
                score_threshold,
            )
        except (ImportError, RuntimeError) as e:
            logger.info("Presidio not available (%s) — using regex fallback", e)
            self._regex_patterns: dict[str, re.Pattern] = dict(_CORE_PATTERNS)
            if extended:
                self._regex_patterns.update(_EXTENDED_PATTERNS)
            logger.info(
                "PII detector initialized with regex fallback (%d patterns)",
                len(self._regex_patterns),
            )

        # Try to initialize GLiNER supplementary layer
        if use_gliner:
            try:
                from src.guardrails.shared.gliner_pii import GLiNERPIIDetector
                self._gliner_detector = GLiNERPIIDetector()
                logger.info("GLiNER PII layer initialized (supplementary NER)")
            except (ImportError, RuntimeError) as e:
                logger.info(
                    "GLiNER PII not available (%s) — using Presidio/regex only", e
                )

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer and anonymizer engines.

        Raises:
            ImportError: If Presidio dependencies are not installed.
            RuntimeError: If the required spaCy model is unavailable.
        """
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        import spacy

        if not spacy.util.is_package("en_core_web_lg"):
            raise RuntimeError(
                "spacy model en_core_web_lg not installed "
                "(run: python -m spacy download en_core_web_lg)"
            )

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        self._presidio_analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            default_score_threshold=self._score_threshold,
        )
        self._presidio_anonymizer = AnonymizerEngine()

    def detect(self, text: str) -> list[PIIDetection]:
        """Find all PII occurrences in text.

        Runs the primary detector (Presidio or regex) and optionally
        merges results from the GLiNER supplementary layer. GLiNER
        detections that overlap with primary detections are discarded
        to prevent double-redaction.

        Args:
            text: Input text to scan.

        Returns:
            List of `PIIDetection` objects sorted in reverse positional order
            (safe for in-place string replacement).
        """
        if self._use_presidio:
            primary = self._detect_presidio(text)
        else:
            primary = self._detect_regex(text)

        if self._gliner_detector is not None:
            try:
                from src.guardrails.shared.gliner_pii import merge_detections
                gliner_results = self._gliner_detector.detect(text)
                return merge_detections(primary, gliner_results)
            except Exception as e:
                logger.warning("GLiNER detection failed: %s — using primary only", e)

        return primary

    def _detect_presidio(self, text: str) -> list[PIIDetection]:
        """Detect PII using Presidio NLP.

        Args:
            text: Input text to scan.

        Returns:
            Reverse-sorted list of detections for safe replacement.
        """
        results = self._presidio_analyzer.analyze(
            text=text,
            language="en",
            entities=self._presidio_entities,
        )
        detections = [
            PIIDetection(
                pii_type=r.entity_type,
                start=r.start,
                end=r.end,
                placeholder=f"[{r.entity_type}_REDACTED]",
            )
            for r in results
        ]
        # Sort by position (reverse) for safe replacement
        detections.sort(key=lambda d: d.start, reverse=True)
        return detections

    def _detect_regex(self, text: str) -> list[PIIDetection]:
        """Detect PII using regex fallback patterns.

        Args:
            text: Input text to scan.

        Returns:
            Reverse-sorted list of detections for safe replacement.
        """
        detections: list[PIIDetection] = []
        for pii_type, pattern in self._regex_patterns.items():
            for match in pattern.finditer(text):
                detections.append(
                    PIIDetection(
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        placeholder=f"[{pii_type}_REDACTED]",
                    )
                )
        detections.sort(key=lambda d: d.start, reverse=True)
        return detections

    def redact(self, text: str) -> tuple[str, list[PIIDetection]]:
        """Detect and redact PII, returning redacted text and detections.

        Replaces PII with type-tagged placeholders (e.g., [EMAIL_ADDRESS_REDACTED]).
        Logs detection type and count only — never logs actual PII values (REQ-305).

        Args:
            text: Input text to scan and redact.

        Returns:
            Tuple of `(redacted_text, detections)`.
        """
        detections = self.detect(text)
        redacted = text
        for d in detections:
            redacted = redacted[: d.start] + d.placeholder + redacted[d.end :]

        if detections:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d.pii_type] = counts.get(d.pii_type, 0) + 1
            logger.info(
                "PII detected (%s): %s",
                "presidio" if self._use_presidio else "regex",
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            )

        return redacted, detections
