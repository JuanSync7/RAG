# @summary
# Output sanitization for generated answers: removes system prompt leakage,
# document boundary markers, and template artifacts.
# Exports: sanitize_answer
# Deps: logging, string
# @end-summary
"""Output sanitization for generated answers (REQ-704).

Uses structural detection rather than regex to identify and remove:
1. System prompt leakage — matches against the actual prompt text
2. Document boundary markers — matches the formatting patterns we use
3. Template variable artifacts — detects unreplaced placeholders

All functions are deterministic and side-effect-free.
"""

from __future__ import annotations

import logging
import string
from typing import Optional

logger = logging.getLogger("rag.output_sanitizer")

# Characters stripped from the edges of each token during normalisation.
# Includes all ASCII punctuation plus common Unicode smart-quote / ellipsis
# characters that may surround words in LLM output. Hyphens that appear
# *inside* a token (e.g. "well-formed") survive because str.strip only
# removes characters from the leading/trailing edge.
_EDGE_PUNCT = (
    string.punctuation
    + "\u201c\u201d"  # left/right double quotation marks
    + "\u2018\u2019"  # left/right single quotation marks
    + "\u2026"         # horizontal ellipsis
    + "\u00ab\u00bb"  # guillemets
)

# Document boundary markers used by the pipeline's formatting stages.
# If any of these appear in the generated answer, they are artifacts
# that leaked from the context into the output.
_BOUNDARY_MARKERS = [
    "--- Document ",
    "--- VERSION CONFLICT WARNING ---",
    "[CONTENT_FILTERED]",
]

# Template variable patterns — unreplaced placeholders
_TEMPLATE_ARTIFACTS = [
    "{context}",
    "{question}",
    "{documents}",
    "{query}",
]


def sanitize_answer(
    answer: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Sanitize a generated answer by removing leaked internal artifacts.

    Uses the actual system prompt text (not regex) to detect leakage.
    This is more robust than pattern matching because it catches exact
    substrings of the real prompt regardless of how they're fragmented.

    Args:
        answer: Raw generated answer text.
        system_prompt: The system prompt used for generation. If provided,
            the sanitizer checks for leaked fragments.

    Returns:
        Sanitized answer with artifacts removed.
    """
    if not answer:
        return answer

    lines = answer.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip document boundary markers
        if _is_boundary_marker(stripped):
            logger.debug("Stripped boundary marker: %s", stripped[:60])
            continue

        # Skip template artifacts
        if _is_template_artifact(stripped):
            logger.debug("Stripped template artifact: %s", stripped[:60])
            continue

        # Skip system prompt fragments (if prompt provided)
        if system_prompt and _is_prompt_fragment(stripped, system_prompt):
            logger.debug("Stripped prompt fragment: %s", stripped[:60])
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    # If sanitization removed everything, return the original
    # (something is better than nothing)
    if not result:
        logger.warning("Sanitization removed all content — returning original")
        return answer

    return result


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens with leading/trailing punctuation stripped.

    Punctuation is removed from the edges of each whitespace-delimited word
    only, so intra-word punctuation (hyphens in "well-formed", apostrophes in
    contractions) is preserved as part of the token identity.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of normalised tokens; empty tokens produced by stripping are
        discarded.
    """
    tokens = []
    for word in text.lower().split():
        token = word.strip(_EDGE_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def _is_boundary_marker(line: str) -> bool:
    """Check if a line is a document boundary marker.

    Requires the marker to appear at column 0 (line starts with the marker
    prefix). For the '--- Document N ---' format, additionally validates that
    the numeric portion consists only of digits and that the line ends with
    ' ---', so embedded mentions in legitimate prose are not flagged.
    """
    if not line.startswith("--- Document "):
        for marker in ("--- VERSION CONFLICT WARNING ---", "[CONTENT_FILTERED]"):
            if line == marker:
                return True
        return False

    # Shape check for '--- Document N ---': everything between the prefix and
    # the trailing ' ---' must be digits.
    prefix = "--- Document "
    suffix = " ---"
    if not line.endswith(suffix):
        return False
    middle = line[len(prefix) : len(line) - len(suffix)]
    return bool(middle) and middle.isdigit()


def _is_template_artifact(line: str) -> bool:
    """Check if a line is an unreplaced template variable placeholder.

    A line is flagged only when it consists of nothing but the placeholder
    token (plus optional surrounding punctuation). Legitimate prose that
    references a placeholder by name is not flagged.

    Braces are not stripped so that '{context}' is preserved as-is for
    comparison against _TEMPLATE_ARTIFACTS. Only edge whitespace and
    non-brace punctuation are removed.
    """
    words = line.split()
    if len(words) != 1:
        return False
    brace_safe_strip = _EDGE_PUNCT.replace("{", "").replace("}", "")
    candidate = words[0].strip(brace_safe_strip)
    return candidate in _TEMPLATE_ARTIFACTS


def _is_prompt_fragment(
    line: str,
    system_prompt: str,
    min_fragment_length: int = 40,
) -> bool:
    """Check if a line is a leaked fragment of the system prompt.

    Uses word-level contiguous sequence matching after stripping edge
    punctuation from each token. This means a line like "Answer the question."
    matches the prompt phrase "Answer the question" even though the raw words
    differ. Intra-word punctuation (hyphens, apostrophes) is preserved so that
    "well-formed" only matches "well-formed" and not "well formed".

    Args:
        line: A single line from the answer.
        system_prompt: The full system prompt text.
        min_fragment_length: Minimum character length of ``line`` to test.
            Lines shorter than this are never flagged.

    Returns:
        True if the line's tokens form a verbatim contiguous run within
        the prompt's token sequence.
    """
    if len(line) < min_fragment_length:
        return False

    line_tokens = _tokenize(line)
    if not line_tokens:
        return False

    prompt_tokens = _tokenize(system_prompt)
    line_len = len(line_tokens)

    # Slide a window of len(line_tokens) across prompt_tokens looking for
    # an exact contiguous match. O(n * m) but prompt and line are both
    # short in practice (< 500 words / < 30 words respectively).
    for i in range(len(prompt_tokens) - line_len + 1):
        if prompt_tokens[i : i + line_len] == line_tokens:
            return True
    return False
