"""The 422 request-validation handler must always render.

Pydantic v2 attaches the raw raised exception to ``ctx['error']`` for
model_validator / field-validator failures (e.g. the retrieval-strategy
mutual-exclusion check). A raw ``ValueError`` is not JSON-serializable, so
without sanitizing it the 422 handler itself raises ``TypeError`` and the
catch-all turns a clean 422 client error into a 500. These tests pin the
sanitizer so conflicting orchestrator flags return a clean 422.
"""
from __future__ import annotations

import json

from fastapi.exceptions import RequestValidationError

from server.api import _jsonsafe_validation_errors


def test_raw_exception_in_ctx_is_stringified_and_serializable():
    exc = RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", "retrieval_strategy"),
                "msg": "only one retrieval orchestrator may be forced at once",
                "ctx": {"error": ValueError("only one retrieval orchestrator may be forced at once")},
            }
        ]
    )
    safe = _jsonsafe_validation_errors(exc)
    # The whole point: the payload must serialize without raising.
    json.dumps({"errors": safe})
    assert isinstance(safe[0]["ctx"]["error"], str)
    assert "only one retrieval orchestrator" in safe[0]["ctx"]["error"]


def test_errors_without_ctx_pass_through():
    exc = RequestValidationError(
        [{"type": "missing", "loc": ("body", "query"), "msg": "Field required"}]
    )
    safe = _jsonsafe_validation_errors(exc)
    json.dumps({"errors": safe})
    assert safe[0]["type"] == "missing"
    assert "ctx" not in safe[0]


def test_primitive_ctx_values_preserved():
    exc = RequestValidationError(
        [
            {
                "type": "greater_than",
                "loc": ("body", "search_limit"),
                "msg": "Input should be greater than 0",
                "ctx": {"gt": 0},
            }
        ]
    )
    safe = _jsonsafe_validation_errors(exc)
    json.dumps({"errors": safe})
    assert safe[0]["ctx"]["gt"] == 0  # primitives untouched
