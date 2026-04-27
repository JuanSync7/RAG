# @summary
# Reranker providers: local BAAI bge-reranker-v2-m3 (in-process) and TEI over HTTP.
# Key exports: LocalBGEReranker, TEIReranker, RankedResult, get_reranker_provider
# Deps: transformers + torch (local path only; lazy-imported), httpx (TEI path),
#       config.settings (RERANKER_MODEL_PATH, RERANK_TOP_K, RERANKER_MAX_LENGTH,
#       RERANKER_BATCH_SIZE, RERANKER_PRECISION, INFERENCE_BACKEND, TEI_*),
#       src.vector_db.common.schemas, src.retrieval.common.schemas,
#       src.retrieval.common.exceptions
# @end-summary
"""Reranker provider implementations and factory.

Two backends:
  local — BAAI/bge-reranker-v2-m3 loaded in-process via transformers. Dev venv
          only; requires the `local-embed` pyproject extra. Heavy deps
          (torch, transformers) are lazy-imported inside the class so the
          module itself can be loaded without them.
  tei   — BAAI/bge-reranker-v2-m3 served by a separate TEI container
          (rag-rerank) over HTTP via the native /rerank endpoint.
"""
from __future__ import annotations


import logging
import math
import time

import httpx

from config.settings import (
    RERANKER_MODEL_PATH,
    RERANK_TOP_K,
    RERANKER_MAX_LENGTH,
    RERANKER_BATCH_SIZE,
    RERANKER_PRECISION,
    INFERENCE_BACKEND,
    TEI_RERANK_URL,
    TEI_RERANKER_MODEL,
    TEI_TIMEOUT_SECONDS,
)
from src.platform.observability import get_tracer
from src.vector_db.common import SearchResult
from src.retrieval.common import RankedResult
from src.retrieval.common import ModelLoadError

logger = logging.getLogger("rag.reranker")


# Mapping from precision name → torch dtype attribute name. Stored as
# strings (resolved via getattr at use time) rather than as torch dtype
# objects so the module can be imported even when torch is not installed
# (worker image slimming). int8/int4 are handled via bitsandbytes
# quantization (not a plain dtype) and fall through to the fp32 load path.
_TORCH_DTYPE_NAME_BY_PRECISION = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


class LocalBGEReranker:
    """Reranker using a local BAAI/bge-reranker-v2-m3 model.

    Heavy imports (torch, transformers) are deferred to ``__init__`` so the
    containing module can be imported in environments without those packages
    (e.g., the slim worker image that runs in TEI mode).

    Attributes:
        device: Compute device string ("cuda" or "cpu").
        precision: Precision mode string (e.g., "fp32", "fp16", "bf16").
        tokenizer: Loaded AutoTokenizer for the reranker model.
        model: Loaded AutoModelForSequenceClassification.
    """

    def __init__(
        self,
        model_path: str = RERANKER_MODEL_PATH,
        precision: str = RERANKER_PRECISION,
    ) -> None:
        # Lazy imports — only pay the torch/transformers import cost when
        # actually instantiating the local reranker. Module-level imports
        # would break the slim worker image, which has neither installed.
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch  # stored so rerank() can use inference_mode without re-import

        _t0 = time.perf_counter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = precision
        _dtype_name = _TORCH_DTYPE_NAME_BY_PRECISION.get(precision)
        torch_dtype = getattr(torch, _dtype_name) if _dtype_name else None
        # SDPA = torch.nn.functional.scaled_dot_product_attention. On transformers
        # >=4.38 with XLM-RoBERTa this routes attention ops through PyTorch's
        # fused Flash / Memory-Efficient / math backends, bypassing the eager
        # 4-matmul attention path. Numerical outputs differ within float
        # tolerance from the eager kernel — drift-guarded by the regression
        # check on KG-query top-K doc IDs.
        attn_impl = "sdpa"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if torch_dtype is not None and torch_dtype != torch.float32:
                # Low-precision fast path (fp16/bf16). Load directly in the
                # target dtype to avoid a wasted fp32 materialization.
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                ).to(self.device)
                logger.info(
                    "Reranker loaded in %s precision on %s (attn=%s)",
                    precision, self.device, attn_impl,
                )
            else:
                # fp32 path. int8/int4 also land here until a future iteration
                # wires bitsandbytes.
                if precision not in ("fp32", None) and precision not in _TORCH_DTYPE_NAME_BY_PRECISION:
                    logger.warning(
                        "Reranker precision=%r is declared but not yet wired "
                        "(int8/int4 require bitsandbytes); loading fp32.",
                        precision,
                    )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    attn_implementation=attn_impl,
                ).to(self.device)
                logger.info(
                    "Reranker loaded in fp32 precision on %s (attn=%s)",
                    self.device, attn_impl,
                )
        except Exception as exc:
            logger.error("Failed to load reranker from %r: %s", model_path, exc)
            raise ModelLoadError(
                f"Failed to load reranker model from {model_path!r}: {exc}",
                model_path=model_path,
            ) from exc
        self.model.eval()
        logger.info(
            "Reranker ready in %.1fms (precision=%s, device=%s, max_length=%d, batch_size=%d).",
            (time.perf_counter() - _t0) * 1000,
            precision,
            self.device,
            RERANKER_MAX_LENGTH,
            RERANKER_BATCH_SIZE,
        )

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int = RERANK_TOP_K,
    ) -> list[RankedResult]:
        """Rerank documents against a query.

        Args:
            query: The search query.
            documents: List of SearchResult objects from the database layer.
            top_k: Number of top results to return.

        Returns:
            Sorted list of RankedResult (highest score first).

        Raises:
            RuntimeError: If the model inference step fails (e.g., CUDA OOM,
                tokenizer version mismatch, unexpected output shape).
        """
        # NOTE: torch.inference_mode is applied as a context manager rather
        # than a @torch.inference_mode() decorator on the method. The
        # decorator form runs `torch.inference_mode()` at class-body
        # definition time, which is import time — and CI's torch install
        # intermittently lacks attributes at that point, breaking pytest
        # collection. The context-manager form defers the call to
        # invocation time, which is functionally equivalent.
        with self._torch.inference_mode(), get_tracer().span(
            "reranker.rerank",
            {"input_count": len(documents), "top_k": top_k},
        ) as span:
            if not documents:
                logger.debug("rerank: empty document list, returning empty result")
                return []

            logger.debug(
                "rerank: scoring %d documents (top_k=%d, batch_size=%d)",
                len(documents), top_k, RERANKER_BATCH_SIZE,
            )
            pairs = [[query, doc.text] for doc in documents]

            # Process in fixed-size batches to bound peak VRAM usage when
            # SEARCH_LIMIT is large (RERANKER_BATCH_SIZE, default 32).
            scores: list[float] = []
            for i in range(0, len(pairs), RERANKER_BATCH_SIZE):
                batch = pairs[i : i + RERANKER_BATCH_SIZE]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=RERANKER_MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.squeeze(-1).float()
                # Materialize to Python floats before sigmoid to avoid _Logits
                # proxy limitations in transformers >= 4.50 (no __neg__, no
                # .sigmoid()) and torch.sigmoid removal in torch 2.x.
                scores_raw = logits.cpu().tolist()
                if isinstance(scores_raw, float):
                    scores_raw = [scores_raw]
                batch_scores: list[float] = [1.0 / (1.0 + math.exp(-x)) for x in scores_raw]
                scores.extend(batch_scores)

            results = [
                RankedResult(
                    text=doc.text,
                    score=score,
                    metadata=doc.metadata,
                )
                for doc, score in zip(documents, scores)
            ]

            results.sort(key=lambda r: r.score, reverse=True)
            top_results = results[:top_k]
            if top_results:
                values = [r.score for r in top_results]
                span.set_attribute("score_min", min(values))
                span.set_attribute("score_max", max(values))
                span.set_attribute("output_count", len(top_results))
            return top_results


class TEIReranker:
    """Reranker backed by TEI's native ``/rerank`` endpoint over HTTP.

    TEI returns a JSON array of ``{"index": i, "score": s}`` objects sorted
    by score descending. We preserve that ordering and build RankedResults.
    """

    def __init__(
        self,
        base_url: str = TEI_RERANK_URL,
        model: str = TEI_RERANKER_MODEL,
        timeout: int = TEI_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self.tracer = get_tracer()

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int = RERANK_TOP_K,
    ) -> list[RankedResult]:
        """Rerank documents against a query via TEI /rerank.

        Args:
            query: The search query.
            documents: List of SearchResult objects from the vector DB layer.
            top_k: Number of top results to return.

        Returns:
            Sorted list of RankedResult (highest relevance score first).
        """
        if not documents:
            return []

        with self.tracer.span(
            "reranker.rerank",
            {"input_count": len(documents), "top_k": top_k},
        ) as span:
            resp = self._client.post(
                f"{self.base_url}/rerank",
                json={
                    "query": query,
                    "texts": [d.text for d in documents],
                    "truncate": True,
                },
            )
            resp.raise_for_status()
            raw = resp.json()
            # TEI returns a top-level list, sorted desc by score.
            results = [
                RankedResult(
                    text=documents[item["index"]].text,
                    score=float(item["score"]),
                    metadata=documents[item["index"]].metadata,
                )
                for item in raw[:top_k]
            ]
            if results:
                values = [r.score for r in results]
                span.set_attribute("score_min", min(values))
                span.set_attribute("score_max", max(values))
                span.set_attribute("output_count", len(results))
            return results


def get_reranker_provider():
    """Return the configured reranker provider.

    Reads ``INFERENCE_BACKEND`` from settings:
      - ``"tei"``   → :class:`TEIReranker` (direct HTTP to rag-rerank container)
      - anything else → :class:`LocalBGEReranker` (in-process transformers;
                         dev venv path — requires the `local-embed` pyproject extra)
    """
    if INFERENCE_BACKEND == "tei":
        return TEIReranker()
    return LocalBGEReranker()


__all__ = ["LocalBGEReranker", "TEIReranker", "RankedResult", "get_reranker_provider"]
