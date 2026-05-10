"""Test bootstrap for the lifecycle test suite.

The root conftest (tests/conftest.py) stubs heavy ML dependencies. This file
stubs additional platform-layer modules that are pulled in through the deep
import chain:

  src.ingest -> src.ingest.impl -> src.ingest.embedding -> PIL / colpali / ...

The lifecycle modules under src.ingest.lifecycle.* only depend on:
  - src.ingest.common.schemas   (PIPELINE_SCHEMA_VERSION, ManifestEntry)
  - src.ingest.common.minio_clean_store (MinioCleanStore)
  - src.ingest.lifecycle.schemas
  - src.vector_db (public facade)
  - src.knowledge_graph (public facade)

None of these require the full src.ingest pipeline to be loaded. We pre-stub
src.ingest itself to prevent it from dragging in PIL, colpali, temporal, etc.
"""

from __future__ import annotations

import sys
import types


def _install_lifecycle_stubs() -> None:
    """Install minimal stubs needed to import src.ingest.lifecycle.*."""

    # ------------------------------------------------------------------ #
    # Stub PIL (Pillow)                                                    #
    # ------------------------------------------------------------------ #
    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")

        class Image:
            @staticmethod
            def open(*a, **kw):
                return Image()

            def resize(self, *a, **kw):
                return self

            def convert(self, *a, **kw):
                return self

            def save(self, *a, **kw):
                pass

        pil_image.Image = Image
        pil.Image = Image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image

    # ------------------------------------------------------------------ #
    # Stub prometheus_client                                               #
    # ------------------------------------------------------------------ #
    if "prometheus_client" not in sys.modules:
        prom = types.ModuleType("prometheus_client")

        class _Metric:
            def __init__(self, *a, **kw):
                pass

            def labels(self, **kw):
                return self

            def inc(self, amount=1):
                pass

            def dec(self, amount=1):
                pass

            def set(self, value):
                pass

            def observe(self, value):
                pass

        class Counter(_Metric):
            pass

        class Gauge(_Metric):
            pass

        class Histogram(_Metric):
            def __init__(self, *a, buckets=None, **kw):
                pass

        class Summary(_Metric):
            pass

        prom.Counter = Counter
        prom.Gauge = Gauge
        prom.Histogram = Histogram
        prom.Summary = Summary
        prom.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
        prom.generate_latest = lambda registry=None: b""
        sys.modules["prometheus_client"] = prom

    # ------------------------------------------------------------------ #
    # Stub langfuse                                                        #
    # ------------------------------------------------------------------ #
    if "langfuse" not in sys.modules:
        lf = types.ModuleType("langfuse")

        class Langfuse:
            def __init__(self, *a, **kw):
                pass

            def trace(self, *a, **kw):
                return self

            def flush(self):
                pass

        lf.Langfuse = Langfuse
        lf_dec = types.ModuleType("langfuse.decorators")

        def observe(*a, **kw):
            def _d(fn):
                return fn
            return _d

        lf_dec.observe = observe
        lf_dec.langfuse_context = types.SimpleNamespace(
            update_current_observation=lambda **kw: None,
            update_current_trace=lambda **kw: None,
        )
        sys.modules["langfuse"] = lf
        sys.modules["langfuse.decorators"] = lf_dec
        sys.modules["langfuse.openai"] = types.ModuleType("langfuse.openai")

    # ------------------------------------------------------------------ #
    # Stub redis                                                           #
    # ------------------------------------------------------------------ #
    if "redis" not in sys.modules:
        redis_mod = types.ModuleType("redis")

        class _Redis:
            def __init__(self, *a, **kw):
                pass

            def get(self, key):
                return None

            def set(self, *a, **kw):
                pass

            def delete(self, *keys):
                pass

            def pipeline(self, *a, **kw):
                return self

            def execute(self):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        redis_asyncio = types.ModuleType("redis.asyncio")
        redis_asyncio.Redis = _Redis
        redis_mod.Redis = _Redis
        redis_mod.StrictRedis = _Redis
        redis_mod.asyncio = redis_asyncio
        redis_mod.ConnectionError = ConnectionError
        redis_mod.exceptions = types.SimpleNamespace(
            ConnectionError=ConnectionError, TimeoutError=TimeoutError
        )
        sys.modules["redis"] = redis_mod
        sys.modules["redis.asyncio"] = redis_asyncio

    # ------------------------------------------------------------------ #
    # Stub temporalio                                                      #
    # ------------------------------------------------------------------ #
    if "temporalio" not in sys.modules:
        for mod_name in [
            "temporalio",
            "temporalio.activity",
            "temporalio.workflow",
            "temporalio.client",
            "temporalio.worker",
            "temporalio.exceptions",
            "temporalio.common",
            "temporalio.converter",
        ]:
            sys.modules[mod_name] = types.ModuleType(mod_name)

        def _noop(*a, **kw):
            if len(a) == 1 and callable(a[0]):
                return a[0]

            def _inner(fn):
                return fn

            return _inner

        sys.modules["temporalio.activity"].defn = _noop
        sys.modules["temporalio.workflow"].defn = _noop
        sys.modules["temporalio.workflow"].run = _noop

    # ------------------------------------------------------------------ #
    # Stub nemoguardrails                                                  #
    # ------------------------------------------------------------------ #
    if "nemoguardrails" not in sys.modules:
        for mod_name in [
            "nemoguardrails",
            "nemoguardrails.integrations",
            "nemoguardrails.integrations.langchain",
            "nemoguardrails.integrations.langchain.runnable_rails",
        ]:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # ------------------------------------------------------------------ #
    # Stub colpali_engine (ColQwen2)                                       #
    # ------------------------------------------------------------------ #
    if "colpali_engine" not in sys.modules:
        for mod_name in [
            "colpali_engine",
            "colpali_engine.models",
            "colpali_engine.models.paligemma",
            "colpali_engine.models.paligemma.colqwen2",
            "colpali_engine.models.paligemma.colqwen2.modeling_colqwen2",
            "colpali_engine.models.paligemma.colqwen2.processing_colqwen2",
        ]:
            sys.modules[mod_name] = types.ModuleType(mod_name)

        class _ColQwen2:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def to(self, *a, **kw):
                return self

            def eval(self):
                return self

        class _ColQwen2Processor:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

        sys.modules[
            "colpali_engine.models.paligemma.colqwen2.modeling_colqwen2"
        ].ColQwen2 = _ColQwen2
        sys.modules[
            "colpali_engine.models.paligemma.colqwen2.processing_colqwen2"
        ].ColQwen2Processor = _ColQwen2Processor

    # ------------------------------------------------------------------ #
    # Stub bitsandbytes                                                    #
    # ------------------------------------------------------------------ #
    if "bitsandbytes" not in sys.modules:
        sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")

    # ------------------------------------------------------------------ #
    # Stub docling                                                         #
    # ------------------------------------------------------------------ #
    if "docling" not in sys.modules:
        for mod_name in [
            "docling",
            "docling.document_converter",
            "docling.datamodel",
            "docling.datamodel.base_models",
            "docling.datamodel.pipeline_options",
            "docling.datamodel.document",
            "docling_core",
            "docling_core.types",
            "docling_core.types.doc",
        ]:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # ------------------------------------------------------------------ #
    # Stub pyjwt / jose                                                    #
    # ------------------------------------------------------------------ #
    if "jwt" not in sys.modules:
        jwt_mod = types.ModuleType("jwt")
        jwt_mod.encode = lambda *a, **kw: "stub_token"
        jwt_mod.decode = lambda *a, **kw: {}
        jwt_mod.PyJWKClient = object
        jwt_mod.exceptions = types.SimpleNamespace(
            PyJWTError=Exception,
            DecodeError=Exception,
            InvalidTokenError=Exception,
        )
        sys.modules["jwt"] = jwt_mod

    # ------------------------------------------------------------------ #
    # Stub langdetect                                                      #
    # ------------------------------------------------------------------ #
    if "langdetect" not in sys.modules:
        ld = types.ModuleType("langdetect")
        ld.detect = lambda text: "en"
        sys.modules["langdetect"] = ld

    # ------------------------------------------------------------------ #
    # Stub tree_sitter (KG dep)                                           #
    # ------------------------------------------------------------------ #
    if "tree_sitter" not in sys.modules:
        for mod_name in ["tree_sitter", "tree_sitter_verilog"]:
            sys.modules[mod_name] = types.ModuleType(mod_name)


_install_lifecycle_stubs()
