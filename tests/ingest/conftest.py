"""Ingest-test conftest: install lightweight stubs ONLY when the real dependency
is genuinely absent from the environment.

The previous version checked ``"X" not in sys.modules``, which incorrectly stubbed
real installed packages whenever they hadn't been imported yet at conftest load
time. This broke tests that exercise real PIL / docling behaviour. The fix:
try to import the real module first; only stub on ImportError.

NOTE: datasketch must be imported (and its transitive deps cached) here, at the
top-level ingest conftest, BEFORE lifecycle/conftest.py has a chance to install a
stub ``redis`` module.  datasketch.storage imports redis at module load time; if
the stub is in sys.modules first, the import fails with AttributeError on
redis.client.Pipeline.  Eagerly importing datasketch here ensures the real redis
(and datasketch itself) are cached before any lower-level conftest runs.
"""

import sys
import types

# Pre-cache datasketch and its real redis dependency before any stub conftest
# can shadow redis.  This is safe — datasketch IS installed in the project venv.
try:
    import datasketch  # noqa: F401
except ImportError:
    pass  # datasketch not installed — minhash tests will be skipped by the engine


def _install_ingest_stubs() -> None:
    """Install lightweight stubs for optional dependencies that are not installed."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        prom = types.ModuleType("prometheus_client")
        prom.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

        def _noop_metric(name="", doc="", *args, **kwargs):
            return types.SimpleNamespace(
                labels=lambda **_kw: types.SimpleNamespace(
                    inc=lambda *_a, **_kw2: None,
                    observe=lambda *_a, **_kw2: None,
                    set=lambda *_a, **_kw2: None,
                ),
                inc=lambda *_a, **_kw: None,
                observe=lambda *_a, **_kw: None,
                set=lambda *_a, **_kw: None,
            )

        prom.Counter = _noop_metric
        prom.Gauge = _noop_metric
        prom.Histogram = _noop_metric
        prom.generate_latest = lambda: b""
        sys.modules["prometheus_client"] = prom

    try:
        import PIL  # noqa: F401
        import PIL.Image  # noqa: F401
    except ImportError:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")

        class _FakeImage:
            def __init__(self, *args, **kwargs):
                self.size = (100, 100)
                self.mode = "RGB"

            def tobytes(self):
                return b""

            def save(self, *args, **kwargs):
                pass

        pil_image.Image = _FakeImage
        pil_image.open = lambda *a, **kw: _FakeImage()
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image


_install_ingest_stubs()
