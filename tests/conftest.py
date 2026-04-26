"""Test bootstrap with lightweight stubs for optional heavy dependencies."""

import math
import sys
import types


def _install_stub_modules() -> None:
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    return [0.1, 0.2, 0.3]
                return [[0.1, 0.2, 0.3] for _ in texts]

        st.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = st

    if "langchain_core.embeddings" not in sys.modules:
        lc_core = types.ModuleType("langchain_core")
        lc_embeddings = types.ModuleType("langchain_core.embeddings")

        class Embeddings:
            pass

        lc_embeddings.Embeddings = Embeddings
        sys.modules["langchain_core"] = lc_core
        sys.modules["langchain_core.embeddings"] = lc_embeddings

    if "langchain_text_splitters" not in sys.modules:
        lts = types.ModuleType("langchain_text_splitters")

        class _Doc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=512, chunk_overlap=0, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text):
                if len(text) <= self.chunk_size:
                    return [text]
                return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        class MarkdownHeaderTextSplitter:
            def __init__(self, headers_to_split_on=None, strip_headers=False):
                pass

            def split_text(self, text):
                return [_Doc(text, {})]

        lts.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        lts.MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter
        sys.modules["langchain_text_splitters"] = lts

    if "langgraph.graph" not in sys.modules:
        lg = types.ModuleType("langgraph")
        lg_graph = types.ModuleType("langgraph.graph")
        END = "__end__"

        class _Compiled:
            def __init__(self, entry):
                self._entry = entry

            def invoke(self, state):
                return state

        class StateGraph:
            def __init__(self, *_args, **_kwargs):
                self.entry = None

            def add_node(self, *_args, **_kwargs):
                return

            def set_entry_point(self, entry):
                self.entry = entry

            def add_conditional_edges(self, *_args, **_kwargs):
                return

            def add_edge(self, *_args, **_kwargs):
                return

            def compile(self):
                return _Compiled(self.entry)

        lg_graph.END = END
        lg_graph.StateGraph = StateGraph
        sys.modules["langgraph"] = lg
        sys.modules["langgraph.graph"] = lg_graph

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        def no_grad():
            def _decorator(fn):
                return fn
            return _decorator

        def tensor(value):
            return value

        def exp(value):
            return math.exp(value)

        class _InferenceMode:
            """Context manager stub for torch.inference_mode()."""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def inference_mode():
            return _InferenceMode()

        torch.cuda = _Cuda()
        torch.no_grad = no_grad
        torch.inference_mode = inference_mode
        torch.tensor = tensor
        torch.exp = exp
        # dtype stubs — needed by LocalBGEReranker and similar modules that
        # call getattr(torch, dtype_name) to resolve a precision string.
        torch.float32 = "float32"
        torch.float16 = "float16"
        torch.bfloat16 = "bfloat16"
        sys.modules["torch"] = torch

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _Model:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, *_args, **_kwargs):
                return self

            def eval(self):
                return self

            def __call__(self, **_kwargs):
                class _Logits:
                    def squeeze(self, *_args):
                        return self

                    def float(self):
                        return self

                    def cpu(self):
                        return self

                    def tolist(self):
                        return [0.1, 0.9]

                class _Out:
                    logits = _Logits()

                return _Out()

        class _Tokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def __call__(self, *_args, **_kwargs):
                class _Batch(dict):
                    def to(self, *_args, **_kwargs):
                        return self

                return _Batch()

        transformers.AutoModelForSequenceClassification = _Model
        transformers.AutoTokenizer = _Tokenizer
        sys.modules["transformers"] = transformers

    if "minio" not in sys.modules:
        minio_mod = types.ModuleType("minio")

        class Minio:
            """Lightweight stub for the Minio client used in document storage tests."""

            def __init__(self, *args, **kwargs):
                pass

            def put_object(self, *args, **kwargs):
                pass

            def fput_object(self, *args, **kwargs):
                pass

            def bucket_exists(self, *args, **kwargs):
                return True

            def make_bucket(self, *args, **kwargs):
                pass

        minio_mod.Minio = Minio
        sys.modules["minio"] = minio_mod

        # minio.error sub-module — needed by src/db/minio/store.py
        minio_error_mod = types.ModuleType("minio.error")

        class S3Error(Exception):
            """Stub for minio.error.S3Error.

            The real minio S3Error stores the error code as ``self.code``.
            Production code (e.g. src/db/minio/store.py) checks ``exc.code``
            to distinguish recoverable NoSuchKey errors from real failures.
            """

            def __init__(self, code: str = "", *args, **kwargs):
                super().__init__(code, *args)
                self.code = code

        minio_error_mod.S3Error = S3Error
        minio_mod.error = minio_error_mod
        sys.modules["minio.error"] = minio_error_mod

        # minio.commonconfig — sometimes imported for ObjectWriteResult etc.
        minio_commonconfig = types.ModuleType("minio.commonconfig")
        sys.modules["minio.commonconfig"] = minio_commonconfig

    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")

        class Image:
            size = (0, 0)

            @staticmethod
            def open(*args, **kwargs):
                return Image()

            def convert(self, mode):
                return self

            def tobytes(self):
                return b""

        # `from PIL import Image` resolves to the PIL.Image module; callers
        # then call Image.open(...) at module level, not on the class.
        pil_image.Image = Image
        pil_image.open = Image.open
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image

    if "temporalio" not in sys.modules:
        temporalio = types.ModuleType("temporalio")

        activity_mod = types.ModuleType("temporalio.activity")
        activity_mod.defn = lambda fn=None, **kw: (fn if fn else lambda f: f)

        workflow_mod = types.ModuleType("temporalio.workflow")
        workflow_mod.defn = lambda fn=None, **kw: (fn if fn else lambda f: f)
        workflow_mod.run = lambda fn: fn

        class _UnsafeCtx:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def imports_passed_through():
                return _UnsafeCtx()

        workflow_mod.unsafe = _UnsafeCtx()

        common_mod = types.ModuleType("temporalio.common")
        common_mod.RetryPolicy = type("RetryPolicy", (), {"__init__": lambda self, **kw: None})

        client_mod = types.ModuleType("temporalio.client")
        client_mod.Client = type("Client", (), {
            "connect": staticmethod(lambda *a, **kw: None),
        })

        worker_mod = types.ModuleType("temporalio.worker")
        worker_mod.Worker = type("Worker", (), {"__init__": lambda self, *a, **kw: None})

        service_mod = types.ModuleType("temporalio.service")
        service_mod.RPCError = type("RPCError", (Exception,), {})

        api_mod = types.ModuleType("temporalio.api")
        enums_mod = types.ModuleType("temporalio.api.enums")
        enums_v1_mod = types.ModuleType("temporalio.api.enums.v1")
        enums_v1_mod.TaskQueueType = type("TaskQueueType", (), {})

        taskqueue_mod = types.ModuleType("temporalio.api.taskqueue")
        taskqueue_v1_mod = types.ModuleType("temporalio.api.taskqueue.v1")
        taskqueue_v1_mod.TaskQueue = type("TaskQueue", (), {"__init__": lambda self, **kw: None})

        workflowservice_mod = types.ModuleType("temporalio.api.workflowservice")
        workflowservice_v1_mod = types.ModuleType("temporalio.api.workflowservice.v1")
        workflowservice_v1_mod.DescribeTaskQueueRequest = type("DescribeTaskQueueRequest", (), {"__init__": lambda self, **kw: None})
        workflowservice_v1_mod.GetSystemInfoRequest = type("GetSystemInfoRequest", (), {"__init__": lambda self, **kw: None})

        temporalio.activity = activity_mod
        temporalio.workflow = workflow_mod
        temporalio.common = common_mod
        temporalio.client = client_mod
        temporalio.worker = worker_mod
        temporalio.service = service_mod
        temporalio.api = api_mod
        sys.modules["temporalio"] = temporalio
        sys.modules["temporalio.activity"] = activity_mod
        sys.modules["temporalio.workflow"] = workflow_mod
        sys.modules["temporalio.common"] = common_mod
        sys.modules["temporalio.client"] = client_mod
        sys.modules["temporalio.worker"] = worker_mod
        sys.modules["temporalio.service"] = service_mod
        sys.modules["temporalio.api"] = api_mod
        sys.modules["temporalio.api.enums"] = enums_mod
        sys.modules["temporalio.api.enums.v1"] = enums_v1_mod
        sys.modules["temporalio.api.taskqueue"] = taskqueue_mod
        sys.modules["temporalio.api.taskqueue.v1"] = taskqueue_v1_mod
        sys.modules["temporalio.api.workflowservice"] = workflowservice_mod
        sys.modules["temporalio.api.workflowservice.v1"] = workflowservice_v1_mod

    if "mcp" not in sys.modules:
        mcp = types.ModuleType("mcp")
        mcp_server = types.ModuleType("mcp.server")
        mcp_server_fastmcp = types.ModuleType("mcp.server.fastmcp")

        class FastMCP:
            def __init__(self, *args, **kwargs):
                pass

            def tool(self, *args, **kwargs):
                return lambda fn: fn

            def resource(self, *args, **kwargs):
                return lambda fn: fn

            def run(self, *args, **kwargs):
                pass

        mcp_server_fastmcp.FastMCP = FastMCP
        mcp.server = mcp_server
        mcp_server.fastmcp = mcp_server_fastmcp
        sys.modules["mcp"] = mcp
        sys.modules["mcp.server"] = mcp_server
        sys.modules["mcp.server.fastmcp"] = mcp_server_fastmcp

    if "prometheus_client" not in sys.modules:
        prom = types.ModuleType("prometheus_client")

        class _MetricBase:
            def __init__(self, *args, **kwargs):
                pass

            def labels(self, **kwargs):
                return self

            def inc(self, amount=1):
                pass

            def dec(self, amount=1):
                pass

            def set(self, value):
                pass

            def observe(self, value):
                pass

        class Counter(_MetricBase):
            pass

        class Gauge(_MetricBase):
            pass

        class Histogram(_MetricBase):
            pass

        prom.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
        prom.generate_latest = lambda registry=None: b""
        prom.Counter = Counter
        prom.Gauge = Gauge
        prom.Histogram = Histogram
        sys.modules["prometheus_client"] = prom

    if "weaviate" not in sys.modules:
        weaviate = types.ModuleType("weaviate")

        class _Obj:
            def __init__(self):
                self.properties = {}
                self.metadata = types.SimpleNamespace(score=0.0)

        class _Result:
            objects = [_Obj()]

        class _DataOps:
            @staticmethod
            def delete_many(where=None):
                return types.SimpleNamespace(matches=0)

        class _QueryOps:
            @staticmethod
            def hybrid(**kwargs):
                return _Result()

        class _Batch:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def add_object(self, **_kwargs):
                return

        class _Collection:
            def __init__(self):
                self.query = _QueryOps()
                self.data = _DataOps()

            class batch:
                @staticmethod
                def dynamic():
                    return _Batch()

        class _Collections:
            def __init__(self):
                self._exists = False

            def exists(self, *_args, **_kwargs):
                return self._exists

            def create(self, *_args, **_kwargs):
                self._exists = True

            def get(self, *_args, **_kwargs):
                return _Collection()

            def delete(self, *_args, **_kwargs):
                self._exists = False

        class WeaviateClient:
            def __init__(self):
                self.collections = _Collections()

            def close(self):
                return

        def connect_to_embedded(**kwargs):
            return WeaviateClient()

        def connect_to_local(**kwargs):
            return WeaviateClient()

        weaviate.connect_to_embedded = connect_to_embedded
        weaviate.connect_to_local = connect_to_local
        weaviate.WeaviateClient = WeaviateClient
        sys.modules["weaviate"] = weaviate

        config_mod = types.ModuleType("weaviate.classes.config")
        query_mod = types.ModuleType("weaviate.classes.query")

        class Configure:
            class Vectorizer:
                @staticmethod
                def none():
                    return None

        class Property:
            def __init__(self, *args, **kwargs):
                pass

        class DataType:
            TEXT = "text"
            INT = "int"

        class Filter:
            @staticmethod
            def by_property(_name):
                class _F:
                    def equal(self, _value):
                        return self

                    def __and__(self, _other):
                        return self

                return _F()

        class HybridFusion:
            RELATIVE_SCORE = "relative"

        class MetadataQuery:
            def __init__(self, **kwargs):
                pass

        config_mod.Configure = Configure
        config_mod.Property = Property
        config_mod.DataType = DataType
        query_mod.Filter = Filter
        query_mod.HybridFusion = HybridFusion
        query_mod.MetadataQuery = MetadataQuery
        sys.modules["weaviate.classes.config"] = config_mod
        sys.modules["weaviate.classes.query"] = query_mod

    if "colpali_engine" not in sys.modules:
        colpali = types.ModuleType("colpali_engine")
        sys.modules["colpali_engine"] = colpali

    if "bitsandbytes" not in sys.modules:
        bnb = types.ModuleType("bitsandbytes")
        sys.modules["bitsandbytes"] = bnb

    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")

        class Image:
            """Minimal PIL.Image stub."""

            LANCZOS = 1

            @staticmethod
            def open(*args, **kwargs):
                return Image()

            @staticmethod
            def new(*args, **kwargs):
                return Image()

            def convert(self, *args, **kwargs):
                return self

            def resize(self, *args, **kwargs):
                return self

            def save(self, *args, **kwargs):
                pass

            def tobytes(self, *args, **kwargs):
                return b""

            @property
            def size(self):
                return (100, 100)

        pil_image.Image = Image
        pil_image.LANCZOS = Image.LANCZOS
        pil.Image = pil_image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image

    if "prometheus_client" not in sys.modules:
        prom = types.ModuleType("prometheus_client")

        CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

        def generate_latest(*args, **kwargs):
            return b""

        class _MetricBase:
            """Lightweight stub for Counter, Gauge, Histogram."""

            def __init__(self, *args, **kwargs):
                pass

            def labels(self, *args, **kwargs):
                return self

            def inc(self, *args, **kwargs):
                pass

            def dec(self, *args, **kwargs):
                pass

            def set(self, *args, **kwargs):
                pass

            def observe(self, *args, **kwargs):
                pass

            def time(self):
                import contextlib
                return contextlib.nullcontext()

        prom.CONTENT_TYPE_LATEST = CONTENT_TYPE_LATEST
        prom.generate_latest = generate_latest
        prom.Counter = _MetricBase
        prom.Gauge = _MetricBase
        prom.Histogram = _MetricBase
        prom.Summary = _MetricBase
        sys.modules["prometheus_client"] = prom

    # httpx: prefer the real package if it's installed (litellm + openai depend
    # on it internally, so stubbing over the top breaks their imports). The
    # stub is only installed when real httpx is unavailable (minimal CI
    # environments that skip the core deps).
    try:
        import httpx as _real_httpx  # noqa: F401
        _has_real_httpx = True
    except Exception:
        _has_real_httpx = False
    if not _has_real_httpx and "httpx" not in sys.modules:
        # Shape-compatible stub for TEI (TEIEmbeddings / TEIReranker).
        # /v1/embeddings  → {"data": [{"embedding": [...]}, ...]}
        # /rerank         → [{"index": i, "score": s}, ...]  (sorted desc)
        # Deterministic small vectors so tests can assert on dimensions.
        httpx_stub = types.ModuleType("httpx")

        class _StubResponse:
            def __init__(self, payload):
                self._payload = payload
                self.status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class _StubClient:
            def __init__(self, *_a, **_kw):
                pass

            def post(self, url, *, json=None, **_kw):
                json = json or {}
                if "/rerank" in url:
                    texts = json.get("texts", []) or []
                    # TEI returns sorted desc; produce stable decreasing scores.
                    return _StubResponse(
                        [{"index": i, "score": 1.0 - i * 0.01} for i in range(len(texts))]
                    )
                # Embeddings path — accept a str or list under "input".
                inputs = json.get("input", [])
                if isinstance(inputs, str):
                    inputs = [inputs]
                return _StubResponse(
                    {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in inputs]}
                )

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class _HTTPError(Exception):
            pass

        httpx_stub.Client = _StubClient
        httpx_stub.AsyncClient = _StubClient  # close-enough for tests
        httpx_stub.HTTPError = _HTTPError
        httpx_stub.HTTPStatusError = _HTTPError
        httpx_stub.RequestError = _HTTPError
        httpx_stub.Response = _StubResponse
        sys.modules["httpx"] = httpx_stub


_install_stub_modules()
