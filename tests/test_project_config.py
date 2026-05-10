# @summary
# Static validation tests for project configuration, dependency declarations,
# README installation instructions, and environment-driven config usage.
# Verifies all third-party imports are declared, ports/URLs are configurable
# via env vars, and README documents installation steps.
# Also contains exhaustive runtime tests for validate_all_config() and
# validate_visual_retrieval_config() covering every validation rule enforced.
# Exports: TestDependencyCompleteness, TestReadmeInstallation,
#          TestConfigNotHardcoded, TestDockerfileBuildDeps, TestValidateAllConfig,
#          TestValidateVisualRetrievalConfig
# Deps: pytest, pathlib, re, ast, tomllib, config.settings
# @end-summary
"""
Static project configuration tests.

These tests verify file contents and structure — no runtime dependencies needed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import config.settings as _settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard library modules (Python 3.10+) — not third-party
_STDLIB_MODULES = frozenset({
    "__future__", "abc", "aifc", "argparse", "array", "ast", "asyncio",
    "atexit", "base64", "binascii", "bisect", "builtins", "bz2",
    "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code",
    "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "cProfile", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils",
    "doctest", "email", "encodings", "enum", "errno", "faulthandler",
    "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
    "glob", "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "importlib",
    "inspect", "io", "ipaddress", "itertools", "json", "keyword",
    "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder",
    "multiprocessing", "netrc", "nis", "nntplib", "numbers",
    "operator", "optparse", "os", "ossaudiodev", "pathlib", "pdb",
    "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil",
    "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
    "socketserver", "spwd", "sqlite3", "ssl", "stat", "statistics",
    "string", "stringprep", "struct", "subprocess", "sunau",
    "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile",
    "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "tomllib", "trace", "traceback", "tracemalloc", "tty", "turtle",
    "turtledemo", "types", "typing", "unicodedata", "unittest",
    "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
    "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
    # Internal/private
    "_thread", "_io", "_collections_abc", "_csv", "typing_extensions",
})

# Mapping from pyproject.toml package name -> importable module name(s)
_PYPROJECT_IMPORT_MAP: dict[str, set[str]] = {
    "weaviate-client": {"weaviate"},
    "httpx": {"httpx"},
    # torch + sentence-transformers + transformers + accelerate are in the
    # [local-embed] optional extra, not core deps. Listed here so the audit
    # recognizes them when someone has the extra installed; they're also
    # valid core imports for the in-process `local` inference backend.
    "sentence-transformers": {"sentence_transformers"},
    "transformers": {"transformers"},
    "torch": {"torch"},
    "networkx": {"networkx"},
    "langchain-core": {"langchain_core"},
    "langchain-text-splitters": {"langchain_text_splitters"},
    "langgraph": {"langgraph"},
    "numpy": {"numpy"},
    "temporalio": {"temporalio"},
    "langfuse": {"langfuse"},
    "fastapi": {"fastapi", "starlette"},  # starlette is a fastapi sub-dependency
    "uvicorn": {"uvicorn"},
    "pydantic": {"pydantic"},
    "accelerate": {"accelerate"},
    "prometheus-client": {"prometheus_client"},
    "redis": {"redis"},
    "pyjwt": {"jwt"},
    "mcp": {"mcp"},
    "docling": {"docling"},
    "nemoguardrails": {"nemoguardrails"},
    "litellm": {"litellm"},
    "pyyaml": {"yaml"},
    "orjson": {"orjson"},
    "minio": {"minio"},
    "langdetect": {"langdetect"},
    # Visual embedding optional deps
    "Pillow": {"PIL"},
    "bitsandbytes": {"bitsandbytes"},
    "colpali-engine": {"colpali_engine"},
    # Transitive deps (stable bundled)
    "docling-core": {"docling_core"},
    # KG optional deps
    "igraph": {"igraph"},
    "leidenalg": {"leidenalg"},
    "neo4j": {"neo4j"},
    "pyverilog": {"pyverilog"},
    "tree-sitter": {"tree_sitter"},
    "tree-sitter-verilog": {"tree_sitter_verilog"},
    "tree-sitter-python": {"tree_sitter_python"},
    "datasketch": {"datasketch"},
    "markdownify": {"markdownify"},
    "pypandoc": {"pypandoc"},
    # Dev deps
    "pytest": {"pytest"},
    "pytest-mock": {"pytest_mock"},
    "import-mend": {"import_mend"},
}

# All importable names that pyproject.toml covers
_DECLARED_IMPORTS = frozenset(
    name for names in _PYPROJECT_IMPORT_MAP.values() for name in names
)

# Optional dependencies that are conditionally/lazily imported inside functions.
# These are NOT required at install time — they are loaded only when specific
# features are enabled (e.g., GLiNER entity extraction, PII detection, alt vector stores).
_OPTIONAL_IMPORTS = frozenset({
    "chromadb", "pinecone", "qdrant_client",  # alternative vector stores (query.py)
    "gliner",                                  # entity extraction (src/core/knowledge_graph.py)
    "presidio_analyzer", "presidio_anonymizer", "spacy",  # PII detection (src/guardrails/pii.py)
})

# Root-level scripts and local packages that are importable but not third-party packages
_LOCAL_ROOT_MODULES = frozenset({"ingest", "query", "cli", "colang_demo"})


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def pyproject_text(project_root: Path) -> str:
    return (project_root / "pyproject.toml").read_text()


@pytest.fixture
def readme_text(project_root: Path) -> str:
    return (project_root / "README.md").read_text()


@pytest.fixture
def settings_text(project_root: Path) -> str:
    return (project_root / "config" / "settings.py").read_text()


@pytest.fixture
def env_example_text(project_root: Path) -> str:
    return (project_root / ".env.example").read_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pyproject_deps(text: str) -> list[str]:
    """Extract dependency names from pyproject.toml dependencies list."""
    deps: list[str] = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies = ["):
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                break
            # Extract bare package name from "package>=version",
            match = re.match(r'"([a-zA-Z0-9_-]+)', stripped)
            if match:
                deps.append(match.group(1).lower())
    return deps




def _collect_third_party_imports(source_dirs: list[Path]) -> set[str]:
    """Walk Python files and return top-level third-party import names."""
    third_party: set[str] = set()
    local_prefixes = {"src", "server", "config", "tests"}

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        py_files = list(source_dir.rglob("*.py"))
        for py_file in py_files:
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in _STDLIB_MODULES and top not in local_prefixes:
                            third_party.add(top)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top not in _STDLIB_MODULES and top not in local_prefixes:
                            third_party.add(top)
    return third_party


# =========================================================================
# Test Class A: Dependency Completeness
# =========================================================================

class TestDependencyCompleteness:
    """Verify that all third-party imports have matching pyproject.toml entries."""

    def test_all_imports_are_declared(self, project_root: Path):
        """Every third-party import in src/server/config/scripts should be
        covered by a pyproject.toml dependency (direct or transitive)."""
        source_dirs = [
            project_root / "src",
            project_root / "server",
            project_root / "config",
            project_root / "scripts",
        ]
        # Also check root-level .py files
        root_py = list(project_root.glob("*.py"))
        # We handle root .py files by adding them to a temp set
        third_party = _collect_third_party_imports(source_dirs)

        # Root-level .py files
        for py_file in root_py:
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top not in _STDLIB_MODULES and top not in {"src", "server", "config", "tests"}:
                            third_party.add(top)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top not in _STDLIB_MODULES and top not in {"src", "server", "config", "tests"}:
                            third_party.add(top)

        undeclared = third_party - _DECLARED_IMPORTS - _OPTIONAL_IMPORTS - _LOCAL_ROOT_MODULES
        assert not undeclared, (
            f"Third-party imports not covered by pyproject.toml: {sorted(undeclared)}. "
            "Add them to [project.dependencies] or update _PYPROJECT_IMPORT_MAP."
        )

    def test_pyproject_has_python_version(self, pyproject_text: str):
        """pyproject.toml should declare a minimum Python version."""
        assert "requires-python" in pyproject_text

    def test_pyproject_has_build_system(self, pyproject_text: str):
        """pyproject.toml should declare a build system."""
        assert "[build-system]" in pyproject_text


# =========================================================================
# Test Class B: README Installation Instructions
# =========================================================================

class TestReadmeInstallation:
    """Verify README.md has complete installation/setup instructions."""

    def test_readme_has_prerequisites_section(self, readme_text: str):
        """README should list prerequisites."""
        assert re.search(r"#+\s+Prerequisites", readme_text), (
            "README.md should have a 'Prerequisites' section"
        )

    def test_readme_mentions_python_version(self, readme_text: str):
        """README should specify the minimum Python version."""
        assert re.search(r"Python\s+3\.1[0-9]", readme_text), (
            "README.md should specify a Python version requirement (3.10+)"
        )

    def test_readme_has_install_command(self, readme_text: str):
        """README should include pip or uv install command."""
        has_pip = "pip install" in readme_text
        has_uv = "uv pip install" in readme_text or "uv venv" in readme_text
        assert has_pip or has_uv, (
            "README.md should include a pip or uv install command"
        )

    def test_readme_mentions_env_setup(self, readme_text: str):
        """README should tell users to copy .env.example."""
        assert ".env.example" in readme_text, (
            "README.md should reference .env.example for environment setup"
        )

    def test_readme_has_compose_start_command(self, readme_text: str):
        """README should show how to start infrastructure services."""
        assert "compose.sh" in readme_text or "compose" in readme_text.lower(), (
            "README.md should show how to start container services"
        )

    def test_readme_has_run_section(self, readme_text: str):
        """README should have a section for running the application."""
        has_run = re.search(r"#+\s+(Run|Quick Start|Getting Started)", readme_text)
        assert has_run, "README.md should have a 'Run' or 'Quick Start' section"

    def test_readme_documents_container_profiles(self, readme_text: str):
        """README should document available container profiles."""
        assert "profile" in readme_text.lower() and (
            "app" in readme_text and "workers" in readme_text
        ), "README.md should document container profiles (app, workers, etc.)"

    def test_readme_has_test_instructions(self, readme_text: str):
        """README should include instructions for running tests."""
        assert "pytest" in readme_text, (
            "README.md should include test running instructions (pytest)"
        )

    def test_readme_documents_podman_setup(self, readme_text: str):
        """README should have Podman setup instructions."""
        assert re.search(r"(?i)podman\s+setup", readme_text), (
            "README.md should have a Podman setup section"
        )

    def test_readme_entry_points_match_actual_files(self, project_root: Path, readme_text: str):
        """Entry point files referenced in README should exist."""
        expected_files = ["ingest.py", "query.py", "cli.py"]
        for filename in expected_files:
            filepath = project_root / filename
            if filename in readme_text:
                assert filepath.exists(), (
                    f"README references '{filename}' but the file does not exist"
                )


# =========================================================================
# Test Class D: Config Not Hardcoded
# =========================================================================

class TestConfigNotHardcoded:
    """Verify that ports and URLs in source code use config/env, not hardcoded values."""

    def test_settings_ports_from_env(self, settings_text: str):
        """config/settings.py should read port numbers from environment variables."""
        # These port variables should use os.environ.get
        port_vars = ["RAG_API_PORT", "RAG_OLLAMA_PORT", "RAG_REDIS_PORT", "RAG_TEMPORAL_PORT"]
        for var in port_vars:
            assert var in settings_text, (
                f"config/settings.py should reference env var {var}"
            )

    def test_settings_urls_from_env(self, settings_text: str):
        """config/settings.py should build URLs from env vars, not hardcode them."""
        url_vars = ["RAG_API_URL", "RAG_OLLAMA_URL", "RAG_CACHE_REDIS_URL", "RAG_TEMPORAL_TARGET_HOST"]
        for var in url_vars:
            assert var in settings_text, (
                f"config/settings.py should reference env var {var}"
            )

    def test_settings_uses_environ_get(self, settings_text: str):
        """config/settings.py should use os.environ.get for configuration."""
        count = settings_text.count("os.environ.get")
        assert count >= 20, (
            f"config/settings.py uses os.environ.get only {count} times — "
            "expected at least 20 for full env-driven config"
        )

    def test_api_port_from_env(self, project_root: Path):
        """server/api.py should read API port from environment, not hardcode it."""
        api_text = (project_root / "server" / "api.py").read_text()
        assert "os.environ.get" in api_text or "RAG_API_PORT" in api_text, (
            "server/api.py should read port from environment variable"
        )

    def test_cli_client_url_from_env(self, project_root: Path):
        """server/cli_client.py should read server URL from environment."""
        cli_text = (project_root / "server" / "cli_client.py").read_text()
        assert "RAG_API_URL" in cli_text or "RAG_API_PORT" in cli_text, (
            "server/cli_client.py should read API URL/port from environment"
        )

    def test_no_hardcoded_localhost_urls_in_source(self, project_root: Path):
        """Source files should not hardcode localhost URLs without env var fallback.

        We check that any http://localhost:NNNN pattern appears within an
        os.environ.get() call or f-string with a config variable.
        Skips comments and docstrings."""
        source_dirs = [project_root / "src", project_root / "server"]
        violations: list[str] = []

        url_pattern = re.compile(r'http://localhost:\d{4,5}')
        env_pattern = re.compile(r'os\.environ\.get|f".*\{.*\}|f\'.*\{.*\}')

        for source_dir in source_dirs:
            for py_file in source_dir.rglob("*.py"):
                source = py_file.read_text()
                lines = source.splitlines()

                # Use AST to find docstring line ranges to skip
                docstring_lines: set[int] = set()
                try:
                    tree = ast.parse(source, filename=str(py_file))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            for ln in range(node.lineno, node.end_lineno + 1):
                                docstring_lines.add(ln)
                except SyntaxError:
                    continue

                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    # Skip comments and docstrings
                    if stripped.startswith("#") or i in docstring_lines:
                        continue
                    if url_pattern.search(line) and not env_pattern.search(line):
                        violations.append(f"{py_file.relative_to(project_root)}:{i}")

        assert not violations, (
            f"Hardcoded localhost URLs found without env var fallback:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_env_example_has_all_port_vars(self, env_example_text: str):
        """'.env.example' should document all port variables."""
        expected_ports = [
            "RAG_API_PORT",
            "RAG_API_HOST_PORT",
            "RAG_OLLAMA_PORT",
            "RAG_REDIS_PORT",
            "RAG_TEMPORAL_PORT",
            "RAG_TEMPORAL_UI_PORT",
        ]
        for var in expected_ports:
            assert var in env_example_text, (
                f".env.example should document {var}"
            )

    def test_env_example_has_derived_urls(self, env_example_text: str):
        """.env.example should show derived URL variables."""
        expected_urls = ["RAG_API_URL", "RAG_LLM_API_BASE"]
        for var in expected_urls:
            assert var in env_example_text, (
                f".env.example should document {var}"
            )


# =========================================================================
# Test Class E: Dockerfile Build Dependencies
# =========================================================================

class TestDockerfileBuildDeps:
    """Verify Dockerfiles install necessary build tools for pip install."""

    def test_api_dockerfile_has_pip_install(self, project_root: Path):
        """Dockerfile.api should install Python dependencies via pip."""
        text = (project_root / "containers" / "Dockerfile.api").read_text()
        assert "pip install" in text, "Dockerfile.api should install pip dependencies"

    def test_runtime_dockerfile_has_pip_install(self, project_root: Path):
        """Dockerfile.runtime should install Python dependencies via pip."""
        text = (project_root / "containers" / "Dockerfile.runtime").read_text()
        assert "pip install" in text, "Dockerfile.runtime should install pip dependencies"

    def test_api_dockerfile_copies_pyproject(self, project_root: Path):
        """Dockerfile.api should copy pyproject.toml for dependency installation."""
        text = (project_root / "containers" / "Dockerfile.api").read_text()
        assert "pyproject.toml" in text, (
            "Dockerfile.api should COPY pyproject.toml for pip install"
        )

    def test_runtime_dockerfile_copies_pyproject(self, project_root: Path):
        """Dockerfile.runtime should copy pyproject.toml for dependency installation."""
        text = (project_root / "containers" / "Dockerfile.runtime").read_text()
        assert "pyproject.toml" in text, (
            "Dockerfile.runtime should COPY pyproject.toml for pip install"
        )

    def test_api_dockerfile_has_curl(self, project_root: Path):
        """Dockerfile.api should install curl for health checks."""
        text = (project_root / "containers" / "Dockerfile.api").read_text()
        assert "curl" in text, "Dockerfile.api should install curl for HEALTHCHECK"

    def test_api_dockerfile_uses_env_for_port(self, project_root: Path):
        """Dockerfile.api should use RAG_API_PORT env var, not hardcode port."""
        text = (project_root / "containers" / "Dockerfile.api").read_text()
        assert "RAG_API_PORT" in text, (
            "Dockerfile.api should use RAG_API_PORT environment variable"
        )


# =========================================================================
# Shared helpers for validate_*_config() tests
# =========================================================================

def _patch_and_call(monkeypatch, overrides: dict, fn) -> None:
    """Apply attribute overrides to _settings, then invoke fn."""
    for attr, val in overrides.items():
        monkeypatch.setattr(_settings, attr, val)
    fn()


def _assert_valid(monkeypatch, overrides: dict, fn) -> None:
    """Apply overrides and assert fn() raises no exception."""
    _patch_and_call(monkeypatch, overrides, fn)


def _assert_invalid(monkeypatch, overrides: dict, fn) -> None:
    """Apply overrides and assert fn() raises ValueError."""
    with pytest.raises(ValueError):
        _patch_and_call(monkeypatch, overrides, fn)


# =========================================================================
# Test Class F: validate_all_config() — exhaustive rule coverage
# =========================================================================


class TestValidateAllConfig:
    """Exhaustive tests for every rule in validate_all_config().

    Each rule gets at least one passing case and one failing case.
    monkeypatch.setattr is used to override module-level settings values
    without touching environment variables or reloading the module.
    """

    @staticmethod
    def _ok(monkeypatch, overrides: dict):
        _assert_valid(monkeypatch, overrides, _settings.validate_all_config)

    @staticmethod
    def _fail(monkeypatch, overrides: dict):
        _assert_invalid(monkeypatch, overrides, _settings.validate_all_config)

    # ------------------------------------------------------------------
    # Rule 1: Confidence threshold ordering — HIGH >= LOW
    # ------------------------------------------------------------------

    def test_confidence_threshold_ordering_valid(self, monkeypatch):
        """HIGH=0.8, LOW=0.3 satisfies the ordering constraint."""
        self._ok(monkeypatch, {
            "RAG_CONFIDENCE_HIGH_THRESHOLD": 0.8,
            "RAG_CONFIDENCE_LOW_THRESHOLD": 0.3,
        })

    def test_confidence_threshold_ordering_equal_valid(self, monkeypatch):
        """HIGH == LOW is a valid edge case (ordering is satisfied)."""
        self._ok(monkeypatch, {
            "RAG_CONFIDENCE_HIGH_THRESHOLD": 0.5,
            "RAG_CONFIDENCE_LOW_THRESHOLD": 0.5,
        })

    def test_confidence_threshold_ordering_invalid(self, monkeypatch):
        """HIGH < LOW must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_CONFIDENCE_HIGH_THRESHOLD": 0.2,
            "RAG_CONFIDENCE_LOW_THRESHOLD": 0.8,
        })

    # ------------------------------------------------------------------
    # Rule 2: Quality threshold ordering — STRONG >= MODERATE >= WEAK
    # ------------------------------------------------------------------

    def test_quality_threshold_ordering_valid(self, monkeypatch):
        """STRONG=0.9, MODERATE=0.6, WEAK=0.3 satisfies the ordering."""
        self._ok(monkeypatch, {
            "RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD": 0.9,
            "RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD": 0.6,
            "RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD": 0.3,
        })

    def test_quality_threshold_strong_equals_moderate_valid(self, monkeypatch):
        """STRONG == MODERATE is still valid."""
        self._ok(monkeypatch, {
            "RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD": 0.5,
            "RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD": 0.5,
            "RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD": 0.2,
        })

    @pytest.mark.parametrize("strong,moderate,weak", [
        (0.4, 0.6, 0.2),   # STRONG < MODERATE
        (0.9, 0.3, 0.5),   # MODERATE < WEAK
        (0.3, 0.6, 0.9),   # all inverted
    ])
    def test_quality_threshold_ordering_invalid(self, monkeypatch, strong, moderate, weak):
        """Any violation of STRONG >= MODERATE >= WEAK must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD": strong,
            "RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD": moderate,
            "RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD": weak,
        })

    # ------------------------------------------------------------------
    # Rule 3: All timeouts must be > 0
    # ------------------------------------------------------------------

    _TIMEOUT_ATTRS = [
        "RAG_WORKFLOW_DEFAULT_TIMEOUT_MS",
        "RAG_RETRIEVAL_TIMEOUT_MS",
        "QUERY_PROCESSING_TIMEOUT",
        "RAG_INGESTION_LLM_TIMEOUT_SECONDS",
        "RAG_INGESTION_VISION_TIMEOUT_SECONDS",
        "RAG_NEMO_RAIL_TIMEOUT_SECONDS",
    ]

    def test_all_timeouts_positive_valid(self, monkeypatch):
        """All timeouts set to 1 (minimal positive) should pass."""
        overrides = {attr: 1 for attr in self._TIMEOUT_ATTRS}
        self._ok(monkeypatch, overrides)

    @pytest.mark.parametrize("attr", _TIMEOUT_ATTRS)
    def test_timeout_zero_invalid(self, monkeypatch, attr):
        """Setting any single timeout to 0 must raise ValueError."""
        self._fail(monkeypatch, {attr: 0})

    @pytest.mark.parametrize("attr", _TIMEOUT_ATTRS)
    def test_timeout_negative_invalid(self, monkeypatch, attr):
        """Setting any single timeout to -1 must raise ValueError."""
        self._fail(monkeypatch, {attr: -1})

    # ------------------------------------------------------------------
    # Rule 4: Port in [1, 65535]
    # ------------------------------------------------------------------

    def test_port_valid_low(self, monkeypatch):
        """Port=1 is the minimum valid value."""
        self._ok(monkeypatch, {"RAG_API_PORT": 1})

    def test_port_valid_high(self, monkeypatch):
        """Port=65535 is the maximum valid value."""
        self._ok(monkeypatch, {"RAG_API_PORT": 65535})

    def test_port_valid_typical(self, monkeypatch):
        """Port=8000 is a typical valid value."""
        self._ok(monkeypatch, {"RAG_API_PORT": 8000})

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_port_invalid(self, monkeypatch, port):
        """Ports outside [1, 65535] must raise ValueError."""
        self._fail(monkeypatch, {"RAG_API_PORT": port})

    # ------------------------------------------------------------------
    # Rule 5: 0.0–1.0 range checks
    # ------------------------------------------------------------------

    _FLOAT_01_ATTRS = [
        "QUERY_CONFIDENCE_THRESHOLD",
        "SEMANTIC_SIMILARITY_THRESHOLD",
        "RAG_NEMO_TOXICITY_THRESHOLD",
        "RAG_NEMO_FAITHFULNESS_THRESHOLD",
        "RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD",
        "RAG_NEMO_PII_SCORE_THRESHOLD",
        "RAG_CONFIDENCE_HIGH_THRESHOLD",
        "RAG_CONFIDENCE_LOW_THRESHOLD",
        "RAG_CONFIDENCE_RETRIEVAL_WEIGHT",
        "RAG_CONFIDENCE_LLM_WEIGHT",
        "RAG_CONFIDENCE_CITATION_WEIGHT",
    ]

    def test_float_thresholds_all_valid(self, monkeypatch):
        """All float thresholds set to 0.5 should pass."""
        overrides = {attr: 0.5 for attr in self._FLOAT_01_ATTRS}
        # Ensure ordering constraints are also satisfied
        overrides["RAG_CONFIDENCE_HIGH_THRESHOLD"] = 0.7
        overrides["RAG_CONFIDENCE_LOW_THRESHOLD"] = 0.3
        self._ok(monkeypatch, overrides)

    def test_float_threshold_boundary_zero_valid(self, monkeypatch):
        """Threshold value of 0.0 is the inclusive lower bound — must pass."""
        # Only test attributes without ordering constraints against other attrs
        self._ok(monkeypatch, {"SEMANTIC_SIMILARITY_THRESHOLD": 0.0})

    def test_float_threshold_boundary_one_valid(self, monkeypatch):
        """Threshold value of 1.0 is the inclusive upper bound — must pass."""
        self._ok(monkeypatch, {"SEMANTIC_SIMILARITY_THRESHOLD": 1.0})

    @pytest.mark.parametrize("attr", _FLOAT_01_ATTRS)
    def test_float_threshold_below_zero_invalid(self, monkeypatch, attr):
        """Any threshold set to -0.1 must raise ValueError."""
        # For confidence ordering attrs, ensure the pair doesn't accidentally satisfy ordering
        overrides = {attr: -0.1}
        if attr == "RAG_CONFIDENCE_HIGH_THRESHOLD":
            overrides["RAG_CONFIDENCE_LOW_THRESHOLD"] = -0.2
        elif attr == "RAG_CONFIDENCE_LOW_THRESHOLD":
            overrides["RAG_CONFIDENCE_HIGH_THRESHOLD"] = 0.5
        self._fail(monkeypatch, overrides)

    @pytest.mark.parametrize("attr", _FLOAT_01_ATTRS)
    def test_float_threshold_above_one_invalid(self, monkeypatch, attr):
        """Any threshold set to 1.1 must raise ValueError."""
        overrides = {attr: 1.1}
        if attr == "RAG_CONFIDENCE_LOW_THRESHOLD":
            overrides["RAG_CONFIDENCE_HIGH_THRESHOLD"] = 1.1
        self._fail(monkeypatch, overrides)

    # ------------------------------------------------------------------
    # Rule 9: Positive integers
    # ------------------------------------------------------------------

    _POSITIVE_INT_ATTRS = [
        "GENERATION_MAX_TOKENS",
        "LLM_MAX_TOKENS",
        "RAG_WORKER_CONCURRENCY",
        "RATE_LIMIT_WINDOW_SECONDS",
        "RERANKER_BATCH_SIZE",
        "RAG_INGESTION_LLM_MAX_KEYWORDS",
    ]

    def test_positive_integers_valid(self, monkeypatch):
        """All positive-int settings set to 1 should pass."""
        overrides = {attr: 1 for attr in self._POSITIVE_INT_ATTRS}
        self._ok(monkeypatch, overrides)

    @pytest.mark.parametrize("attr", _POSITIVE_INT_ATTRS)
    def test_positive_integer_zero_invalid(self, monkeypatch, attr):
        """Setting any positive-integer setting to 0 must raise ValueError."""
        self._fail(monkeypatch, {attr: 0})

    @pytest.mark.parametrize("attr", _POSITIVE_INT_ATTRS)
    def test_positive_integer_negative_invalid(self, monkeypatch, attr):
        """Setting any positive-integer setting to -5 must raise ValueError."""
        self._fail(monkeypatch, {attr: -5})

    # ------------------------------------------------------------------
    # Vision feature dependency checks
    # ------------------------------------------------------------------

    def test_vision_enabled_with_provider_and_model_valid(self, monkeypatch):
        """Vision enabled with both provider and model set should pass."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISION_ENABLED": True,
            "RAG_INGESTION_VISION_PROVIDER": "openai",
            "RAG_INGESTION_VISION_MODEL": "gpt-4o",
        })

    def test_vision_disabled_no_provider_valid(self, monkeypatch):
        """Vision disabled with no provider/model is valid."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISION_ENABLED": False,
            "RAG_INGESTION_VISION_PROVIDER": "",
            "RAG_INGESTION_VISION_MODEL": "",
        })

    def test_vision_enabled_missing_provider_invalid(self, monkeypatch):
        """Vision enabled but no provider must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_INGESTION_VISION_ENABLED": True,
            "RAG_INGESTION_VISION_PROVIDER": "",
            "RAG_INGESTION_VISION_MODEL": "gpt-4o",
        })

    def test_vision_enabled_missing_model_invalid(self, monkeypatch):
        """Vision enabled but no model must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_INGESTION_VISION_ENABLED": True,
            "RAG_INGESTION_VISION_PROVIDER": "openai",
            "RAG_INGESTION_VISION_MODEL": "",
        })

    def test_vision_enabled_missing_both_invalid(self, monkeypatch):
        """Vision enabled but neither provider nor model must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_INGESTION_VISION_ENABLED": True,
            "RAG_INGESTION_VISION_PROVIDER": "",
            "RAG_INGESTION_VISION_MODEL": "",
        })

    # ------------------------------------------------------------------
    # Passing baseline — unmodified defaults must pass
    # ------------------------------------------------------------------

    def test_default_config_passes(self):
        """The module's default (env-derived) values must pass validation."""
        _settings.validate_all_config()


# =========================================================================
# Test Class G: validate_visual_retrieval_config()
# =========================================================================


class TestValidateVisualRetrievalConfig:
    """Tests for validate_visual_retrieval_config() — called when visual
    retrieval is enabled."""

    @staticmethod
    def _ok(monkeypatch, overrides: dict):
        _assert_valid(monkeypatch, overrides, _settings.validate_visual_retrieval_config)

    @staticmethod
    def _fail(monkeypatch, overrides: dict):
        _assert_invalid(monkeypatch, overrides, _settings.validate_visual_retrieval_config)

    # ------------------------------------------------------------------
    # Target collection must be non-empty
    # ------------------------------------------------------------------

    def test_target_collection_non_empty_valid(self, monkeypatch):
        """Non-empty target collection with valid thresholds should pass."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "MyCollection",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.5,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 3600,
        })

    def test_target_collection_empty_invalid(self, monkeypatch):
        """Empty target collection must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.5,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 3600,
        })

    # ------------------------------------------------------------------
    # Score threshold in [0.0, 1.0]
    # ------------------------------------------------------------------

    def test_score_threshold_valid_min(self, monkeypatch):
        """Score threshold of 0.0 is valid."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "Col",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.0,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 3600,
        })

    def test_score_threshold_valid_max(self, monkeypatch):
        """Score threshold of 1.0 is valid."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "Col",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 1.0,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 3600,
        })

    @pytest.mark.parametrize("score", [-0.1, 1.1, -1.0, 2.0])
    def test_score_threshold_out_of_range_invalid(self, monkeypatch, score):
        """Score threshold outside [0.0, 1.0] must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "Col",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": score,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 3600,
        })

    # ------------------------------------------------------------------
    # URL expiry in [60, 86400]
    # ------------------------------------------------------------------

    def test_url_expiry_valid_min(self, monkeypatch):
        """URL expiry of 60 is the minimum valid value."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "Col",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.3,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 60,
        })

    def test_url_expiry_valid_max(self, monkeypatch):
        """URL expiry of 86400 is the maximum valid value."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "Col",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.3,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 86400,
        })

    @pytest.mark.parametrize("expiry", [0, 59, 86401, -1])
    def test_url_expiry_out_of_range_invalid(self, monkeypatch, expiry):
        """URL expiry outside [60, 86400] must raise ValueError."""
        self._fail(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "Col",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.3,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": expiry,
        })

    # ------------------------------------------------------------------
    # Passing baseline
    # ------------------------------------------------------------------

    def test_default_visual_config_passes(self, monkeypatch):
        """Default visual config values (non-empty collection, valid thresholds) pass."""
        self._ok(monkeypatch, {
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION": "RAGVisualPages",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE": 0.3,
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS": 3600,
        })
