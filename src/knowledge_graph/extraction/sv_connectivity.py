# @summary
# Cross-module SV port connectivity analysis using pyverilog DataflowAnalyzer.
# Produces only connects_to triples (no entity upserts) to prevent duplication.
# Exports: SVConnectivityAnalyzer, SV_CONNECTIVITY_SOURCE
# Deps: pyverilog (optional), src.knowledge_graph.backend, src.knowledge_graph.common.schemas
# @end-summary
"""Cross-module SV port connectivity analysis using pyverilog.

Uses pyverilog's ``DataflowAnalyzer`` to resolve port-to-signal connections
across module boundaries.  Produces ``connects_to`` triples only — entity
nodes are created by the per-file tree-sitter parser, not by this analyzer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.common import Triple

__all__ = ["SVConnectivityAnalyzer", "SV_CONNECTIVITY_SOURCE"]

logger = logging.getLogger("rag.knowledge_graph.extraction.sv_connectivity")

# Synthetic source key for all pyverilog-generated triples.
SV_CONNECTIVITY_SOURCE = "__sv_connectivity_batch__"

# Guard pyverilog import
try:
    from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer  # type: ignore[import-untyped]
    _PYVERILOG_AVAILABLE = True
except ImportError:
    _PYVERILOG_AVAILABLE = False


class SVConnectivityAnalyzer:
    """Cross-module SV port connectivity analysis using pyverilog.

    Accepts a ``.f`` filelist path and optional top module name.  Uses
    pyverilog's ``DataflowAnalyzer`` to resolve cross-module port connections
    and produce ``connects_to`` triples.

    Produces ONLY triples — no entity upserts — to prevent duplication
    with tree-sitter entities.
    """

    def __init__(
        self,
        filelist_path: str,
        backend: GraphStorageBackend,
        top_module: Optional[str] = None,
    ) -> None:
        self._filelist_path = filelist_path
        self._backend = backend
        self._top_module = top_module or ""

    def analyze(self) -> List[Triple]:
        """Run pyverilog DataflowAnalyzer and return connects_to triples.

        Returns:
            List of connects_to Triple objects.  Empty list on any failure.
        """
        if not _PYVERILOG_AVAILABLE:
            logger.warning(
                "pyverilog not installed — SV connectivity analysis disabled. "
                "Install with: pip install pyverilog"
            )
            return []

        file_paths, include_dirs = self.parse_filelist(self._filelist_path)
        if not file_paths:
            logger.warning("No SV files found in filelist: %s", self._filelist_path)
            return []

        top_module = self._top_module
        if not top_module:
            top_module = self._auto_detect_top_module()
            if not top_module:
                return []

        return self._run_pyverilog(file_paths, include_dirs, top_module)

    # ------------------------------------------------------------------
    # Filelist parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_filelist(
        filelist_path: str, _visited: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Parse a .f filelist into file paths and include directories.

        Supports one file path per line, ``//`` line comments,
        ``+incdir+<path>`` include directives, and ``-f <path>``
        recursive filelist inclusion.  Relative paths are resolved
        relative to the filelist's parent directory.

        Args:
            filelist_path: Path to the .f filelist file.

        Returns:
            Tuple of (file_paths, include_dirs).
        """
        if _visited is None:
            _visited = set()

        real_path = str(Path(filelist_path).resolve())
        if real_path in _visited:
            logger.warning("Circular filelist reference: %s", filelist_path)
            return [], []
        _visited.add(real_path)

        base_dir = Path(filelist_path).parent
        file_paths: List[str] = []
        include_dirs: List[str] = []

        try:
            with open(filelist_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("//"):
                        continue
                    # Include directory directive
                    if line.startswith("+incdir+"):
                        inc_path = line[len("+incdir+"):]
                        resolved = str((base_dir / inc_path).resolve())
                        include_dirs.append(resolved)
                    # Recursive filelist
                    elif line.startswith("-f "):
                        sub_path = line[3:].strip()
                        resolved = str((base_dir / sub_path).resolve())
                        sub_files, sub_incs = SVConnectivityAnalyzer.parse_filelist(
                            resolved, _visited
                        )
                        file_paths.extend(sub_files)
                        include_dirs.extend(sub_incs)
                    # Regular file path
                    else:
                        resolved = str((base_dir / line).resolve())
                        file_paths.append(resolved)
        except FileNotFoundError:
            logger.warning("Filelist not found: %s", filelist_path)

        return file_paths, include_dirs

    # ------------------------------------------------------------------
    # Top module auto-detection
    # ------------------------------------------------------------------

    def _auto_detect_top_module(self) -> Optional[str]:
        """Auto-detect the top module from the graph backend.

        Heuristic: find RTL_Module entities that are never the target
        of an ``instantiates`` edge.

        Returns:
            Module name if exactly one candidate found, None otherwise.
        """
        all_entities = self._backend.get_all_entities()
        modules = {e.name for e in all_entities if e.type == "RTL_Module"}

        if not modules:
            logger.warning("No RTL_Module entities in graph — cannot auto-detect top module")
            return None

        # Find modules that are instantiated by others
        instantiated: Set[str] = set()
        for module_name in modules:
            edges = self._backend.get_outgoing_edges(module_name)
            for triple in edges:
                if triple.predicate == "instantiates":
                    instantiated.add(triple.object)

        tops = modules - instantiated
        if len(tops) == 1:
            top = next(iter(tops))
            logger.info("Auto-detected top module: %s", top)
            return top
        elif len(tops) == 0:
            logger.warning("No uninstantiated modules found — circular hierarchy?")
            return None
        else:
            # Multiple tops — run analysis for each
            logger.info(
                "Multiple top module candidates: %s — running analysis for each",
                ", ".join(sorted(tops)),
            )
            return next(iter(sorted(tops)))  # Use first alphabetically as primary

    # ------------------------------------------------------------------
    # Pyverilog integration
    # ------------------------------------------------------------------

    def _run_pyverilog(
        self,
        file_paths: List[str],
        include_dirs: List[str],
        top_module: str,
    ) -> List[Triple]:
        """Run pyverilog DataflowAnalyzer and extract connectivity triples."""
        if not _PYVERILOG_AVAILABLE:
            return []

        try:
            analyzer = VerilogDataflowAnalyzer(
                file_paths,
                top_module,
                noreorder=True,
                nobind=False,
                preprocess_include=include_dirs if include_dirs else None,
            )
            analyzer.generate()
            terms = analyzer.getTerms()
            binds = analyzer.getBinddict()
        except Exception as exc:
            logger.warning(
                "pyverilog DataflowAnalyzer failed for top=%s: %s",
                top_module, exc,
            )
            return []

        triples: List[Triple] = []
        # Extract connectivity from bind dict
        for signal_name, bind_list in binds.items():
            signal_str = str(signal_name)
            for bind in bind_list:
                dest_str = str(bind.dest) if hasattr(bind, "dest") else None
                if dest_str and dest_str != signal_str:
                    triples.append(
                        Triple(
                            subject=signal_str,
                            predicate="connects_to",
                            object=dest_str,
                            source=SV_CONNECTIVITY_SOURCE,
                            extractor_source="sv_connectivity",
                        )
                    )

        logger.info(
            "pyverilog analysis (top=%s): produced %d connects_to triples",
            top_module, len(triples),
        )
        return triples
