# @summary
# SystemVerilog parser-based structural entity extraction using tree-sitter.
# Walks the concrete syntax tree to extract modules, ports, parameters,
# instances, signals, interfaces, packages, generates, and tasks/functions.
# Falls back gracefully if tree-sitter-verilog is not installed.
# Exports: SVParserExtractor, AST_TO_SCHEMA_MAP, VALID_EXTENSIONS, KNOWN_UNSUPPORTED
# Deps: tree_sitter, tree_sitter_verilog, src.knowledge_graph.common.schemas,
#        src.knowledge_graph.common.types
# @end-summary
"""SystemVerilog parser-based structural entity extraction.

Uses tree-sitter-verilog for deterministic extraction of RTL structural
entities (modules, ports, parameters, instances, signals, interfaces,
packages, generates, tasks/functions) from SystemVerilog source files.
All results are tagged with ``extractor_source="sv_parser"``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.knowledge_graph.common import (
    Entity,
    ExtractionResult,
    Triple,
)
from src.knowledge_graph.common import (
    KGConfig,
    SchemaDefinition,
)

__all__ = [
    "SVParserExtractor",
    "AST_TO_SCHEMA_MAP",
    "VALID_EXTENSIONS",
    "KNOWN_UNSUPPORTED",
]

logger = logging.getLogger("rag.knowledge_graph.sv_parser")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VALID_EXTENSIONS: frozenset = frozenset({".sv", ".v", ".svh"})

AST_TO_SCHEMA_MAP: Dict[str, str] = {
    "module_declaration": "RTL_Module",
    "port_declaration": "Port",
    "ansi_port_declaration": "Port",
    "parameter_declaration": "Parameter",
    "local_parameter_declaration": "Parameter",
    "module_instantiation": "Instance",
    "net_declaration": "Signal",
    "data_declaration": "Signal",
    "interface_declaration": "Interface",
    "package_declaration": "Package",
    # SHOULD-level (REQ-KG-1b-207):
    "generate_region": "Generate",
    "task_declaration": "Task_Function",
    "function_declaration": "Task_Function",
}

KNOWN_UNSUPPORTED: List[str] = [
    "bind",
    "cross_module_reference",
    "uvm_macro",
    "clock_domain_crossing",  # REQ-KG-1b-208 — deferred stretch goal
]

# ---------------------------------------------------------------------------
# tree-sitter bootstrap (wrapped in try/except)
# ---------------------------------------------------------------------------

_TS_AVAILABLE = False
_VERILOG_LANGUAGE = None

try:
    import tree_sitter_verilog as tsverilog  # type: ignore[import-untyped]
    from tree_sitter import Language, Parser  # type: ignore[import-untyped]

    _VERILOG_LANGUAGE = Language(tsverilog.language())
    _TS_AVAILABLE = True
except Exception:  # pragma: no cover — optional dependency
    pass


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class SVParserExtractor:
    """Deterministic structural extractor for SystemVerilog using tree-sitter.

    Extracts entities and structural relationships from the parsed CST.
    All results are tagged with ``extractor_source="sv_parser"``.
    """

    @property
    def name(self) -> str:
        """Extractor identifier reported in Entity.extractor_source."""
        return "sv_parser"

    def __init__(
        self,
        schema: SchemaDefinition,
        config: KGConfig,
    ) -> None:
        """Initialise with schema and config. Creates the tree-sitter parser.

        Args:
            schema: Parsed YAML schema for type validation.
            config: KG runtime configuration.

        Raises:
            ImportError: If tree-sitter or tree-sitter-verilog is not installed.
        """
        self._schema = schema
        self._config = config
        self._parser: Any = None

        if not _TS_AVAILABLE:
            logger.warning(
                "tree-sitter or tree-sitter-verilog not installed. "
                "SVParserExtractor will return empty results. "
                "Install with: pip install tree-sitter tree-sitter-verilog"
            )
            return

        self._parser = Parser(_VERILOG_LANGUAGE)

    # -- Public API ----------------------------------------------------------

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        """Extract structural entities and relations from SV source text.

        Args:
            text: SystemVerilog source code as a string.
            source: File path or URI for provenance.

        Returns:
            ExtractionResult with entities and triples. Partial results
            on parse errors (ERROR subtrees are skipped).
        """
        if self._parser is None:
            return ExtractionResult()

        tree = self._parse_tree(text)
        if tree is None:
            return ExtractionResult()

        entities, triples = self._walk_tree(tree.root_node, source)

        return ExtractionResult(entities=entities, triples=triples)

    def extract_file(self, file_path: str) -> ExtractionResult:
        """Extract from a SystemVerilog file on disk.

        Args:
            file_path: Path to a .sv, .v, or .svh file.

        Returns:
            ExtractionResult. Empty result for non-SV extensions.

        Raises:
            FileNotFoundError: If file_path does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix not in VALID_EXTENSIONS:
            logger.debug(
                "Skipping non-SV file: %s (extension %s)", file_path, path.suffix
            )
            return ExtractionResult()

        text = path.read_text(encoding="utf-8", errors="replace")
        return self.extract(text, source=str(path))

    def extract_entities(self, text: str) -> Set[str]:
        """Return entity name strings from *text* (EntityExtractor protocol)."""
        result = self.extract(text)
        return {e.name for e in result.entities}

    def extract_relations(self, text: str, known_entities: Set[str]) -> List[Triple]:
        """Return relation triples from *text* (EntityExtractor protocol)."""
        return self.extract(text).triples

    # -- Internal tree-walking methods ---------------------------------------

    def _parse_tree(self, text: str) -> Any:
        """Parse text into a tree-sitter CST.

        Args:
            text: Source code string (encoded to UTF-8 internally).

        Returns:
            tree-sitter Tree object, or None on failure.
        """
        if self._parser is None:
            return None
        try:
            return self._parser.parse(bytes(text, "utf-8"))
        except Exception:
            logger.warning("tree-sitter parse failed", exc_info=True)
            return None

    def _walk_tree(
        self, node: Any, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Recursively walk the CST and extract entities and triples.

        Skips ERROR and MISSING nodes with a logged warning.
        Dispatches to type-specific extraction methods based on
        ``AST_TO_SCHEMA_MAP`` membership.

        Args:
            node: tree-sitter Node to process.
            source: File path for provenance.

        Returns:
            Tuple of (entities, triples) collected from this subtree.
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        if node.type in ("ERROR", "MISSING"):
            logger.warning(
                "Skipping %s node at byte %d in %s",
                node.type,
                node.start_byte,
                source,
            )
            return entities, triples

        # Dispatch based on node type
        if node.type == "module_declaration":
            e, t = self._extract_module(node, source)
            entities.extend(e)
            triples.extend(t)
        elif node.type == "interface_declaration":
            e, t = self._extract_interface(node, source)
            entities.extend(e)
            triples.extend(t)
        elif node.type == "package_declaration":
            e, t = self._extract_package(node, source)
            entities.extend(e)
            triples.extend(t)
        else:
            # Recurse into children for top-level nodes we don't directly handle
            for child in node.children:
                e, t = self._walk_tree(child, source)
                entities.extend(e)
                triples.extend(t)

        return entities, triples

    def _extract_module(
        self, node: Any, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract a module declaration and its children.

        Produces:
        - RTL_Module entity for the module itself.
        - Port, Parameter, Signal, Instance entities from children.
        - ``contains`` triples from module to each child entity.
        - ``instantiates`` triples from Instance entities to their module types.

        Args:
            node: module_declaration tree-sitter node.
            source: File path for provenance.

        Returns:
            Tuple of (entities, triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        module_name = self._extract_node_name(node)
        if module_name is None:
            logger.warning("Could not extract module name at byte %d", node.start_byte)
            return entities, triples

        module_entity = self._make_entity(module_name, "RTL_Module", source)
        entities.append(module_entity)

        # Extract child constructs
        port_e, port_t = self._extract_ports(node, module_name, source)
        entities.extend(port_e)
        triples.extend(port_t)

        param_e, param_t = self._extract_parameters(node, module_name, source)
        entities.extend(param_e)
        triples.extend(param_t)

        inst_e, inst_t = self._extract_instances(node, module_name, source)
        entities.extend(inst_e)
        triples.extend(inst_t)

        sig_e, sig_t = self._extract_signals(node, module_name, source)
        entities.extend(sig_e)
        triples.extend(sig_t)

        # SHOULD-level: generates, tasks, functions inside the module
        gen_e, gen_t = self._extract_generates(node, module_name, source)
        entities.extend(gen_e)
        triples.extend(gen_t)

        tf_e, tf_t = self._extract_tasks_functions(node, module_name, source)
        entities.extend(tf_e)
        triples.extend(tf_t)

        # Extract import statements for depends_on relationships
        import_t = self._extract_imports(node, module_name, source)
        triples.extend(import_t)

        return entities, triples

    def _extract_ports(
        self, node: Any, module_name: str, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract port declarations from a module's port list.

        Extracts direction (input/output/inout) and width from range nodes.
        Looks specifically for ``port_identifier`` nodes to get the correct
        port name (rather than identifiers in type/range expressions).

        Args:
            node: Module node whose children are scanned for ports.
            module_name: Parent module name for ``contains`` triples.
            source: File path for provenance.

        Returns:
            Tuple of (port entities, contains triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []
        seen_ports: Set[str] = set()

        for child in self._find_descendants(node, {"port_declaration", "ansi_port_declaration"}):
            port_name = self._extract_port_name(child)
            if port_name is None or port_name in seen_ports:
                continue
            seen_ports.add(port_name)

            entity = self._make_entity(port_name, "Port", source)
            entities.append(entity)
            triples.append(
                self._make_triple(module_name, "contains", port_name, source)
            )

        return entities, triples

    def _extract_instances(
        self, node: Any, module_name: str, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract module instantiations.

        Produces Instance entities, ``contains`` triples (parent -> instance),
        and ``instantiates`` triples (parent module -> instantiated module type).

        Args:
            node: Module body node to scan for instantiations.
            module_name: Parent module name.
            source: File path for provenance.

        Returns:
            Tuple of (entities, triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        for inst_node in self._find_descendants(node, {"module_instantiation"}):
            # module_instantiation typically has: module_type_name instance_name(...)
            module_type = None
            instance_name = None

            # The first identifier child is the module type being instantiated
            for child in inst_node.children:
                if child.type in ("simple_identifier", "identifier"):
                    if module_type is None:
                        module_type = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                    else:
                        instance_name = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                        break
                elif child.type == "hierarchical_instance":
                    # The instance name is inside hierarchical_instance
                    inst_name_node = self._find_first_identifier(child)
                    if inst_name_node is not None:
                        instance_name = (
                            inst_name_node.text.decode("utf-8")
                            if isinstance(inst_name_node.text, bytes)
                            else inst_name_node.text
                        )
                    break

            if module_type is None:
                # Try extracting from the first named child
                module_type = self._extract_node_name(inst_node)

            if instance_name is None and module_type is not None:
                # Fallback: use module_type + _inst as instance name
                instance_name = f"{module_type}_inst"

            if instance_name is None:
                continue

            entity = self._make_entity(instance_name, "Instance", source)
            entities.append(entity)

            triples.append(
                self._make_triple(module_name, "contains", instance_name, source)
            )
            if module_type is not None:
                triples.append(
                    self._make_triple(module_name, "instantiates", module_type, source)
                )

        return entities, triples

    def _extract_parameters(
        self, node: Any, module_name: str, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract parameter and localparam declarations.

        Looks for identifiers inside ``param_assignment`` or
        ``parameter_identifier`` nodes to get the correct parameter name.

        Args:
            node: Module node to scan.
            module_name: Parent module name for ``contains`` triples.
            source: File path for provenance.

        Returns:
            Tuple of (parameter entities, contains triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []
        seen: Set[str] = set()

        for child in self._find_descendants(
            node, {"parameter_declaration", "local_parameter_declaration"}
        ):
            names = self._extract_param_names(child)
            for param_name in names:
                if param_name in seen:
                    continue
                seen.add(param_name)

                entity = self._make_entity(param_name, "Parameter", source)
                entities.append(entity)
                triples.append(
                    self._make_triple(module_name, "contains", param_name, source)
                )

        return entities, triples

    def _extract_signals(
        self, node: Any, module_name: str, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract net/data declarations (wire, reg, logic).

        Looks for identifier names inside ``variable_decl_assignment``,
        ``net_decl_assignment``, or ``list_of_variable_decl_assignments``
        to avoid picking up identifiers from range expressions.

        Args:
            node: Module body node to scan.
            module_name: Parent module name for ``contains`` triples.
            source: File path for provenance.

        Returns:
            Tuple of (signal entities, contains triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []
        seen: Set[str] = set()

        for child in self._find_descendants(node, {"net_declaration", "data_declaration"}):
            names = self._extract_declaration_names(child)
            for sig_name in names:
                if sig_name in seen:
                    continue
                seen.add(sig_name)

                entity = self._make_entity(sig_name, "Signal", source)
                entities.append(entity)
                triples.append(
                    self._make_triple(module_name, "contains", sig_name, source)
                )

        return entities, triples

    def _extract_interface(
        self, node: Any, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract an interface declaration.

        Args:
            node: interface_declaration tree-sitter node.
            source: File path for provenance.

        Returns:
            Tuple of (entities, triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        iface_name = self._extract_node_name(node)
        if iface_name is None:
            logger.warning(
                "Could not extract interface name at byte %d", node.start_byte
            )
            return entities, triples

        entity = self._make_entity(iface_name, "Interface", source)
        entities.append(entity)

        # Extract ports declared inside the interface
        port_e, port_t = self._extract_ports(node, iface_name, source)
        entities.extend(port_e)
        triples.extend(port_t)

        return entities, triples

    def _extract_package(
        self, node: Any, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract a package declaration.

        Produces a Package entity and ``contains`` triples for items
        declared inside the package.

        Args:
            node: package_declaration tree-sitter node.
            source: File path for provenance.

        Returns:
            Tuple of (entities, triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        pkg_name = self._extract_node_name(node)
        if pkg_name is None:
            logger.warning(
                "Could not extract package name at byte %d", node.start_byte
            )
            return entities, triples

        entity = self._make_entity(pkg_name, "Package", source)
        entities.append(entity)

        # Extract parameters and type declarations inside the package
        param_e, param_t = self._extract_parameters(node, pkg_name, source)
        entities.extend(param_e)
        triples.extend(param_t)

        return entities, triples

    def _extract_generates(
        self, node: Any, module_name: str, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract generate regions (SHOULD-level REQ-KG-1b-207).

        Args:
            node: Module node to scan.
            module_name: Parent module name for ``contains`` triples.
            source: File path for provenance.

        Returns:
            Tuple of (generate entities, contains triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        for idx, child in enumerate(
            self._find_descendants(node, {"generate_region"})
        ):
            gen_name = self._extract_node_name(child)
            if gen_name is None:
                gen_name = f"{module_name}_gen_{idx}"

            entity = self._make_entity(gen_name, "Generate", source)
            entities.append(entity)
            triples.append(
                self._make_triple(module_name, "contains", gen_name, source)
            )

        return entities, triples

    def _extract_tasks_functions(
        self, node: Any, module_name: str, source: str
    ) -> Tuple[List[Entity], List[Triple]]:
        """Extract task and function declarations (SHOULD-level REQ-KG-1b-207).

        Args:
            node: Module node to scan.
            module_name: Parent module name for ``contains`` triples.
            source: File path for provenance.

        Returns:
            Tuple of (task/function entities, contains triples).
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        for child in self._find_descendants(
            node, {"task_declaration", "function_declaration"}
        ):
            tf_name = self._extract_node_name(child)
            if tf_name is None:
                continue

            entity = self._make_entity(tf_name, "Task_Function", source)
            entities.append(entity)
            triples.append(
                self._make_triple(module_name, "contains", tf_name, source)
            )

        return entities, triples

    def _extract_imports(
        self, node: Any, module_name: str, source: str
    ) -> List[Triple]:
        """Extract package import statements for depends_on relationships.

        Args:
            node: Module node to scan.
            module_name: Module that depends on imported packages.
            source: File path for provenance.

        Returns:
            List of depends_on triples.
        """
        triples: List[Triple] = []
        seen_packages: Set[str] = set()

        for imp in self._find_descendants(
            node, {"package_import_declaration", "package_import_item"}
        ):
            # Look for package_identifier inside the import node
            for pkg_id in self._find_descendants(imp, {"package_identifier"}):
                ident = self._find_first_identifier(pkg_id)
                if ident is not None:
                    text = ident.text
                    pkg = text.decode("utf-8") if isinstance(text, bytes) else text
                    if pkg and pkg not in seen_packages:
                        seen_packages.add(pkg)
                        triples.append(
                            self._make_triple(
                                module_name, "depends_on", pkg, source
                            )
                        )

        return triples

    # -- Utility helpers -----------------------------------------------------

    def _extract_port_name(self, node: Any) -> Optional[str]:
        """Extract port name from a port_declaration or ansi_port_declaration.

        Looks specifically for ``port_identifier`` child nodes to avoid
        picking up identifiers from type or range expressions.

        Args:
            node: A port_declaration or ansi_port_declaration node.

        Returns:
            Port name string, or None if not found.
        """
        # ansi_port_declaration has a port_identifier child
        for desc in self._find_descendants(node, {"port_identifier"}):
            ident = self._find_first_identifier(desc)
            if ident is not None:
                text = ident.text
                return text.decode("utf-8") if isinstance(text, bytes) else text

        # Fallback for port_declaration (non-ANSI): look for direct identifier
        return self._extract_node_name(node)

    def _extract_param_names(self, node: Any) -> List[str]:
        """Extract parameter names from a parameter_declaration node.

        Looks for ``param_assignment`` or ``parameter_identifier`` nodes
        to find the actual parameter names, not identifiers in expressions.

        Args:
            node: A parameter_declaration or local_parameter_declaration node.

        Returns:
            List of parameter name strings.
        """
        names: List[str] = []

        # Look for param_assignment nodes (parameter WIDTH = 8)
        for pa in self._find_descendants(node, {"param_assignment"}):
            for pi in self._find_descendants(pa, {"parameter_identifier"}):
                ident = self._find_first_identifier(pi)
                if ident is not None:
                    text = ident.text
                    name = text.decode("utf-8") if isinstance(text, bytes) else text
                    names.append(name)
                    break
            else:
                # No parameter_identifier found; try first identifier in param_assignment
                ident = self._find_first_identifier(pa)
                if ident is not None:
                    text = ident.text
                    name = text.decode("utf-8") if isinstance(text, bytes) else text
                    names.append(name)

        if not names:
            # Fallback: use generic name extraction
            name = self._extract_node_name(node)
            if name is not None:
                names.append(name)

        return names

    def _is_import_declaration(self, node: Any) -> bool:
        """Check if a data_declaration node is actually an import statement.

        Args:
            node: A data_declaration or net_declaration node.

        Returns:
            True if the node contains a package_import_declaration.
        """
        for child in node.children:
            if child.type == "package_import_declaration":
                return True
        return False

    def _extract_declaration_names(self, node: Any) -> List[str]:
        """Extract signal names from net_declaration or data_declaration.

        Looks for ``variable_decl_assignment``, ``net_decl_assignment``, or
        similar assignment nodes where the actual signal name lives, avoiding
        identifiers in range or type expressions. Skips import declarations.

        Args:
            node: A net_declaration or data_declaration node.

        Returns:
            List of signal name strings.
        """
        # Skip import statements disguised as data_declaration
        if self._is_import_declaration(node):
            return []

        names: List[str] = []

        # net_declaration uses net_decl_assignment for each declared name
        # data_declaration uses variable_decl_assignment
        target_types = {"variable_decl_assignment", "net_decl_assignment"}
        for assign in self._find_descendants(node, target_types):
            ident = self._find_first_identifier(assign)
            if ident is not None:
                text = ident.text
                name = text.decode("utf-8") if isinstance(text, bytes) else text
                names.append(name)

        if not names:
            # Fallback: use generic extraction
            name = self._extract_node_name(node)
            if name is not None:
                names.append(name)

        return names

    def _extract_node_name(self, node: Any) -> Optional[str]:
        """Extract the identifier name from a declaration node.

        Uses a recursive search to find the first ``simple_identifier``
        in the subtree, skipping keyword tokens and ERROR nodes.

        Args:
            node: Any declaration node.

        Returns:
            Name string, or None if no identifier found.
        """
        # Keywords that look like identifiers but aren't names
        _KEYWORDS = frozenset({
            "module", "endmodule", "interface", "endinterface",
            "package", "endpackage", "parameter", "localparam",
            "input", "output", "inout", "wire", "reg", "logic",
            "task", "endtask", "function", "endfunction",
            "generate", "endgenerate",
        })

        result = self._find_first_identifier(node)
        if result is not None:
            text = result.text
            name = text.decode("utf-8") if isinstance(text, bytes) else text
            if name not in _KEYWORDS:
                return name

        return None

    def _find_descendants(
        self, node: Any, target_types: Set[str]
    ) -> List[Any]:
        """Find all descendant nodes matching the given types.

        Uses iterative BFS but does not recurse into nodes that are
        themselves targets (to avoid double-counting nested constructs).

        Args:
            node: Root node to search from.
            target_types: Set of tree-sitter node type strings.

        Returns:
            List of matching descendant nodes.
        """
        results: List[Any] = []
        queue = list(node.children)

        while queue:
            current = queue.pop(0)
            if current.type in ("ERROR", "MISSING"):
                logger.debug(
                    "Skipping %s node at byte %d during descendant search",
                    current.type,
                    current.start_byte,
                )
                continue
            if current.type in target_types:
                results.append(current)
                # Don't recurse into matched nodes to avoid double-counting
            else:
                queue.extend(current.children)

        return results

    def _find_first_identifier(self, node: Any) -> Optional[Any]:
        """Find the first identifier node in the subtree.

        Args:
            node: Root of subtree to search.

        Returns:
            The first simple_identifier or identifier node, or None.
        """
        if node.type in ("simple_identifier", "identifier"):
            return node
        for child in node.children:
            result = self._find_first_identifier(child)
            if result is not None:
                return result
        return None

    def _make_entity(
        self, name: str, entity_type: str, source: str
    ) -> Entity:
        """Create an Entity with sv_parser attribution.

        Args:
            name: Entity canonical name.
            entity_type: YAML schema type name.
            source: File path for provenance.

        Returns:
            Entity with ``extractor_source=["sv_parser"]`` and ``sources=[source]``.
        """
        sources = [source] if source else []
        return Entity(
            name=name,
            type=entity_type,
            sources=sources,
            extractor_source=["sv_parser"],
        )

    def _make_triple(
        self, subject: str, predicate: str, obj: str, source: str
    ) -> Triple:
        """Create a Triple with sv_parser attribution.

        Args:
            subject: Source entity name.
            predicate: Edge type label.
            obj: Target entity name.
            source: File path for provenance.

        Returns:
            Triple with ``extractor_source="sv_parser"``.
        """
        return Triple(
            subject=subject,
            predicate=predicate,
            object=obj,
            source=source,
            extractor_source="sv_parser",
        )
