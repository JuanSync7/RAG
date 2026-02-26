# @summary
# Bridges document chunks into a NetworkX DiGraph for entity extraction and querying,
# exports knowledge graph nodes as .md files, merges functionality;
# imports nx, re, json, Path, collections, os.path, pathlib; exports KnowledgeGraphBuilder, GraphQueryExpander, export_obsidian.
# @end-summary
"""
Knowledge graph builder and query-time expander.

Extracts entities and relationships from document chunks using rule-based
patterns, stores them in a NetworkX directed graph, and provides query-time
expansion to augment BM25 search with related terms.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# --- Entity Extraction ---

# Common words that look like entities but aren't
_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "There", "Here",
    "It", "Its", "In", "Is", "Are", "Was", "Were", "Be", "Been",
    "For", "And", "But", "Or", "Nor", "Not", "No", "So",
    "A", "AN", "THE", "AND", "OR", "FOR", "IS", "TO", "OF",
    "BY", "AT", "ON", "AS", "IF", "DO", "UP", "WE", "MY",
    "HE", "ME", "US", "AM", "AN", "IF", "GO", "VS",
    "ALL", "HAS", "HAD", "GET", "GOT", "DID", "MAY", "CAN",
    "LET", "USE", "SET", "HOW", "WHO", "WHY", "NEW", "OLD",
    "ONE", "TWO", "KEY", "SEE", "MAX", "MIN", "TOP", "END",
    "ALSO", "MANY", "EACH", "BOTH", "SUCH", "SOME", "MORE",
    "MOST", "VERY", "WELL", "MUCH", "THAN", "THEN", "WHEN",
    "WITH", "FROM", "HAVE", "WILL", "BEEN", "INTO", "ONLY",
    "OVER", "JUST", "ALSO", "LIKE", "WHAT", "MAKE", "TAKE",
    "USED", "HELP", "MAKE", "DOES", "WIDE", "TYPE", "BEST",
    "HIGH", "LOOK", "ARGS", "NOTE", "TODO", "NONE", "TRUE",
    "LAST", "MUST", "SAME", "LONG", "NEXT", "NEED",
})

# CamelCase: TensorFlow, PyTorch, NumPy
_CAMEL_PAT = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]+)+\b")

# ALL-CAPS acronyms (2-10 chars): RAG, BM25, CNN, NLP
_ACRONYM_PAT = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

# Multi-word capitalized phrases (2-3 words): Machine Learning, Deep Learning
# Each word must start uppercase, be 2+ lowercase chars, max 3 words
# Uses [ ]+ (not \s+) to avoid matching across newlines
_MULTI_WORD_PAT = re.compile(
    r"\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){1,2}\b"
)

# Acronym expansion: "Retrieval-Augmented Generation (RAG)" or "RAG (Retrieval-Augmented Generation)"
_EXPAND_PAT_1 = re.compile(
    r"([A-Z][a-z]+(?:[\s\-]+[A-Za-z]+){1,5})\s+\(([A-Z][A-Z0-9]{1,9})\)"
)
_EXPAND_PAT_2 = re.compile(
    r"([A-Z][A-Z0-9]{1,9})\s+\(([A-Z][a-z]+(?:[\s\-]+[a-z]+){1,5})\)"
)


class GLiNEREntityExtractor:
    """Zero-shot NER entity extractor using GLiNER.

    Uses GLiNER for entity extraction while delegating acronym alias
    detection and relation extraction to the regex-based EntityExtractor.
    """

    def __init__(self, model_path: str = None):
        from gliner import GLiNER
        from config.settings import GLINER_MODEL_PATH, GLINER_ENTITY_LABELS

        model_path = model_path or GLINER_MODEL_PATH
        self.model = GLiNER.from_pretrained(model_path, local_files_only=True)
        self._regex_extractor = None  # lazy init (avoids circular ref at class def time)
        self._labels = GLINER_ENTITY_LABELS

    def _get_regex_extractor(self):
        if self._regex_extractor is None:
            self._regex_extractor = EntityExtractor()
        return self._regex_extractor

    def extract_entities(self, text: str) -> Set[str]:
        """Extract entities using GLiNER zero-shot NER."""
        clean = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)

        entities = set()
        predictions = self.model.predict_entities(clean, self._labels, threshold=0.5)
        for pred in predictions:
            entity_text = pred["text"].strip()
            if len(entity_text) <= 2:
                continue
            if entity_text in _STOPWORDS or entity_text.upper() in _STOPWORDS:
                continue
            entities.add(entity_text)

        return entities

    def extract_acronym_aliases(self, text: str) -> Dict[str, str]:
        """Delegate to regex extractor (GLiNER doesn't handle acronym patterns)."""
        return self._get_regex_extractor().extract_acronym_aliases(text)

    def extract_relations(
        self, text: str, known_entities: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """Delegate to regex extractor (GLiNER doesn't extract relations)."""
        return self._get_regex_extractor().extract_relations(text, known_entities)


class EntityExtractor:
    """Rule-based entity and relationship extractor."""

    # Words that commonly start sentences but aren't entity-leading words
    _SENTENCE_STARTERS = frozenset({
        "These", "Those", "This", "That", "There", "Their", "They",
        "Some", "Many", "Most", "Each", "Every", "Several", "Both",
        "One", "Two", "Three", "Four", "Five", "Other", "Another",
        "Common", "Various", "Popular", "Important",
    })

    def extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        # Strip entire markdown header lines before extraction
        clean = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        entities = set()

        # CamelCase terms
        for m in _CAMEL_PAT.finditer(clean):
            entities.add(m.group())

        # ALL-CAPS acronyms
        for m in _ACRONYM_PAT.finditer(clean):
            term = m.group()
            if term not in _STOPWORDS:
                entities.add(term)

        # Multi-word capitalized phrases
        for m in _MULTI_WORD_PAT.finditer(clean):
            term = m.group()
            words = term.split()
            # Filter: skip if first word is a common sentence starter
            if words[0] in self._SENTENCE_STARTERS:
                continue
            # Filter: skip if any word is a stopword
            if any(w in _STOPWORDS for w in words):
                continue
            # Filter: must have at least 2 meaningful words
            if len(words) >= 2:
                entities.add(term)

        return entities

    def extract_acronym_aliases(self, text: str) -> Dict[str, str]:
        """Find acronym expansions like 'Long Form (ACRO)' → {ACRO: Long Form}."""
        aliases = {}

        # Pattern 1: "Long Form (ACRO)"
        for m in _EXPAND_PAT_1.finditer(text):
            long_form = m.group(1).strip()
            acronym = m.group(2).strip()
            aliases[acronym] = long_form

        # Pattern 2: "ACRO (long form)"
        for m in _EXPAND_PAT_2.finditer(text):
            acronym = m.group(1).strip()
            long_form = m.group(2).strip()
            aliases[acronym] = long_form

        return aliases

    def extract_relations(
        self, text: str, known_entities: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """Extract (subject, relation, object) triples from text.

        Uses sentence-level regex patterns for common relationship forms.
        """
        relations = []
        # Strip markdown headers and split into sentences
        clean_text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        sentences = re.split(r"[.!?\n]\s*", clean_text)

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue

            # "X is a subset of Y"
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+is\s+a\s+subset\s+of\s+"
                r"(\b[a-z][a-z\s\-]{2,40})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                relations.append((subj, "subset_of", obj))
                continue

            # "X is a/an Y" — subject must be a known entity, object capped at ~5 words
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+is\s+an?\s+"
                r"(\b[a-z][a-z\s\-]{2,50})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                # Cap object to first ~5 words
                obj_words = obj.split()[:4]
                obj = " ".join(obj_words)
                if any(subj in e or e in subj for e in known_entities):
                    relations.append((subj, "is_a", obj))

            # "X is/are used in/for Y"
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+(?:is|are)\s+(?:widely\s+)?used\s+"
                r"(?:in|for)\s+(\b[a-z][a-z\s\-,]{2,50})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                obj_words = obj.split()[:4]
                obj = " ".join(obj_words)
                relations.append((subj, "used_for", obj))

            # "X includes/uses/supports/combines Y"
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+"
                r"(?:includes?|uses?|supports?|combines?|provides?|enables?)\s+"
                r"(\b[a-z][a-z\s\-]{2,50})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                obj_words = obj.split()[:4]
                obj = " ".join(obj_words)
                if any(subj in e or e in subj for e in known_entities):
                    relations.append((subj, "uses", obj))

            # "X such as Y, Z, ..." — expands to multiple is_a relations
            m = re.search(
                r"(\b[a-z][a-z\s\-]{2,30}?)\s+such\s+as\s+([A-Z][\w\s,\-]{2,80})",
                sentence,
            )
            if m:
                category = m.group(1).strip()
                examples_str = m.group(2).strip().rstrip("., ")
                examples = [e.strip() for e in re.split(r",\s*(?:and\s+)?", examples_str)]
                for example in examples:
                    if example and example[0].isupper() and len(example.split()) <= 4:
                        relations.append((example, "is_a", category))

        return relations


# --- Knowledge Graph Builder ---


# Trailing prepositions/conjunctions to strip from relation objects
_TRAILING_JUNK = re.compile(
    r"\s+(?:with|of|for|in|to|from|by|at|on|and|or|that|which|where|through)$",
    re.IGNORECASE,
)

# Words that indicate a phrase is a verb fragment, not an entity
_VERB_STARTS = frozenset({
    "are", "is", "was", "were", "has", "have", "had", "can", "could",
    "will", "would", "should", "may", "might", "do", "does", "did",
    "being", "been",
})

# Adverbs that shouldn't trail entity subjects
_TRAILING_ADVERBS = frozenset({
    "natively", "typically", "commonly", "generally", "usually",
    "effectively", "essentially", "primarily", "mainly", "also",
})


class KnowledgeGraphBuilder:
    """Builds a NetworkX DiGraph from document chunks."""

    def __init__(self, use_gliner: bool = False):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._aliases: Dict[str, str] = {}
        # Case-insensitive lookup: lowercase name → canonical node name
        self._case_index: Dict[str, str] = {}

        if use_gliner:
            try:
                self._extractor = GLiNEREntityExtractor()
                print("Using GLiNER for entity extraction.")
            except Exception as e:
                print(f"Warning: GLiNER unavailable ({e}), falling back to regex.")
                self._extractor = EntityExtractor()
        else:
            self._extractor = EntityExtractor()

    def add_chunk(self, text: str, source: str) -> None:
        """Process one chunk: extract entities and relations, add to graph."""
        # Pass 1: find acronym expansions
        new_aliases = self._extractor.extract_acronym_aliases(text)
        self._aliases.update(new_aliases)

        # Pass 2: extract entities
        entities = self._extractor.extract_entities(text)
        canonical_entities = set()
        for entity in entities:
            canonical = self._resolve(entity)
            canonical_entities.add(canonical)
            self._upsert_node(
                canonical,
                source,
                aliases=[entity] if entity != canonical else [],
            )

        # Pass 3: extract relations with cleanup
        relations = self._extractor.extract_relations(text, canonical_entities)
        for subj, rel, obj in relations:
            # Skip if subject or object look like sentence fragments
            if len(subj.split()) > 4 or len(obj.split()) > 4:
                continue

            # Clean subject: skip if starts with article or ends with adverb
            subj_words = subj.split()
            if subj_words and subj_words[0].lower() in ("the", "a", "an"):
                continue
            if subj_words and subj_words[-1].lower() in _TRAILING_ADVERBS:
                continue

            # Clean object: skip if starts with a verb (fragment indicator)
            obj_words = obj.split()
            if obj_words and obj_words[0].lower() in _VERB_STARTS:
                continue

            # Strip trailing prepositions from object
            obj = _TRAILING_JUNK.sub("", obj).strip()
            if not obj or len(obj) < 3:
                continue

            # Skip single generic lowercase words as relation-created nodes
            if len(obj.split()) == 1 and obj[0].islower() and len(obj) < 5:
                continue

            subj_c = self._resolve(subj)
            obj_c = self._resolve(obj)
            self._upsert_node(subj_c, source)
            self._upsert_node(obj_c, source)
            self._upsert_edge(subj_c, obj_c, rel, source)

    def _resolve(self, term: str) -> str:
        """Resolve an acronym to its long form, then deduplicate by case.

        Priority: acronym alias → case-insensitive existing node → original term.
        First-seen form becomes canonical (preserves original casing).
        """
        # Acronym expansion first
        term = self._aliases.get(term, term)
        # Case-insensitive dedup: reuse existing canonical form
        lower = term.lower()
        if lower in self._case_index:
            return self._case_index[lower]
        # First time seeing this (case-insensitive) — register it
        self._case_index[lower] = term
        return term

    def _upsert_node(
        self, name: str, source: str, aliases: Optional[List[str]] = None
    ) -> None:
        if self.graph.has_node(name):
            data = self.graph.nodes[name]
            data["mention_count"] += 1
            if source not in data["sources"]:
                data["sources"].append(source)
            if aliases:
                for a in aliases:
                    if a not in data["aliases"]:
                        data["aliases"].append(a)
        else:
            self.graph.add_node(
                name,
                type=self._classify_type(name),
                sources=[source],
                mention_count=1,
                aliases=aliases or [],
            )

    def _upsert_edge(
        self, subj: str, obj: str, relation: str, source: str
    ) -> None:
        if subj == obj:
            return
        if self.graph.has_edge(subj, obj):
            self.graph[subj][obj]["weight"] += 1.0
            if source not in self.graph[subj][obj]["sources"]:
                self.graph[subj][obj]["sources"].append(source)
        else:
            self.graph.add_edge(
                subj,
                obj,
                relation=relation,
                weight=1.0,
                sources=[source],
            )

    @staticmethod
    def _classify_type(name: str) -> str:
        """Heuristic entity type classification."""
        if re.match(r"^[A-Z][a-z]+[A-Z]", name):
            return "technology"
        if re.match(r"^[A-Z][A-Z0-9]+$", name):
            return "acronym"
        return "concept"

    def save(self, path: Path) -> None:
        """Save graph to JSON."""
        data = nx.node_link_data(self.graph, edges="edges")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraphBuilder":
        """Load graph from JSON."""
        builder = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        builder.graph = nx.node_link_graph(data, directed=True, edges="edges")
        # Rebuild aliases and case index from node data
        for node, node_data in builder.graph.nodes(data=True):
            builder._case_index[node.lower()] = node
            for alias in node_data.get("aliases", []):
                builder._aliases[alias] = node
        return builder

    def stats(self) -> dict:
        """Return basic graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "top_entities": sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("mention_count", 0),
                reverse=True,
            )[:10],
        }


# --- Query-Time Expansion ---


class GraphQueryExpander:
    """Find entities in a query and expand via graph neighbors."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        # Build lowercase lookup index: name/alias -> canonical node name
        self._index: Dict[str, str] = {}
        for node, data in graph.nodes(data=True):
            self._index[node.lower()] = node
            for alias in data.get("aliases", []):
                self._index[alias.lower()] = node

    def find_entities_in_query(self, query: str) -> List[str]:
        """Match graph nodes against query text by substring."""
        query_lower = query.lower()
        matched = set()
        # Check longest keys first to prefer specific matches
        for key in sorted(self._index, key=len, reverse=True):
            if key in query_lower:
                matched.add(self._index[key])
        return list(matched)

    def expand(self, query: str, depth: int = 1) -> List[str]:
        """Return related entity names to augment the search query.

        Traverses the graph outward (and inward) from matched entities
        up to `depth` hops. Returns only terms not already in the query.
        """
        seed_entities = self.find_entities_in_query(query)
        expanded = set(seed_entities)

        for entity in seed_entities:
            if not self.graph.has_node(entity):
                continue
            # Forward neighbors within depth hops
            for neighbor in nx.single_source_shortest_path_length(
                self.graph, entity, cutoff=depth
            ):
                expanded.add(neighbor)
            # Reverse neighbors (entities that point to this one)
            for predecessor in self.graph.predecessors(entity):
                expanded.add(predecessor)

        # Return only terms not already in the query
        query_lower = query.lower()
        return [e for e in expanded if e.lower() not in query_lower]

    def get_context_summary(self, entities: List[str], max_lines: int = 5) -> str:
        """Build a short text summary of entity relationships."""
        lines = []
        for entity in entities:
            if not self.graph.has_node(entity):
                continue
            for _, target, data in self.graph.out_edges(entity, data=True):
                lines.append(f"{entity} {data['relation']} {target}")
                if len(lines) >= max_lines:
                    return "; ".join(lines)
        return "; ".join(lines)


# --- Obsidian Export ---


def export_obsidian(graph: nx.DiGraph, output_dir: Path) -> int:
    """Write one .md file per node with [[wikilinks]] to neighbors.

    Returns number of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for node, data in graph.nodes(data=True):
        safe_name = re.sub(r"[^\w\s\-]", "", node).strip()
        if not safe_name:
            continue

        lines = [f"# {node}"]
        lines.append(f"\n**Type**: {data.get('type', 'unknown')}")
        if data.get("aliases"):
            lines.append(f"**Aliases**: {', '.join(data['aliases'])}")
        lines.append(f"**Mentions**: {data.get('mention_count', 0)}")
        lines.append(f"**Sources**: {', '.join(data.get('sources', []))}")

        out_edges = list(graph.out_edges(node, data=True))
        if out_edges:
            lines.append("\n## Relationships")
            for _, target, edata in out_edges:
                lines.append(f"- {edata['relation']}: [[{target}]]")

        in_edges = list(graph.in_edges(node, data=True))
        if in_edges:
            lines.append("\n## Referenced by")
            for source_node, _, edata in in_edges:
                lines.append(f"- [[{source_node}]] ({edata['relation']})")

        (output_dir / f"{safe_name}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        count += 1

    return count
