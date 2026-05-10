# @summary
# Typed data contracts for KG retrieval query results.
# Exports: PathHop, PathResult, ExpansionResult
# Deps: dataclasses, typing
# @end-summary
"""Typed data contracts for KG retrieval query results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List


__all__ = ["ExpansionResult", "PathHop", "PathResult"]


@dataclass
class ExpansionResult:
    """Return type for GraphQueryExpander.expand().

    Backward-compat: iterating yields the same strings as List[str].
    """

    terms: List[str]
    graph_context: str = field(default="")

    def __iter__(self) -> Iterator[str]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __getitem__(self, index: int) -> str:
        return self.terms[index]


@dataclass
class PathHop:
    """One directed hop in a matched traversal path."""

    from_entity: str
    edge_type: str
    to_entity: str


@dataclass
class PathResult:
    """A fully matched traversal path."""

    pattern_label: str
    seed_entity: str
    hops: List[PathHop]
    terminal_entity: str

    @property
    def length(self) -> int:
        """Number of hops in this path."""
        return len(self.hops)
