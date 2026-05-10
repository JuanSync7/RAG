# @summary
# Interactive HTML graph visualization using Sigma.js + graphology from CDN.
# Exports: export_html
# Deps: json, pathlib, src.knowledge_graph.backend
# @end-summary
"""Interactive HTML graph visualization using Sigma.js.

Generates a single self-contained HTML file with embedded graph data,
Sigma.js v3 and graphology loaded from CDN.  No pip dependency required.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge_graph.community import CommunityDetector

from src.knowledge_graph.backend import GraphStorageBackend

__all__ = ["export_html"]

logger = logging.getLogger("rag.knowledge_graph.export.sigma")

# Deterministic color palette for types/communities
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

# Edge style categories
_EDGE_STYLES: Dict[str, Dict[str, str]] = {
    "connects_to":  {"color": "#e74c3c", "type": "dashed"},
    "contains":     {"color": "#95a5a6", "type": "line"},
    "instantiates": {"color": "#3498db", "type": "line"},
    "depends_on":   {"color": "#2ecc71", "type": "line"},
    "specified_by": {"color": "#9b59b6", "type": "dotted"},
}
_DEFAULT_EDGE_STYLE = {"color": "#cccccc", "type": "line"}


def _type_color(type_name: str) -> str:
    """Deterministic color from type name."""
    idx = int(hashlib.md5(type_name.encode()).hexdigest(), 16) % len(_PALETTE)
    return _PALETTE[idx]


def _build_graph_json(
    backend: GraphStorageBackend,
    community_detector: Optional["CommunityDetector"] = None,
) -> Dict[str, Any]:
    """Build the graph data structure for Sigma.js rendering."""
    entities = backend.get_all_entities()
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    type_colors: Dict[str, str] = {}
    edge_id = 0

    for entity in entities:
        community_id = None
        if community_detector and community_detector.is_ready:
            community_id = community_detector.get_community_for_entity(entity.name)

        color = _type_color(entity.type)
        type_colors[entity.type] = color

        if community_id is not None and community_id >= 0:
            color = _PALETTE[community_id % len(_PALETTE)]

        size = max(3, min(20, entity.mention_count or 1))

        nodes.append({
            "key": entity.name,
            "attributes": {
                "label": entity.name,
                "type": entity.type,
                "community": community_id,
                "size": size,
                "color": color,
                "sources": ", ".join(entity.sources[:3]),
                "mentions": entity.mention_count,
            },
        })

        # Outgoing edges
        out_edges = backend.get_outgoing_edges(entity.name)
        for triple in out_edges:
            style = _EDGE_STYLES.get(triple.predicate, _DEFAULT_EDGE_STYLE)
            edges.append({
                "key": f"e{edge_id}",
                "source": triple.subject,
                "target": triple.object,
                "attributes": {
                    "label": triple.predicate,
                    "color": style["color"],
                    "type": style["type"],
                },
            })
            edge_id += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "legend": {
            "type_colors": type_colors,
            "edge_styles": {k: v["color"] for k, v in _EDGE_STYLES.items()},
        },
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RagWeave Knowledge Graph</title>
  <script src="https://unpkg.com/graphology@0.25.4/dist/graphology.umd.min.js"></script>
  <script src="https://unpkg.com/sigma@3/build/sigma.min.js"></script>
  <script src="https://unpkg.com/graphology-layout-forceatlas2@0.10.1/dist/graphology-layout-forceatlas2.umd.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; overflow: hidden; }
    #graph-container { width: 100vw; height: 100vh; }
    #search-container {
      position: absolute; top: 10px; left: 10px; z-index: 10;
      background: rgba(255,255,255,0.95); padding: 8px; border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    #search-input {
      width: 250px; padding: 6px 10px; border: 1px solid #ddd;
      border-radius: 4px; font-size: 14px;
    }
    #legend {
      position: absolute; bottom: 10px; left: 10px; z-index: 10;
      background: rgba(255,255,255,0.95); padding: 10px; border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15); max-height: 200px; overflow-y: auto;
      font-size: 12px;
    }
    .legend-item { display: flex; align-items: center; margin: 3px 0; }
    .legend-dot {
      width: 10px; height: 10px; border-radius: 50%;
      margin-right: 6px; flex-shrink: 0;
    }
    #tooltip {
      position: absolute; z-index: 20; background: rgba(0,0,0,0.85);
      color: white; padding: 8px 12px; border-radius: 4px; font-size: 12px;
      pointer-events: none; display: none; max-width: 300px;
    }
    #info {
      position: absolute; top: 10px; right: 10px; z-index: 10;
      background: rgba(255,255,255,0.95); padding: 8px 12px; border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15); font-size: 13px;
    }
  </style>
</head>
<body>
  <div id="search-container">
    <input id="search-input" type="text" placeholder="Search entities..." />
  </div>
  <div id="graph-container"></div>
  <div id="legend"></div>
  <div id="tooltip"></div>
  <div id="info"></div>
  <script>
    const graphData = __GRAPH_DATA__;

    // Build graph
    const graph = new graphology.Graph({multi: false, type: "directed"});
    graphData.nodes.forEach(n => graph.addNode(n.key, n.attributes));
    graphData.edges.forEach(e => {
      if (graph.hasNode(e.source) && graph.hasNode(e.target)) {
        try { graph.addEdge(e.source, e.target, e.attributes); } catch(err) {}
      }
    });

    // Layout
    const settings = graphologyLayoutForceAtlas2.inferSettings(graph);
    graphologyLayoutForceAtlas2.assign(graph, {settings, iterations: 100});

    // Render
    const container = document.getElementById("graph-container");
    const renderer = new Sigma(graph, container, {
      renderEdgeLabels: false,
      labelRenderedSizeThreshold: 8,
      defaultEdgeType: "arrow",
    });

    // Info
    document.getElementById("info").textContent =
      `${graph.order} nodes, ${graph.size} edges`;

    // Legend
    const legendEl = document.getElementById("legend");
    const types = graphData.legend.type_colors;
    Object.entries(types).forEach(([type, color]) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `<span class="legend-dot" style="background:${color}"></span>${type}`;
      legendEl.appendChild(item);
    });

    // Search
    const searchInput = document.getElementById("search-input");
    searchInput.addEventListener("input", () => {
      const query = searchInput.value.toLowerCase();
      graph.forEachNode((node, attrs) => {
        const match = !query || attrs.label.toLowerCase().includes(query);
        graph.setNodeAttribute(node, "hidden", !match && query.length > 0);
      });
      renderer.refresh();
    });

    // Tooltip
    const tooltip = document.getElementById("tooltip");
    renderer.on("enterNode", ({node}) => {
      const attrs = graph.getNodeAttributes(node);
      tooltip.innerHTML = `<b>${attrs.label}</b><br>Type: ${attrs.type}<br>Mentions: ${attrs.mentions}<br>Sources: ${attrs.sources || "n/a"}`;
      tooltip.style.display = "block";
    });
    renderer.on("leaveNode", () => { tooltip.style.display = "none"; });
    renderer.getMouseCaptor().on("mousemove", (e) => {
      tooltip.style.left = e.x + 15 + "px";
      tooltip.style.top = e.y + 15 + "px";
    });
  </script>
</body>
</html>"""


def export_html(
    backend: GraphStorageBackend,
    output_path: str,
    community_detector: Optional["CommunityDetector"] = None,
) -> int:
    """Generate a self-contained interactive HTML graph visualization.

    Args:
        backend: Graph storage backend to export from.
        output_path: Path for the output HTML file.
        community_detector: Optional detector for community-based coloring.

    Returns:
        Number of nodes rendered.
    """
    graph_data = _build_graph_json(backend, community_detector)
    html = _HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(graph_data))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    node_count = len(graph_data["nodes"])
    logger.info("Exported %d nodes to %s", node_count, output_path)
    return node_count
