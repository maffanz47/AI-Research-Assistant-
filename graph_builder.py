"""
graph_builder.py
================
Build a NetworkX directed citation graph (1-hop scope).

Edge categorisation
-------------------
- **Extends** : `isInfluential == True` OR intent ∈ {"methodology", "result", "extends"}
- **Mentions** : anything else (background-only or unclassified intent)

Nodes store metadata: title, abstract, year, authors, citationCount, node_type (seed | reference | citation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import networkx as nx
from pyvis.network import Network

logger = logging.getLogger(__name__)

_EXTENDS_INTENTS = {"methodology", "result", "extends"}


def _edge_label(intents: list[str], is_influential: bool) -> str:
    """Determine edge bucket ('Extends' vs 'Mentions')."""
    normalised = {str(i).lower().strip() for i in intents if i}
    if is_influential or bool(normalised & _EXTENDS_INTENTS):
        return "Extends"
    return "Mentions"


class GraphBuilder:
    """Constructs and visualises directed citation networks using NetworkX and Pyvis."""

    def build(self, neighbourhood: dict[str, Any]) -> nx.DiGraph:
        """
        Build a directed graph from a neighbourhood dictionary.

        Edges:
        - seed -> reference (seed cites reference)
        - citation -> seed (citing paper cites seed)
        """
        G = nx.DiGraph()
        seed = neighbourhood.get("seed", {})
        seed_id = seed.get("paperId", "seed")

        # Add seed node
        G.add_node(seed_id, node_type="seed", **seed)

        # Outbound references (seed -> reference)
        for paper in neighbourhood.get("references", []):
            pid = paper.get("paperId")
            if not pid:
                continue
            node_attrs = {k: v for k, v in paper.items() if k not in ("intents", "isInfluential")}
            node_attrs["node_type"] = "reference"
            G.add_node(pid, **node_attrs)

            label = _edge_label(paper.get("intents", []), paper.get("isInfluential", False))
            G.add_edge(
                seed_id,
                pid,
                edge_type=label,
                intents=paper.get("intents", []),
                isInfluential=paper.get("isInfluential", False),
            )

        # Inbound citations (citation -> seed)
        for paper in neighbourhood.get("citations", []):
            pid = paper.get("paperId")
            if not pid:
                continue
            node_attrs = {k: v for k, v in paper.items() if k not in ("intents", "isInfluential")}
            node_attrs["node_type"] = "citation"
            G.add_node(pid, **node_attrs)

            label = _edge_label(paper.get("intents", []), paper.get("isInfluential", False))
            G.add_edge(
                pid,
                seed_id,
                edge_type=label,
                intents=paper.get("intents", []),
                isInfluential=paper.get("isInfluential", False),
            )

        extends_count = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "Extends")
        mentions_count = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "Mentions")
        logger.info(
            "Graph constructed: %d nodes, %d edges (%d Extends, %d Mentions)",
            G.number_of_nodes(),
            G.number_of_edges(),
            extends_count,
            mentions_count,
        )
        return G

    def to_pyvis_html(self, G: nx.DiGraph, seed_id: str) -> str:
        """
        Render graph as self-contained Pyvis HTML.
        - Seed node: Gold (#FFD700)
        - Reference nodes: Steel blue (#4682B4)
        - Citation nodes: Medium sea green (#3CB371)
        - Extends edge: Crimson red (#FF4B4B)
        - Mentions edge: Muted grey (#888888)
        """
        net = Network(
            height="600px",
            width="100%",
            directed=True,
            bgcolor="#0e1117",
            font_color="#ffffff",
        )
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

        color_map = {
            "seed": "#FFD700",
            "reference": "#4682B4",
            "citation": "#3CB371",
        }
        edge_color_map = {
            "Extends": "#FF4B4B",
            "Mentions": "#888888",
        }

        for node, attrs in G.nodes(data=True):
            ntype = attrs.get("node_type", "citation")
            title_str = attrs.get("title", str(node))
            year = attrs.get("year", "?")
            abstract = (attrs.get("abstract") or "")[:200]
            tooltip = f"<b>{title_str}</b> ({year})<br/>{abstract}"

            net.add_node(
                str(node),
                label=title_str[:40] + "…" if len(title_str) > 40 else title_str,
                title=tooltip,
                color=color_map.get(ntype, "#888888"),
                size=22 if str(node) == str(seed_id) else 12,
                borderWidth=3 if str(node) == str(seed_id) else 1,
            )

        for src, dst, attrs in G.edges(data=True):
            etype = attrs.get("edge_type", "Mentions")
            net.add_edge(
                str(src),
                str(dst),
                color=edge_color_map.get(etype, "#888888"),
                title=f"Edge type: {etype}",
                width=3 if etype == "Extends" else 1,
                arrows="to",
            )

        return net.generate_html()

    @staticmethod
    def graph_to_dict(G: nx.DiGraph) -> dict[str, Any]:
        """Serialise NetworkX graph for API JSON responses."""
        return {
            "nodes": [
                {
                    "id": str(n),
                    **{
                        k: v
                        for k, v in d.items()
                        if isinstance(v, (str, int, float, bool, type(None), list))
                    },
                }
                for n, d in G.nodes(data=True)
            ],
            "edges": [
                {
                    "source": str(u),
                    "target": str(v),
                    **{
                        k: vv
                        for k, vv in d.items()
                        if isinstance(vv, (str, int, float, bool, type(None), list))
                    },
                }
                for u, v, d in G.edges(data=True)
            ],
        }
