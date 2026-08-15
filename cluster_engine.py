"""
cluster_engine.py
=================
Embed paper abstracts with BAAI/bge-small-en-v1.5 (CPU-friendly, 384-dim),
store vectors in an ephemeral ChromaDB collection, and cluster forward
citations using sklearn.cluster.HDBSCAN.

Outliers / Noise
----------------
Papers assigned cluster label -1 by HDBSCAN are placed into a "noise" bucket
rather than being discarded.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_COLLECTION_NAME = "research_papers"


class ClusterEngine:
    """Embed, store in ChromaDB, query text, and cluster paper abstracts."""

    def __init__(self, min_cluster_size: int = 3, min_samples: int = 1):
        logger.info("Loading SentenceTransformer model: %s", _MODEL_NAME)
        self.model = SentenceTransformer(_MODEL_NAME)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self._chroma = chromadb.EphemeralClient()
        self._active_collection: chromadb.Collection | None = None

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Create or reset ephemeral ChromaDB collection."""
        try:
            self._chroma.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        collection = self._chroma.create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )
        self._active_collection = collection
        return collection

    def embed_and_store(
        self, papers: list[dict[str, Any]]
    ) -> tuple[chromadb.Collection, np.ndarray]:
        """
        Embed all paper abstracts and store them in ChromaDB.
        Returns the active Chroma collection and embedding matrix (numpy array).
        """
        collection = self._get_or_create_collection()

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, str]] = []

        for p in papers:
            pid = p.get("paperId", "")
            if not pid:
                continue
            abstract = p.get("abstract") or "Abstract not available."
            title = p.get("title") or "Untitled"
            texts.append(abstract)
            ids.append(pid)
            metadatas.append(
                {
                    "title": str(title)[:512],
                    "year": str(p.get("year") or ""),
                    "citationCount": str(p.get("citationCount", 0)),
                }
            )

        if not texts:
            logger.warning("No valid paper abstracts to embed.")
            return collection, np.array([])

        logger.info("Embedding %d abstracts with %s...", len(texts), _MODEL_NAME)
        embeddings: np.ndarray = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Stored %d vectors into ChromaDB ephemeral collection.", len(ids))
        return collection, embeddings

    def cluster(
        self, papers: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Cluster forward citation papers using HDBSCAN.
        Returns mapping of cluster label -> list of paper dicts.
        Outliers are stored under key "noise".
        """
        if len(papers) < 2:
            logger.warning("Fewer than 2 papers to cluster (%d). Returning single cluster '0'.", len(papers))
            return {"0": papers}

        collection, embeddings = self.embed_and_store(papers)

        if embeddings.size == 0:
            return {"noise": papers}

        effective_min = min(self.min_cluster_size, max(2, len(papers) // 3))

        clusterer = HDBSCAN(
            min_cluster_size=effective_min,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels: np.ndarray = clusterer.fit_predict(embeddings)

        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        valid_papers = [p for p in papers if p.get("paperId")]

        for paper, label in zip(valid_papers, labels):
            cluster_key = "noise" if label == -1 else str(label)
            clusters[cluster_key].append(paper)

        n_clusters = len([k for k in clusters if k != "noise"])
        n_noise = len(clusters.get("noise", []))
        logger.info(
            "HDBSCAN clustering complete: %d thematic clusters found, %d noise papers.",
            n_clusters,
            n_noise,
        )
        return dict(clusters)

    def retrieve_text_chunks(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Query stored ChromaDB vectors for text chunks matching a query string."""
        if not self._active_collection:
            return []
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True).tolist()
            res = self._active_collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self._active_collection.count() or 1),
            )
            results: list[dict[str, Any]] = []
            if res and res.get("documents") and len(res["documents"]) > 0:
                docs = res["documents"][0]
                ids = res["ids"][0] if res.get("ids") else []
                metas = res["metadatas"][0] if res.get("metadatas") else []
                for doc, pid, meta in zip(docs, ids, metas):
                    results.append({"paperId": pid, "text": doc, "metadata": meta})
            return results
        except Exception as exc:
            logger.warning("Error retrieving text chunks from ChromaDB: %s", exc)
            return []

    def get_cluster_summary(self, cluster_papers: list[dict[str, Any]]) -> str:
        """Format cluster paper titles and metadata into a summary list."""
        lines = [
            f"- [{p.get('year', '?')}] {p.get('title', 'Untitled')} (Citations: {p.get('citationCount', 0)})"
            for p in cluster_papers[:10]
        ]
        if len(cluster_papers) > 10:
            lines.append(f"  … and {len(cluster_papers) - 10} more papers.")
        return "\n".join(lines)
