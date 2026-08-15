"""
data_client.py
==============
Semantic Scholar Graph API client with two-tier caching:

  Tier 1 — `.data_cache/`  (keyed by endpoint + paper ID)
      Fast per-request cache for individual API calls (paper, refs, citations).

  Tier 2 — `cache/`  (keyed by seed paper ID)
      Offline neighbourhood cache: stores the complete
      {seed, references, citations} bundle produced by `fetch_neighbourhood`.
      Checked first so that a single cache hit skips ALL API calls.
      This directory is tracked by DVC for reproducible ML pipelines.

  Both caches use plain JSON files for maximum portability.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------
BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,year,authors,citationCount"
REF_FIELDS = "paperId,title,abstract,year,authors,citationCount,intents,isInfluential"
PAGE_LIMIT = 100
MAX_RETRIES = 4
BACKOFF_BASE = 2

# ---------------------------------------------------------------------------
# Cache directories
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# Tier-1: fine-grained per-call cache (not tracked by DVC)
_DATA_CACHE_DIR = os.path.join(_HERE, ".data_cache")

# Tier-2: offline full-neighbourhood cache (tracked by DVC)
NEIGHBOURHOOD_CACHE_DIR = os.path.join(_HERE, "cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_paper_id(paper_id: str) -> str:
    """Normalize paper IDs: arXiv shorthand, DOIs, and raw S2 hex IDs."""
    pid = paper_id.strip()
    if not pid:
        return pid
    if pid.lower().startswith("arxiv:"):
        return f"ARXIV:{pid[6:].strip()}"
    if pid.startswith("10.") and not pid.lower().startswith("doi:"):
        return f"DOI:{pid}"
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", pid):
        return f"ARXIV:{pid}"
    return pid


def _safe_filename(text: str) -> str:
    """Convert arbitrary string to a filesystem-safe filename stem."""
    return re.sub(r"[^\w\-_]", "_", text)


# ------------------------------------------------------------------
# Tier-1 helper functions (.data_cache/)
# ------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _tier1_path(key_type: str, item_id: str) -> str:
    _ensure_dir(_DATA_CACHE_DIR)
    return os.path.join(_DATA_CACHE_DIR, f"{key_type}_{_safe_filename(item_id)}.json")


def _read_cache(key_type: str, item_id: str) -> Optional[Any]:
    filepath = _tier1_path(key_type, item_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                logger.debug("Tier-1 cache hit: %s:%s", key_type, item_id)
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read tier-1 cache %s: %s", filepath, exc)
    return None


def _write_cache(key_type: str, item_id: str, data: Any) -> None:
    filepath = _tier1_path(key_type, item_id)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write tier-1 cache %s: %s", filepath, exc)


# ------------------------------------------------------------------
# Tier-2 helper functions (cache/)
# ------------------------------------------------------------------

def _neighbourhood_cache_path(norm_id: str) -> str:
    _ensure_dir(NEIGHBOURHOOD_CACHE_DIR)
    return os.path.join(NEIGHBOURHOOD_CACHE_DIR, f"neighbourhood_{_safe_filename(norm_id)}.json")


def _read_neighbourhood_cache(norm_id: str) -> Optional[dict[str, Any]]:
    filepath = _neighbourhood_cache_path(norm_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                logger.info("Tier-2 (offline) cache hit for neighbourhood: %s", norm_id)
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read neighbourhood cache %s: %s", filepath, exc)
    return None


def _write_neighbourhood_cache(norm_id: str, data: dict[str, Any]) -> None:
    filepath = _neighbourhood_cache_path(norm_id)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Neighbourhood written to tier-2 cache: %s", filepath)
    except Exception as exc:
        logger.warning("Failed to write neighbourhood cache %s: %s", filepath, exc)


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------

def _http_request(
    method: str,
    url: str,
    params: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """HTTP request with exponential backoff on 429 / 5xx responses."""
    headers = {"x-api-key": api_key} if api_key else {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method, url, params=params, json=json_data, headers=headers, timeout=30
            )
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Rate-limited (429). Retrying in %ss (attempt %d/%d).",
                    wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES:
                logger.error("Request failed permanently: %s", exc)
                raise RuntimeError(
                    f"Semantic Scholar API failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                "Request error: %s. Retrying in %ss (%d/%d).", exc, wait, attempt, MAX_RETRIES
            )
            time.sleep(wait)
    return {}


def _clean_paper(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw API paper dict into a standard shape."""
    if not isinstance(raw, dict):
        return {
            "paperId": "", "title": "Untitled",
            "abstract": "Abstract not available.", "year": None,
            "authors": [], "citationCount": 0,
        }
    return {
        "paperId": raw.get("paperId", "") or "",
        "title": raw.get("title") or "Untitled",
        "abstract": raw.get("abstract") or "Abstract not available.",
        "year": raw.get("year"),
        "authors": [
            a.get("name", "")
            for a in (raw.get("authors") or [])
            if isinstance(a, dict) and a.get("name")
        ],
        "citationCount": raw.get("citationCount", 0) or 0,
    }


def _paginate(url: str, fields: str, api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Iterate all pages from a Semantic Scholar list endpoint."""
    results: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = _http_request(
            "GET", url,
            params={"fields": fields, "limit": PAGE_LIMIT, "offset": offset},
            api_key=api_key,
        )
        if not isinstance(payload, dict):
            break
        data: list[dict[str, Any]] = payload.get("data", [])
        results.extend(data)
        next_offset = payload.get("next")
        if not next_offset or not data:
            break
        offset = next_offset
    return results


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class SemanticScholarClient:
    """
    Semantic Scholar Graph API client.

    Caching hierarchy (fetch_neighbourhood):
      1. Tier-2 offline cache (`cache/`) — full neighbourhood bundle
      2. Tier-1 per-call cache (`.data_cache/`) — individual requests
      3. Live API call
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("S2_API_KEY")

    # ------------------------------------------------------------------
    def get_paper(self, paper_id: str) -> dict[str, Any]:
        """Fetch single paper metadata (tier-1 cached)."""
        norm_id = normalize_paper_id(paper_id)
        cached = _read_cache("paper", norm_id)
        if cached is not None:
            return cached
        url = f"{BASE_URL}/paper/{norm_id}"
        raw = _http_request("GET", url, params={"fields": PAPER_FIELDS}, api_key=self.api_key)
        cleaned = _clean_paper(raw if isinstance(raw, dict) else {})
        _write_cache("paper", norm_id, cleaned)
        return cleaned

    # ------------------------------------------------------------------
    def get_papers_batch(self, paper_ids: list[str]) -> list[dict[str, Any]]:
        """Batch-fetch metadata for multiple IDs via POST /paper/batch."""
        if not paper_ids:
            return []
        norm_ids = [normalize_paper_id(pid) for pid in paper_ids]
        results: list[dict[str, Any]] = []
        missing: list[str] = []
        for pid in norm_ids:
            cached = _read_cache("paper", pid)
            if cached is not None:
                results.append(cached)
            else:
                missing.append(pid)
        if not missing:
            return results
        url = f"{BASE_URL}/paper/batch"
        payload = _http_request(
            "POST", url,
            params={"fields": PAPER_FIELDS},
            json_data={"ids": missing},
            api_key=self.api_key,
        )
        if isinstance(payload, list):
            for raw_item in payload:
                if raw_item and isinstance(raw_item, dict):
                    cleaned = _clean_paper(raw_item)
                    if cleaned["paperId"]:
                        _write_cache("paper", cleaned["paperId"], cleaned)
                        results.append(cleaned)
        return results

    # ------------------------------------------------------------------
    def get_references(self, paper_id: str) -> list[dict[str, Any]]:
        """Fetch cited references with intent / isInfluential flags (tier-1 cached)."""
        norm_id = normalize_paper_id(paper_id)
        cached = _read_cache("references", norm_id)
        if cached is not None:
            return cached
        url = f"{BASE_URL}/paper/{norm_id}/references"
        rows = _paginate(url, REF_FIELDS, api_key=self.api_key)
        results: list[dict[str, Any]] = []
        for row in rows:
            paper = _clean_paper(row.get("citedPaper") or {})
            paper["intents"] = row.get("intents") or []
            paper["isInfluential"] = bool(row.get("isInfluential", False))
            if paper["paperId"]:
                results.append(paper)
        _write_cache("references", norm_id, results)
        return results

    # ------------------------------------------------------------------
    def get_citations(self, paper_id: str) -> list[dict[str, Any]]:
        """Fetch forward citations with intent / isInfluential flags (tier-1 cached)."""
        norm_id = normalize_paper_id(paper_id)
        cached = _read_cache("citations", norm_id)
        if cached is not None:
            return cached
        url = f"{BASE_URL}/paper/{norm_id}/citations"
        rows = _paginate(url, REF_FIELDS, api_key=self.api_key)
        results: list[dict[str, Any]] = []
        for row in rows:
            paper = _clean_paper(row.get("citingPaper") or {})
            paper["intents"] = row.get("intents") or []
            paper["isInfluential"] = bool(row.get("isInfluential", False))
            if paper["paperId"]:
                results.append(paper)
        _write_cache("citations", norm_id, results)
        return results

    # ------------------------------------------------------------------
    def fetch_neighbourhood(self, paper_id: str) -> dict[str, Any]:
        """
        Fetch seed + references + citations (1-hop neighbourhood).

        Checks Tier-2 offline cache first. On a cache miss, calls the API
        and persists the result to the offline cache directory (`cache/`).
        """
        norm_id = normalize_paper_id(paper_id)

        # --- Tier-2: offline neighbourhood cache ---
        cached_neighbourhood = _read_neighbourhood_cache(norm_id)
        if cached_neighbourhood is not None:
            return cached_neighbourhood

        # --- Live API calls (tier-1 caching applied per call) ---
        logger.info("Fetching seed paper: %s", norm_id)
        seed = self.get_paper(norm_id)
        logger.info("Fetching references for: %s", norm_id)
        references = self.get_references(norm_id)
        logger.info("Fetching citations for: %s", norm_id)
        citations = self.get_citations(norm_id)

        neighbourhood: dict[str, Any] = {
            "seed": seed,
            "references": references,
            "citations": citations,
        }

        # Persist to tier-2 offline cache
        _write_neighbourhood_cache(norm_id, neighbourhood)
        return neighbourhood
