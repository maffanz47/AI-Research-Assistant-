"""
api.py
======
FastAPI serving backend for Research Gap Finder.

Endpoints:
- GET  /health   -> {"status": "ok"}
- POST /analyze  -> executes full ReAct pipeline for seed paper ID
- GET  /graph    -> returns self-contained Pyvis HTML citation network
- GET  /report   -> returns gap report text and abandonment ratios
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from agent_engine import ResearchGapAgent  # noqa: E402

app = FastAPI(
    title="Research Gap Finder API",
    description="Fetch citation networks, cluster thematic branches with HDBSCAN, and identify research gaps.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = ResearchGapAgent()

# In-memory cache for graph HTML, reports, and metadata
_cache: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    paper_id: str = Field(
        ...,
        example="1706.03762",
        description="Semantic Scholar Paper ID, arXiv ID (e.g. 1706.03762), or DOI",
    )


class AnalyzeResponse(BaseModel):
    paper_id: str
    report: str
    trace: list[dict[str, Any]]
    clusters: dict[str, int]
    abandonment_ratios: dict[str, float]


class ReportResponse(BaseModel):
    paper_id: str
    report: str
    abandonment_ratios: dict[str, float]
    clusters: dict[str, int]


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Pipeline"])
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full Research Gap Finder analysis pipeline.
    1. Fetches paper neighbourhood (references + forward citations)
    2. Builds directed NetworkX graph with Extends vs Mentions edge buckets
    3. Embeds abstracts with BGE-small & clusters with HDBSCAN
    4. Calculates Abandonment Ratios per branch
    5. Executes ReAct agent loop and returns report & trace
    """
    paper_id = req.paper_id.strip()
    if not paper_id:
        raise HTTPException(status_code=422, detail="paper_id must not be empty.")

    try:
        logger.info("Executing pipeline for paper_id: %s", paper_id)
        result = _agent.run(paper_id)
    except RuntimeError as exc:
        logger.exception("Pipeline failed for paper_id=%s", paper_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected pipeline error for paper_id=%s", paper_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    # Cache result
    cluster_counts = {k: len(v) for k, v in result.clusters.items()}
    _cache[paper_id] = {
        "graph_html": result.graph_html,
        "report": result.report,
        "trace": result.trace,
        "clusters": cluster_counts,
        "abandonment_ratios": result.abandonment_ratios,
    }

    return AnalyzeResponse(
        paper_id=paper_id,
        report=result.report,
        trace=result.trace,
        clusters=cluster_counts,
        abandonment_ratios=result.abandonment_ratios,
    )


@app.get("/graph", response_class=HTMLResponse, tags=["Pipeline"])
def get_graph(paper_id: str = Query(..., description="Paper ID or arXiv ID")) -> HTMLResponse:
    """Return Pyvis interactive HTML visualization for an analyzed paper ID."""
    paper_id = paper_id.strip()
    if paper_id not in _cache:
        # Run pipeline dynamically if not cached
        try:
            analyze(AnalyzeRequest(paper_id=paper_id))
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"No graph found or generated for '{paper_id}': {exc}",
            ) from exc

    return HTMLResponse(content=_cache[paper_id]["graph_html"])


@app.get("/report", response_model=ReportResponse, tags=["Pipeline"])
def get_report(paper_id: str = Query(..., description="Paper ID or arXiv ID")) -> ReportResponse:
    """Return Gap Report markdown and Abandonment Ratios for an analyzed paper ID."""
    paper_id = paper_id.strip()
    if paper_id not in _cache:
        # Run pipeline dynamically if not cached
        try:
            analyze(AnalyzeRequest(paper_id=paper_id))
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"No report found or generated for '{paper_id}': {exc}",
            ) from exc

    cached = _cache[paper_id]
    return ReportResponse(
        paper_id=paper_id,
        report=cached["report"],
        abandonment_ratios=cached["abandonment_ratios"],
        clusters=cached["clusters"],
    )
