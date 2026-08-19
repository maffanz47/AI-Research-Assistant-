"""
api.py
======
FastAPI serving backend for Research Gap Finder.

Endpoints:
- GET  /         -> {"status": "online", "gpu": bool}   ← root health check
- GET  /health   -> {"status": "ok"}                    ← legacy health check
- POST /infer    -> accepts abstract text, returns research gaps via Qwen/mock
- POST /analyze  -> executes full ReAct pipeline for seed paper ID
- GET  /graph    -> returns self-contained Pyvis HTML citation network
- GET  /report   -> returns gap report text and abandonment ratios
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import requests as _requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Load .env if present (python-dotenv is optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime config — resolved from env so no .env file is *required*
# ---------------------------------------------------------------------------
USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").strip().lower() not in ("false", "0", "no")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")

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

# Mount the inference API so both run on the same port
try:
    from app.main import app as inference_app
    app.mount("/inference", inference_app)
    logger.info("Successfully mounted Inference API at /inference")
except ImportError as e:
    logger.warning(f"Could not mount Inference API: {e}")

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


@app.get("/", tags=["System"])
def root() -> dict[str, Any]:
    """
    Root health check — reports live GPU availability.
    Safe to call without any setup; torch import is lazy so it never crashes.
    """
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    return {"status": "online", "gpu": gpu_available}


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """Legacy health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /infer — lightweight single-abstract inference (no full ReAct pipeline)
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    abstract: str = Field(
        ...,
        min_length=50,
        description="Full text of a scientific paper abstract.",
        example=(
            "We propose a novel attention mechanism that scales linearly "
            "with sequence length, enabling transformer models to process "
            "documents of up to 1 million tokens on a single GPU."
        ),
    )
    top_k_gaps: int = Field(default=3, ge=1, le=10, description="Number of research gaps to identify.")


class InferResponse(BaseModel):
    gaps: list[str]
    model_used: str
    gpu: bool


@app.post("/infer", response_model=InferResponse, tags=["Inference"])
def infer(req: InferRequest) -> InferResponse:
    """
    Lightweight inference endpoint.

    Accepts a raw paper abstract and returns a bullet-point list of
    research gaps produced by the configured Qwen model (or mock).

    Does NOT run the full ReAct pipeline — suitable for direct frontend
    calls where only gap text is needed.
    """
    abstract = req.abstract.strip()
    if not abstract:
        raise HTTPException(status_code=422, detail="abstract must not be empty.")

    # Detect GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    if USE_MOCK_LLM:
        logger.info("/infer called in MOCK mode.")
        gaps = [
            f"Gap {i}: Placeholder gap #{i} — set USE_MOCK_LLM=false to enable real Qwen inference."
            for i in range(1, req.top_k_gaps + 1)
        ]
        return InferResponse(gaps=gaps, model_used="mock", gpu=gpu_available)

    # ── Real Ollama inference ──────────────────────────────────────────────
    prompt = (
        f"You are an expert research analyst.\n\n"
        f"Given the following paper abstract, identify exactly {req.top_k_gaps} "
        f"specific, actionable research gaps that are NOT addressed by the paper.\n"
        f"Format your response as a numbered list with one gap per line.\n\n"
        f"Abstract:\n{abstract}\n\nResearch Gaps:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1024},
    }
    try:
        resp = _requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        raw_text: str = resp.json().get("response", "")
    except Exception as exc:
        logger.exception("/infer Ollama request failed.")
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    # Parse numbered list lines into individual gap strings
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    gaps = [ln for ln in lines if ln[0].isdigit() or ln.startswith("-")][:req.top_k_gaps]
    if not gaps:
        gaps = lines[:req.top_k_gaps]  # fallback: take whatever lines we got

    logger.info("/infer returned %d gaps via %s.", len(gaps), OLLAMA_MODEL)
    return InferResponse(gaps=gaps, model_used=OLLAMA_MODEL, gpu=gpu_available)


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
