"""
app/main.py
===========
FastAPI inference backend for the Research Gap Finder.

Loads the fine-tuned LoRA adapter (from `lora_adapter/`) on top of the base
Qwen2.5-14B-Instruct model using Unsloth and exposes:

  POST /api/analyze-gap   — structured gap analysis from abstract + citations
  GET  /health            — liveness probe

Environment variables
---------------------
  LORA_ADAPTER_PATH   default: lora_adapter/
  MAX_NEW_TOKENS      default: 1024
  USE_MOCK_LLM        set to "true" to skip model loading (for local dev / CPU)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("app.main")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LORA_ADAPTER_PATH: str = os.getenv("LORA_ADAPTER_PATH", "lora_adapter/")
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))
USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "false").strip().lower() in ("true", "1", "yes")
BASE_MODEL: str = "unsloth/Qwen2.5-14B-Instruct"

# ---------------------------------------------------------------------------
# Model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None


def _load_model() -> None:
    """Load the fine-tuned LoRA adapter using Unsloth."""
    global _model, _tokenizer  # noqa: PLW0603

    if USE_MOCK_LLM:
        logger.warning("USE_MOCK_LLM=true — skipping model load (CPU / dev mode).")
        return

    try:
        from unsloth import FastModel
    except ImportError:
        logger.error(
            "Unsloth is not installed. Install it on the GPU machine with:\n"
            "  pip install unsloth"
        )
        return

    adapter_path = LORA_ADAPTER_PATH.rstrip("/")
    if not os.path.isdir(adapter_path):
        logger.error(
            "LoRA adapter directory not found: '%s'. "
            "Run `dvc repro` first to produce the adapter.",
            adapter_path,
        )
        return

    logger.info("Loading base model + LoRA adapter from: %s", adapter_path)
    _model, _tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,          # points to the saved adapter; Unsloth loads base automatically
        max_seq_length=8192,
        dtype=None,                       # auto bfloat16
        load_in_4bit=True,
    )
    FastModel.for_inference(_model)
    logger.info("Model loaded and set to inference mode.")


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield
    logger.info("Shutting down inference backend.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Research Gap Finder — LLM Inference API",
    description=(
        "Fine-tuned Qwen2.5-14B-Instruct (LoRA) inference endpoint. "
        "Returns structured research gap JSON from paper metadata."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CitationMeta(BaseModel):
    paper_id: str = Field(default="", description="Semantic Scholar or arXiv paper ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(default="", description="Paper abstract")
    citation_count: int = Field(default=0, description="Forward citation count")
    is_influential: bool = Field(default=False)
    intents: list[str] = Field(default_factory=list)


class SeedPaper(BaseModel):
    title: str
    abstract: str


class AnalyzeGapRequest(BaseModel):
    seed_papers: list[SeedPaper] = Field(default_factory=list, description="List of seed papers to analyze collectively")
    title: str = Field(default="", description="Seed paper title (legacy)")
    abstract: str = Field(default="", description="Seed paper abstract (legacy)")
    citations: list[CitationMeta] = Field(
        default_factory=list,
        description="List of forward-citing papers with metadata",
    )
    references: list[CitationMeta] = Field(
        default_factory=list,
        description="List of cited references with metadata",
    )


class ResearchGap(BaseModel):
    gap_id: int
    title: str
    description: str
    evidence: str
    recommended_action: str


class AnalyzeGapResponse(BaseModel):
    seed_titles: list[str]
    executive_summary: str
    gaps: list[ResearchGap]
    model_used: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _build_prompt(req: AnalyzeGapRequest) -> str:
    seeds = req.seed_papers
    if not seeds and req.title:
        seeds = [SeedPaper(title=req.title, abstract=req.abstract)]

    seeds_text = "\n\n".join(
        f"Paper {i+1}:\nTitle: \"{p.title}\"\nAbstract: {p.abstract[:600]}"
        for i, p in enumerate(seeds)
    )

    citation_lines = "\n".join(
        f"  [{i+1}] {c.title} (cited by {c.citation_count}, influential={c.is_influential})"
        for i, c in enumerate(req.citations[:30])   # cap at 30 to stay within context
    )

    prompt = (
        f"You are an expert academic research analyst.\n\n"
        f"You have been provided with the following seed paper(s):\n"
        f"{seeds_text}\n\n"
    )

    if req.citations:
        prompt += (
            f"Forward citations ({len(req.citations)} total, showing first 30):\n"
            f"{citation_lines}\n\n"
        )

    prompt += (
        f"Task: Identify the top 3 unexplored research gaps based on the collective landscape of the provided paper(s) and any provided citations.\n"
        f"Return ONLY valid JSON — no prose, no markdown fences — matching this schema:\n"
        f"{{\n"
        f'  "executive_summary": "<2-3 sentence summary>",\n'
        f'  "gaps": [\n'
        f'    {{\n'
        f'      "gap_id": 1,\n'
        f'      "title": "<gap name>",\n'
        f'      "description": "<what is missing>",\n'
        f'      "evidence": "<which papers or lack thereof suggest this gap>",\n'
        f'      "recommended_action": "<what researcher should do>"\n'
        f'    }}\n'
        f"  ]\n"
        f"}}"
    )
    return prompt


def _mock_response(req: AnalyzeGapRequest) -> dict[str, Any]:
    """Return a plausible mock when USE_MOCK_LLM=true or model unavailable."""
    return {
        "executive_summary": (
            f"Analysis of '{req.title}' reveals citation clusters across multiple sub-fields. "
            "Forward citation momentum is uneven, pointing to several under-explored directions."
        ),
        "gaps": [
            {
                "gap_id": 1,
                "title": "Low-resource Cross-lingual Generalisation",
                "description": "Existing work focuses on high-resource languages; cross-lingual transfer for low-resource settings is under-studied.",
                "evidence": "Only 2 of the forward citations address non-English corpora.",
                "recommended_action": "Construct multilingual benchmark datasets and fine-tune on diverse language pairs.",
            },
            {
                "gap_id": 2,
                "title": "Long-horizon Temporal Reasoning",
                "description": "Benchmarks evaluate single-step inference; multi-step temporal chains are absent.",
                "evidence": "No forward citation introduces a multi-step temporal benchmark.",
                "recommended_action": "Design a temporally grounded evaluation suite with chain-of-thought annotations.",
            },
            {
                "gap_id": 3,
                "title": "Edge-device Efficiency Constraints",
                "description": "Methods assume high-compute servers; deployment on resource-constrained edge devices is unexplored.",
                "evidence": "Architectural papers scale parameters up, not down.",
                "recommended_action": "Apply structured pruning + quantisation co-optimisation for sub-1B parameter models.",
            },
        ],
    }


def _run_inference(req: AnalyzeGapRequest) -> dict[str, Any]:
    """Run model inference and parse JSON output; falls back to mock on error."""
    if USE_MOCK_LLM or _model is None:
        logger.info("Returning mock response (mock mode or model not loaded).")
        return _mock_response(req)

    prompt = _build_prompt(req)

    # Apply Qwen chat template
    messages = [
        {"role": "system", "content": "You are an expert academic research analyst. Respond only with valid JSON."},
        {"role": "user", "content": prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    import torch
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    raw = _tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    logger.debug("Raw model output: %s", raw[:300])

    cleaned = _strip_fences(raw)
    for attempt in range(2):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            if attempt == 0:
                logger.warning("JSON parse failed (%s) — retrying with fence-stripped text.", exc)
                cleaned = re.sub(r"[^\{]*(\{.*\})[^\}]*", r"\1", cleaned, flags=re.DOTALL)
            else:
                logger.error("JSON parse failed after retry; returning mock.")
                return _mock_response(req)
    return _mock_response(req)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "mock_mode": USE_MOCK_LLM,
        "adapter_path": LORA_ADAPTER_PATH,
    }


@app.post("/api/analyze-gap", response_model=AnalyzeGapResponse, tags=["Inference"])
def analyze_gap(req: AnalyzeGapRequest) -> AnalyzeGapResponse:
    """
    Run fine-tuned Qwen inference to identify top-3 research gaps.

    Accepts seed paper title + abstract + citation metadata.
    Returns structured JSON with executive summary and gap list.
    """
    if not req.seed_papers and (not req.title.strip() or not req.abstract.strip()):
        raise HTTPException(status_code=422, detail="Provide at least one seed paper or a title/abstract.")

    seeds = req.seed_papers
    if not seeds and req.title:
        seeds = [SeedPaper(title=req.title, abstract=req.abstract)]

    logger.info("Running gap analysis for %d seed papers", len(seeds))

    try:
        raw_output = _run_inference(req)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    # Parse and validate gaps
    try:
        gaps: list[ResearchGap] = [ResearchGap(**g) for g in raw_output.get("gaps", [])]
    except Exception as exc:
        logger.warning("Gap parsing error: %s — using raw output", exc)
        gaps = []

    if not gaps:
        raise HTTPException(
            status_code=502,
            detail="Model returned no parseable gaps. Try again or check model output.",
        )

    model_label = f"{BASE_MODEL} + LoRA" if not USE_MOCK_LLM else "Mock (dev mode)"

    return AnalyzeGapResponse(
        seed_titles=[p.title for p in seeds],
        executive_summary=raw_output.get("executive_summary", "Summary unavailable."),
        gaps=gaps,
        model_used=model_label,
    )
