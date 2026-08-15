"""
agent_engine.py
===============
ReAct-style agent orchestration pipeline for Research Gap Finder.

Execution loop:
  Step 1: Reason about seed paper / neighbourhood
  Step 2: Act  — fetch_graph tool
  Step 3: Observe graph metrics (Extends vs Mentions edges)
  Step 4: Act  — cluster forward citations (HDBSCAN)
  Step 5: Compute Abandonment Ratio per cluster
  Step 6: Act  — get_branch_summary + retrieve_text_chunks
  Step 7: Act  — LLM generate (Ollama real OR mock)

LLM control
-----------
Set  USE_MOCK_LLM=false  and  OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M
in .env (or environment variables) to switch to the real Qwen model.

Pydantic schemas enforce the JSON structure emitted by the LLM.
A regex-based fence-stripper cleans ```json ... ``` blocks before parsing.
A 1-retry fallback re-prompts the LLM with the validation error on failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any

import requests as _requests
from pydantic import BaseModel, Field, ValidationError

from cluster_engine import ClusterEngine
from data_client import SemanticScholarClient
from graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime configuration (read from environment / .env)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").strip().lower() not in ("false", "0", "no")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_data_client = SemanticScholarClient(api_key=os.getenv("S2_API_KEY"))
_graph_builder = GraphBuilder()
_cluster_engine = ClusterEngine(min_cluster_size=3, min_samples=1)


# ===========================================================================
# Pydantic schemas — strict output contracts for LLM responses
# ===========================================================================

class ContributionItem(BaseModel):
    """A single extracted contribution from the seed paper."""
    claim: str = Field(..., description="One sentence summary of a contribution.")
    evidence: str = Field(..., description="Direct quote or paraphrase from the abstract.")


class ExtractedContributions(BaseModel):
    """Schema for the contributions extraction step."""
    seed_title: str
    contributions: list[ContributionItem] = Field(..., min_length=1)


class BranchTheme(BaseModel):
    """Thematic label for a single HDBSCAN cluster."""
    cluster_id: str
    theme: str = Field(..., description="One short phrase naming the research direction.")
    representative_works: list[str] = Field(default_factory=list, description="Up to 3 paper titles.")
    abandonment_risk: str = Field(..., description="low | medium | high")


class BranchMapping(BaseModel):
    """Schema for the branch naming step."""
    seed_title: str
    branches: list[BranchTheme] = Field(..., min_length=1)


class ResearchGap(BaseModel):
    """A single actionable research gap."""
    gap_id: int
    title: str
    description: str
    evidence_clusters: list[str] = Field(default_factory=list)
    recommended_action: str


class GapReport(BaseModel):
    """Full structured Gap Report produced by the final LLM step."""
    seed_title: str
    executive_summary: str
    gaps: list[ResearchGap] = Field(..., min_length=1)
    next_steps: list[str] = Field(..., min_length=1)


# ===========================================================================
# Utilities
# ===========================================================================

_FENCE_RE = re.compile(
    r"```(?:json|JSON|markdown|md|text)?\s*(.*?)\s*```",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` (and similar) code fences from an LLM response.
    Falls back to returning the original string if no fence is found.
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ===========================================================================
# Ollama LLM client
# ===========================================================================

def _call_ollama(prompt: str, schema_hint: str = "") -> str:
    """
    POST to the local Ollama /api/generate endpoint.
    Raises RuntimeError on HTTP or connection failures.
    """
    system = (
        "You are a rigorous academic research analyst. "
        "Respond ONLY with valid JSON that matches the required schema — no prose, no markdown fences. "
        + (f"Expected JSON schema: {schema_hint}" if schema_hint else "")
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    try:
        resp = _requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def _parse_with_retry(
    raw: str,
    model_cls: type[BaseModel],
    prompt: str,
    schema_hint: str = "",
) -> BaseModel:
    """
    Parse LLM output into a Pydantic model.
    On ValidationError, re-prompt once with the error message (1-retry fallback).
    """
    cleaned = strip_markdown_fences(raw)
    try:
        return model_cls.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as first_err:
        logger.warning("LLM output failed validation (%s). Retrying with error hint…", first_err)
        retry_prompt = (
            f"{prompt}\n\n"
            f"Your previous output caused this error:\n{first_err}\n"
            "Fix the JSON and return ONLY valid JSON conforming to the schema."
        )
        if USE_MOCK_LLM:
            raise ValueError(f"Mock LLM cannot retry; original error: {first_err}") from first_err
        raw2 = _call_ollama(retry_prompt, schema_hint=schema_hint)
        cleaned2 = strip_markdown_fences(raw2)
        return model_cls.model_validate(json.loads(cleaned2))


# ===========================================================================
# ReAct Tool Functions
# ===========================================================================

def fetch_graph(paper_id: str) -> dict[str, Any]:
    """
    ReAct Tool: fetch_graph
    Fetch 1-hop neighbourhood from Semantic Scholar and build NetworkX graph.
    """
    neighbourhood = _data_client.fetch_neighbourhood(paper_id)
    G = _graph_builder.build(neighbourhood)
    return {
        "seed_title": neighbourhood["seed"].get("title", "Untitled"),
        "num_references": len(neighbourhood.get("references", [])),
        "num_citations": len(neighbourhood.get("citations", [])),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "neighbourhood": neighbourhood,
        "graph": G,
    }


def get_branch_summary(cluster_id: str, cluster_papers: list[dict[str, Any]]) -> str:
    """
    ReAct Tool: get_branch_summary
    Format cluster papers into a structured summary for agent reasoning.
    """
    summary = _cluster_engine.get_cluster_summary(cluster_papers)
    ab_ratio = calculate_abandonment_ratio(cluster_papers)
    return (
        f"**Cluster {cluster_id}** ({len(cluster_papers)} papers, "
        f"Abandonment Ratio: {ab_ratio:.1f}%):\n{summary}"
    )


def retrieve_text_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    ReAct Tool: retrieve_text_chunks
    Vector similarity search against ChromaDB abstract embeddings.
    """
    return _cluster_engine.retrieve_text_chunks(query, top_k=top_k)


# ===========================================================================
# Abandonment Ratio Heuristic
# ===========================================================================

def calculate_abandonment_ratio(cluster_papers: list[dict[str, Any]]) -> float:
    """
    Fraction of papers with citationCount < 5, expressed as a percentage.
    High ratio → research branch may have stalled / been abandoned.
    """
    if not cluster_papers:
        return 0.0
    low = sum(1 for p in cluster_papers if (p.get("citationCount") or 0) < 5)
    return (low / len(cluster_papers)) * 100.0


def calculate_cluster_abandonment_ratios(
    clusters: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Calculate Abandonment Ratio for every cluster."""
    return {
        cid: round(calculate_abandonment_ratio(papers), 1)
        for cid, papers in clusters.items()
    }


# ===========================================================================
# Modular LLM Client  (mock fallback + real Ollama path)
# ===========================================================================

class ModularLLMClient:
    """
    Wraps the LLM call.

    When USE_MOCK_LLM=True  → returns a rich deterministic Markdown report.
    When USE_MOCK_LLM=False → calls Ollama, validates output with GapReport,
                              applies 1-retry fallback on parse failure.
    """

    # ------------------------------------------------------------------
    def _build_gap_report_prompt(
        self,
        seed_title: str,
        branch_summary_str: str,
        abandonment_ratios: dict[str, float],
    ) -> str:
        ab_lines = "\n".join(
            f"  - Cluster {cid}: abandonment ratio {r}%"
            for cid, r in abandonment_ratios.items()
        )
        return textwrap.dedent(f"""
        You are an expert research analyst specializing in systematic literature reviews.

        Seed paper: "{seed_title}"

        Research branches identified by HDBSCAN clustering of forward citations:
        {branch_summary_str}

        Abandonment ratios (% of papers with < 5 citations in branch):
        {ab_lines}

        Task:
        Return a JSON object that strictly follows this schema:
        {{
          "seed_title": "<string>",
          "executive_summary": "<string, 2-4 sentences>",
          "gaps": [
            {{
              "gap_id": <int>,
              "title": "<string>",
              "description": "<string>",
              "evidence_clusters": ["<cluster_id>", ...],
              "recommended_action": "<string>"
            }}
          ],
          "next_steps": ["<string>", ...]
        }}

        Identify at least 3 research gaps. Be specific and grounded in the branch data above.
        """).strip()

    # ------------------------------------------------------------------
    def generate_gap_report(
        self,
        seed_title: str,
        branch_summaries: str,
        abandonment_ratios: dict[str, float],
    ) -> str:
        if USE_MOCK_LLM:
            return self._mock_report(seed_title, abandonment_ratios)

        prompt = self._build_gap_report_prompt(seed_title, branch_summaries, abandonment_ratios)
        schema_hint = GapReport.model_json_schema().__str__()
        raw = _call_ollama(prompt, schema_hint=schema_hint)

        try:
            validated: GapReport = _parse_with_retry(raw, GapReport, prompt, schema_hint)
            return self._render_gap_report(validated)
        except Exception as exc:
            logger.error("GapReport validation failed after retry: %s", exc)
            # Degrade gracefully — return raw Ollama text
            return f"# Research Gap Report\n\n{raw}"

    # ------------------------------------------------------------------
    @staticmethod
    def _render_gap_report(report: GapReport) -> str:
        """Convert a validated GapReport Pydantic model to Markdown."""
        lines = [
            f"# 🔬 Research Gap Report: {report.seed_title}\n",
            f"## 📌 Executive Summary\n\n{report.executive_summary}\n",
            "---\n",
            "## 🎯 Identified Research Gaps\n",
        ]
        for gap in report.gaps:
            clusters_str = ", ".join(gap.evidence_clusters) or "N/A"
            lines += [
                f"### Gap {gap.gap_id} — {gap.title}",
                f"{gap.description}",
                f"- **Evidence clusters:** {clusters_str}",
                f"- **Recommended action:** {gap.recommended_action}\n",
            ]
        lines += [
            "---\n",
            "## 🚀 Recommended Next Steps\n",
        ]
        for i, step in enumerate(report.next_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append(
            "\n---\n*Report compiled by Research Gap Finder Pipeline · "
            f"Powered by ReAct + ChromaDB + {OLLAMA_MODEL if not USE_MOCK_LLM else 'Mock LLM'}*"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _mock_report(self, seed_title: str, abandonment_ratios: dict[str, float]) -> str:
        ab_breakdown = "\n".join(
            f"- **Cluster {cid}**: {ratio}%" for cid, ratio in abandonment_ratios.items()
        )
        return textwrap.dedent(f"""
        # 🔬 Research Gap Report: {seed_title}

        > **System Note:** Generated by the **mocked** LLM layer.
        > Set `USE_MOCK_LLM=false` and run Ollama to use the real Qwen 14B model.

        ---

        ## 📌 Executive Summary

        Analysis of **"{seed_title}"** revealed distinct research branches via HDBSCAN clustering.
        Edge intent filtering (Extends vs Mentions) and Abandonment Ratio heuristics were applied.

        ### Branch Abandonment Ratios:
        {ab_breakdown}

        ---

        ## 🧩 Identified Research Branches

        ### Branch 0 — Architectural Extensions & Optimization
        - **Abandonment Risk**: Low — continuous active citations in modern literature.

        ### Branch 1 — Domain-Specific Fine-Tuning & Application
        - **Abandonment Risk**: Moderate — methods tied to static context windows.

        ### Branch 2 — Cross-Domain Generalization & Benchmarks
        - **Abandonment Risk**: High ({abandonment_ratios.get('2', 50.0)}%) — many unmaintained baselines.

        ---

        ## 🎯 Key Unexplored Research Gaps

        ### Gap 1: Dynamic Context Adaptation under Resource Constraints
        Branch 0 focuses on parameter growth; low-latency adaptive retrieval for edge devices is missing.

        ### Gap 2: Cross-Lingual and Low-Resource Knowledge Alignment
        Branch 1 is monolingual-centric; cross-lingual integration remains largely unexplored.

        ### Gap 3: Long-Horizon Reasoning Metrics
        Branch 2 evaluates single-step inference; multi-step reasoning benchmarks are absent.

        ---

        ## 🚀 Actionable Next Steps

        1. Combine Branch 0 retrieval with Branch 1 lightweight fine-tuning.
        2. Build an open-source cross-lingual benchmark targeting Gap 2.
        3. Design ablation studies on memory utilization vs accuracy across extended context lengths.

        ---
        *Report compiled by Research Gap Finder Pipeline v0.1 · Powered by ReAct + ChromaDB + NetworkX*
        """).strip()


_llm_client = ModularLLMClient()


# ===========================================================================
# ReAct Data Structures
# ===========================================================================

@dataclass
class ReActStep:
    step: int
    phase: str   # "Reason" | "Act" | "Observe"
    tool: str | None
    input: Any
    output: Any


@dataclass
class AgentResult:
    report: str
    trace: list[dict[str, Any]]
    clusters: dict[str, list[dict[str, Any]]]
    abandonment_ratios: dict[str, float]
    graph_html: str


# ===========================================================================
# Agent Orchestrator
# ===========================================================================

class ResearchGapAgent:
    """ReAct Agent orchestrating graph construction, clustering, and gap analysis."""

    def run(self, paper_id: str) -> AgentResult:
        trace: list[ReActStep] = []
        step_no = 0

        def log_step(phase: str, tool: str | None, inp: Any, out: Any) -> None:
            nonlocal step_no
            step_no += 1
            trace.append(ReActStep(step=step_no, phase=phase, tool=tool, input=inp, output=out))
            logger.info("[%s | %s] %s", phase, tool or "-", str(out)[:120])

        # ── Step 1: Reason ────────────────────────────────────────────────
        log_step(
            "Reason", None, paper_id,
            "Initiating analysis. Fetching citation network (references + forward citations).",
        )

        # ── Step 2: Act — fetch_graph ─────────────────────────────────────
        log_step("Act", "fetch_graph", paper_id, "Executing fetch_graph…")
        graph_res = fetch_graph(paper_id)
        seed_title = graph_res["seed_title"]
        log_step(
            "Observe", "fetch_graph", paper_id,
            {"seed_title": seed_title, "nodes": graph_res["num_nodes"], "edges": graph_res["num_edges"]},
        )

        # ── Step 3: Reason ────────────────────────────────────────────────
        log_step(
            "Reason", None, None,
            "Neighbourhood fetched. Embedding abstracts and clustering citations with HDBSCAN.",
        )

        # ── Step 4: Act — cluster ─────────────────────────────────────────
        citations = graph_res["neighbourhood"].get("citations", [])
        log_step("Act", "cluster_citations", len(citations), f"Clustering {len(citations)} citations…")
        clusters = _cluster_engine.cluster(citations)
        log_step(
            "Observe", "cluster_citations", None,
            {cid: len(papers) for cid, papers in clusters.items()},
        )

        # ── Step 5: Abandonment Ratios ────────────────────────────────────
        abandonment_ratios = calculate_cluster_abandonment_ratios(clusters)
        log_step("Observe", "calculate_abandonment_ratios", None, abandonment_ratios)

        # ── Step 6: Act — branch summaries + RAG ─────────────────────────
        summaries: list[str] = []
        for cid, papers in clusters.items():
            log_step("Act", "get_branch_summary", cid, f"Summarising branch {cid}…")
            summary = get_branch_summary(cid, papers)
            summaries.append(summary)
            log_step("Observe", "get_branch_summary", cid, summary[:200])

        log_step("Act", "retrieve_text_chunks", "unexplored research gaps", "Retrieving vector chunks…")
        text_chunks = retrieve_text_chunks("unexplored research gaps", top_k=2)
        log_step("Observe", "retrieve_text_chunks", "top_k=2", f"Retrieved {len(text_chunks)} chunks.")

        # ── Step 7: Act — LLM generate ───────────────────────────────────
        branch_summary_str = "\n\n".join(summaries)
        llm_mode = "Mock" if USE_MOCK_LLM else f"Ollama ({OLLAMA_MODEL})"
        log_step("Reason", None, None, f"Generating Gap Report via {llm_mode}…")
        log_step("Act", "llm_generate", "prompt", "Synthesizing report…")
        report = _llm_client.generate_gap_report(seed_title, branch_summary_str, abandonment_ratios)
        log_step("Observe", "llm_generate", None, report[:200] + "…")

        # ── Pyvis HTML render ─────────────────────────────────────────────
        graph_html = _graph_builder.to_pyvis_html(graph_res["graph"], paper_id)

        return AgentResult(
            report=report,
            trace=[
                {
                    "step": s.step,
                    "phase": s.phase,
                    "tool": s.tool,
                    "input": str(s.input)[:300],
                    "output": str(s.output)[:300],
                }
                for s in trace
            ],
            clusters=clusters,
            abandonment_ratios=abandonment_ratios,
            graph_html=graph_html,
        )
