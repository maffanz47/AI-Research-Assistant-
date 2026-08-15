"""
app/ui.py
=========
Streamlit frontend for Research Gap Finder — LLM Inference Layer.

Connects to the FastAPI backend at /api/analyze-gap and displays
the returned research gaps as formatted markdown cards.

Features
--------
- Two input modes:
    1. Manual — paste a paper title + abstract + optional arXiv / DOI
    2. Auto   — enter arXiv / S2 ID; Semantic Scholar metadata is fetched
                automatically via the existing /analyze pipeline (legacy API)
- Renders each gap as a styled card with evidence and recommended action
- Shows health status of both the Inference API and the Legacy Pipeline API
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INFERENCE_API = os.getenv("INFERENCE_API_URL", "http://localhost:8001")
PIPELINE_API  = os.getenv("PIPELINE_API_URL",  "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Gap Finder — LLM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px; padding: 2rem 2rem; margin-bottom: 1.4rem;
        text-align: center; border: 1px solid #303651;
      }
      .hero h1  { font-size: 2.3rem; font-weight: 700; color: #fff; margin: 0; }
      .hero p   { color: #b0b8d8; font-size: 1rem; margin-top: .5rem; }
      .badge {
        display: inline-block; background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.18); border-radius: 50px;
        padding: .2rem .8rem; font-size: .75rem; color: #a5b4fc; margin: .2rem;
      }

      .gap-card {
        background: #161b2e; border: 1px solid #2a3152; border-radius: 14px;
        padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
        border-left: 5px solid #6366f1;
        transition: box-shadow .2s;
      }
      .gap-card:hover { box-shadow: 0 4px 24px rgba(99,102,241,.18); }
      .gap-title  { font-size: 1.1rem; font-weight: 600; color: #e0e7ff; margin-bottom: .5rem; }
      .gap-badge  {
        display: inline-block; background: #312e81; color: #a5b4fc;
        border-radius: 6px; padding: .15rem .6rem; font-size: .72rem;
        font-weight: 600; margin-bottom: .7rem;
      }
      .gap-label  { font-size: .8rem; color: #6b7280; text-transform: uppercase;
                    letter-spacing: .05em; margin-top: .6rem; }
      .gap-text   { color: #c7d2fe; font-size: .92rem; margin-top: .2rem; }
      .action-box {
        background: #0f1729; border: 1px solid #1e3a5f;
        border-radius: 8px; padding: .7rem 1rem; margin-top: .6rem;
        color: #38bdf8; font-size: .88rem;
      }

      .stat-card {
        background: #161b2e; border: 1px solid #2a3152;
        border-radius: 12px; padding: 1rem; text-align: center;
      }
      .stat-num   { font-size: 2rem; font-weight: 700; color: #818cf8; }
      .stat-label { font-size: .8rem; color: #6b7280; margin-top: .2rem; }

      section[data-testid="stSidebar"] {
        background: #0e1117; border-right: 1px solid #1e2230;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔬 Research Gap Finder</h1>
      <p>Fine-tuned Qwen2.5-14B · LoRA Adapter · Unsloth Inference</p>
      <div>
        <span class="badge">Qwen2.5-14B-Instruct</span>
        <span class="badge">LoRA Fine-Tuned</span>
        <span class="badge">Semantic Scholar API</span>
        <span class="badge">HDBSCAN Clustering</span>
        <span class="badge">MLflow Tracking</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Analysis Mode")
    st.markdown("---")

    mode = st.radio(
        "Input Mode",
        options=["📝 Manual (Title + Abstract)", "🔗 Auto (arXiv / DOI ID)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🖥️ API Status")

    # Inference API health
    try:
        h = requests.get(f"{INFERENCE_API}/health", timeout=3).json()
        model_loaded = h.get("model_loaded", False)
        mock = h.get("mock_mode", True)
        if mock:
            st.warning("⚠️ Inference API — Mock Mode")
        elif model_loaded:
            st.success("✅ Inference API Online (GPU Model Loaded)")
        else:
            st.error("❌ Inference API — Model not loaded")
    except Exception:
        st.error("❌ Inference API Offline")

    # Legacy pipeline API health
    try:
        ph = requests.get(f"{PIPELINE_API}/health", timeout=3)
        if ph.status_code == 200:
            st.success("✅ Pipeline API Online (Port 8000)")
        else:
            st.warning("⚠️ Pipeline API — Status " + str(ph.status_code))
    except Exception:
        st.warning("⚠️ Pipeline API Offline (Port 8000)")

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280;'>Research Gap Finder v1.0<br/>"
        "Fine-tuned · GPU Inference · MLflow Tracked</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper – render gap cards
# ---------------------------------------------------------------------------

def _render_gap_card(gap: dict, idx: int) -> None:
    border_colors = ["#6366f1", "#ec4899", "#10b981"]
    color = border_colors[idx % len(border_colors)]
    st.markdown(
        f"""
        <div class="gap-card" style="border-left-color:{color};">
          <div class="gap-badge">Gap #{gap.get('gap_id', idx + 1)}</div>
          <div class="gap-title">{gap.get('title', 'Untitled Gap')}</div>
          <div class="gap-label">Description</div>
          <div class="gap-text">{gap.get('description', '—')}</div>
          <div class="gap-label">Evidence</div>
          <div class="gap-text">{gap.get('evidence', '—')}</div>
          <div class="gap-label">Recommended Action</div>
          <div class="action-box">💡 {gap.get('recommended_action', '—')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_results(data: dict) -> None:
    """Render the full response from /api/analyze-gap."""
    gaps = data.get("gaps", [])

    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{len(gaps)}</div>'
            '<div class="stat-label">Gaps Identified</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num" style="font-size:1.1rem;color:#38bdf8;">'
            f'{data.get("model_used", "—")}</div>'
            '<div class="stat-label">Model</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        seed = data.get("seed_title", "—")[:40]
        st.markdown(
            f'<div class="stat-card"><div class="stat-num" style="font-size:1rem;color:#10b981;">'
            f'{seed}</div><div class="stat-label">Seed Paper</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Executive summary
    st.markdown("### 📌 Executive Summary")
    st.info(data.get("executive_summary", "No summary available."))
    st.markdown("---")

    # Gap cards
    st.markdown("### 🎯 Identified Research Gaps")
    for i, gap in enumerate(gaps):
        _render_gap_card(gap, i)

    # Raw JSON expander
    with st.expander("📦 Raw JSON Response"):
        st.json(data)


# ---------------------------------------------------------------------------
# Input form — Manual mode
# ---------------------------------------------------------------------------
if "📝 Manual" in mode:
    st.markdown("## 📝 Manual Input")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        title_input = st.text_input(
            "📄 Paper Title",
            placeholder="e.g. Attention Is All You Need",
        )
        abstract_input = st.text_area(
            "📃 Abstract",
            height=200,
            placeholder="Paste the paper abstract here…",
        )

    with col_r:
        st.markdown("#### Optional Citation Metadata")
        st.markdown(
            "<small style='color:#6b7280;'>Enter DOIs or arXiv IDs of forward-citing papers "
            "(one per line). Metadata will be fetched from Semantic Scholar.</small>",
            unsafe_allow_html=True,
        )
        raw_citations = st.text_area(
            "Forward Citation IDs (optional)",
            height=130,
            placeholder="1706.03762\n10.1145/3442188.3445922",
        )

        st.markdown("---")
        num_gaps = st.slider("Number of gaps to request", 1, 5, 3)

    analyze_btn = st.button("🚀 Find Research Gaps", type="primary", use_container_width=True)

    if analyze_btn:
        if not title_input.strip() or not abstract_input.strip():
            st.error("Please provide both a title and an abstract.")
        else:
            with st.spinner("🤖 Running fine-tuned Qwen inference…"):
                # Build citation list from raw IDs (basic; no S2 lookup for manual mode)
                citations = []
                for raw_id in raw_citations.strip().splitlines():
                    raw_id = raw_id.strip()
                    if raw_id:
                        citations.append({
                            "paper_id": raw_id,
                            "title": raw_id,
                            "abstract": "",
                            "citation_count": 0,
                            "is_influential": False,
                            "intents": [],
                        })

                payload = {
                    "title": title_input.strip(),
                    "abstract": abstract_input.strip(),
                    "citations": citations,
                    "references": [],
                }

                t0 = time.time()
                try:
                    resp = requests.post(
                        f"{INFERENCE_API}/api/analyze-gap",
                        json=payload,
                        timeout=300,
                    )
                    elapsed = round(time.time() - t0, 1)

                    if resp.status_code == 200:
                        st.success(f"✅ Analysis complete in {elapsed}s")
                        _render_results(resp.json())
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"API error {resp.status_code}: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Inference API is unreachable. Make sure `app/main.py` is running on port 8001.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Input form — Auto mode (uses existing pipeline + inference API)
# ---------------------------------------------------------------------------
else:
    st.markdown("## 🔗 Auto Mode — arXiv / DOI / S2 ID")
    st.info(
        "Enter a Semantic Scholar-compatible paper ID. The pipeline will fetch the citation "
        "neighbourhood, then run the fine-tuned model to identify gaps."
    )

    paper_id_input = st.text_input(
        "📄 Seed Paper ID",
        placeholder="e.g. 1706.03762  or  10.1145/3442188.3445922",
        value="1706.03762",
    )
    auto_btn = st.button("🚀 Auto-Analyze", type="primary", use_container_width=True)

    if auto_btn:
        if not paper_id_input.strip():
            st.error("Please enter a paper ID.")
        else:
            pid = paper_id_input.strip()

            # Step 1: Fetch neighbourhood from legacy pipeline API
            with st.spinner("🔍 Fetching citation neighbourhood from Semantic Scholar…"):
                try:
                    pipe_resp = requests.post(
                        f"{PIPELINE_API}/analyze",
                        json={"paper_id": pid},
                        timeout=120,
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Pipeline API is offline. Start it with:\n"
                             "`uvicorn api:app --port 8000`")
                    st.stop()

            if pipe_resp.status_code != 200:
                st.error(f"Pipeline API error {pipe_resp.status_code}: {pipe_resp.text[:200]}")
                st.stop()

            pipe_data = pipe_resp.json()

            # For auto mode we can't easily reconstruct the abstract from /analyze,
            # so we call Semantic Scholar directly for the seed paper metadata
            with st.spinner("📡 Fetching seed paper metadata…"):
                try:
                    from data_client import SemanticScholarClient, normalize_paper_id
                    client = SemanticScholarClient()
                    norm_id = normalize_paper_id(pid)
                    seed = client.get_paper(norm_id)
                    seed_title = seed.get("title", pid)
                    seed_abstract = seed.get("abstract", "Abstract not available.")
                except Exception as exc:
                    st.warning(f"Could not fetch seed metadata: {exc}. Using placeholder.")
                    seed_title = pid
                    seed_abstract = "Abstract not available."

            # Step 2: Send to inference API
            payload = {
                "title": seed_title,
                "abstract": seed_abstract,
                "citations": [],
                "references": [],
            }

            with st.spinner("🤖 Running fine-tuned Qwen inference…"):
                t0 = time.time()
                try:
                    inf_resp = requests.post(
                        f"{INFERENCE_API}/api/analyze-gap",
                        json=payload,
                        timeout=300,
                    )
                    elapsed = round(time.time() - t0, 1)

                    if inf_resp.status_code == 200:
                        st.success(f"✅ Analysis complete in {elapsed}s")

                        # Also show legacy report from pipeline
                        with st.expander("📋 Legacy HDBSCAN / Abandonment Ratio Report"):
                            st.markdown(pipe_data.get("report", "No report returned."))

                        st.markdown("---")
                        _render_results(inf_resp.json())
                    else:
                        detail = inf_resp.json().get("detail", inf_resp.text)
                        st.error(f"Inference API error {inf_resp.status_code}: {detail}")

                except requests.exceptions.ConnectionError:
                    st.error("Inference API is unreachable. Start it with:\n"
                             "`uvicorn app.main:app --port 8001`")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
