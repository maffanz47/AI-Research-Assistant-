"""
app.py
======
Streamlit frontend interface for Research Gap Finder.

Features:
- Seed Paper / arXiv ID / DOI input
- Interactive Pyvis citation network graph
- Thematic research branch breakdown with Abandonment Ratios
- Markdown Gap Report viewer
- ReAct Agent execution step trace
"""

from __future__ import annotations

import time

import requests
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Config
st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# Custom Styling (Dark theme & modern typography)
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

      html, body, [class*="css"] {
          font-family: 'Inter', sans-serif;
      }

      .hero {
          background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
          border-radius: 16px;
          padding: 2.2rem 2rem;
          margin-bottom: 1.5rem;
          text-align: center;
          border: 1px solid #303651;
      }
      .hero h1 {
          font-size: 2.5rem;
          font-weight: 700;
          color: #ffffff;
          margin: 0;
          letter-spacing: -0.5px;
      }
      .hero p {
          color: #b0b8d8;
          font-size: 1.05rem;
          margin-top: 0.5rem;
      }
      .badge {
          display: inline-block;
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.18);
          border-radius: 50px;
          padding: 0.25rem 0.85rem;
          font-size: 0.78rem;
          color: #a5b4fc;
          margin: 0.2rem;
      }

      section[data-testid="stSidebar"] {
          background: #0e1117;
          border-right: 1px solid #1e2230;
      }

      div[data-testid="stMetric"] {
          background: #161b2e;
          border: 1px solid #2a3152;
          border-radius: 12px;
          padding: 1rem;
      }

      button[data-baseweb="tab"] {
          font-weight: 600;
          font-size: 0.92rem;
      }

      .info-box {
          background: #1a2035;
          border-left: 4px solid #6366f1;
          border-radius: 0 8px 8px 0;
          padding: 1rem 1.2rem;
          margin: 0.8rem 0;
          color: #c7d2fe;
      }

      .react-step {
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 10px;
          padding: 0.8rem 1rem;
          margin: 0.5rem 0;
          font-size: 0.85rem;
      }
      .phase-reason  { border-left: 4px solid #f59e0b; }
      .phase-act     { border-left: 4px solid #3b82f6; }
      .phase-observe { border-left: 4px solid #10b981; }

      .branch-card {
          background: #161b2e;
          border: 1px solid #2a3152;
          border-radius: 12px;
          padding: 1.2rem;
          margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Header
st.markdown(
    """
    <div class="hero">
        <h1>🔬 Research Gap Finder</h1>
        <p>Map citation networks, discover research branches, and identify unexplored frontiers.</p>
        <div>
            <span class="badge">Semantic Scholar Graph API</span>
            <span class="badge">NetworkX Citation Graphs</span>
            <span class="badge">BGE-Small-v1.5 + HDBSCAN</span>
            <span class="badge">Abandonment Ratio Heuristics</span>
            <span class="badge">ReAct Agent</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Inputs
with st.sidebar:
    st.markdown("## ⚙️ Paper Input")
    st.markdown("---")

    paper_id = st.text_input(
        "📄 Seed Paper ID / arXiv ID / DOI",
        value="1706.03762",
        help="Enter arXiv ID (e.g. 1706.03762), DOI (e.g. 10.1145/...), or 40-char Semantic Scholar ID",
        placeholder="e.g. 1706.03762 or 204e30738...",
    )

    st.markdown(
        """
        <div class="info-box">
          <b>💡 Quick Input Examples:</b><br/>
          • <b>arXiv ID:</b> <code>1706.03762</code> (Attention Is All You Need)<br/>
          • <b>S2 ID:</b> <code>204e3073870fae3d05bcbc2f6a8e263d9b72e776</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Research Domain", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Backend Health Check**")
    try:
        hc = requests.get(f"{API_BASE}/health", timeout=3)
        if hc.status_code == 200:
            st.success("✅ FastAPI Backend Online (Port 8000)")
        else:
            st.error("⚠️ Backend returned status " + str(hc.status_code))
    except Exception:
        st.error("❌ Backend Offline (Run uvicorn api:app)")

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280;'>Research Gap Finder v0.1<br/>CPU Mode · Modular LLM Layer</small>",
        unsafe_allow_html=True,
    )

# Session State Storage
if "result" not in st.session_state:
    st.session_state.result = None
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None

# Analysis Execution Trigger
if analyze_btn:
    if not paper_id.strip():
        st.error("Please enter a valid paper ID, arXiv ID, or DOI.")
    else:
        with st.spinner("🔍 Fetching citation neighbourhood & executing ReAct pipeline..."):
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"paper_id": paper_id.strip()},
                    timeout=120,
                )
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    st.session_state.result = resp.json()
                    st.session_state.result["elapsed"] = round(elapsed, 1)

                    # Fetch Pyvis Graph HTML
                    gresp = requests.get(
                        f"{API_BASE}/graph",
                        params={"paper_id": paper_id.strip()},
                        timeout=30,
                    )
                    if gresp.status_code == 200:
                        st.session_state.graph_html = gresp.text
                    else:
                        st.session_state.graph_html = None

                    st.success(f"✅ Analysis completed in {elapsed:.1f}s")
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"API error {resp.status_code}: {detail}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Backend unreachable. Ensure uvicorn backend is running on port 8000."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

# Display Results
result = st.session_state.result
graph_html = st.session_state.graph_html

if result:
    clusters = result.get("clusters", {})
    ab_ratios = result.get("abandonment_ratios", {})
    n_clusters = len([k for k in clusters if k != "noise"])
    n_noise = clusters.get("noise", 0)
    n_trace = len(result.get("trace", []))
    elapsed = result.get("elapsed", "—")

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🧩 Thematic Clusters", n_clusters)
    m2.metric("🔇 Noise Outliers", n_noise)
    m3.metric("🔄 ReAct Steps", n_trace)
    m4.metric("⏱️ Processing Time", f"{elapsed}s")

    st.markdown("---")

    # Main Tabs
    tab_graph, tab_report, tab_branches, tab_trace = st.tabs(
        [
            "🕸️ Interactive Citation Graph",
            "📝 Gap Report",
            "📊 Research Branches & Abandonment",
            "🔄 ReAct Agent Trace",
        ]
    )

    # Tab 1: Pyvis Network Graph
    with tab_graph:
        st.markdown("### 🕸️ Interactive Citation Network")
        st.markdown(
            "<small style='color:#9ca3af;'>🟡 Seed Paper &nbsp;|&nbsp; 🔵 Cited References &nbsp;|&nbsp; "
            "🟢 Forward Citations &nbsp;|&nbsp; 🔴 <b>Extends</b> edge &nbsp;|&nbsp; ⚪ <b>Mentions</b> edge</small>",
            unsafe_allow_html=True,
        )
        if graph_html:
            components.html(graph_html, height=620, scrolling=False)
        else:
            st.info("Graph visualisation unavailable. Click 'Analyze Research Domain' again.")

    # Tab 2: Gap Report
    with tab_report:
        st.markdown("### 📋 Generated Research Gap Report")
        st.markdown(result.get("report", "No report generated."))

    # Tab 3: Research Branches & Abandonment Ratios
    with tab_branches:
        st.markdown("### 📊 Branch Analysis & Abandonment Ratios")
        st.markdown(
            "The **Abandonment Ratio** measures the percentage of papers in a research branch "
            "with low citation momentum (citation count < 5)."
        )
        if clusters:
            for cid, count in sorted(clusters.items(), key=lambda x: str(x[0])):
                icon = "🔇" if cid == "noise" else "🧩"
                ab_val = ab_ratios.get(cid, 0.0)
                badge_color = "#ef4444" if ab_val > 50 else ("#f59e0b" if ab_val > 25 else "#10b981")
                
                with st.expander(f"{icon} Cluster **{cid}** — {count} Paper(s) | Abandonment Ratio: {ab_val:.1f}%"):
                    st.markdown(f"**Total Papers:** {count}")
                    st.markdown(f"**Abandonment Ratio:** <span style='color:{badge_color}; font-weight:bold;'>{ab_val:.1f}%</span>", unsafe_allow_html=True)
                    if ab_val > 50:
                        st.warning("⚠️ High Abandonment Risk: Most papers in this branch have stalled forward citation growth.")
                    elif ab_val > 25:
                        st.info("ℹ️ Moderate Activity: Branch exhibits mixed citation momentum.")
                    else:
                        st.success("🟢 Active Research Branch: High forward citation frequency.")
        else:
            st.info("No cluster data available.")

    # Tab 4: ReAct Agent Execution Trace
    with tab_trace:
        st.markdown("### 🔄 ReAct Execution Trace")
        st.markdown("Step-by-step audit log of the agent's **Reason → Act → Observe** decision chain.")
        phase_colors = {
            "Reason": ("🤔", "phase-reason"),
            "Act": ("⚡", "phase-act"),
            "Observe": ("👁️", "phase-observe"),
        }
        for step in result.get("trace", []):
            icon, css_class = phase_colors.get(step["phase"], ("•", ""))
            tool_str = f" → <code>{step['tool']}</code>" if step.get("tool") else ""
            st.markdown(
                f"""
                <div class="react-step {css_class}">
                  <b>Step {step['step']}</b> &nbsp; {icon} <b>{step['phase']}</b>{tool_str}<br/>
                  <div style="margin-top:0.3rem; color:#d1d5db;"><b>Input:</b> {step.get('input', '-')}</div>
                  <div style="margin-top:0.2rem; color:#9ca3af;"><b>Output:</b> {step.get('output', '-')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    # Initial Welcome Screen
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; color: #6b7280;">
            <div style="font-size:4.5rem;">🔬</div>
            <h3 style="color:#9ca3af; font-weight:500; margin-top:1rem;">
                Enter a Seed Paper ID, arXiv ID, or DOI in the sidebar and click <b>Analyze Research Domain</b>.
            </h3>
            <p>The system will construct the 1-hop citation network, segment edges into Extends vs Mentions,<br/>
               cluster forward citations into thematic branches, calculate Abandonment Ratios, and synthesize a Gap Report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
