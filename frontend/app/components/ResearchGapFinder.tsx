"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types matching backend schemas
// ---------------------------------------------------------------------------
interface ResearchGap {
  gap_id: number;
  title: string;
  description: string;
  evidence: string;
  recommended_action: string;
}

interface AnalyzeGapResponse {
  seed_titles: string[];
  executive_summary: string;
  gaps: ResearchGap[];
  model_used: string;
}

interface HealthResponse {
  status: string;
  model_loaded?: boolean;
  mock_mode?: boolean;
  gpu?: boolean;
}

interface Toast {
  id: number;
  message: string;
  type: "error" | "success" | "info";
}

type Mode = "manual" | "batch" | "auto";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const STORAGE_KEY = "rgf_backend_url";
const ENV_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

const EXAMPLE_ABSTRACT = `We introduce Transformer, a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.`;

// ---------------------------------------------------------------------------
// Utility sub-components
// ---------------------------------------------------------------------------
function Spinner({ size = 16 }: { size?: number }) {
  return (
    <span
      className="inline-block border-2 border-current/30 border-t-current rounded-full"
      style={{ width: size, height: size, animation: "spin 0.7s linear infinite" }}
    />
  );
}

function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  if (!toasts.length) return null;
  const styles = {
    error: "bg-[var(--color-error-container)] text-[var(--color-error)] border-[var(--color-error)]",
    success: "bg-[var(--color-success-container)] text-[var(--color-success)] border-[var(--color-success)]",
    info: "bg-[var(--color-primary-container)] text-[var(--color-primary)] border-[var(--color-primary)]",
  };
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => (
        <div key={t.id} className={`animate-slide-in rounded-lg px-4 py-3 border text-sm flex items-center gap-2 shadow-md ${styles[t.type]}`}>
          <span className="flex-1">{t.message}</span>
          <button onClick={() => onDismiss(t.id)} className="opacity-60 hover:opacity-100 text-lg leading-none">×</button>
        </div>
      ))}
    </div>
  );
}

function StatusChip({ label, type }: { label: string; type: "success" | "error" | "warning" | "neutral" }) {
  const cls = { success: "chip-success", error: "chip-error", warning: "chip-warning", neutral: "chip-neutral" };
  const dots = { success: "bg-[var(--color-success)]", error: "bg-[var(--color-error)]", warning: "bg-amber-500", neutral: "bg-gray-400" };
  return (
    <span className={`chip ${cls[type]}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dots[type]}`} />
      {label}
    </span>
  );
}

function GapCard({ gap, index }: { gap: ResearchGap; index: number }) {
  const colors = ["border-l-blue-600", "border-l-pink-500", "border-l-green-600", "border-l-amber-500", "border-l-purple-600"];
  return (
    <div className={`card p-5 border-l-4 ${colors[index % colors.length]} animate-fade-up`} style={{ animationDelay: `${index * 0.08}s` }}>
      <div className="flex items-center gap-2 mb-2">
        <span className="chip-neutral chip text-xs">Gap #{gap.gap_id}</span>
        <h3 className="font-semibold text-[var(--color-on-surface)]">{gap.title}</h3>
      </div>
      <div className="space-y-3 text-sm">
        <div>
          <p className="text-xs font-medium text-[var(--color-on-surface-variant)] uppercase tracking-wider mb-1">Description</p>
          <p className="text-[var(--color-on-surface)]">{gap.description}</p>
        </div>
        <div>
          <p className="text-xs font-medium text-[var(--color-on-surface-variant)] uppercase tracking-wider mb-1">Evidence</p>
          <p className="text-[var(--color-on-surface)]">{gap.evidence}</p>
        </div>
        <div className="bg-[var(--color-surface-dim)] border border-[var(--color-outline-variant)] rounded-lg p-3">
          <p className="text-xs font-medium text-[var(--color-primary)] uppercase tracking-wider mb-1">💡 Recommended Action</p>
          <p className="text-[var(--color-on-surface)]">{gap.recommended_action}</p>
        </div>
      </div>
    </div>
  );
}

function ResultsView({ data, onCopy }: { data: AnalyzeGapResponse; onCopy: () => void }) {
  const [showJson, setShowJson] = useState(false);
  return (
    <div className="space-y-5 animate-fade-up">
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Gaps Found", value: String(data.gaps.length), color: "text-[var(--color-primary)]" },
          { label: "Model", value: data.model_used, color: "text-[var(--color-on-surface)]" },
          { label: "Seed Papers", value: data.seed_titles.length > 1 ? `${data.seed_titles.length} papers` : (data.seed_titles[0]?.substring(0, 35) || "—"), color: "text-[var(--color-success)]" },
        ].map((s, i) => (
          <div key={i} className="card p-4 text-center">
            <p className={`text-lg font-bold ${s.color} truncate`}>{s.value}</p>
            <p className="text-xs text-[var(--color-on-surface-variant)] mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Executive summary */}
      <div className="bg-[var(--color-primary-container)] border border-blue-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-[var(--color-primary)] uppercase tracking-wider mb-1">Executive Summary</p>
        <p className="text-sm text-[var(--color-on-surface)]">{data.executive_summary}</p>
      </div>

      {/* Gap cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-[var(--color-on-surface)]">Identified Research Gaps</h2>
          <button onClick={onCopy} className="btn-outlined px-3 py-1.5 text-xs">Copy All</button>
        </div>
        <div className="space-y-4">
          {data.gaps.map((gap, i) => <GapCard key={gap.gap_id} gap={gap} index={i} />)}
        </div>
      </div>

      {/* Raw JSON toggle */}
      <div className="border-t border-[var(--color-outline-variant)] pt-3">
        <button onClick={() => setShowJson(!showJson)} className="btn-outlined px-3 py-1.5 text-xs">
          {showJson ? "Hide" : "Show"} Raw JSON
        </button>
        {showJson && (
          <pre className="mt-3 bg-[var(--color-surface-dim)] border border-[var(--color-outline)] rounded-lg p-4 text-xs overflow-auto max-h-80">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export default function ResearchGapFinder() {
  const [mode, setMode] = useState<Mode>("manual");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthErr, setHealthErr] = useState(false);

  // Backend URL state (localStorage-backed)
  const [backendUrl, setBackendUrl] = useState<string>(ENV_URL);
  const [urlInput, setUrlInput]     = useState<string>(ENV_URL);
  const [urlSaved, setUrlSaved]     = useState(false);

  // Load saved URL from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) { setBackendUrl(saved); setUrlInput(saved); }
      else if (ENV_URL) { setBackendUrl(ENV_URL); setUrlInput(ENV_URL); }
    } catch { /* private mode */ }
  }, []);

  const saveUrl = useCallback(() => {
    const trimmed = urlInput.trim().replace(/\/$/, "");
    try { localStorage.setItem(STORAGE_KEY, trimmed); } catch { /* noop */ }
    setBackendUrl(trimmed);
    setUrlSaved(true);
    setTimeout(() => setUrlSaved(false), 2000);
  }, [urlInput]);

  // Manual mode state
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [citations, setCitations] = useState("");

  // Batch mode state
  const [batchPapers, setBatchPapers] = useState([{ title: "", abstract: "" }, { title: "", abstract: "" }, { title: "", abstract: "" }]);

  // Auto mode state
  const [paperId, setPaperId] = useState("1706.03762");

  // Shared state
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeGapResponse | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);

  // Toast helper
  const toast = useCallback((message: string, type: Toast["type"] = "error") => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  // Health check — re-run whenever backendUrl changes
  useEffect(() => {
    if (!backendUrl) { setHealth(null); setHealthErr(false); return; }
    setHealth(null); setHealthErr(false);
    fetch(`${backendUrl}/inference/health`, { signal: AbortSignal.timeout(5000) })
      .then((r) => r.json())
      .then((d) => { setHealth(d); setHealthErr(false); })
      .catch(() => setHealthErr(true));
  }, [backendUrl]);

  // Copy gaps to clipboard
  const copyGaps = useCallback(() => {
    if (!result) return;
    const text = result.gaps.map((g) => `${g.gap_id}. ${g.title}\n   ${g.description}\n   Evidence: ${g.evidence}\n   Action: ${g.recommended_action}`).join("\n\n");
    navigator.clipboard.writeText(text);
    toast("Copied to clipboard!", "success");
  }, [result, toast]);

  // ── Manual mode submit ──
  const submitManual = useCallback(async () => {
    if (!backendUrl) { toast("Save a Backend URL first."); return; }
    if (!title.trim() || !abstract.trim()) { toast("Please provide both a title and an abstract."); return; }
    setIsLoading(true); setResult(null);
    const citationList = citations.trim().split("\n").filter(Boolean).map((id) => ({
      paper_id: id.trim(), title: id.trim(), abstract: "", citation_count: 0, is_influential: false, intents: [],
    }));
    try {
      const res = await fetch(`${backendUrl}/inference/api/analyze-gap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim(), abstract: abstract.trim(), citations: citationList, references: [] }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
      setResult(await res.json());
      toast("Analysis complete!", "success");
    } catch (e) { toast((e as Error).message); }
    finally { setIsLoading(false); }
  }, [title, abstract, citations, toast]);

  // ── Batch mode submit ──
  const submitBatch = useCallback(async () => {
    if (!backendUrl) { toast("Save a Backend URL first."); return; }
    const valid = batchPapers.filter((p) => p.title.trim() && p.abstract.trim());
    if (!valid.length) { toast("Add at least one paper with title and abstract."); return; }
    setIsLoading(true); setResult(null);
    try {
      const res = await fetch(`${backendUrl}/inference/api/analyze-gap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed_papers: valid, citations: [], references: [] }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
      setResult(await res.json());
      toast(`Analyzed ${valid.length} papers!`, "success");
    } catch (e) { toast((e as Error).message); }
    finally { setIsLoading(false); }
  }, [batchPapers, toast]);

  // ── Auto mode submit ──
  const submitAuto = useCallback(async () => {
    if (!backendUrl) { toast("Save a Backend URL first."); return; }
    if (!paperId.trim()) { toast("Enter a paper ID."); return; }
    setIsLoading(true); setResult(null);
    try {
      // Step 1: Fetch from pipeline API
      const pipeRes = await fetch(`${backendUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_id: paperId.trim() }),
      });
      if (!pipeRes.ok) throw new Error(`Pipeline error ${pipeRes.status}`);

      // Step 2: Fetch seed metadata via Semantic Scholar
      const s2Res = await fetch(`https://api.semanticscholar.org/graph/v1/paper/${paperId.trim()}?fields=title,abstract`, { signal: AbortSignal.timeout(10000) });
      let seedTitle = paperId.trim(), seedAbstract = "Abstract not available.";
      if (s2Res.ok) {
        const s2 = await s2Res.json();
        seedTitle = s2.title || seedTitle;
        seedAbstract = s2.abstract || seedAbstract;
      }

      // Step 3: Run inference
      const infRes = await fetch(`${backendUrl}/inference/api/analyze-gap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: seedTitle, abstract: seedAbstract, citations: [], references: [] }),
      });
      if (!infRes.ok) throw new Error(`Inference error ${infRes.status}`);
      setResult(await infRes.json());
      toast("Auto-analysis complete!", "success");
    } catch (e) { toast((e as Error).message); }
    finally { setIsLoading(false); }
  }, [paperId, toast]);

  // Batch row helpers
  const updateBatch = (i: number, field: "title" | "abstract", value: string) => {
    setBatchPapers((p) => p.map((r, j) => (j === i ? { ...r, [field]: value } : r)));
  };
  const addBatchRow = () => setBatchPapers((p) => [...p, { title: "", abstract: "" }]);
  const removeBatchRow = (i: number) => setBatchPapers((p) => p.filter((_, j) => j !== i));

  const modes: { key: Mode; label: string; icon: string }[] = [
    { key: "manual", label: "Manual (Single)", icon: "📝" },
    { key: "batch", label: "Batch (Multiple)", icon: "📚" },
    { key: "auto", label: "Auto (arXiv / DOI)", icon: "🔗" },
  ];

  return (
    <>
      <ToastContainer toasts={toasts} onDismiss={(id) => setToasts((t) => t.filter((x) => x.id !== id))} />

      {/* Header */}
      <header className="border-b border-[var(--color-outline)] bg-[var(--color-surface)]">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🔬</span>
            <div>
              <h1 className="text-lg font-semibold text-[var(--color-on-surface)]">Research Gap Finder</h1>
              <p className="text-xs text-[var(--color-on-surface-variant)]">Qwen 2.5-14B · LoRA · HDBSCAN · Semantic Scholar</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!backendUrl ? (
              <StatusChip label="No URL set" type="neutral" />
            ) : healthErr ? (
              <StatusChip label="API Offline" type="error" />
            ) : health ? (
              <>
                <StatusChip label={health.mock_mode ? "Mock Mode" : "GPU Model Loaded"} type={health.mock_mode ? "warning" : "success"} />
                <StatusChip label="API Online" type="success" />
              </>
            ) : (
              <StatusChip label="Checking…" type="neutral" />
            )}
          </div>
        </div>
        {/* URL Config Bar */}
        <div className="border-t border-[var(--color-outline-variant)] bg-[var(--color-surface-dim)]">
          <div className="max-w-4xl mx-auto px-4 py-2 flex items-center gap-2">
            <label htmlFor="backend-url" className="text-xs text-[var(--color-on-surface-variant)] whitespace-nowrap font-medium">Backend URL:</label>
            <input
              id="backend-url"
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveUrl()}
              placeholder="https://xxxx.lhr.life  (paste tunnel URL from GPU terminal)"
              className="input-field flex-1 py-1.5 text-xs"
            />
            <button
              onClick={saveUrl}
              className="btn-primary px-4 py-1.5 text-xs whitespace-nowrap"
            >
              {urlSaved ? "Saved!" : "Save & Connect"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-6 space-y-6">
        {/* Mode tabs */}
        <div className="flex border-b border-[var(--color-outline)]">
          {modes.map((m) => (
            <button
              key={m.key}
              onClick={() => { setMode(m.key); setResult(null); }}
              className={`tab ${mode === m.key ? "tab-active" : ""}`}
            >
              {m.icon} {m.label}
            </button>
          ))}
        </div>

        {/* ── MANUAL MODE ── */}
        {mode === "manual" && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-4">
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Paper Title</label>
                  <input value={title} onChange={(e) => setTitle(e.target.value)} className="input-field" placeholder="e.g. Attention Is All You Need" disabled={isLoading} />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Abstract</label>
                  <textarea value={abstract} onChange={(e) => setAbstract(e.target.value)} rows={7} className="input-field resize-y" placeholder="Paste the paper abstract here…" disabled={isLoading} />
                  <button onClick={() => { setTitle("Attention Is All You Need"); setAbstract(EXAMPLE_ABSTRACT); }} className="text-xs text-[var(--color-primary)] hover:underline mt-1" disabled={isLoading}>
                    Try example
                  </button>
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Citation IDs <span className="text-[var(--color-on-surface-variant)] font-normal">(optional)</span></label>
                  <p className="text-xs text-[var(--color-on-surface-variant)] mb-2">arXiv IDs or DOIs, one per line</p>
                  <textarea value={citations} onChange={(e) => setCitations(e.target.value)} rows={5} className="input-field resize-y" placeholder={"1706.03762\n10.1145/3442188.3445922"} disabled={isLoading} />
                </div>
              </div>
            </div>
            <button onClick={submitManual} disabled={isLoading || !title.trim() || !abstract.trim()} className="btn-primary w-full py-3 flex items-center justify-center gap-2">
              {isLoading ? <><Spinner /> Analyzing…</> : "🚀 Find Research Gaps"}
            </button>
          </div>
        )}

        {/* ── BATCH MODE ── */}
        {mode === "batch" && (
          <div className="space-y-4">
            <p className="text-sm text-[var(--color-on-surface-variant)]">Add multiple papers to collectively identify research gaps across the series.</p>
            <div className="space-y-3">
              {batchPapers.map((paper, i) => (
                <div key={i} className="card p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-[var(--color-on-surface-variant)]">Paper {i + 1}</span>
                    {batchPapers.length > 1 && (
                      <button onClick={() => removeBatchRow(i)} className="text-xs text-[var(--color-error)] hover:underline">Remove</button>
                    )}
                  </div>
                  <input value={paper.title} onChange={(e) => updateBatch(i, "title", e.target.value)} className="input-field" placeholder="Paper title" disabled={isLoading} />
                  <textarea value={paper.abstract} onChange={(e) => updateBatch(i, "abstract", e.target.value)} rows={3} className="input-field resize-y" placeholder="Abstract" disabled={isLoading} />
                </div>
              ))}
            </div>
            <button onClick={addBatchRow} className="btn-outlined px-4 py-2 w-full" disabled={isLoading}>+ Add Paper</button>
            <button onClick={submitBatch} disabled={isLoading} className="btn-primary w-full py-3 flex items-center justify-center gap-2">
              {isLoading ? <><Spinner /> Analyzing {batchPapers.filter((p) => p.title && p.abstract).length} papers…</> : "🚀 Analyze Paper Series"}
            </button>
          </div>
        )}

        {/* ── AUTO MODE ── */}
        {mode === "auto" && (
          <div className="space-y-4">
            <div className="bg-[var(--color-primary-container)] border border-blue-200 rounded-lg p-4 text-sm text-[var(--color-on-surface)]">
              Enter a Semantic Scholar-compatible paper ID. The pipeline will fetch the citation neighbourhood, then run the fine-tuned model to identify gaps.
            </div>
            <div>
              <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Seed Paper ID</label>
              <input value={paperId} onChange={(e) => setPaperId(e.target.value)} className="input-field" placeholder="e.g. 1706.03762 or 10.1145/3442188.3445922" disabled={isLoading} />
            </div>
            <button onClick={submitAuto} disabled={isLoading || !paperId.trim()} className="btn-primary w-full py-3 flex items-center justify-center gap-2">
              {isLoading ? <><Spinner /> Fetching citations & analyzing…</> : "🚀 Auto-Analyze"}
            </button>
          </div>
        )}

        {/* Loading skeletons */}
        {isLoading && (
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="card p-5 space-y-3">
                <div className="skeleton h-4 w-1/3" />
                <div className="skeleton h-3 w-full" />
                <div className="skeleton h-3 w-4/5" />
                <div className="skeleton h-3 w-2/3" />
              </div>
            ))}
          </div>
        )}

        {/* Results */}
        {!isLoading && result && <ResultsView data={result} onCopy={copyGaps} />}

        {/* Empty state */}
        {!isLoading && !result && (
          <div className="text-center py-16 text-[var(--color-on-surface-variant)]">
            <p className="text-4xl mb-3 opacity-30">🧬</p>
            <p className="text-sm">Select a mode and provide paper details to discover research gaps.</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--color-outline)] mt-8">
        <div className="max-w-4xl mx-auto px-4 py-4 text-center text-xs text-[var(--color-on-surface-variant)]">
          Research Gap Finder · Qwen 2.5-14B · LoRA Fine-Tuned · HDBSCAN Clustering · Semantic Scholar API · MLflow Tracked
        </div>
      </footer>
    </>
  );
}
