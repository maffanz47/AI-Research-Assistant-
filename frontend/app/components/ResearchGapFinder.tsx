"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface InferResponse {
  gaps: string[];
  model_used: string;
  gpu: boolean;
}

interface Toast {
  id: number;
  message: string;
  type: "error" | "success" | "info";
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = "rgf_backend_url";
const DEFAULT_URL = "";

const EXAMPLE_ABSTRACT = `We introduce Transformer, a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.`;

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Spinner() {
  return (
    <span
      className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
      style={{ animation: "spin 0.8s linear infinite" }}
      aria-label="Loading"
    />
  );
}

function GapCardSkeleton({ index }: { index: number }) {
  return (
    <div
      className="glass rounded-2xl p-5"
      style={{ animationDelay: `${index * 0.08}s` }}
    >
      <div className="flex items-start gap-4">
        <div className="skeleton w-9 h-9 rounded-full flex-shrink-0" />
        <div className="flex-1 space-y-2.5 pt-1">
          <div className="skeleton h-3.5 w-full rounded" />
          <div className="skeleton h-3.5 w-4/5 rounded" />
          <div className="skeleton h-3.5 w-3/5 rounded" />
        </div>
      </div>
    </div>
  );
}

function GapCard({ gap, index }: { gap: string; index: number }) {
  // Strip any leading "1. " / "Gap 1:" prefix the LLM might have added
  const clean = gap.replace(/^(\d+[\.\)]\s*|Gap\s*\d+[:\-\s]*)/i, "").trim();

  return (
    <div
      className="glass rounded-2xl p-5 hover:border-indigo-500/30 transition-all duration-300 animate-fade-up"
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="flex items-start gap-4">
        {/* Number badge */}
        <div className="gap-badge flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold text-white">
          {index + 1}
        </div>
        <p className="text-slate-200 leading-relaxed text-sm md:text-base flex-1">
          {clean}
        </p>
      </div>
    </div>
  );
}

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: (id: number) => void;
}) {
  const colors = {
    error: "bg-red-950/90 border-red-500/40 text-red-200",
    success: "bg-emerald-950/90 border-emerald-500/40 text-emerald-200",
    info: "bg-indigo-950/90 border-indigo-500/40 text-indigo-200",
  };
  const icons = { error: "✕", success: "✓", info: "ℹ" };

  return (
    <div
      className={`animate-slide-in glass rounded-xl px-4 py-3 flex items-center gap-3 border text-sm max-w-sm ${colors[toast.type]}`}
    >
      <span className="font-bold text-base">{icons[toast.type]}</span>
      <span className="flex-1">{toast.message}</span>
      <button
        onClick={() => onDismiss(toast.id)}
        className="opacity-60 hover:opacity-100 transition-opacity ml-1 text-lg leading-none"
        aria-label="Dismiss notification"
      >
        ×
      </button>
    </div>
  );
}

function StatusBadge({ url }: { url: string }) {
  const [status, setStatus] = useState<"idle" | "checking" | "online" | "offline">("idle");
  const [gpu, setGpu] = useState<boolean | null>(null);

  const check = useCallback(async () => {
    if (!url) { setStatus("idle"); return; }
    setStatus("checking");
    try {
      const res = await fetch(url.replace(/\/$/, "") + "/", { signal: AbortSignal.timeout(5000) });
      const data = await res.json();
      setStatus("online");
      setGpu(data.gpu ?? null);
    } catch {
      setStatus("offline");
      setGpu(null);
    }
  }, [url]);

  useEffect(() => { check(); }, [check]);

  if (status === "idle") return null;

  const map = {
    checking: { dot: "bg-yellow-400 animate-pulse", label: "Checking…" },
    online:   { dot: "bg-emerald-400", label: `Online${gpu !== null ? ` · ${gpu ? "🚀 GPU" : "💻 CPU"}` : ""}` },
    offline:  { dot: "bg-red-400", label: "Unreachable" },
  } as const;

  const cfg = map[status as keyof typeof map];

  return (
    <div className="flex items-center gap-2 text-xs text-slate-400">
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${cfg.dot}`} />
      <span>{cfg.label}</span>
      <button onClick={check} className="text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-2 hover:underline" title="Re-check">
        Recheck
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function ResearchGapFinder() {
  // Saved backend URL (persisted in localStorage)
  const [backendUrl, setBackendUrl] = useState<string>(DEFAULT_URL);
  const [urlInput, setUrlInput] = useState<string>(DEFAULT_URL);
  const [urlSaved, setUrlSaved] = useState(false);

  // Abstract input
  const [abstract, setAbstract] = useState("");

  // Analysis state
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<InferResponse | null>(null);
  const [topK, setTopK] = useState(3);

  // Toasts
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastCounter = useRef(0);

  // ---------------------------------------------------------------------------
  // Load persisted URL from localStorage on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY) ?? "";
      setBackendUrl(saved);
      setUrlInput(saved);
    } catch {
      // localStorage not available (e.g. private mode)
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Toast helpers
  // ---------------------------------------------------------------------------
  const addToast = useCallback((message: string, type: Toast["type"] = "error") => {
    const id = ++toastCounter.current;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  // ---------------------------------------------------------------------------
  // Save backend URL
  // ---------------------------------------------------------------------------
  const saveUrl = useCallback(() => {
    const trimmed = urlInput.trim().replace(/\/$/, "");
    try { localStorage.setItem(STORAGE_KEY, trimmed); } catch { /* noop */ }
    setBackendUrl(trimmed);
    setUrlSaved(true);
    setTimeout(() => setUrlSaved(false), 2000);
    addToast("Backend URL saved!", "success");
  }, [urlInput, addToast]);

  // ---------------------------------------------------------------------------
  // Analyze (client-side fetch to GPU backend)
  // ---------------------------------------------------------------------------
  const analyze = useCallback(async () => {
    if (!backendUrl) {
      addToast("Please save a Backend API URL first.", "error");
      return;
    }
    if (abstract.trim().length < 50) {
      addToast("Abstract must be at least 50 characters long.", "error");
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      // NOTE: Backend exposes /infer (not /predict)
      const endpoint = backendUrl.replace(/\/$/, "") + "/infer";
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ abstract: abstract.trim(), top_k_gaps: topK }),
      });

      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(`Server returned ${res.status}: ${errBody}`);
      }

      const data: InferResponse = await res.json();
      setResult(data);
    } catch (err) {
      const message =
        err instanceof TypeError
          ? "Could not reach the backend. Is your tunnel URL correct and the server running?"
          : (err as Error).message;
      addToast(message, "error");
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl, abstract, topK, addToast]);

  // Keyboard shortcut: Ctrl+Enter / Cmd+Enter to analyze
  const handleTextareaKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!isLoading) analyze();
      }
    },
    [isLoading, analyze]
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <>
      {/* Toast Container */}
      <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 items-center pointer-events-none">
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto">
            <ToastItem toast={t} onDismiss={dismissToast} />
          </div>
        ))}
      </div>

      <div className="max-w-3xl mx-auto space-y-6">
        {/* ── Header ─────────────────────────────────────────────────────── */}
        <header className="text-center pt-6 pb-2">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl glass mb-4">
            <span className="text-3xl">🔬</span>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold gradient-text tracking-tight mb-2">
            Research Gap Finder
          </h1>
          <p className="text-slate-400 text-sm md:text-base max-w-lg mx-auto">
            Paste a paper abstract to discover unexplored research opportunities,
            powered by Qwen&nbsp;14B on a remote GPU.
          </p>
        </header>

        {/* ── Backend URL Config ──────────────────────────────────────────── */}
        <section className="glass rounded-2xl p-4 space-y-3" aria-label="Backend configuration">
          <div className="flex items-center justify-between">
            <label
              htmlFor="backend-url-input"
              className="text-xs font-semibold text-slate-400 uppercase tracking-widest"
            >
              Backend API URL
            </label>
            <StatusBadge url={backendUrl} />
          </div>
          <div className="flex gap-2">
            <input
              id="backend-url-input"
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveUrl()}
              placeholder="https://xxxx.lhr.life"
              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 transition-all"
              aria-label="Backend API URL input"
            />
            <button
              onClick={saveUrl}
              className="btn-glow rounded-xl px-5 py-2.5 text-sm font-semibold text-white whitespace-nowrap"
              aria-label="Save backend URL"
            >
              {urlSaved ? "✓ Saved" : "Save"}
            </button>
          </div>
          <p className="text-xs text-slate-600">
            Paste your tunnel URL each time the GPU server restarts. Saved in <code className="text-slate-500">localStorage</code>.
          </p>
        </section>

        {/* ── Abstract Input ──────────────────────────────────────────────── */}
        <section className="glass rounded-2xl p-4 space-y-3" aria-label="Abstract input">
          <div className="flex items-center justify-between">
            <label
              htmlFor="abstract-input"
              className="text-xs font-semibold text-slate-400 uppercase tracking-widest"
            >
              Paper Abstract
            </label>
            <span className="text-xs text-slate-600">
              {abstract.length} chars
              {abstract.length > 0 && abstract.length < 50 && (
                <span className="text-red-400 ml-1">(min 50)</span>
              )}
            </span>
          </div>
          <textarea
            id="abstract-input"
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
            onKeyDown={handleTextareaKeyDown}
            rows={7}
            placeholder="Paste your paper abstract here…"
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 transition-all resize-y leading-relaxed"
            aria-label="Paper abstract"
            disabled={isLoading}
          />
          <div className="flex items-center justify-between flex-wrap gap-3">
            {/* Quick fill button */}
            <button
              onClick={() => setAbstract(EXAMPLE_ABSTRACT)}
              className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-2 hover:underline"
              disabled={isLoading}
            >
              Try example abstract
            </button>
            {/* Number of gaps control */}
            <div className="flex items-center gap-2">
              <label htmlFor="top-k-input" className="text-xs text-slate-500">
                Gaps to find:
              </label>
              <div className="flex items-center glass rounded-lg overflow-hidden">
                {[3, 5, 7].map((n) => (
                  <button
                    key={n}
                    onClick={() => setTopK(n)}
                    disabled={isLoading}
                    className={`px-3 py-1 text-xs font-medium transition-colors ${
                      topK === n
                        ? "bg-indigo-600 text-white"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                    aria-pressed={topK === n}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── Analyze Button ──────────────────────────────────────────────── */}
        <button
          id="analyze-button"
          onClick={analyze}
          disabled={isLoading || abstract.trim().length < 50 || !backendUrl}
          className="btn-glow w-full rounded-2xl py-3.5 text-base font-semibold text-white flex items-center justify-center gap-3 transition-all"
          aria-busy={isLoading}
        >
          {isLoading ? (
            <>
              <Spinner />
              <span>Analyzing on GPU…</span>
            </>
          ) : (
            <>
              <span>🔍</span>
              <span>Analyze Research Gaps</span>
              <span className="text-xs text-white/50 ml-1 hidden sm:inline">
                ⌘ Enter
              </span>
            </>
          )}
        </button>

        {/* ── Loading Skeletons ───────────────────────────────────────────── */}
        {isLoading && (
          <section aria-label="Loading results" className="space-y-3">
            <div className="flex items-center gap-2 px-1">
              <div className="skeleton h-3 w-3 rounded-full" />
              <div className="skeleton h-3 w-40 rounded" />
            </div>
            {Array.from({ length: topK }).map((_, i) => (
              <GapCardSkeleton key={i} index={i} />
            ))}
          </section>
        )}

        {/* ── Results ─────────────────────────────────────────────────────── */}
        {!isLoading && result && (
          <section aria-label="Analysis results" className="space-y-4">
            {/* Meta bar */}
            <div className="flex items-center justify-between px-1 text-xs text-slate-500">
              <span>
                {result.gaps.length} gap{result.gaps.length !== 1 ? "s" : ""} identified
              </span>
              <div className="flex items-center gap-3">
                <span>
                  Model:{" "}
                  <span className="text-slate-400 font-medium">{result.model_used}</span>
                </span>
                <span>
                  {result.gpu ? "🚀 GPU" : "💻 CPU"}
                </span>
              </div>
            </div>

            {/* Gap cards */}
            <div className="space-y-3">
              {result.gaps.map((gap, i) => (
                <GapCard key={i} gap={gap} index={i} />
              ))}
            </div>

            {/* Export */}
            <div className="flex justify-end pt-2">
              <button
                onClick={() => {
                  const text = result.gaps
                    .map((g, i) => `${i + 1}. ${g}`)
                    .join("\n\n");
                  navigator.clipboard.writeText(text);
                  addToast("Gaps copied to clipboard!", "success");
                }}
                className="glass rounded-xl px-4 py-2 text-xs text-slate-400 hover:text-slate-200 hover:border-indigo-500/30 transition-all"
              >
                Copy all gaps
              </button>
            </div>
          </section>
        )}

        {/* ── Empty State ─────────────────────────────────────────────────── */}
        {!isLoading && !result && (
          <div className="text-center py-12 text-slate-600 select-none">
            <div className="text-5xl mb-3 opacity-30">🧬</div>
            <p className="text-sm">
              Paste an abstract and click Analyze to discover research gaps.
            </p>
          </div>
        )}

        {/* ── Footer ──────────────────────────────────────────────────────── */}
        <footer className="text-center text-xs text-slate-700 pb-6">
          Research Gap Finder · Qwen 2.5 14B · HDBSCAN · NetworkX · FastAPI
        </footer>
      </div>
    </>
  );
}
