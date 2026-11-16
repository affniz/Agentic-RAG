import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { api } from "./api";
import { Icon } from "./components";
import type { DocChunks, DocStatus, DocumentsResponse, Health, IngestState, SearchResponse } from "./types";

// ---------------------------------------------------------------- helpers

function size(bytes: number | null): string {
  if (bytes == null) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function when(ts: number | null): string {
  if (ts == null) return "-";
  return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

const STATUS: Record<DocStatus, { label: string; cls: string; hint: string }> = {
  indexed: { label: "Indexed", cls: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300", hint: "Searchable and up to date" },
  changed: { label: "Changed", cls: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300", hint: "Edited since the last index - re-index to update" },
  new: { label: "Not indexed", cls: "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300", hint: "Not searchable yet - re-index to add it" },
  deleted: { label: "Deleted", cls: "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300", hint: "File removed but still in the index - re-index to drop it" },
};

const btn = "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-50";
const btnPrimary = `${btn} bg-indigo-600 text-white hover:bg-indigo-700`;
const btnSecondary = `${btn} border border-zinc-200 hover:bg-zinc-100 dark:border-zinc-800 dark:hover:bg-zinc-900`;

// ---------------------------------------------------------------- page

export function DocumentsPage({ health, onIndexChanged }: { health: Health | null; onIndexChanged: () => void }) {
  const [data, setData] = useState<DocumentsResponse | null>(null);
  const [ingest, setIngest] = useState<IngestState>({ state: "idle" });
  const [tab, setTab] = useState<"files" | "search">("files");
  const [viewing, setViewing] = useState<{ name: string; focus?: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rejected, setRejected] = useState<{ name: string; reason: string }[]>([]);
  const [uploading, setUploading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const d = await api.documents();
      setData(d);
      setIngest(d.ingest);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll while an ingest runs; refresh everything when it finishes.
  useEffect(() => {
    if (ingest.state !== "running") return;
    const t = setInterval(async () => {
      const s = await api.ingestStatus().catch(() => null);
      if (!s) return;
      if (s.state === "running") {
        setIngest(s);
        return;
      }
      clearInterval(t);
      await refresh(); // updates the list and the banner together
      onIndexChanged();
    }, 700);
    return () => clearInterval(t);
  }, [ingest.state, refresh, onIndexChanged]);

  async function upload(files: File[]) {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const r = await api.upload(files);
      setRejected(r.rejected);
      if (r.ingest) setIngest(r.ingest);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function reindex(rebuild = false) {
    setError(null);
    try {
      setIngest(await api.ingest(rebuild));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(name: string) {
    setError(null);
    try {
      const r = (await api.deleteDocument(name)) as { ingest: IngestState };
      setIngest(r.ingest);
      if (viewing?.name === name) setViewing(null);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const running = ingest.state === "running";
  const docs = data?.documents ?? [];

  return (
    <div className="flex min-h-0 flex-1">
      <div className="min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4 py-6">
          <div className="flex flex-wrap items-start gap-3">
            <div className="min-w-0 flex-1">
              <h2 className="text-xl font-semibold tracking-tight">Documents</h2>
              <p className="mt-1 text-sm text-zinc-500">
                {data ? `${docs.filter((d) => d.status !== "deleted").length} documents · ${data.chunks} chunks indexed` : "Loading…"}
              </p>
            </div>
            <div className="flex gap-2">
              <button className={btnSecondary} onClick={() => reindex(true)} disabled={running}
                title="Re-embed every document from scratch">Rebuild all</button>
              <button className={data?.needs_ingest ? btnPrimary : btnSecondary} onClick={() => reindex(false)} disabled={running}
                title="Index new and changed documents, drop deleted ones">
                {running ? Icon.spinner() : Icon.refresh()} Re-index
              </button>
            </div>
          </div>

          <IngestBanner ingest={ingest} />
          {error && <Alert tone="error">{error}</Alert>}
          {data?.needs_ingest && !running && (
            <Alert tone="info">Some documents aren't indexed yet or have changed. Re-index to make them searchable.</Alert>
          )}

          <div className="mt-5 flex gap-1 border-b border-zinc-200 dark:border-zinc-800" role="tablist">
            {(["files", "search"] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium ${tab === t
                  ? "border-indigo-600 text-indigo-700 dark:text-indigo-400"
                  : "border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"}`}>
                {t === "files" ? "Files" : "Search tester"}
              </button>
            ))}
          </div>

          {tab === "files" ? (
            <div className="mt-4 space-y-4">
              <DropZone onFiles={upload} busy={uploading} />
              {rejected.length > 0 && (
                <Alert tone="error">
                  Not uploaded: {rejected.map((r) => `${r.name} (${r.reason})`).join("; ")}
                </Alert>
              )}
              <FileTable docs={docs} onView={(name) => setViewing({ name })} onDelete={remove} busy={running} />
            </div>
          ) : (
            <SearchTester reranker={!!health?.reranker} onOpen={(name, focus) => setViewing({ name, focus })} />
          )}
        </div>
      </div>

      {viewing && (
        <>
          <div className="fixed inset-0 z-20 bg-black/30 lg:hidden" onClick={() => setViewing(null)} />
          <aside className="fixed inset-y-0 right-0 z-30 w-full max-w-md border-l border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950 lg:static lg:w-[28rem]">
            <DocumentPanel name={viewing.name} focus={viewing.focus} onClose={() => setViewing(null)} />
          </aside>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- pieces

function Alert({ tone, children }: { tone: "error" | "info"; children: React.ReactNode }) {
  const cls = tone === "error"
    ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300"
    : "border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-300";
  return <div className={`mt-3 rounded-lg border px-3 py-2 text-sm ${cls}`} role={tone === "error" ? "alert" : "status"}>{children}</div>;
}

function IngestBanner({ ingest }: { ingest: IngestState }) {
  if (ingest.state === "running") {
    const total = ingest.total ?? 0;
    const pct = total ? Math.round(((ingest.done ?? 0) / total) * 100) : 0;
    const label = ingest.stage === "embedding"
      ? `Embedding ${ingest.done ?? 0} of ${total} chunks`
      : ingest.stage === "reading" ? `Reading ${ingest.done ?? 0} of ${total} files` : "Starting…";
    return (
      <div className="mt-4 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2.5 dark:border-indigo-900 dark:bg-indigo-950/40" role="status">
        <div className="flex items-center gap-2 text-sm text-indigo-800 dark:text-indigo-300">{Icon.spinner()} Indexing - {label}</div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-indigo-100 dark:bg-indigo-900">
          <div className="h-full rounded-full bg-indigo-600 transition-all" style={{ width: `${ingest.stage === "embedding" ? pct : 5}%` }} />
        </div>
      </div>
    );
  }
  if (ingest.state === "error") return <Alert tone="error">Indexing failed: {ingest.error}</Alert>;
  if (ingest.state === "done" && ingest.report && ingest.finished_at && Date.now() / 1000 - ingest.finished_at < 600) {
    const r = ingest.report;
    const parts = [r.added.length && `${r.added.length} added`, r.updated.length && `${r.updated.length} updated`,
      r.removed.length && `${r.removed.length} removed`].filter(Boolean);
    return (
      <div className="mt-4 flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300" role="status">
        {Icon.check()} Index up to date{parts.length ? `: ${parts.join(", ")}` : " - nothing changed"}. {r.chunks} chunks.
      </div>
    );
  }
  return null;
}

function DropZone({ onFiles, busy }: { onFiles: (f: File[]) => void; busy: boolean }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const drop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    onFiles(Array.from(e.dataTransfer.files));
  };
  return (
    <div onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={drop}
      className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-7 text-center transition
        ${over ? "border-indigo-400 bg-indigo-50 dark:bg-indigo-950/30" : "border-zinc-300 dark:border-zinc-700"}`}>
      <div className="text-zinc-400">{busy ? Icon.spinner("h-6 w-6") : Icon.upload("h-6 w-6")}</div>
      <p className="mt-2 text-sm">
        {busy ? "Uploading…" : <>Drop files here, or{" "}
          <button className="font-medium text-indigo-600 hover:underline dark:text-indigo-400" onClick={() => input.current?.click()}>browse</button></>}
      </p>
      <p className="mt-1 text-xs text-zinc-500">.md, .txt or .pdf, up to 20 MB each. They're indexed automatically.</p>
      <input ref={input} type="file" multiple accept=".md,.txt,.pdf" className="hidden"
        onChange={(e) => { onFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
    </div>
  );
}

function FileTable({ docs, onView, onDelete, busy }: {
  docs: DocumentsResponse["documents"];
  onView: (name: string) => void;
  onDelete: (name: string) => void;
  busy: boolean;
}) {
  const [confirming, setConfirming] = useState<string | null>(null);
  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => setConfirming(null), 4000);
    return () => clearTimeout(t);
  }, [confirming]);

  if (!docs.length) return <p className="py-6 text-center text-sm text-zinc-500">No documents yet. Upload some to get started.</p>;
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-sm">
        <thead className="bg-zinc-50 text-left text-xs uppercase tracking-wide text-zinc-500 dark:bg-zinc-900">
          <tr>
            <th className="w-full px-3 py-2 font-medium">Name</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="hidden px-3 py-2 text-right font-medium sm:table-cell">Chunks</th>
            <th className="hidden px-3 py-2 text-right font-medium md:table-cell">Size</th>
            <th className="hidden px-3 py-2 font-medium xl:table-cell">Modified</th>
            <th className="px-3 py-2"><span className="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
          {docs.map((d) => {
            const st = STATUS[d.status];
            return (
              <tr key={d.name} className="hover:bg-zinc-50 dark:hover:bg-zinc-900/50">
                <td className="w-full max-w-0 px-3 py-2.5">
                  <button onClick={() => d.chunks && onView(d.name)} disabled={!d.chunks}
                    className="flex w-full min-w-0 items-center gap-2 text-left enabled:hover:text-indigo-600 disabled:cursor-default">
                    <span className="shrink-0 text-zinc-400">{Icon.doc()}</span>
                    <span className="truncate font-medium">{d.name}</span>
                  </button>
                </td>
                <td className="px-3 py-2.5">
                  <span className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${st.cls}`} title={st.hint}>{st.label}</span>
                </td>
                <td className="hidden px-3 py-2.5 text-right tabular-nums text-zinc-600 dark:text-zinc-400 sm:table-cell">{d.chunks || "-"}</td>
                <td className="hidden whitespace-nowrap px-3 py-2.5 text-right tabular-nums text-zinc-600 dark:text-zinc-400 md:table-cell">{size(d.size)}</td>
                <td className="hidden whitespace-nowrap px-3 py-2.5 text-zinc-600 dark:text-zinc-400 xl:table-cell">{when(d.modified)}</td>
                <td className="whitespace-nowrap px-3 py-2.5 text-right">
                  {d.status !== "deleted" && (confirming === d.name ? (
                    <button onClick={() => { setConfirming(null); onDelete(d.name); }} disabled={busy}
                      className="rounded-md bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-50">
                      Confirm delete
                    </button>
                  ) : (
                    <button onClick={() => setConfirming(d.name)} disabled={busy} aria-label={`Delete ${d.name}`}
                      className="rounded p-1.5 text-zinc-400 hover:bg-zinc-100 hover:text-red-600 disabled:opacity-40 dark:hover:bg-zinc-800">
                      {Icon.trash()}
                    </button>
                  ))}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const MODES = [
  { id: "hybrid", label: "Hybrid" },
  { id: "bm25", label: "Keyword" },
  { id: "vector", label: "Semantic" },
] as const;

function SearchTester({ reranker, onOpen }: { reranker: boolean; onOpen: (name: string, index: number) => void }) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<string>("hybrid");
  const [rerank, setRerank] = useState(true);
  const [k, setK] = useState(8);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(e?: React.FormEvent) {
    e?.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setRes(await api.search({ q, k, mode, rerank }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // Re-run when settings change after a first search.
  useEffect(() => { if (res) run(); }, [mode, rerank, k]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mt-4">
      <p className="text-sm text-zinc-500">
        See exactly what the agent's search tool returns for a query - no model call, no cost.
      </p>
      <form onSubmit={run} className="mt-3 flex gap-2">
        <label htmlFor="search-q" className="sr-only">Search query</label>
        <input id="search-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. hotel cap Tier 1"
          className="min-w-0 flex-1 rounded-lg border border-zinc-300 bg-transparent px-3 py-2 text-sm outline-none focus:border-indigo-400 dark:border-zinc-700" />
        <button className={btnPrimary} disabled={busy || !q.trim()}>{busy ? Icon.spinner() : Icon.search()} Search</button>
      </form>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <div className="inline-flex rounded-lg border border-zinc-200 p-0.5 dark:border-zinc-800" role="radiogroup" aria-label="Search mode">
          {MODES.map((m) => (
            <button key={m.id} role="radio" aria-checked={mode === m.id} onClick={() => setMode(m.id)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium ${mode === m.id ? "bg-zinc-200 dark:bg-zinc-800" : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"}`}>
              {m.label}
            </button>
          ))}
        </div>
        <label className={`flex items-center gap-1.5 ${reranker ? "" : "text-zinc-400"}`}
          title={reranker ? "Re-score results with the cross-encoder" : "Reranker unavailable on this server"}>
          <input type="checkbox" checked={rerank && reranker} disabled={!reranker} onChange={(e) => setRerank(e.target.checked)} />
          Rerank
        </label>
        <label className="flex items-center gap-1.5">
          Results
          <select value={k} onChange={(e) => setK(Number(e.target.value))}
            className="rounded-md border border-zinc-200 bg-transparent px-1.5 py-0.5 dark:border-zinc-800">
            {[5, 8, 15, 25].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
      </div>

      {error && <Alert tone="error">{error}</Alert>}
      {res && (
        <div className="mt-4">
          <p className="text-xs text-zinc-500">
            {res.results.length} results · {res.ms} ms · {MODES.find((m) => m.id === res.mode)?.label}{res.reranked ? " + rerank" : ""}
          </p>
          <ol className="mt-2 space-y-2">
            {res.results.map((r) => (
              <li key={`${r.source}#${r.index}`} className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <div className="flex items-baseline gap-2">
                  <span className="text-xs font-semibold tabular-nums text-zinc-400">{r.rank}</span>
                  <button onClick={() => onOpen(r.source, r.index)}
                    className="min-w-0 truncate text-sm font-medium text-indigo-700 hover:underline dark:text-indigo-400">
                    {r.source} #{r.index}
                  </button>
                  <span className="ml-auto shrink-0 text-xs tabular-nums text-zinc-400" title="Fusion or reranker score - only comparable within one search">
                    {r.score.toFixed(3)}
                  </span>
                </div>
                {r.section && <div className="mt-0.5 truncate text-xs text-zinc-500">{r.section}</div>}
                <p className="mt-1.5 line-clamp-4 whitespace-pre-wrap text-sm text-zinc-700 dark:text-zinc-300">{r.text}</p>
              </li>
            ))}
          </ol>
          {!res.results.length && <p className="mt-2 text-sm text-zinc-500">No results.</p>}
        </div>
      )}
    </div>
  );
}

function DocumentPanel({ name, focus, onClose }: { name: string; focus?: number; onClose: () => void }) {
  const [doc, setDoc] = useState<DocChunks | null>(null);
  const [error, setError] = useState<string | null>(null);
  const focusRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    setDoc(null);
    setError(null);
    api.documentChunks(name).then((d) => live && setDoc(d)).catch((e) => live && setError(String(e.message ?? e)));
    return () => { live = false; };
  }, [name]);

  useEffect(() => {
    // Block body: scrollIntoView may return a Promise, which React would treat as a cleanup.
    focusRef.current?.scrollIntoView({ block: "center" });
  }, [doc, focus]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start gap-2 border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <div className="mt-0.5 text-zinc-400">{Icon.doc()}</div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{name}</div>
          <div className="text-xs text-zinc-500">{doc ? `${doc.chunks.length} chunks` : " "}</div>
        </div>
        <button onClick={onClose} aria-label="Close document"
          className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800">{Icon.x()}</button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {!doc && !error && <p className="text-sm text-zinc-500">Loading…</p>}
        {doc?.doc_note && (
          <div className="rounded-lg bg-zinc-100 p-3 text-xs text-zinc-600 dark:bg-zinc-900 dark:text-zinc-400">
            <span className="font-medium">Document note</span> - shown to the model with every chunk: {doc.doc_note}
          </div>
        )}
        {doc?.chunks.map((c) => (
          <div key={c.index} ref={c.index === focus ? focusRef : undefined}
            className={`rounded-lg border p-3 text-sm leading-relaxed ${c.index === focus
              ? "border-amber-300 bg-amber-50 dark:border-amber-700/60 dark:bg-amber-950/30"
              : "border-zinc-200 dark:border-zinc-800"}`}>
            <div className="mb-1 flex gap-2 text-xs text-zinc-500">
              <span className="font-semibold tabular-nums">#{c.index}</span>
              {c.section && <span className="truncate">{c.section}</span>}
            </div>
            <div className="whitespace-pre-wrap">{c.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
