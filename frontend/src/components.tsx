import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type { ChunkView, Citation, ConversationSummary, Health, Step, Turn } from "./types";

// ---------------------------------------------------------------- icons

const svg = (d: ReactNode, cls = "h-4 w-4") => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"
       strokeLinejoin="round" className={cls} aria-hidden>{d}</svg>
);
export const Icon = {
  search: (c?: string) => svg(<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>, c),
  doc: (c?: string) => svg(<><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></>, c),
  list: (c?: string) => svg(<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />, c),
  check: (c?: string) => svg(<path d="M20 6 9 17l-5-5" />, c),
  x: (c?: string) => svg(<path d="M18 6 6 18M6 6l12 12" />, c),
  alert: (c?: string) => svg(<><circle cx="12" cy="12" r="9" /><path d="M12 8v4M12 16h.01" /></>, c),
  plus: (c?: string) => svg(<path d="M12 5v14M5 12h14" />, c),
  menu: (c?: string) => svg(<path d="M4 6h16M4 12h16M4 18h16" />, c),
  send: (c?: string) => svg(<path d="M5 12h14M13 6l6 6-6 6" />, c),
  chevron: (c?: string) => svg(<path d="m9 6 6 6-6 6" />, c),
  upload: (c?: string) => svg(<path d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />, c),
  refresh: (c?: string) => svg(<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" />, c),
  chat: (c?: string) => svg(<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z" />, c),
  folder: (c?: string) => svg(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />, c),
  trash: (c?: string) => svg(<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />, c),
  spinner: (c = "h-4 w-4") => (
    <svg viewBox="0 0 24 24" className={`${c} animate-spin`} aria-hidden>
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="3" opacity=".2" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  ),
};

// ---------------------------------------------------------------- sidebar

function ago(ts: number): string {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export type View = "chat" | "documents";

export function Sidebar(props: {
  conversations: ConversationSummary[];
  activeId: string | null;
  health: Health | null;
  view: View;
  onNavigate: (v: View) => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  const { conversations, activeId, health } = props;
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 px-4 pb-3 pt-4">
        <div className="grid h-7 w-7 place-items-center rounded-md bg-indigo-600 text-white">{Icon.search("h-4 w-4")}</div>
        <span className="font-semibold tracking-tight">Agentic RAG</span>
      </div>
      <nav className="space-y-0.5 px-2 pb-3" aria-label="Sections">
        {([["chat", "Chat", Icon.chat()], ["documents", "Documents", Icon.folder()]] as const).map(([id, label, icon]) => (
          <button key={id} onClick={() => props.onNavigate(id)} aria-current={props.view === id ? "page" : undefined}
            className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ${props.view === id
              ? "bg-zinc-200/70 dark:bg-zinc-800" : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-900"}`}>
            {icon} {label}
          </button>
        ))}
      </nav>
      <div className="px-3">
        <button onClick={props.onNew}
          className="flex w-full items-center gap-2 rounded-lg border border-zinc-200 px-3 py-2 text-sm font-medium hover:bg-zinc-100 dark:border-zinc-800 dark:hover:bg-zinc-900">
          {Icon.plus()} New chat
        </button>
      </div>
      <nav className="mt-3 flex-1 overflow-y-auto px-2" aria-label="Conversations">
        {conversations.length === 0 && <p className="px-3 py-2 text-sm text-zinc-500">No conversations yet.</p>}
        {conversations.map((c) => (
          <div key={c.id}
            className={`group flex items-center rounded-lg ${c.id === activeId && props.view === "chat" ? "bg-zinc-200/70 dark:bg-zinc-800" : "hover:bg-zinc-100 dark:hover:bg-zinc-900"}`}>
            <button onClick={() => props.onSelect(c.id)} className="min-w-0 flex-1 px-3 py-2 text-left">
              <div className="truncate text-sm">{c.title}</div>
              <div className="mt-0.5 text-xs text-zinc-500">
                {ago(c.updated_at)} · ${c.cost_usd.toFixed(c.cost_usd < 0.01 ? 4 : 2)}
              </div>
            </button>
            <button onClick={() => props.onDelete(c.id)} aria-label={`Delete ${c.title}`}
              className="mr-1 rounded p-1.5 text-zinc-400 opacity-0 hover:text-red-600 focus:opacity-100 group-hover:opacity-100">
              {Icon.trash()}
            </button>
          </div>
        ))}
      </nav>
      <div className="border-t border-zinc-200 px-4 py-3 text-xs text-zinc-500 dark:border-zinc-800">
        {health ? (
          <div className="space-y-1">
            <div>{health.documents} documents · {health.chunks} chunks</div>
            <div>Reranker {health.reranker ? "on" : "off"}</div>
            {health.demo_mode && (
              <div className="inline-block rounded bg-amber-100 px-1.5 py-0.5 font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
                Demo mode: no model calls
              </div>
            )}
          </div>
        ) : "Connecting…"}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- steps

function stepLabel(s: Step): { icon: ReactNode; text: ReactNode } {
  const a = s.args as Record<string, string | number | null>;
  if (s.name === "search_knowledge_base")
    return { icon: Icon.search(), text: <>Searched <q className="font-medium text-zinc-700 dark:text-zinc-200">{a.query}</q>{a.source ? <> in {a.source}</> : null}</> };
  if (s.name === "read_chunks")
    return { icon: Icon.doc(), text: <>Read <span className="font-medium text-zinc-700 dark:text-zinc-200">{a.source}</span> #{a.start}{a.end !== a.start ? `–${a.end}` : ""}</> };
  if (s.name === "list_documents") return { icon: Icon.list(), text: "Listed documents" };
  return { icon: Icon.list(), text: s.name };
}

export function Steps({ steps, streaming }: { steps: Step[]; streaming: boolean }) {
  const [open, setOpen] = useState(false);
  if (steps.length === 0) return null;
  const running = steps.some((s) => s.status === "running");
  const searches = steps.filter((s) => s.name === "search_knowledge_base").length;
  const reads = steps.filter((s) => s.name === "read_chunks").length;
  const summary = [searches && `${searches} search${searches > 1 ? "es" : ""}`, reads && `${reads} read${reads > 1 ? "s" : ""}`]
    .filter(Boolean).join(", ") || `${steps.length} steps`;
  const expanded = open || streaming;
  return (
    <div className="mb-2">
      <button onClick={() => setOpen(!open)} aria-expanded={expanded}
        className="flex items-center gap-1.5 text-sm text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">
        {running ? Icon.spinner() : <span className={`transition ${expanded ? "rotate-90" : ""}`}>{Icon.chevron()}</span>}
        {running ? "Researching…" : summary}
      </button>
      {expanded && (
        <ol className="mt-2 space-y-1.5 border-l border-zinc-200 pl-4 dark:border-zinc-800">
          {steps.map((s) => {
            const { icon, text } = stepLabel(s);
            return (
              <li key={s.id} className="flex items-start gap-2 text-sm text-zinc-500">
                <span className="mt-0.5 shrink-0">{icon}</span>
                <span className="min-w-0 flex-1">{text}</span>
                <span className="mt-0.5 shrink-0">
                  {s.status === "running" ? Icon.spinner("h-3.5 w-3.5")
                    : s.status === "error" ? <span className="text-red-500">{Icon.alert("h-3.5 w-3.5")}</span>
                    : <span className="text-emerald-600">{Icon.check("h-3.5 w-3.5")}</span>}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- answer + citations

const CITE = /\[([^[\]#\n]+)#(\d+)\]/g;

export type CiteRef = { source: string; index: number };

/** Unique citations in order of first appearance - matches the numbering shown in the text. */
export function citationOrder(text: string): CiteRef[] {
  const out: CiteRef[] = [];
  for (const m of text.matchAll(CITE)) {
    const ref = { source: m[1], index: Number(m[2]) };
    if (!out.some((r) => r.source === ref.source && r.index === ref.index)) out.push(ref);
  }
  return out;
}

export function Answer({ turn, onCite }: { turn: Turn; onCite: (c: CiteRef) => void }) {
  const order = useMemo(() => citationOrder(turn.text), [turn.text]);
  const md = useMemo(
    () => turn.text.replace(CITE, (_, s: string, i: string) => `[${s}#${i}](cite:${encodeURIComponent(s)}:${i})`),
    [turn.text],
  );
  const found = (c: CiteRef): Citation | undefined =>
    turn.citations.find((x) => x.source === c.source && x.index === c.index);

  return (
    <div>
      <div className={`prose-answer text-[15px] ${turn.streaming ? "cursor" : ""} ${turn.stop === "error" ? "text-red-600" : ""}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          urlTransform={(u) => u}
          components={{
            a: ({ href, children }) => {
              if (href?.startsWith("cite:")) {
                const [, src, idx] = href.match(/^cite:(.*):(\d+)$/) ?? [];
                const ref = { source: decodeURIComponent(src), index: Number(idx) };
                const n = order.findIndex((r) => r.source === ref.source && r.index === ref.index) + 1;
                const missing = found(ref)?.found === false;
                return (
                  <button onClick={() => onCite(ref)} title={`${ref.source} #${ref.index}`}
                    className={`mx-0.5 inline-grid h-5 min-w-5 -translate-y-px place-items-center rounded-full px-1 align-middle text-[11px] font-semibold
                      ${missing ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300"
                        : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200 dark:bg-indigo-900/50 dark:text-indigo-300 dark:hover:bg-indigo-900"}`}>
                    {n || "?"}
                  </button>
                );
              }
              return <a href={href} target="_blank" rel="noreferrer" className="text-indigo-600 underline">{children}</a>;
            },
          }}
        >
          {md}
        </ReactMarkdown>
      </div>

      {!turn.streaming && order.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {order.map((c, i) => {
            const meta = found(c);
            return (
              <button key={`${c.source}#${c.index}`} onClick={() => onCite(c)}
                className="flex max-w-full items-center gap-1.5 rounded-md border border-zinc-200 px-2 py-1 text-xs text-zinc-600 hover:border-indigo-300 hover:bg-indigo-50 dark:border-zinc-800 dark:text-zinc-400 dark:hover:border-indigo-800 dark:hover:bg-indigo-950/40">
                <span className="font-semibold text-indigo-600 dark:text-indigo-400">{i + 1}</span>
                <span className="truncate">{c.source}</span>
                {meta?.section && <span className="hidden truncate text-zinc-400 sm:inline">· {meta.section.split(" > ").pop()}</span>}
              </button>
            );
          })}
        </div>
      )}

      {!turn.streaming && turn.usage && (
        <div className="mt-2 text-xs text-zinc-400">
          {turn.usage.api_calls} API call{turn.usage.api_calls === 1 ? "" : "s"} ·{" "}
          {(turn.usage.input_tokens + turn.usage.cache_read_tokens + turn.usage.cache_write_tokens).toLocaleString()} in /{" "}
          {turn.usage.output_tokens.toLocaleString()} out
          {turn.cost_usd != null && <> · ~${turn.cost_usd.toFixed(4)}</>}
          {turn.seconds != null && <> · {turn.seconds.toFixed(1)}s</>}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- citation panel

export function CitationPanel({ cite, onClose, onNavigate }: {
  cite: CiteRef;
  onClose: () => void;
  onNavigate: (c: CiteRef) => void;
}) {
  const [view, setView] = useState<ChunkView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const citedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    setError(null);
    api.chunk(cite.source, cite.index)
      .then((v) => live && setView(v))
      .catch((e) => live && (setView(null), setError(String(e.message ?? e))));
    return () => { live = false; };
  }, [cite.source, cite.index]);

  useEffect(() => {
    // Block body on purpose: scrollIntoView can return a Promise, which React would treat as a cleanup.
    citedRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [view]);

  const cited = view?.chunks.find((c) => c.cited);
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start gap-2 border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <div className="mt-0.5 text-zinc-400">{Icon.doc()}</div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{cite.source}</div>
          <div className="truncate text-xs text-zinc-500">
            Chunk {cite.index + 1}{view ? ` of ${view.total_chunks}` : ""}{cited?.section ? ` · ${cited.section}` : ""}
          </div>
        </div>
        <button onClick={onClose} aria-label="Close source panel"
          className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800">{Icon.x()}</button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {error && <p className="text-sm text-red-600">Couldn't load this source: {error}</p>}
        {!view && !error && <p className="text-sm text-zinc-500">Loading…</p>}
        {view?.chunks.map((c) => (
          <div key={c.index} ref={c.cited ? citedRef : undefined}
            className={`rounded-lg p-3 text-sm leading-relaxed ${c.cited
              ? "border border-amber-300 bg-amber-50 dark:border-amber-700/60 dark:bg-amber-950/30"
              : "text-zinc-500"}`}>
            {c.section && <div className="mb-1 text-xs font-medium text-zinc-500">{c.section}</div>}
            <div className="whitespace-pre-wrap">{c.text}</div>
          </div>
        ))}
      </div>
      {view && (
        <div className="flex justify-between border-t border-zinc-200 px-4 py-2 text-sm dark:border-zinc-800">
          <button disabled={cite.index === 0} onClick={() => onNavigate({ ...cite, index: cite.index - 1 })}
            className="rounded px-2 py-1 text-zinc-600 hover:bg-zinc-100 disabled:opacity-30 dark:text-zinc-400 dark:hover:bg-zinc-800">← Earlier</button>
          <button disabled={cite.index >= view.total_chunks - 1} onClick={() => onNavigate({ ...cite, index: cite.index + 1 })}
            className="rounded px-2 py-1 text-zinc-600 hover:bg-zinc-100 disabled:opacity-30 dark:text-zinc-400 dark:hover:bg-zinc-800">Later →</button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- composer

export function Composer({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => void }) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);
  const submit = () => {
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText("");
  };
  return (
    <div className="flex items-end gap-2 rounded-2xl border border-zinc-300 bg-white p-2 shadow-sm focus-within:border-indigo-400 dark:border-zinc-700 dark:bg-zinc-900">
      <textarea ref={ref} rows={1} value={text} onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); } }}
        placeholder="Ask about your documents…" aria-label="Message"
        className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] outline-none placeholder:text-zinc-400" />
      <button onClick={submit} disabled={disabled || !text.trim()} aria-label="Send"
        className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-zinc-300 dark:disabled:bg-zinc-700">
        {disabled ? Icon.spinner() : Icon.send()}
      </button>
    </div>
  );
}
