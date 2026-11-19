import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamChat } from "./api";
import { Answer, CitationPanel, Composer, Icon, Sidebar, Steps, type CiteRef, type View } from "./components";
import { DocumentsPage } from "./documents";
import { EvalsPage } from "./evals";
import { SettingsPage, modelLabel } from "./settings";
import type { ConversationSummary, Feedback, Health, Settings, Turn } from "./types";

const SUGGESTIONS = [
  "How many vacation days can I carry over, and when do they expire?",
  "I want to attend a conference in Tokyo with my learning budget. What's my budget and the hotel cap?",
  "I left my laptop on the train. What do I do?",
  "If I put 6% of my salary in my 401(k), how much does Acme add?",
];

const EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;
const VIEWS: View[] = ["chat", "documents", "evals", "settings"];
const viewFromHash = (): View => VIEWS.find((v) => window.location.hash === `#/${v}`) ?? "chat";

const emptyAssistant = (): Turn => ({
  role: "assistant", text: "", steps: [], citations: [], usage: null,
  cost_usd: null, seconds: null, stop: null, streaming: true,
});

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [cite, setCite] = useState<CiteRef | null>(null);
  const [navOpen, setNavOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>(viewFromHash);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = (v: View) => {
    window.location.hash = v === "chat" ? "#/" : `#/${v}`;
    setView(v);
    setNavOpen(false);
  };

  const refreshList = useCallback(() => api.conversations().then(setConversations).catch(() => {}), []);

  const refreshHealth = useCallback(() => api.health().then(setHealth).catch(() => {}), []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError("Can't reach the API server. Is it running on :8000?"));
    api.settings().then(setSettings).catch(() => {});
    refreshList();
  }, [refreshList]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ block: "end" }); }, [turns]);

  const patchLast = (fn: (t: Turn) => Turn) =>
    setTurns((ts) => (ts.length ? [...ts.slice(0, -1), fn(ts[ts.length - 1])] : ts));

  async function send(message: string) {
    setError(null);
    setBusy(true);
    setTurns((ts) => [...ts, { ...emptyAssistant(), role: "user", text: message, streaming: false }, emptyAssistant()]);
    try {
      for await (const ev of streamChat({ message, conversation_id: activeId })) {
        switch (ev.type) {
          case "conversation":
            setActiveId(ev.id);
            patchLast((t) => ({ ...t, id: ev.turn_id, model: ev.model }));
            refreshList();
            break;
          case "tool_call":
            patchLast((t) => ({ ...t, steps: [...t.steps, { id: ev.id, name: ev.name, args: ev.args, status: "running", step: ev.step }] }));
            break;
          case "tool_result":
            patchLast((t) => ({
              ...t,
              steps: t.steps.map((s) => (s.id === ev.id ? { ...s, status: ev.is_error ? "error" : "done", chars: ev.chars, results: ev.results, seconds: ev.seconds } : s)),
            }));
            break;
          case "model_call": {
            const { type: _, ...call } = ev;
            patchLast((t) => ({ ...t, timings: [...(t.timings ?? []), call] }));
            break;
          }
          case "text_delta":
            patchLast((t) => ({ ...t, text: t.text + ev.text }));
            break;
          case "answer":
            patchLast((t) => ({ ...t, text: ev.text, citations: ev.citations, stop: ev.stop }));
            break;
          case "usage":
            patchLast((t) => ({ ...t, usage: ev.usage, cost_usd: ev.cost_usd }));
            break;
          case "error":
            patchLast((t) => ({ ...t, text: `Something went wrong: ${ev.message}`, stop: "error" }));
            break;
          case "done":
            patchLast((t) => ({ ...t, seconds: ev.seconds, streaming: false }));
            refreshList();
            break;
        }
      }
    } catch (e) {
      patchLast((t) => ({ ...t, text: `Request failed: ${(e as Error).message}`, stop: "error" }));
    } finally {
      patchLast((t) => ({ ...t, streaming: false }));
      setBusy(false);
    }
  }

  async function open(id: string) {
    navigate("chat");
    setCite(null);
    const c = await api.conversation(id);
    setActiveId(id);
    setTurns(c.messages);
  }

  function newChat() {
    navigate("chat");
    setActiveId(null);
    setTurns([]);
    setCite(null);
    setNavOpen(false);
  }

  async function resetAll() {
    if (!window.confirm("Delete every conversation? Feedback you gave is kept.")) return;
    await api.resetConversations();
    newChat();
    refreshList();
  }

  async function saveSettings(next: Settings) {
    const saved = await api.saveSettings(next);
    setSettings(saved);
    return saved;
  }

  async function giveFeedback(turnId: string, rating: Feedback["rating"] | null, comment = "") {
    if (!activeId) return;
    const r = await api.feedback(turnId, activeId, rating, comment);
    setTurns((ts) => ts.map((t) => (t.id === turnId ? { ...t, feedback: r.feedback } : t)));
  }

  async function remove(id: string) {
    await api.deleteConversation(id);
    if (id === activeId) newChat();
    refreshList();
  }

  const title = conversations.find((c) => c.id === activeId)?.title ?? "New chat";

  return (
    <div className="flex h-full">
      {/* Sidebar: static on desktop, drawer on mobile */}
      <aside className={`fixed inset-y-0 left-0 z-30 w-72 border-r border-zinc-200 bg-zinc-50 transition-transform dark:border-zinc-800 dark:bg-zinc-900
        md:static md:translate-x-0 ${navOpen ? "translate-x-0" : "-translate-x-full"}`}>
        <Sidebar conversations={conversations} activeId={activeId} health={health} view={view}
          onNavigate={navigate} onSelect={open} onNew={newChat} onDelete={remove} onReset={resetAll} />
      </aside>
      {navOpen && <div className="fixed inset-0 z-20 bg-black/30 md:hidden" onClick={() => setNavOpen(false)} />}

      {view !== "chat" ? (
        <main className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-center gap-2 border-b border-zinc-200 px-4 py-2.5 md:hidden dark:border-zinc-800">
            <button className="rounded p-1" onClick={() => setNavOpen(true)} aria-label="Open navigation">{Icon.menu("h-5 w-5")}</button>
            <span className="text-sm font-medium capitalize">{view}</span>
          </header>
          {view === "documents" && <DocumentsPage health={health} onIndexChanged={refreshHealth} />}
          {view === "evals" && <EvalsPage health={health} settings={settings} />}
          {view === "settings" && <SettingsPage health={health} settings={settings} onSave={saveSettings} />}
        </main>
      ) : (<>
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-800">
          <button className="rounded p-1 md:hidden" onClick={() => setNavOpen(true)} aria-label="Open conversations">{Icon.menu("h-5 w-5")}</button>
          <h1 className="min-w-0 flex-1 truncate text-sm font-medium">{title}</h1>
          <label className="sr-only" htmlFor="model">Model</label>
          <select id="model" value={settings?.model ?? ""} disabled={busy || !settings}
            onChange={(e) => settings && saveSettings({ ...settings, model: e.target.value })}
            className="rounded-md border border-zinc-200 bg-transparent px-2 py-1 text-xs dark:border-zinc-800">
            {health?.models.map((m) => <option key={m} value={m}>{modelLabel(m)}</option>)}
          </select>
          <label className="sr-only" htmlFor="effort">Effort</label>
          <select id="effort" value={settings?.effort ?? "medium"} disabled={busy || !settings}
            onChange={(e) => settings && saveSettings({ ...settings, effort: e.target.value as Settings["effort"] })}
            className="hidden rounded-md border border-zinc-200 bg-transparent px-2 py-1 text-xs sm:block dark:border-zinc-800">
            {EFFORTS.map((e) => <option key={e} value={e}>{e} effort</option>)}
          </select>
          <button onClick={() => navigate("settings")} aria-label="Settings" title="All settings"
            className="rounded p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-200">
            {Icon.gear()}
          </button>
        </header>

        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6">
            {error && <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">{error}</div>}

            {turns.length === 0 ? (
              <div className="pt-[12vh] text-center">
                <h2 className="text-2xl font-semibold tracking-tight">Ask your documents</h2>
                <p className="mt-2 text-sm text-zinc-500">
                  {health ? `${health.documents} documents · ${health.chunks} chunks indexed` : "…"}. Answers cite their sources.
                </p>
                <div className="mx-auto mt-8 grid max-w-2xl gap-2 sm:grid-cols-2">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} onClick={() => send(s)} disabled={busy}
                      className="rounded-xl border border-zinc-200 px-4 py-3 text-left text-sm text-zinc-600 hover:border-indigo-300 hover:bg-indigo-50/50 dark:border-zinc-800 dark:text-zinc-400 dark:hover:border-indigo-800 dark:hover:bg-indigo-950/30">
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {turns.map((t, i) =>
                  t.role === "user" ? (
                    <div key={i} className="flex justify-end">
                      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-zinc-100 px-4 py-2.5 text-[15px] dark:bg-zinc-800">{t.text}</div>
                    </div>
                  ) : (
                    <div key={i}>
                      <Steps steps={t.steps} timings={t.timings ?? []} streaming={!!t.streaming} />
                      {t.text || !t.streaming ? <Answer turn={t} onCite={setCite}
                        onFeedback={t.id && !t.streaming ? (r, c) => giveFeedback(t.id!, r, c) : undefined} />
                        : !t.steps.length && <div className="flex items-center gap-2 text-sm text-zinc-500">{Icon.spinner()} Thinking…</div>}
                    </div>
                  ),
                )}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="mx-auto w-full max-w-3xl px-4 pb-4">
          <Composer disabled={busy} onSend={send} />
          <p className="mt-1.5 text-center text-xs text-zinc-400">Enter to send · Shift+Enter for a new line</p>
        </div>
      </main>

      {cite && (
        <>
          <div className="fixed inset-0 z-20 bg-black/30 lg:hidden" onClick={() => setCite(null)} />
          <aside className="fixed inset-y-0 right-0 z-30 w-full max-w-md border-l border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950 lg:static lg:w-[26rem]">
            <CitationPanel cite={cite} onClose={() => setCite(null)} onNavigate={setCite} />
          </aside>
        </>
      )}
      </>)}
    </div>
  );
}
