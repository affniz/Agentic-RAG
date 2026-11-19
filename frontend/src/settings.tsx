import { useEffect, useState, type ReactNode } from "react";
import type { Effort, Health, Settings } from "./types";

export const modelLabel = (m: string) =>
  m.replace("claude-", "").replace(/-(\d+)-(\d+)$/, " $1.$2").replace(/^./, (c) => c.toUpperCase());

const EFFORTS: { id: Effort; hint: string }[] = [
  { id: "low", hint: "Fastest and cheapest; fine for single-fact questions." },
  { id: "medium", hint: "Default. Good balance for most questions." },
  { id: "high", hint: "Helps on hard multi-step questions." },
  { id: "xhigh", hint: "More thinking; slower." },
  { id: "max", hint: "Most thorough; slowest and most expensive." },
];

function Field({ label, hint, htmlFor, children }: { label: string; hint: ReactNode; htmlFor?: string; children: ReactNode }) {
  return (
    <div className="grid gap-2 border-b border-zinc-200 py-5 last:border-0 sm:grid-cols-[1fr_16rem] sm:gap-6 dark:border-zinc-800">
      <div>
        <label htmlFor={htmlFor} className="text-sm font-medium">{label}</label>
        <p className="mt-0.5 text-sm text-zinc-500">{hint}</p>
      </div>
      <div className="sm:pt-0.5">{children}</div>
    </div>
  );
}

const input = "w-full rounded-lg border border-zinc-200 bg-transparent px-2.5 py-1.5 text-sm dark:border-zinc-800 dark:bg-zinc-950";

export function SettingsPage({ health, settings, onSave }: {
  health: Health | null;
  settings: Settings | null;
  onSave: (s: Settings) => Promise<Settings>;
}) {
  const [draft, setDraft] = useState<Settings | null>(settings);
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => { setDraft(settings); }, [settings]);

  if (!draft || !settings) return <div className="p-6 text-sm text-zinc-500">Loading settings…</div>;

  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => { setDraft({ ...draft, [key]: value }); setStatus(null); };

  async function save() {
    if (!draft) return;
    setSaving(true);
    try {
      await onSave(draft);
      setStatus({ ok: true, text: "Saved. Applies from the next question, in every conversation." });
    } catch (e) {
      setStatus({ ok: false, text: (e as Error).message });
    } finally {
      setSaving(false);
    }
  }

  const rerankAvailable = !!health?.reranker_configured;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6">
        <h1 className="text-xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-zinc-500">Used by chat and by answer evals, so an eval run measures what you'd get in chat.</p>

        <div className="mt-4">
          <Field label="Model" htmlFor="s-model" hint="Haiku with the agent loop scored 95-100% on the hard eval set at about $0.001 per question. Sonnet and Opus add margin at 10-20× the cost.">
            <select id="s-model" value={draft.model} onChange={(e) => set("model", e.target.value)} className={input}>
              {health?.models.map((m) => <option key={m} value={m}>{modelLabel(m)}</option>)}
            </select>
          </Field>

          <Field label="Effort" htmlFor="s-effort" hint={EFFORTS.find((e) => e.id === draft.effort)?.hint}>
            <select id="s-effort" value={draft.effort} onChange={(e) => set("effort", e.target.value as Effort)} className={input}>
              {EFFORTS.map((e) => <option key={e.id} value={e.id}>{e.id}</option>)}
            </select>
          </Field>

          <Field label="Results per search" htmlFor="s-k"
            hint={draft.search_k == null
              ? "The model picks how many chunks each search returns (usually 5)."
              : `Every search returns ${draft.search_k} chunk${draft.search_k === 1 ? "" : "s"}, whatever the model asks for. More results means more context and more input tokens.`}>
            <div className="flex items-center gap-3">
              <label className="flex shrink-0 items-center gap-1.5 text-sm">
                <input type="checkbox" checked={draft.search_k == null} onChange={(e) => set("search_k", e.target.checked ? null : 5)} />
                Auto
              </label>
              <input id="s-k" type="range" min={1} max={10} value={draft.search_k ?? 5} disabled={draft.search_k == null}
                onChange={(e) => set("search_k", Number(e.target.value))} className="min-w-0 flex-1 disabled:opacity-40" />
              <span className="w-5 text-right text-sm tabular-nums">{draft.search_k ?? "-"}</span>
            </div>
          </Field>

          <Field label="Reranking" htmlFor="s-rerank"
            hint={rerankAvailable
              ? "A local cross-encoder re-scores the top candidates after keyword + semantic search. Free, adds a few hundred milliseconds per search."
              : "The reranker is turned off on the server (RAG_RERANK=0), so this has no effect."}>
            <label className="flex items-center gap-2 text-sm">
              <input id="s-rerank" type="checkbox" checked={draft.rerank} disabled={!rerankAvailable}
                onChange={(e) => set("rerank", e.target.checked)} />
              {draft.rerank && rerankAvailable ? "On" : "Off"}
              {rerankAvailable && !health?.reranker && <span className="text-xs text-zinc-500">(model loads on first search)</span>}
            </label>
          </Field>

          <Field label="Step limit" htmlFor="s-steps"
            hint="Most model calls the agent may make for one question. When it runs out, it must answer with what it has found. Questions on the hard eval set average about 3.">
            <div className="flex items-center gap-3">
              <input id="s-steps" type="range" min={1} max={30} value={draft.max_steps}
                onChange={(e) => set("max_steps", Number(e.target.value))} className="min-w-0 flex-1" />
              <span className="w-6 text-right text-sm tabular-nums">{draft.max_steps}</span>
            </div>
          </Field>

          <Field label="Cost limit per question" htmlFor="s-cost"
            hint={draft.max_cost_usd == null
              ? "No limit: a question can make as many model calls as the step limit allows."
              : `Once a question has cost $${draft.max_cost_usd.toFixed(2)}, the agent stops searching and answers with what it has (that last call can take it a little over). Chat only; evals show their own estimate before running.`}>
            <div className="flex items-center gap-3">
              <label className="flex shrink-0 items-center gap-1.5 text-sm">
                <input type="checkbox" checked={draft.max_cost_usd == null} onChange={(e) => set("max_cost_usd", e.target.checked ? null : 0.5)} />
                No limit
              </label>
              <span className="text-sm text-zinc-500">$</span>
              <input id="s-cost" type="number" min={0.01} max={100} step={0.05} value={draft.max_cost_usd ?? ""}
                disabled={draft.max_cost_usd == null}
                onChange={(e) => { const v = Number(e.target.value); if (v > 0) set("max_cost_usd", v); }}
                className={`${input} disabled:opacity-40`} />
            </div>
          </Field>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-3">
          <button onClick={save} disabled={!dirty || saving}
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
            {saving ? "Saving…" : "Save"}
          </button>
          {dirty && (
            <button onClick={() => setDraft(settings)} className="rounded-lg px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800">
              Discard changes
            </button>
          )}
          {status && <span className={`text-sm ${status.ok ? "text-emerald-700 dark:text-emerald-400" : "text-red-600"}`}>{status.text}</span>}
        </div>
      </div>
    </div>
  );
}
