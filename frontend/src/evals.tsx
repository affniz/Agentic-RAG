import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import { Icon } from "./components";
import { modelLabel } from "./settings";
import type {
  Estimate, EvalJob, EvalRequest, EvalRow, EvalRun, EvalSet, EvalSystem, ExportResult, FeedbackRecord, Health, Settings,
} from "./types";

// ---------------------------------------------------------------- helpers

const pct = (v: number | null | undefined) => (v == null ? "-" : `${Math.round(v * 100)}%`);
const usd = (v: number) => (v === 0 ? "$0" : v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
const date = (ts: number) =>
  new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

const SYSTEMS: { id: EvalSystem; label: string; hint: string }[] = [
  { id: "agentic", label: "Agent", hint: "The agent as used in chat" },
  { id: "naive", label: "One-shot baseline", hint: "Retrieve once, answer once - shows what the agent loop adds" },
  { id: "agentic-v1", label: "Agent (v1 prompt)", hint: "Previous agent instructions, for A/B comparison" },
];
const systemLabel = (s: string) => SYSTEMS.find((x) => x.id === s)?.label ?? s;

const VERDICT = {
  correct: { label: "Correct", cls: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300", icon: Icon.check },
  partially_correct: { label: "Partial", cls: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300", icon: Icon.alert },
  incorrect: { label: "Incorrect", cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300", icon: Icon.x },
} as const;

const btn = "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-50";
const btnPrimary = `${btn} bg-indigo-600 text-white hover:bg-indigo-700`;
const btnSecondary = `${btn} border border-zinc-200 hover:bg-zinc-100 dark:border-zinc-800 dark:hover:bg-zinc-900`;
const card = "rounded-xl border border-zinc-200 dark:border-zinc-800";
const input = "rounded-lg border border-zinc-200 bg-transparent px-2 py-1 text-sm dark:border-zinc-800 dark:bg-zinc-950";

// ---------------------------------------------------------------- page

export function EvalsPage({ health, settings }: { health: Health | null; settings: Settings | null }) {
  const [sets, setSets] = useState<EvalSet[]>([]);
  const [setName, setSetName] = useState<string>("");
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [job, setJob] = useState<EvalJob>({ state: "idle" });
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, r, j] = await Promise.all([api.evalSets(), api.evalRuns(), api.evalJob()]);
      setSets(s);
      setRuns(r);
      setJob(j);
      setSetName((cur) => cur || (s.find((x) => x.name === "hard") ?? s[0])?.name || "");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (job.state !== "running") return;
    const t = setInterval(async () => {
      const j = await api.evalJob().catch(() => null);
      if (!j) return;
      setJob(j);
      if (j.state !== "running") {
        clearInterval(t);
        const r = await api.evalRuns();
        setRuns(r);
        if (j.state === "done" && j.result_file) setSelected(j.result_file);
      }
    }, 800);
    return () => clearInterval(t);
  }, [job.state]);

  const evalSet = sets.find((s) => s.name === setName);
  const setRuns_ = runs.filter((r) => r.set === setName);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Evals</h1>
            <p className="mt-1 text-sm text-zinc-500">Check whether a change to the documents, settings or prompts helped.</p>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="eval-set" className="text-sm text-zinc-500">Eval set</label>
            <select id="eval-set" value={setName} onChange={(e) => { setSetName(e.target.value); setSelected(null); }} className={input}>
              {sets.map((s) => <option key={s.name} value={s.name}>{s.name} ({s.questions})</option>)}
            </select>
          </div>
        </div>

        {error && <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">{error}</div>}
        {health?.demo_mode && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300">
            Demo mode: answers come from the stand-in model and a keyword-matching judge, so scores here don't mean anything. Runs are free.
          </div>
        )}

        {evalSet && <RunPanel evalSet={evalSet} settings={settings} job={job} onStarted={setJob} />}

        <HistoryChart runs={setRuns_} selected={selected} onSelect={setSelected} />

        <RunsTable runs={setRuns_} selected={selected} onSelect={setSelected} />

        {selected && <RunDetail key={selected} file={selected} onClose={() => setSelected(null)} />}

        <FeedbackPanel onExported={refresh} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- run panel

function RunPanel({ evalSet, settings, job, onStarted }: {
  evalSet: EvalSet;
  settings: Settings | null;
  job: EvalJob;
  onStarted: (j: EvalJob) => void;
}) {
  const [systems, setSystems] = useState<EvalSystem[]>(["agentic"]);
  const [limit, setLimit] = useState<number | null>(null);
  const [repeats, setRepeats] = useState(1);
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const running = job.state === "running";

  const request = (kind: EvalRequest["kind"]): EvalRequest =>
    ({ kind, set: evalSet.name, systems, limit, repeats });

  async function start(kind: EvalRequest["kind"]) {
    setBusy(true);
    setError(null);
    try {
      onStarted(await api.startEval(request(kind)));
      setEstimate(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function askEstimate() {
    setBusy(true);
    setError(null);
    try {
      setEstimate(await api.estimate(request("answers")));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const toggle = (s: EvalSystem) =>
    setSystems((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));

  return (
    <section className={`${card} p-4`} aria-labelledby="run-heading">
      <h2 id="run-heading" className="text-sm font-semibold">Run an eval</h2>
      <p className="mt-0.5 text-sm text-zinc-500">
        {evalSet.description || `${evalSet.questions} questions.`}{" "}
        <span className="text-zinc-400">{Object.entries(evalSet.categories).map(([c, n]) => `${c} ${n}`).join(" · ")}</span>
      </p>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
          <div className="text-sm font-medium">Retrieval eval <span className="font-normal text-zinc-500">· free, no API calls</span></div>
          <p className="mt-1 text-sm text-zinc-500">Searches with each question's text and checks whether the evidence lands in the top 5, for every search mode.</p>
          <button className={`${btnSecondary} mt-3`} disabled={running || busy} onClick={() => start("retrieval")}>Run retrieval eval</button>
        </div>

        <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
          <div className="text-sm font-medium">Answer eval <span className="font-normal text-zinc-500">· calls Claude</span></div>
          <p className="mt-1 text-sm text-zinc-500">
            Answers every question{settings && <> with {modelLabel(settings.model)}, {settings.effort} effort, step limit {settings.max_steps}</>}, then a judge grades each answer against the reference.
          </p>
          <fieldset className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
            <legend className="sr-only">Systems</legend>
            {SYSTEMS.map((s) => (
              <label key={s.id} className="flex items-center gap-1.5 text-sm" title={s.hint}>
                <input type="checkbox" checked={systems.includes(s.id)} onChange={() => toggle(s.id)} /> {s.label}
              </label>
            ))}
          </fieldset>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
            <label className="flex items-center gap-1.5">Questions
              <select value={limit ?? ""} onChange={(e) => setLimit(e.target.value ? Number(e.target.value) : null)} className={input}>
                <option value="">all {evalSet.questions}</option>
                {[5, 10, 20].filter((n) => n < evalSet.questions).map((n) => <option key={n} value={n}>first {n}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-1.5" title="Run everything N times; shows the range and which questions flip">Repeats
              <select value={repeats} onChange={(e) => setRepeats(Number(e.target.value))} className={input}>
                {[1, 2, 3, 5].map((n) => <option key={n} value={n}>{n}×</option>)}
              </select>
            </label>
          </div>
          <button className={`${btnPrimary} mt-3`} disabled={running || busy || systems.length === 0} onClick={askEstimate}>
            Run answer eval…
          </button>
        </div>
      </div>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      {estimate && <ConfirmCost estimate={estimate} busy={busy} onCancel={() => setEstimate(null)} onConfirm={() => start("answers")} />}
      {job.state !== "idle" && <JobStatus job={job} />}
    </section>
  );
}

function ConfirmCost({ estimate, busy, onCancel, onConfirm }: {
  estimate: Estimate; busy: boolean; onCancel: () => void; onConfirm: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { ref.current?.showModal(); }, []);
  const basis = {
    history: "Based on your last run of the same model and system.",
    rough: "Rough estimate from typical token counts - no past run with this model to go by.",
    demo: "Demo mode: no model is called.",
    free: "",
  }[estimate.basis];
  return (
    <dialog ref={ref} onClose={onCancel} aria-labelledby="confirm-title"
      className="m-auto w-[min(28rem,calc(100vw-2rem))] rounded-xl border border-zinc-200 bg-white p-5 text-zinc-900 backdrop:bg-black/40 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100">
      <h3 id="confirm-title" className="font-semibold">Run the answer eval?</h3>
      <div className="mt-3 text-3xl font-semibold tabular-nums">~{usd(estimate.total_usd)}</div>
      <p className="mt-1 text-sm text-zinc-500">{basis}</p>
      <dl className="mt-4 space-y-1 text-sm">
        {Object.entries(estimate.per_system).map(([s, c]) => (
          <div key={s} className="flex justify-between"><dt className="text-zinc-500">{systemLabel(s)} ({estimate.answers} answers)</dt><dd className="tabular-nums">{usd(c)}</dd></div>
        ))}
        <div className="flex justify-between"><dt className="text-zinc-500">Judge</dt><dd className="tabular-nums">{usd(estimate.judge_usd)}</dd></div>
        <div className="flex justify-between"><dt className="text-zinc-500">API calls, about</dt><dd className="tabular-nums">{estimate.api_calls}</dd></div>
      </dl>
      <div className="mt-5 flex justify-end gap-2">
        <button className={btnSecondary} onClick={() => ref.current?.close()}>Cancel</button>
        <button className={btnPrimary} disabled={busy} autoFocus onClick={() => { onConfirm(); ref.current?.close(); }}>
          Run for ~{usd(estimate.total_usd)}
        </button>
      </div>
    </dialog>
  );
}

function JobStatus({ job }: { job: EvalJob }) {
  const [cancelling, setCancelling] = useState(false);
  const share = job.total ? (job.done ?? 0) / job.total : 0;
  const what = `${job.kind === "answers" ? "Answer" : "Retrieval"} eval on ${job.set}`;
  if (job.state === "running") {
    return (
      <div className="mt-4" role="status">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="flex items-center gap-2">{Icon.spinner()} {what}: {job.stage}{job.total ? ` · ${job.done}/${job.total}` : ""}</span>
          <span className="flex items-center gap-3">
            {job.kind === "answers" && <span className="tabular-nums text-zinc-500">{usd(job.cost_usd ?? 0)} so far</span>}
            {job.kind === "answers" && (
              <button className="text-sm text-red-600 hover:underline disabled:opacity-50" disabled={cancelling}
                onClick={() => { setCancelling(true); api.cancelEval().catch(() => setCancelling(false)); }}>
                {cancelling ? "Stopping…" : "Stop"}
              </button>
            )}
          </span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
          <div className="h-full rounded-full bg-indigo-600 transition-[width]" style={{ width: `${Math.max(3, share * 100)}%` }} />
        </div>
      </div>
    );
  }
  const msg = {
    done: <span className="text-emerald-700 dark:text-emerald-400">{what} finished{job.kind === "answers" ? `, cost ${usd(job.cost_usd ?? 0)}` : ""}</span>,
    cancelled: <span className="text-zinc-500">{what} stopped; nothing was saved ({usd(job.cost_usd ?? 0)} spent)</span>,
    error: <span className="text-red-600">{what} failed: {job.error}</span>,
    idle: null, running: null,
  }[job.state];
  return <p className="mt-4 text-sm">{msg}</p>;
}

// ---------------------------------------------------------------- chart

type Series = { key: string; label: string; color: string; points: { i: number; v: number; run: EvalRun }[] };

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(640);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function HistoryChart({ runs, selected, onSelect }: { runs: EvalRun[]; selected: string | null; onSelect: (f: string) => void }) {
  const [kind, setKind] = useState<"answers" | "retrieval">("answers");
  const [hover, setHover] = useState<number | null>(null);
  const [ref, width] = useWidth<HTMLDivElement>();
  const shown = runs.filter((r) => r.kind === kind);

  const series = useMemo<Series[]>(() => {
    const map = new Map<string, Series>();
    shown.forEach((run, i) => {
      const entries: [string, string, number | undefined][] = kind === "answers"
        ? Object.entries(run.systems ?? {}).map(([s, v]) => [`${s}|${run.model}`, `${systemLabel(s)} · ${modelLabel(run.model ?? "")}`, v.correct])
        : Object.entries(run.configs ?? {}).map(([c, v]) => [c, c, v["recall@k"]]);
      for (const [key, label, v] of entries) {
        if (v == null) continue;
        if (!map.has(key)) map.set(key, { key, label, color: "", points: [] });
        map.get(key)!.points.push({ i, v, run });
      }
    });
    // Color follows the entity: slots are assigned by sorted key, so filtering never repaints a series.
    const all = [...map.values()].sort((a, b) => a.key.localeCompare(b.key)).slice(0, 8);
    all.forEach((s, n) => { s.color = `var(--series-${n + 1})`; });
    return all;
  }, [shown, kind]);

  const H = 220, pad = { l: 40, r: 16, t: 12, b: 28 };
  const W = Math.max(280, width);
  const n = shown.length;
  const x = (i: number) => pad.l + (n <= 1 ? (W - pad.l - pad.r) / 2 : (i * (W - pad.l - pad.r)) / (n - 1));
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const metric = kind === "answers" ? "Correct" : `Recall@${shown[0]?.k ?? 5}`;

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!n) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    setHover(best);
  }

  const hovered = hover != null ? shown[hover] : null;

  return (
    <section className={`${card} p-4`} aria-labelledby="chart-heading">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="chart-heading" className="text-sm font-semibold">{metric} over past runs</h2>
        <div className="flex rounded-lg border border-zinc-200 p-0.5 text-sm dark:border-zinc-800" role="tablist">
          {(["answers", "retrieval"] as const).map((k) => (
            <button key={k} role="tab" aria-selected={kind === k} onClick={() => { setKind(k); setHover(null); }}
              className={`rounded-md px-2.5 py-1 capitalize ${kind === k ? "bg-zinc-100 font-medium dark:bg-zinc-800" : "text-zinc-500"}`}>
              {k}
            </button>
          ))}
        </div>
      </div>

      {series.length > 1 && (
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-600 dark:text-zinc-400" aria-label="Legend">
          {series.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: s.color }} />{s.label}
            </li>
          ))}
        </ul>
      )}

      <div ref={ref} className="relative mt-2">
        {n === 0 ? (
          <p className="py-12 text-center text-sm text-zinc-500">No {kind} runs on this set yet.</p>
        ) : (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} className="block touch-none select-none"
              onPointerMove={onMove} onPointerLeave={() => setHover(null)}
              onClick={() => hovered && kind === "answers" && onSelect(hovered.file)}
              role="img" aria-label={`${metric} for ${n} runs`}>
              {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                <g key={v}>
                  <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} className="stroke-zinc-200 dark:stroke-zinc-800" strokeWidth={1} />
                  <text x={pad.l - 6} y={y(v)} dy="0.32em" textAnchor="end" className="fill-zinc-500 text-[10px]">{v * 100}%</text>
                </g>
              ))}
              {shown.map((r, i) => (n <= 8 || i === 0 || i === n - 1) && (
                <text key={r.file} x={x(i)} y={H - 8} textAnchor={n > 1 && i === 0 ? "start" : n > 1 && i === n - 1 ? "end" : "middle"}
                  className="fill-zinc-500 text-[10px]">
                  {new Date(r.created_at * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                </text>
              ))}
              {hover != null && <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={H - pad.b} className="stroke-zinc-400 dark:stroke-zinc-600" strokeWidth={1} />}
              {series.map((s) => (
                <g key={s.key}>
                  {s.points.length > 1 && (
                    <polyline fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
                      points={s.points.map((p) => `${x(p.i)},${y(p.v)}`).join(" ")} />
                  )}
                  {s.points.map((p) => (
                    <circle key={p.i} cx={x(p.i)} cy={y(p.v)} r={p.run.file === selected || p.i === hover ? 5 : 4}
                      fill={s.color} stroke="var(--chart-surface)" strokeWidth={2} />
                  ))}
                </g>
              ))}
            </svg>
            {hovered && (
              <div className="pointer-events-none absolute top-0 z-10 w-56 rounded-lg border border-zinc-200 bg-white p-2.5 text-xs shadow-lg dark:border-zinc-700 dark:bg-zinc-900"
                style={{ left: Math.min(Math.max(0, (x(hover!) / W) * width - 112), width - 224) }}>
                <div className="font-medium">{date(hovered.created_at)}</div>
                <div className="text-zinc-500">
                  {kind === "answers"
                    ? `${hovered.questions} questions${hovered.repeats && hovered.repeats > 1 ? ` × ${hovered.repeats}` : ""} · ${hovered.effort} effort`
                    : `${hovered.questions} questions · k=${hovered.k}`}
                </div>
                <ul className="mt-1.5 space-y-0.5">
                  {series.flatMap((s) => s.points.filter((p) => p.i === hover).map((p) => (
                    <li key={s.key} className="flex items-center gap-1.5">
                      <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: s.color }} />
                      <span className="min-w-0 flex-1 truncate">{s.label}</span>
                      <span className="font-medium tabular-nums">{pct(p.v)}</span>
                    </li>
                  )))}
                </ul>
                {kind === "answers" && <div className="mt-1.5 text-zinc-400">Click for per-question results</div>}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- runs table

function RunsTable({ runs, selected, onSelect }: { runs: EvalRun[]; selected: string | null; onSelect: (f: string) => void }) {
  const answers = [...runs].filter((r) => r.kind === "answers").reverse();
  const retrieval = [...runs].filter((r) => r.kind === "retrieval").reverse()[0];
  if (!answers.length && !retrieval) return null;
  return (
    <section className={card} aria-labelledby="runs-heading">
      <h2 id="runs-heading" className="px-4 pt-4 text-sm font-semibold">Answer runs</h2>
      {answers.length === 0 ? <p className="px-4 pb-4 pt-1 text-sm text-zinc-500">None yet.</p> : (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-zinc-500">
              <tr>
                <th className="px-4 py-2 font-medium">When</th>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">System</th>
                <th className="px-3 py-2 text-right font-medium">Correct</th>
                <th className="hidden px-3 py-2 text-right font-medium sm:table-cell">Evidence</th>
                <th className="hidden px-3 py-2 text-right font-medium md:table-cell">API calls</th>
                <th className="px-4 py-2 text-right font-medium">Cost / q</th>
              </tr>
            </thead>
            <tbody>
              {answers.flatMap((r) => Object.entries(r.systems ?? {}).map(([s, v], j) => (
                <tr key={`${r.file}-${s}`} onClick={() => onSelect(r.file)}
                  className={`cursor-pointer border-t border-zinc-100 dark:border-zinc-900 ${r.file === selected ? "bg-indigo-50 dark:bg-indigo-950/30" : "hover:bg-zinc-50 dark:hover:bg-zinc-900"}`}>
                  <td className="whitespace-nowrap px-4 py-2 text-zinc-600 dark:text-zinc-400">
                    {j === 0 && <button className="text-left hover:underline" onClick={() => onSelect(r.file)}>{date(r.created_at)}</button>}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2">{j === 0 && <>{modelLabel(r.model ?? "")} <span className="text-zinc-400">· {r.effort}</span></>}</td>
                  <td className="whitespace-nowrap px-3 py-2">{systemLabel(s)}</td>
                  <td className="px-3 py-2 text-right font-medium tabular-nums">
                    {pct(v.correct)}{v.range?.correct && v.range.correct[0] !== v.range.correct[1] &&
                      <span className="ml-1 font-normal text-zinc-400">({pct(v.range.correct[0])}-{pct(v.range.correct[1])})</span>}
                  </td>
                  <td className="hidden px-3 py-2 text-right tabular-nums sm:table-cell">{pct(v.evidence_recall)}</td>
                  <td className="hidden px-3 py-2 text-right tabular-nums md:table-cell">{v.avg_api_calls.toFixed(1)}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{usd(v.cost_usd / Math.max(1, r.questions))}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      )}
      {retrieval && (
        <div className="border-t border-zinc-200 px-4 py-3 dark:border-zinc-800">
          <div className="text-sm font-semibold">Latest retrieval run <span className="font-normal text-zinc-500">· {date(retrieval.created_at)}, k={retrieval.k}</span></div>
          <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {Object.entries(retrieval.configs ?? {}).map(([c, m]) => (
              <div key={c} className="rounded-lg bg-zinc-50 px-3 py-2 dark:bg-zinc-900">
                <div className="text-xs text-zinc-500">{c}</div>
                <div className="text-lg font-semibold tabular-nums">{pct(m["recall@k"])}</div>
                <div className="text-xs text-zinc-500">hit {pct(m["hit@k"])} · MRR {m.mrr.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------- run detail

function RunDetail({ file, onClose }: { file: string; onClose: () => void }) {
  const [data, setData] = useState<{ summary: EvalRun; rows: EvalRow[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "wrong">("all");
  const [system, setSystem] = useState<string>("all");
  const [open, setOpen] = useState<string | null>(null);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    api.evalRun(file).then(setData).catch((e) => setError(e.message));
  }, [file]);
  useEffect(() => { if (data) ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }, [data]);

  if (error) return <p className="text-sm text-red-600">Couldn't load the run: {error}</p>;
  if (!data) return <p className="text-sm text-zinc-500">Loading run…</p>;

  const { summary, rows } = data;
  const systems = Object.keys(summary.systems ?? {});
  const shown = rows.filter((r) => (filter === "all" || r.verdict !== "correct") && (system === "all" || r.system === system));

  return (
    <section ref={ref} className={`${card} scroll-mt-4`} aria-labelledby="detail-heading">
      <div className="flex items-start justify-between gap-3 px-4 pt-4">
        <div>
          <h2 id="detail-heading" className="text-sm font-semibold">Per-question results</h2>
          <p className="text-sm text-zinc-500">
            {date(summary.created_at)} · {modelLabel(summary.model ?? "")} · {summary.effort} effort
            {summary.max_steps != null && <> · step limit {summary.max_steps}</>}
            {summary.search_k != null && <> · {summary.search_k} results/search</>}
            {summary.rerank === false && <> · no reranking</>} · judged by {modelLabel(summary.judge_model ?? "")}
          </p>
        </div>
        <button onClick={onClose} aria-label="Close results" className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800">{Icon.x()}</button>
      </div>

      <div className="mt-3 grid gap-2 px-4 sm:grid-cols-2 lg:grid-cols-3">
        {systems.map((s) => {
          const v = summary.systems![s];
          return (
            <div key={s} className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
              <div className="text-xs text-zinc-500">{systemLabel(s)}</div>
              <div className="text-2xl font-semibold tabular-nums">{pct(v.correct)}</div>
              <div className="text-xs text-zinc-500">
                partial {pct(v.partially_correct)} · evidence {pct(v.evidence_recall)} · {v.avg_api_calls.toFixed(1)} calls · {v.avg_seconds.toFixed(1)}s · {usd(v.cost_usd)}
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {Object.entries(v.by_category).map(([c, st]) => (
                  <span key={c} className="rounded bg-white px-1.5 py-0.5 text-xs text-zinc-600 dark:bg-zinc-950 dark:text-zinc-400">{c} {pct(st.correct)}</span>
                ))}
              </div>
              {v.unstable && v.unstable.length > 0 && <div className="mt-1 text-xs text-amber-700 dark:text-amber-400">Flipped between runs: {v.unstable.join(", ")}</div>}
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 px-4 text-sm">
        <div className="flex rounded-lg border border-zinc-200 p-0.5 dark:border-zinc-800">
          {(["all", "wrong"] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)} aria-pressed={filter === f}
              className={`rounded-md px-2.5 py-0.5 ${filter === f ? "bg-zinc-100 font-medium dark:bg-zinc-800" : "text-zinc-500"}`}>
              {f === "all" ? `All (${rows.length})` : `Not correct (${rows.filter((r) => r.verdict !== "correct").length})`}
            </button>
          ))}
        </div>
        {systems.length > 1 && (
          <select value={system} onChange={(e) => setSystem(e.target.value)} className={input} aria-label="System">
            <option value="all">All systems</option>
            {systems.map((s) => <option key={s} value={s}>{systemLabel(s)}</option>)}
          </select>
        )}
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-2 font-medium">Question</th>
              {systems.length > 1 && <th className="hidden px-3 py-2 font-medium sm:table-cell">System</th>}
              <th className="px-3 py-2 font-medium">Verdict</th>
              <th className="hidden px-3 py-2 text-right font-medium md:table-cell">Evidence</th>
              <th className="hidden px-4 py-2 text-right font-medium md:table-cell">Cost</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const key = `${r.id}|${r.system}|${r.run}`;
              const v = VERDICT[r.verdict];
              const isOpen = open === key;
              return [
                <tr key={key} onClick={() => setOpen(isOpen ? null : key)}
                  className="cursor-pointer border-t border-zinc-100 align-top hover:bg-zinc-50 dark:border-zinc-900 dark:hover:bg-zinc-900">
                  <td className="px-4 py-2">
                    <button className="flex gap-1.5 text-left" aria-expanded={isOpen}>
                      <span className={`mt-0.5 shrink-0 text-zinc-400 transition ${isOpen ? "rotate-90" : ""}`}>{Icon.chevron("h-3.5 w-3.5")}</span>
                      <span>
                        <span className="text-xs text-zinc-400">{r.id} · {r.category}{summary.repeats! > 1 ? ` · run ${r.run}` : ""}</span><br />
                        {r.question}
                      </span>
                    </button>
                  </td>
                  {systems.length > 1 && <td className="hidden whitespace-nowrap px-3 py-2 sm:table-cell">{systemLabel(r.system)}</td>}
                  <td className="px-3 py-2">
                    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${v.cls}`}>{v.icon("h-3 w-3")}{v.label}</span>
                  </td>
                  <td className="hidden px-3 py-2 text-right tabular-nums md:table-cell">{pct(r.evidence_recall)}</td>
                  <td className="hidden px-4 py-2 text-right tabular-nums md:table-cell">{usd(r.cost_usd)}</td>
                </tr>,
                isOpen && (
                  <tr key={`${key}-detail`} className="bg-zinc-50/60 dark:bg-zinc-900/40">
                    <td colSpan={5} className="px-4 pb-4 pt-1">
                      <div className="ml-5 space-y-3">
                        <div>
                          <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">Judge's reasoning</div>
                          <p className="mt-1 text-sm">{r.judge_reasoning}</p>
                        </div>
                        <div>
                          <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                            Answer <span className="font-normal normal-case">· {r.api_calls} API calls, {r.tool_calls} tool calls, {r.seconds}s{r.stop !== "end_turn" ? `, stopped: ${r.stop}` : ""}</span>
                          </div>
                          <div className="prose-answer mt-1 text-sm"><ReactMarkdown remarkPlugins={[remarkGfm]}>{r.answer}</ReactMarkdown></div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ),
              ];
            })}
          </tbody>
        </table>
        {shown.length === 0 && <p className="px-4 py-6 text-center text-sm text-zinc-500">Nothing to show.</p>}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- feedback

function FeedbackPanel({ onExported }: { onExported: () => void }) {
  const [items, setItems] = useState<FeedbackRecord[] | null>(null);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.feedbackList().then(setItems).catch((e) => setError(e.message)); }, []);

  async function exportSet() {
    setError(null);
    try {
      setResult(await api.exportFeedback());
      onExported();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const up = items?.filter((f) => f.rating === "up").length ?? 0;
  const down = items?.filter((f) => f.rating === "down").length ?? 0;
  return (
    <section className={`${card} p-4`} aria-labelledby="fb-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="fb-heading" className="text-sm font-semibold">Feedback from chat</h2>
          <p className="mt-0.5 text-sm text-zinc-500">
            {items == null ? "Loading…" : items.length === 0
              ? "Rate answers in the chat with 👍/👎 and they'll collect here."
              : `${up} 👍 · ${down} 👎. Export turns them into the eval set "feedback": 👍 answers become the reference, 👎 need a correction.`}
          </p>
        </div>
        <button className={btnSecondary} disabled={!items?.length} onClick={exportSet}>{Icon.upload()} Export as eval set</button>
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
      {result && (
        <div className="mt-3 text-sm">
          <p className="text-emerald-700 dark:text-emerald-400">Wrote {result.questions} question{result.questions === 1 ? "" : "s"} to {result.path}. Pick "feedback" above to run it.</p>
          {result.skipped.length > 0 && (
            <ul className="mt-1 list-disc pl-5 text-zinc-500">
              {result.skipped.map((s) => <li key={s.id}>Skipped "{s.question}": {s.reason}</li>)}
            </ul>
          )}
        </div>
      )}
      {!!items?.length && (
        <ul className="mt-3 divide-y divide-zinc-100 text-sm dark:divide-zinc-900">
          {items.slice(0, 8).map((f) => (
            <li key={f.turn_id} className="flex gap-2 py-2">
              <span className={f.rating === "up" ? "text-emerald-600" : "text-red-600"}>{f.rating === "up" ? Icon.thumbUp() : Icon.thumbDown()}</span>
              <div className="min-w-0 flex-1">
                <div className="truncate">{f.question}</div>
                {f.comment && <div className="truncate text-zinc-500">Correction: {f.comment}</div>}
              </div>
              <span className="shrink-0 text-xs text-zinc-400">{date(f.updated_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
