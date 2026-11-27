import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { api } from "./api";
import { Icon } from "./components";
import type { DocChunks, DocStatus, DocumentRow, DocumentsResponse, DriveFolders, DriveStatus, DriveSyncReport, Health, IngestState, SearchResponse } from "./types";

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
  empty: { label: "No text found", cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300", hint: "Indexed, but no text could be extracted (most likely a scanned PDF), so it isn't searchable" },
  error: { label: "Couldn't read", cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300", hint: "The last re-index couldn't read this file, so it isn't searchable" },
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
  const [confirm, setConfirm] = useState<DocumentRow | null>(null);
  const [replacing, setReplacing] = useState<File[] | null>(null); // uploads whose names already exist

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

  async function upload(files: File[], overwrite = false) {
    setReplacing(null);
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const r = await api.upload(files, overwrite);
      setRejected((prev) => (overwrite ? [...prev, ...r.rejected] : r.rejected));
      if (r.conflicts.length) setReplacing(files.filter((f) => r.conflicts.includes(f.name)));
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

  async function syncDrive() {
    setError(null);
    try {
      setIngest(await api.syncDrive());
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

  async function transcribe(row: DocumentRow) {
    setConfirm(null);
    setError(null);
    try {
      setIngest(await api.transcribe(row.name));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function untranscribe(name: string) {
    setError(null);
    try {
      setIngest(await api.clearTranscription(name));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const running = ingest.state === "running";
  const docs = data?.documents ?? [];
  const viewingRow = viewing ? docs.find((d) => d.name === viewing.name) : undefined;

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
            <div className="flex flex-wrap gap-2">
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
          {!running && <EmptyFilesNotice docs={docs} ocr={!!data?.ocr} onReindex={() => reindex(false)} onTranscribe={setConfirm} />}
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
              <DrivePanel running={running} lastJob={ingest.finished_at} onSync={syncDrive} onChanged={refresh} />
              <DropZone onFiles={(files) => upload(files)} busy={uploading} />
              {rejected.length > 0 && (
                <Alert tone="error">
                  Not uploaded: {rejected.map((r) => `${r.name} (${r.reason})`).join("; ")}
                </Alert>
              )}
              <FileTable docs={docs} onView={(name) => setViewing({ name })} onDelete={remove} onTranscribe={setConfirm} busy={running} compact={!!viewing} />
            </div>
          ) : (
            <SearchTester reranker={!!health?.reranker} onOpen={(name, focus) => setViewing({ name, focus })} />
          )}
        </div>
      </div>

      {confirm && <ConfirmTranscribe row={confirm} onCancel={() => setConfirm(null)} onConfirm={() => transcribe(confirm)} />}
      {replacing && <ConfirmReplace files={replacing} onCancel={() => setReplacing(null)} onConfirm={() => upload(replacing, true)} />}

      {viewing && (
        <>
          <div className="fixed inset-0 z-20 bg-black/30 lg:hidden" onClick={() => setViewing(null)} />
          <aside className="fixed inset-y-0 right-0 z-30 w-full max-w-md border-l border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950 lg:static lg:w-[28rem]">
            <DocumentPanel key={`${viewing.name}:${viewingRow?.chunks}:${viewingRow?.transcribed}:${viewingRow?.modified}`} name={viewing.name} focus={viewing.focus}
              row={viewingRow} busy={running} onTranscribe={setConfirm} onUntranscribe={untranscribe}
              onClose={() => setViewing(null)} />
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
    const label = ingest.stage === "transcribing"
      ? `Claude is transcribing ${ingest.file ?? "the PDF"} - this can take a minute for long files`
      : ingest.stage === "listing" ? "Listing the Google Drive folder"
      : ingest.stage === "downloading" ? `Downloading ${Math.min((ingest.done ?? 0) + 1, total)} of ${total} files from Google Drive`
      : ingest.stage === "embedding"
        ? `Embedding ${ingest.done ?? 0} of ${total} chunks`
        : ingest.stage === "reading" ? `Reading ${ingest.done ?? 0} of ${total} files` : "Starting…";
    return (
      <div className="mt-4 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2.5 dark:border-indigo-900 dark:bg-indigo-950/40" role="status">
        <div className="flex items-center gap-2 text-sm text-indigo-800 dark:text-indigo-300">{Icon.spinner()} Indexing - {label}</div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-indigo-100 dark:bg-indigo-900">
          <div className="h-full rounded-full bg-indigo-600 transition-all" style={{ width: `${ingest.stage === "embedding" || ingest.stage === "downloading" ? pct : 5}%` }} />
        </div>
      </div>
    );
  }
  if (ingest.state === "error") {
    return <Alert tone="error">{ingest.reason === "gdrive" ? "Google Drive sync" : "Indexing"} failed: {ingest.error}</Alert>;
  }
  if (ingest.state === "done" && ingest.report && ingest.finished_at && Date.now() / 1000 - ingest.finished_at < 600) {
    return <>{ingest.sync && <DriveSyncSummary sync={ingest.sync} />}<IndexSummary report={ingest.report} /></>;
  }
  return null;
}

function DriveSyncSummary({ sync }: { sync: DriveSyncReport }) {
  const parts = [sync.downloaded.length && `${sync.downloaded.length} downloaded`, sync.removed.length && `${sync.removed.length} removed`,
    `${sync.unchanged} unchanged`].filter(Boolean);
  const problems = [...Object.entries(sync.failed).map(([n, r]) => [n, `failed: ${r}`]),
    ...Object.entries(sync.skipped).map(([n, r]) => [n, `skipped: ${r}`])];
  return (
    <Alert tone={Object.keys(sync.failed).length ? "error" : "info"}>
      Google Drive: {parts.join(", ")}.
      {problems.length > 0 && (
        <ul className="mt-1 list-disc pl-5">
          {problems.map(([name, reason]) => <li key={name}><span className="font-medium">{name}</span> {reason}</li>)}
        </ul>
      )}
    </Alert>
  );
}

function IndexSummary({ report }: { report: NonNullable<IngestState["report"]> }) {
  const r = report;
  const parts = [r.added.length && `${r.added.length} added`, r.updated.length && `${r.updated.length} updated`,
    r.removed.length && `${r.removed.length} removed`, r.empty?.length && `${r.empty.length} with no text`].filter(Boolean);
  const failed = Object.entries(r.failed ?? {});
  if (failed.length) {
    return (
      <Alert tone="error">
        {failed.length === 1 ? "1 file was" : `${failed.length} files were`} skipped; the rest is indexed{parts.length ? ` (${parts.join(", ")})` : ""}. A skipped file that was indexed before stays searchable in its previous version.
        <ul className="mt-1 list-disc pl-5">
          {failed.map(([name, reason]) => <li key={name}><span className="font-medium">{name}</span>: {reason}</li>)}
        </ul>
      </Alert>
    );
  }
  return (
    <div className="mt-4 flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300" role="status">
      {Icon.check()} Index up to date{parts.length ? `: ${parts.join(", ")}` : " - nothing changed"}. {r.chunks} chunks{r.embedded ? ` (${r.embedded} newly embedded)` : ""}.
    </div>
  );
}

function EmptyFilesNotice({ docs, ocr, onReindex, onTranscribe }: {
  docs: DocumentsResponse["documents"];
  ocr: boolean;
  onReindex: () => void;
  onTranscribe: (row: DocumentRow) => void;
}) {
  const empty = docs.filter((d) => d.status === "empty");
  if (!empty.length) return null;
  const names = empty.map((d) => d.name).join(", ");
  const pdfs = empty.some((d) => d.type === "pdf");
  return (
    <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300" role="status">
      <p>
        <span className="font-medium">No searchable text in {empty.length === 1 ? "1 file" : `${empty.length} files`}:</span> {names}.
        {pdfs && (empty.length === 1
          ? " This is most likely a scanned PDF: images of pages with no text layer."
          : " These are most likely scanned PDFs: images of pages with no text layer.")}
      </p>
      {pdfs && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {empty.filter((d) => d.type === "pdf").map((d) => (
            <button key={d.name} onClick={() => onTranscribe(d)}
              className="inline-flex max-w-full items-center gap-1.5 rounded-md bg-red-700 px-2.5 py-1 text-xs font-medium text-white hover:bg-red-800 dark:bg-red-800 dark:hover:bg-red-700">
              {Icon.scan("h-3.5 w-3.5 shrink-0")} <span className="truncate">Transcribe {empty.length > 1 ? d.name : "with Claude"}</span>
            </button>
          ))}
          {ocr && <span className="text-xs">or <button onClick={onReindex} className="font-medium underline">re-index</button> - automatic transcription is on.</span>}
        </div>
      )}
    </div>
  );
}

const input = "w-full rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-sm outline-none focus:border-indigo-500 dark:border-zinc-800 dark:bg-zinc-950";
const link = "font-medium text-indigo-700 underline decoration-indigo-300 underline-offset-2 hover:text-indigo-900 dark:text-indigo-400 dark:decoration-indigo-700";

/** Connect a Google Drive folder: one-time OAuth client setup, "Connect Google Drive", choose a folder, sync. */
/** onChanged: the connection changed (folder unselected, disconnected, ...), which changes which files are Drive's. */
function DrivePanel({ running, lastJob, onSync, onChanged }: {
  running: boolean; lastJob?: number; onSync: () => void; onChanged: () => void;
}) {
  const [st, setSt] = useState<DriveStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [setup, setSetup] = useState(false); // OAuth client form open
  const [editFolder, setEditFolder] = useState(false);

  // Reload after each job (a sync updates "last synced") and when the user comes back from Google's sign-in.
  useEffect(() => { api.drive().then(setSt).catch((e) => setError((e as Error).message)); }, [lastJob]);

  async function act(fn: () => Promise<DriveStatus | void>) {
    setBusy(true);
    setError(null);
    try {
      const next = await fn();
      if (next) setSt(next);
      onChanged();
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function connect() {
    setBusy(true);
    setError(null);
    try {
      window.location.href = (await api.connectDrive()).url; // Google's sign-in; it sends the browser back here
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  if (!st) return error ? <Alert tone="error">{error}</Alert> : null;
  const synced = st.last_sync ? `Last synced ${when(st.last_sync.finished_at)}` : "Not synced yet";
  const ready = st.mode === "env" || (st.account && st.folder);

  return (
    <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800" aria-label="Google Drive">
      <div className="flex flex-wrap items-start gap-3">
        <span className="mt-0.5 rounded-lg bg-zinc-100 p-2 text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300">{Icon.cloud()}</span>
        <div className="min-w-[12rem] flex-1">
          <h3 className="text-sm font-semibold">Google Drive</h3>
          <p className="mt-0.5 text-sm text-zinc-500">
            {st.mode === "env" ? <>Folder set by the server · synced into <code>{st.dir}/</code> · {synced}</>
              : st.account ? <>Connected as <span className="font-medium text-zinc-700 dark:text-zinc-300">{st.account}</span>
                {st.folder ? <> · Folder <span className="font-medium text-zinc-700 dark:text-zinc-300">{st.folder_name ?? st.folder}</span> · {synced}</>
                  : <> · No folder selected</>}</>
              : "Sync Google Docs, Slides, PDFs and text files from a Drive folder. Read-only."}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {ready && (
            <button className={btnPrimary} onClick={onSync} disabled={running || busy}
              title={`Download new and changed files into ${st.dir}/, then index them`}>
              {running ? Icon.spinner() : Icon.refresh()} Sync now
            </button>
          )}
          {st.mode === "oauth" && st.account && !editFolder && (
            <button className={st.folder ? btnSecondary : btnPrimary} onClick={() => setEditFolder(true)} disabled={busy}>
              {st.folder ? "Change folder" : "Choose folder"}
            </button>
          )}
          {st.mode === "oauth" && st.account && st.folder && (
            <button className={btnSecondary} disabled={busy || running}
              onClick={() => window.confirm(`Stop syncing “${st.folder_name ?? st.folder}”? Files already synced stay in your documents (and can then be deleted there); choosing another folder later replaces them.`)
                && act(api.unselectDriveFolder).then((ok) => ok && setEditFolder(false))}>
              Unselect folder
            </button>
          )}
          {st.mode === "oauth" && st.account && (
            <button className={btnSecondary} disabled={busy || running}
              onClick={() => window.confirm("Disconnect Google Drive? Files already synced stay in your documents.")
                && act(api.disconnectDrive).then(() => setEditFolder(false))}>
              Disconnect
            </button>
          )}
          {st.mode === "oauth" && !st.account && st.client.configured && (
            <button className={btnPrimary} onClick={connect} disabled={busy}>
              {busy ? Icon.spinner() : Icon.cloud()} Connect Google Drive
            </button>
          )}
          {st.mode === "oauth" && !st.client.configured && !setup && (
            <button className={btnPrimary} onClick={() => setSetup(true)}>Set up Google sign-in</button>
          )}
        </div>
      </div>

      {st.error && (
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300" role="alert">
          <span className="flex-1">Couldn't connect Google Drive: {st.error}</span>
          <button aria-label="Dismiss" onClick={() => act(api.dismissDriveError)}>{Icon.x()}</button>
        </div>
      )}
      {error && <Alert tone="error">{error}</Alert>}

      {st.mode === "oauth" && !st.account && st.client.configured && (
        <p className="mt-3 text-xs text-zinc-500">
          Google may say it hasn't verified this app; it's your own OAuth client, so choose <b>Continue</b>, and tick
          the Google Drive permission.{" "}
          {!st.client.from_env && <button className={link} onClick={() => setSetup((v) => !v)}>Change OAuth client</button>}
        </p>
      )}

      {st.mode === "oauth" && setup && !st.client.from_env && (!st.account || !st.client.configured) && (
        <ClientSetup status={st} busy={busy} onCancel={st.client.configured ? () => setSetup(false) : undefined}
          onSave={(id, secret) => act(() => api.setDriveClient(id, secret)).then((ok) => ok && setSetup(false))} />
      )}

      {st.mode === "oauth" && st.account && editFolder && (
        <FolderPicker current={st.folder} busy={busy || running} onCancel={() => setEditFolder(false)}
          onSave={(folder) => {
            const switching = st.folder && folder !== st.folder && st.last_sync;
            if (switching && !window.confirm(`Sync this folder instead of “${st.folder_name ?? st.folder}”? Files synced from that folder will be removed from your documents.`)) return;
            act(() => api.setDriveFolder(folder)).then((ok) => {
              if (!ok) return;
              setEditFolder(false);
              onSync(); // picking a folder means "sync this": start right away
            });
          }} />
      )}
    </section>
  );
}

function ClientSetup({ status, busy, onSave, onCancel }: {
  status: DriveStatus; busy: boolean; onSave: (id: string, secret: string) => void; onCancel?: () => void;
}) {
  const [id, setId] = useState("");
  const [secret, setSecret] = useState("");
  const [copied, setCopied] = useState(false);
  const step = "pl-1";
  return (
    <form className="mt-4 space-y-3 border-t border-zinc-200 pt-4 text-sm dark:border-zinc-800"
      onSubmit={(e) => { e.preventDefault(); onSave(id, secret); }}>
      <p className="font-medium">One-time setup: create a Google OAuth client for this app</p>
      <ol className="list-decimal space-y-1.5 pl-5 text-zinc-600 dark:text-zinc-400">
        <li className={step}>In Google Cloud, <a className={link} href="https://console.cloud.google.com/apis/library/drive.googleapis.com" target="_blank" rel="noreferrer">enable the Google Drive API</a> for a project.</li>
        <li className={step}><a className={link} href="https://console.cloud.google.com/auth/overview" target="_blank" rel="noreferrer">Set up Google Auth Platform</a>: audience <b>External</b>, and add the Google accounts that will connect as <b>test users</b>. While the app is in <b>Testing</b>, Google ends the sign-in after 7 days; <b>Publish app</b> (under Audience) to stay connected.</li>
        <li className={step}>
          <a className={link} href="https://console.cloud.google.com/auth/clients/create" target="_blank" rel="noreferrer">Create a client</a> of type <b>Web application</b> with this authorized redirect URI:
          <div className="mt-1.5 flex items-center gap-2">
            <code className="min-w-0 flex-1 truncate rounded-md bg-zinc-100 px-2 py-1 text-xs text-zinc-800 dark:bg-zinc-900 dark:text-zinc-200">{status.redirect_uri}</code>
            <button type="button" className={btnSecondary}
              onClick={() => navigator.clipboard?.writeText(status.redirect_uri).then(() => setCopied(true))}>
              {copied ? Icon.check() : null} {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </li>
        <li className={step}>Paste the client's ID and secret:</li>
      </ol>
      <div className="grid gap-2 sm:grid-cols-2">
        <input className={input} placeholder="Client ID (…apps.googleusercontent.com)" value={id} onChange={(e) => setId(e.target.value)}
          autoComplete="off" spellCheck={false} required aria-label="OAuth client ID" />
        <input className={input} placeholder="Client secret" type="password" value={secret} onChange={(e) => setSecret(e.target.value)}
          autoComplete="off" required aria-label="OAuth client secret" />
      </div>
      <div className="flex gap-2">
        <button className={btnPrimary} type="submit" disabled={busy || !id.trim() || !secret.trim()}>Save</button>
        {onCancel && <button className={btnSecondary} type="button" onClick={onCancel}>Cancel</button>}
      </div>
      <p className="text-xs text-zinc-500">The secret is stored on this server only and is never shown again.</p>
    </form>
  );
}

type Crumb = { id: string; name: string; q?: string }; // id: a folder id, a location ("root", "shared", "drives") or "search"
const LOCATIONS: Crumb[] = [{ id: "root", name: "My Drive" }, { id: "shared", name: "Shared with me" }, { id: "drives", name: "Shared drives" }];

/** Browse the signed-in account's Drive and choose the folder to sync. */
function FolderPicker({ current, busy, onSave, onCancel }: {
  current: string | null; busy: boolean; onSave: (folderId: string) => void; onCancel?: () => void;
}) {
  const [trail, setTrail] = useState<Crumb[]>([LOCATIONS[0]]);
  const [search, setSearch] = useState("");
  const [res, setRes] = useState<DriveFolders | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const here = trail[trail.length - 1];
  const inFolder = !LOCATIONS.some((l) => l.id === here.id) && here.id !== "search";

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    setRes(null); // don't show the previous folder's list under the new path
    api.driveFolders(here.id === "search" ? { q: here.q } : { parent: here.id })
      .then((r) => live && setRes(r))
      .catch((e) => live && (setRes(null), setError((e as Error).message)))
      .finally(() => live && setLoading(false));
    return () => { live = false; };
  }, [here.id, here.q]);

  // Searching replaces the trail; clearing the box goes back to My Drive.
  useEffect(() => {
    const q = search.trim();
    if (!q) {
      setTrail((t) => (t[0].id === "search" ? [LOCATIONS[0]] : t));
      return;
    }
    const timer = setTimeout(() => setTrail([{ id: "search", name: `Folders named “${q}”`, q }]), 300);
    return () => clearTimeout(timer);
  }, [search]);

  const open = (f: Crumb) => setTrail((t) => [...t, f]);
  const tab = (l: Crumb) => `rounded-full px-3 py-1 text-xs font-medium ${trail[0].id === l.id
    ? "bg-indigo-600 text-white" : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"}`;

  return (
    <div className="mt-4 border-t border-zinc-200 pt-4 dark:border-zinc-800">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="mr-auto text-sm font-medium">Choose the folder to sync</h4>
        {onCancel && <button className={btnSecondary} type="button" onClick={onCancel}>Cancel</button>}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {LOCATIONS.map((l) => (
          <button key={l.id} type="button" className={tab(l)} onClick={() => { setSearch(""); setTrail([l]); }}>{l.name}</button>
        ))}
        <div className="relative min-w-[10rem] flex-1">
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-400">{Icon.search("h-3.5 w-3.5")}</span>
          <input className={`${input} pl-8`} type="search" placeholder="Search folders by name" value={search}
            onChange={(e) => setSearch(e.target.value)} aria-label="Search Google Drive folders" />
        </div>
      </div>

      <nav className="mt-3 flex flex-wrap items-center gap-1 text-sm" aria-label="Folder path">
        {trail.map((c, i) => (
          <span key={`${c.id}:${i}`} className="flex items-center gap-1">
            {i > 0 && <span className="text-zinc-400">{Icon.chevron("h-3.5 w-3.5")}</span>}
            {i < trail.length - 1
              ? <button type="button" className="text-zinc-500 hover:text-zinc-900 hover:underline dark:hover:text-zinc-100" onClick={() => setTrail(trail.slice(0, i + 1))}>{c.name}</button>
              : <span className="font-medium">{c.name}</span>}
          </span>
        ))}
        {loading && <span className="ml-1 text-zinc-400">{Icon.spinner("h-3.5 w-3.5")}</span>}
      </nav>

      {error ? <Alert tone="error">{error}</Alert> : (
        <ul className="mt-2 max-h-72 divide-y divide-zinc-100 overflow-y-auto rounded-lg border border-zinc-200 dark:divide-zinc-900 dark:border-zinc-800" aria-busy={loading}>
          {res?.folders.map((f) => (
            <li key={f.id} className="group flex items-center gap-2 px-2 py-1.5 hover:bg-zinc-50 dark:hover:bg-zinc-900/60">
              <button type="button" className="flex min-w-0 flex-1 items-center gap-2 rounded px-1 py-0.5 text-left text-sm"
                onClick={() => { setSearch(""); open(f); }} title={`Open ${f.name}`}>
                <span className="text-zinc-400">{Icon.folder("h-4 w-4")}</span>
                <span className="truncate">{f.name}</span>
                {f.id === current && <span className="shrink-0 text-xs font-medium text-emerald-700 dark:text-emerald-400">(syncing now)</span>}
              </button>
              <button type="button" className={`${btnSecondary} !px-2 !py-1 text-xs sm:opacity-0 sm:group-hover:opacity-100 sm:focus:opacity-100`}
                disabled={busy} onClick={() => onSave(f.id)} aria-label={`Sync ${f.name}`}>Sync</button>
            </li>
          ))}
          {res && !res.folders.length && !loading && (
            <li className="px-3 py-6 text-center text-sm text-zinc-500">
              {here.id === "search" ? "No folders with that name." : here.id === "drives" ? "You aren't a member of any shared drives."
                : inFolder ? "No subfolders here." : "No folders here."}
            </li>
          )}
          {!res && loading && <li className="px-3 py-6 text-center text-sm text-zinc-400">Loading…</li>}
        </ul>
      )}
      {res?.truncated && <p className="mt-1 text-xs text-zinc-500">Showing the first {res.folders.length}; search to find others.</p>}

      {inFolder && (
        <div className="mt-3 flex flex-wrap items-center gap-3 rounded-lg bg-zinc-50 px-3 py-2 dark:bg-zinc-900/60">
          <p className="min-w-0 flex-1 text-xs text-zinc-500">
            {res?.files ? `${res.files.total} file${res.files.total === 1 ? "" : "s"} directly in this folder, ${res.files.supported} of them can be synced (Docs, Slides, PDF, .md, .txt).` : "\u00a0"}
            {" "}Subfolders are included.
          </p>
          <button className={btnPrimary} type="button" disabled={busy} onClick={() => onSave(here.id)}>
            {busy && Icon.spinner()} Sync “{here.name}”
          </button>
        </div>
      )}
    </div>
  );
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

function FileTable({ docs, onView, onDelete, onTranscribe, busy, compact = false }: {
  docs: DocumentsResponse["documents"];
  onView: (name: string) => void;
  onDelete: (name: string) => void;
  onTranscribe: (row: DocumentRow) => void;
  busy: boolean;
  compact?: boolean; // side panel open: drop the size and date columns
}) {
  const wide = compact ? "hidden" : "hidden md:table-cell";
  const widest = compact ? "hidden" : "hidden xl:table-cell";
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
            <th className={`${wide} px-3 py-2 text-right font-medium`}>Size</th>
            <th className={`${widest} px-3 py-2 font-medium`}>Last indexed</th>
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
                    {d.gdrive && (
                      <span className="shrink-0 text-zinc-400" title="Synced from Google Drive">{Icon.cloud("h-3.5 w-3.5")}</span>
                    )}
                  </button>
                  {d.error && <p className="ml-6 mt-0.5 text-xs text-red-700 dark:text-red-300">{d.error}</p>}
                </td>
                <td className="px-3 py-2.5">
                  <span className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${st.cls}`} title={d.error ?? st.hint}>{st.label}</span>
                  {d.transcribed && (
                    <div className="mt-1 whitespace-nowrap text-[11px] text-zinc-500" title="Text written by Claude from the page images">Transcribed by Claude</div>
                  )}
                </td>
                <td className="hidden px-3 py-2.5 text-right tabular-nums text-zinc-600 dark:text-zinc-400 sm:table-cell">{d.chunks || "-"}</td>
                <td className={`${wide} whitespace-nowrap px-3 py-2.5 text-right tabular-nums text-zinc-600 dark:text-zinc-400`}>{size(d.size)}</td>
                <td className={`${widest} whitespace-nowrap px-3 py-2.5 text-zinc-600 dark:text-zinc-400`}
                  title={d.modified != null ? `File modified ${when(d.modified)}` : undefined}>{d.indexed_at != null ? when(d.indexed_at) : "Never"}</td>
                <td className="whitespace-nowrap px-3 py-2.5 text-right">
                  {d.status === "empty" && d.type === "pdf" && confirming !== d.name && (
                    <button onClick={() => onTranscribe(d)} disabled={busy}
                      className="mr-1 rounded-md border border-zinc-200 px-2 py-1 text-xs font-medium hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800">
                      Transcribe
                    </button>
                  )}
                  {d.gdrive ? (
                    <span className="inline-block rounded p-1.5 text-zinc-300 dark:text-zinc-600"
                      title="Synced from Google Drive: delete it in Drive, or unselect the folder first. Deleting it here would only bring it back on the next sync.">
                      {Icon.trash()}
                    </span>
                  ) : d.status !== "deleted" && (confirming === d.name ? (
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

function DocumentPanel({ name, focus, row, busy, onTranscribe, onUntranscribe, onClose }: {
  name: string;
  focus?: number;
  row?: DocumentRow;
  busy: boolean;
  onTranscribe: (row: DocumentRow) => void;
  onUntranscribe: (name: string) => void;
  onClose: () => void;
}) {
  const [doc, setDoc] = useState<DocChunks | null>(null);
  const [error, setError] = useState<string | null>(null);
  const focusRef = useRef<HTMLDivElement>(null);

  const noText = row !== undefined && row.chunks === 0;
  useEffect(() => {
    let live = true;
    setDoc(null);
    setError(null);
    if (noText) {
      setDoc({ name, doc_note: "", chunks: [] }); // nothing indexed to fetch
    } else {
      api.documentChunks(name).then((d) => live && setDoc(d)).catch((e) => live && setError(String(e.message ?? e)));
    }
    return () => { live = false; };
  }, [name, noText]);

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
          <div className="text-xs text-zinc-500">{doc ? `${doc.chunks.length} chunk${doc.chunks.length === 1 ? "" : "s"}` : " "}</div>
          {row?.type === "pdf" && (
            <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
              {row.transcribed ? (
                <>
                  <span className="rounded-full bg-violet-100 px-2 py-0.5 font-medium text-violet-800 dark:bg-violet-900/40 dark:text-violet-300">
                    Text transcribed by Claude
                  </span>
                  <button onClick={() => onUntranscribe(name)} disabled={busy}
                    className="text-zinc-600 underline hover:text-zinc-900 disabled:opacity-40 dark:text-zinc-400 dark:hover:text-zinc-100">
                    Use the PDF's own text instead
                  </button>
                </>
              ) : (
                <>
                  <span className="text-zinc-500">Text looks wrong or incomplete?</span>
                  <button onClick={() => onTranscribe(row)} disabled={busy}
                    className="inline-flex items-center gap-1 font-medium text-indigo-700 hover:underline disabled:opacity-40 dark:text-indigo-400">
                    {Icon.scan("h-3.5 w-3.5")} Transcribe with Claude
                  </button>
                </>
              )}
            </div>
          )}
        </div>
        <button onClick={onClose} aria-label="Close document"
          className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800">{Icon.x()}</button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {!doc && !error && <p className="text-sm text-zinc-500">Loading…</p>}
        {doc && !doc.chunks.length && (
          <p className="text-sm text-zinc-500">
            No text could be extracted from this file, so it isn't searchable.
            {row?.type === "pdf" && " It's most likely a scanned PDF - use Transcribe with Claude above."}
          </p>
        )}
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

// ---------------------------------------------------------------- transcription consent

function ConfirmTranscribe({ row, onCancel, onConfirm }: { row: DocumentRow; onCancel: () => void; onConfirm: () => void }) {
  const confirmRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    confirmRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const cost = row.transcribe_cost_usd ?? 0;
  const costText = cost < 0.01 ? "less than $0.01" : `about $${cost.toFixed(2)}`;
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4" onClick={onCancel}>
      <div role="dialog" aria-modal="true" aria-labelledby="transcribe-title" onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
        <h3 id="transcribe-title" className="text-base font-semibold">Send this file to Claude?</h3>
        <p className="mt-2 break-words text-sm text-zinc-600 dark:text-zinc-400">
          <span className="font-medium text-zinc-900 dark:text-zinc-100">{row.name}</span>
          {row.pages ? ` (${row.pages} page${row.pages === 1 ? "" : "s"})` : ""} will be sent to the Anthropic API, where
          Claude Haiku reads the page images and writes out the text. The transcription then replaces the PDF's own
          text for search.
        </p>
        <ul className="mt-3 space-y-1 text-sm text-zinc-600 dark:text-zinc-400">
          <li>• The whole file is sent, including any personal details in it.</li>
          <li>• Estimated cost: {costText}.</li>
          <li>• It's saved, so this file is only sent once. You can switch back to the PDF's own text later.</li>
        </ul>
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onCancel} className={btnSecondary}>Cancel</button>
          <button ref={confirmRef} onClick={onConfirm} className={btnPrimary}>{Icon.scan()} Transcribe</button>
        </div>
      </div>
    </div>
  );
}

function ConfirmReplace({ files, onCancel, onConfirm }: { files: File[]; onCancel: () => void; onConfirm: () => void }) {
  const confirmRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    confirmRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const names = files.map((f) => f.name).join(", ");
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4" onClick={onCancel}>
      <div role="dialog" aria-modal="true" aria-labelledby="replace-title" onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
        <h3 id="replace-title" className="break-words text-base font-semibold">Replace {names}?</h3>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          {files.length === 1 ? "A document with this name already exists." : "Documents with these names already exist."}{" "}
          Replacing overwrites the current file{files.length === 1 ? "" : "s"} and re-indexes the new version.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onCancel} className={btnSecondary}>Keep existing</button>
          <button ref={confirmRef} onClick={onConfirm} className={btnPrimary}>Replace</button>
        </div>
      </div>
    </div>
  );
}
