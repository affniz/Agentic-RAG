import type {
  ChatEvent, ChunkView, ConversationSummary, DocChunks, DocumentsResponse, Estimate, EvalJob, EvalRequest, EvalRow,
  EvalRun, EvalSet, ExportResult, Feedback, FeedbackRecord, Health, IngestState, SearchResponse, Settings, Turn,
  UploadResponse,
} from "./types";

const send = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

const path = (name: string) => name.split("/").map(encodeURIComponent).join("/");

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
  return res.json();
}

export const api = {
  health: () => fetch("/api/health").then(json<Health>),
  conversations: () => fetch("/api/conversations").then(json<ConversationSummary[]>),
  conversation: (id: string) =>
    fetch(`/api/conversations/${id}`).then(json<ConversationSummary & { messages: Turn[] }>),
  deleteConversation: (id: string) => fetch(`/api/conversations/${id}`, { method: "DELETE" }).then(json),
  resetConversations: () => fetch("/api/conversations", { method: "DELETE" }).then(json<{ kept_busy: number }>),
  settings: () => fetch("/api/settings").then(json<Settings>),
  saveSettings: (s: Settings) => fetch("/api/settings", send("PUT", s)).then(json<Settings>),
  feedback: (turnId: string, conversationId: string, rating: Feedback["rating"] | null, comment = "") =>
    fetch(`/api/feedback/${turnId}`, send("PUT", { conversation_id: conversationId, rating, comment }))
      .then(json<{ feedback: Feedback | null }>),
  feedbackList: () => fetch("/api/feedback").then(json<FeedbackRecord[]>),
  exportFeedback: () => fetch("/api/feedback/export", send("POST")).then(json<ExportResult>),

  evalSets: () => fetch("/api/evals/sets").then(json<EvalSet[]>),
  evalRuns: () => fetch("/api/evals/runs").then(json<EvalRun[]>),
  evalRun: (file: string) => fetch(`/api/evals/runs/${encodeURIComponent(file)}`).then(json<{ summary: EvalRun; rows: EvalRow[] }>),
  estimate: (r: EvalRequest) => fetch("/api/evals/estimate", send("POST", r)).then(json<Estimate>),
  startEval: (r: EvalRequest) => fetch("/api/evals/run", send("POST", r)).then(json<EvalJob>),
  evalJob: () => fetch("/api/evals/job").then(json<EvalJob>),
  cancelEval: () => fetch("/api/evals/job/cancel", send("POST")).then(json<EvalJob>),
  chunk: (source: string, index: number) => fetch(`/api/chunks/${path(source)}/${index}`).then(json<ChunkView>),

  documents: () => fetch("/api/documents").then(json<DocumentsResponse>),
  documentChunks: (name: string) => fetch(`/api/documents/${path(name)}/chunks`).then(json<DocChunks>),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return fetch("/api/documents", { method: "POST", body: form }).then(json<UploadResponse>);
  },
  deleteDocument: (name: string) => fetch(`/api/documents/${path(name)}`, { method: "DELETE" }).then(json),
  transcribe: (name: string) => fetch(`/api/documents/${path(name)}/transcribe`, { method: "POST" }).then(json<IngestState>),
  clearTranscription: (name: string) =>
    fetch(`/api/documents/${path(name)}/transcription`, { method: "DELETE" }).then(json<IngestState>),
  ingest: (rebuild = false) =>
    fetch("/api/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rebuild }) })
      .then(json<IngestState>),
  ingestStatus: () => fetch("/api/ingest").then(json<IngestState>),
  search: (p: { q: string; k: number; mode: string; rerank: boolean }) =>
    fetch(`/api/search?${new URLSearchParams({ q: p.q, k: String(p.k), mode: p.mode, rerank: String(p.rerank) })}`)
      .then(json<SearchResponse>),
};

/** POST a question and yield Server-Sent Events as they arrive. */
export async function* streamChat(
  body: { message: string; conversation_id?: string | null },
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = (await res.json().catch(() => null))?.detail;
    throw new Error(detail ?? `HTTP ${res.status}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data: "))
        .map((l) => l.slice(6))
        .join("\n");
      if (data) yield JSON.parse(data) as ChatEvent;
    }
  }
}
