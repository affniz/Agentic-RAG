import type {
  ChatEvent, ChunkView, ConversationSummary, DocChunks, DocumentsResponse, Health, IngestState,
  SearchResponse, Turn, UploadResponse,
} from "./types";

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
  chunk: (source: string, index: number) => fetch(`/api/chunks/${path(source)}/${index}`).then(json<ChunkView>),

  documents: () => fetch("/api/documents").then(json<DocumentsResponse>),
  documentChunks: (name: string) => fetch(`/api/documents/${path(name)}/chunks`).then(json<DocChunks>),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return fetch("/api/documents", { method: "POST", body: form }).then(json<UploadResponse>);
  },
  deleteDocument: (name: string) => fetch(`/api/documents/${path(name)}`, { method: "DELETE" }).then(json),
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
  body: { message: string; conversation_id?: string | null; model?: string; effort?: string },
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
