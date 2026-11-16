export type Step = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  chars?: number;
};

export type Citation = { source: string; index: number; section: string | null; found: boolean };

export type Usage = {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  api_calls: number;
};

export type Turn = {
  role: "user" | "assistant";
  text: string;
  steps: Step[];
  citations: Citation[];
  usage: Usage | null;
  cost_usd: number | null;
  seconds: number | null;
  stop: string | null;
  streaming?: boolean;
};

export type ConversationSummary = {
  id: string;
  title: string;
  updated_at: number;
  turns: number;
  cost_usd: number;
  model: string;
};

export type Health = {
  documents: number;
  chunks: number;
  default_model: string;
  models: string[];
  demo_mode: boolean;
  reranker: boolean;
};

export type ChunkView = {
  source: string;
  index: number;
  total_chunks: number;
  chunks: { index: number; section: string; text: string; cited: boolean }[];
};

export type ChatEvent =
  | ({ type: "conversation" } & ConversationSummary)
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; id: string; name: string; is_error: boolean; chars: number }
  | { type: "answer"; text: string; stop: string; citations: Citation[] }
  | { type: "usage"; usage: Usage; cost_usd: number | null }
  | { type: "error"; message: string }
  | { type: "done"; seconds: number; conversation: ConversationSummary };

export type DocStatus = "indexed" | "changed" | "new" | "deleted";

export type DocumentRow = {
  name: string;
  type: string;
  size: number | null;
  modified: number | null;
  chunks: number;
  status: DocStatus;
};

export type IngestState = {
  state: "idle" | "running" | "done" | "error";
  stage?: string;
  done?: number;
  total?: number;
  reason?: string;
  started_at?: number;
  finished_at?: number;
  error?: string;
  report?: { added: string[]; updated: string[]; removed: string[]; unchanged: number; chunks: number };
};

export type DocumentsResponse = {
  documents: DocumentRow[];
  chunks: number;
  ingest: IngestState;
  needs_ingest: boolean;
};

export type UploadResponse = {
  saved: string[];
  rejected: { name: string; reason: string }[];
  ingest: IngestState | null;
};

export type SearchResult = {
  rank: number;
  score: number;
  source: string;
  index: number;
  section: string;
  doc_note: string;
  text: string;
};

export type SearchResponse = { query: string; mode: string; reranked: boolean; ms: number; results: SearchResult[] };

export type DocChunks = { name: string; doc_note: string; chunks: { index: number; section: string; text: string }[] };
