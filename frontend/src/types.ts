export type Step = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  chars?: number;
  results?: number | null;
  step?: number | null; // which model call asked for it (absent on older saved answers)
  seconds?: number | null;
};

/** One model call: total time, and how much of it was waiting for the first streamed byte. */
export type ModelCall = {
  step: number;
  seconds: number;
  wait_seconds: number;
  retries: number | null;
  output_tokens: number | null;
  stop_reason: string;
};

// found: the chunk exists. shown: the model was shown it (absent on answers saved before this was tracked).
export type Citation = { source: string; index: number; section: string | null; found: boolean; shown?: boolean };

export type Usage = {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  api_calls: number;
};

export type Feedback = { rating: "up" | "down"; comment: string };

export type Turn = {
  id?: string;
  role: "user" | "assistant";
  text: string;
  steps: Step[];
  citations: Citation[];
  usage: Usage | null;
  cost_usd: number | null;
  seconds: number | null;
  stop: string | null;
  model?: string | null;
  feedback?: Feedback | null;
  streaming?: boolean;
  timings?: ModelCall[];
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
  reranker_configured: boolean;
  ocr: boolean;
};

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

export type Settings = {
  model: string;
  effort: Effort;
  search_k: number | null;
  rerank: boolean;
  max_steps: number;
  max_cost_usd: number | null;
};

export type ChunkView = {
  source: string;
  index: number;
  total_chunks: number;
  chunks: { index: number; section: string; text: string; cited: boolean }[];
};

export type ChatEvent =
  | ({ type: "conversation"; turn_id: string } & ConversationSummary)
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; id: string; name: string; args: Record<string, unknown>; step?: number }
  | { type: "tool_result"; id: string; name: string; is_error: boolean; chars: number; results?: number; seconds?: number }
  | ({ type: "model_call" } & ModelCall)
  | { type: "answer"; text: string; stop: string; citations: Citation[] }
  | { type: "usage"; usage: Usage; cost_usd: number | null }
  | { type: "error"; message: string }
  | { type: "done"; seconds: number; conversation: ConversationSummary };

export type DocStatus = "indexed" | "changed" | "new" | "deleted" | "empty" | "error";

export type DocumentRow = {
  name: string;
  type: string;
  size: number | null;
  modified: number | null;
  indexed_at: number | null;
  error: string | null;
  chunks: number;
  status: DocStatus;
  pages: number | null;
  transcribed: boolean;
  transcribe_cost_usd: number | null;
};

export type IngestState = {
  state: "idle" | "running" | "done" | "error";
  stage?: string;
  done?: number;
  total?: number;
  reason?: string;
  file?: string | null;
  started_at?: number;
  finished_at?: number;
  error?: string;
  report?: { added: string[]; updated: string[]; removed: string[]; unchanged: number; chunks: number; empty?: string[]; failed?: Record<string, string>; embedded?: number };
};

export type DocumentsResponse = {
  documents: DocumentRow[];
  chunks: number;
  ingest: IngestState;
  needs_ingest: boolean;
  ocr: boolean;
};

export type UploadResponse = {
  saved: string[];
  rejected: { name: string; reason: string }[];
  conflicts: string[]; // names that already exist; not saved unless re-sent with overwrite
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

// ---------------------------------------------------------------- evals

export type EvalSystem = "agentic" | "agentic-v1" | "naive";

export type EvalSet = { name: string; questions: number; categories: Record<string, number>; description: string; docs_found: boolean };

export type SystemSummary = {
  correct: number;
  partially_correct: number;
  by_category: Record<string, { n: number; correct: number; range?: [number, number] }>;
  cited_expected_sources: number | null;
  cited_unseen?: number; // share of answers citing a chunk the model was never shown
  evidence_recall: number | null;
  avg_tool_calls: number;
  avg_api_calls: number;
  avg_seconds: number;
  cost_usd: number;
  judge_cost_usd: number;
  runs?: number;
  range?: Record<string, [number, number]>;
  unstable?: string[];
  errors?: number; // questions that failed with an error; left out of the metrics above
};

export type RetrievalMetrics = { "recall@k": number; "hit@k": number; mrr: number; note?: string };

export type EvalRun = {
  file: string;
  kind: "answers" | "retrieval";
  set: string | null;
  created_at: number;
  questions: number;
  // answers
  model?: string;
  effort?: string;
  repeats?: number;
  naive_k?: number;
  judge_model?: string;
  judge_same_as_answer?: boolean;
  max_steps?: number;
  search_k?: number | null;
  rerank?: boolean;
  systems?: Record<string, SystemSummary>;
  // retrieval
  k?: number;
  configs?: Record<string, RetrievalMetrics>;
};

export type EvalRow = {
  id: string;
  category: string;
  system: string;
  run: number;
  question: string;
  answer: string;
  verdict: "correct" | "partially_correct" | "incorrect" | "error";
  judge_reasoning: string;
  error?: string;
  cited_expected_sources: boolean | null;
  unseen_citations?: string[];
  evidence_recall: number | null;
  tool_calls: number;
  api_calls: number;
  cost_usd: number;
  seconds: number;
  stop: string;
};

export type EvalRequest = {
  kind: "retrieval" | "answers";
  set: string;
  systems?: EvalSystem[];
  model?: string;
  effort?: Effort;
  limit?: number | null;
  repeats?: number;
  naive_k?: number;
};

export type Estimate = {
  total_usd: number;
  questions: number;
  answers: number;
  per_system: Record<string, number>;
  judge_usd: number;
  api_calls: number;
  basis: "history" | "rough" | "demo" | "free";
  model?: string;
};

export type EvalJob = {
  state: "idle" | "running" | "done" | "error" | "cancelled";
  stage?: string;
  kind?: "retrieval" | "answers";
  set?: string;
  model?: string | null;
  done?: number;
  total?: number;
  cost_usd?: number;
  started_at?: number;
  finished_at?: number;
  result_file?: string;
  error?: string;
};

export type FeedbackRecord = {
  turn_id: string;
  conversation_id: string;
  rating: "up" | "down";
  comment: string;
  question: string;
  answer: string;
  model: string | null;
  followup: boolean;
  created_at: number;
  updated_at: number;
};

export type ExportResult = { path: string; questions: number; skipped: { id: string; question: string; reason: string }[] };
