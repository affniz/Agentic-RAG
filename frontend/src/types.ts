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
