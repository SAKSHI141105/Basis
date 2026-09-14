export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type ClassifyResponse = {
  intent: string;
  confidence: number;
  all_scores: Record<string, number>;
};

export type DraftReplyResponse = {
  draft: string;
  grounded_on: string[];
  retrieval_scores: number[];
};

export type DecideResponse = {
  decision: "auto_handle" | "escalate";
  reason: string;
  signals: {
    intent_confidence: number;
    max_retrieval_similarity: number;
    contact_count: number;
    sentiment_delta: number;
    risk_flags: string[];
  };
};

export type PipelineResponse = {
  classify: ClassifyResponse;
  draft_reply: DraftReplyResponse;
  decision: DecideResponse;
};

export type SampleMessage = {
  thread_id: string;
  message: string;
};

export type IntentMetrics = {
  accuracy: number;
  macro_f1: number;
  per_intent_f1: Record<string, number>;
  confusion_matrix: number[][];
  labels: string[];
};

export type EscalationMetrics = {
  precision: number;
  recall: number;
  f1: number;
  cost_weighted_score: number;
  false_auto_handle_count: number;
  false_escalate_count: number;
};

export type JudgeScore = {
  thread_id: string;
  groundedness: number;
  correctness: number;
  tone: number;
  actionability: number;
  rationale: string;
  mean_score: number;
};

export type FailureExample = {
  thread_id: string;
  customer_msg: string;
  true_intent: string;
  pred_intent: string;
  true_escalation: string;
  pred_escalation: string;
};

export type EvalReport = {
  golden_set_size: number;
  main_system_examples_evaluated?: number;
  skipped_example_ids?: string[];
  intent_metrics: Record<"trivial" | "simple" | "main", IntentMetrics>;
  escalation_metrics: Record<"trivial" | "simple" | "main", EscalationMetrics>;
  judge_scores: JudgeScore[];
  judge_summary: Record<string, number>;
  human_agreement: {
    n: number;
    cohen_kappa: number;
    spearman_r: number;
    spearman_p: number;
    mean_absolute_diff: number;
  } | null;
  failure_examples: FailureExample[];
  misleading_number_note: string;
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body?.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function runPipeline(message: string): Promise<PipelineResponse> {
  return apiFetch("/pipeline", { method: "POST", body: JSON.stringify({ message }) });
}

export function getSamples(): Promise<SampleMessage[]> {
  return apiFetch("/samples");
}

export function getEvalReport(): Promise<EvalReport> {
  return apiFetch("/eval-report");
}
