export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      cache: "no-store",
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Network request failed";
    throw new ApiError(
      `Could not reach the PAPER2PRODUCT API at ${API_BASE_URL}: ${detail}`,
      0,
    );
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        message = body.detail;
      }
    } catch {
      message = response.statusText || message;
    }
    throw new ApiError(message, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export interface Paper {
  id: number;
  title: string;
  source: string;
  source_ref: string | null;
  abstract: string | null;
  status: string;
  section_count: number;
}

export interface PaperDetail extends Paper {
  authors: string[];
  metadata: Record<string, unknown>;
  sections: Array<{
    id: number;
    heading: string;
    content: string;
    page_start: number | null;
    page_end: number | null;
  }>;
}

export interface Fact {
  kind: string;
  statement: string;
  source_section_id: number;
  evidence_quote: string;
  confidence: number;
}

export interface Understanding {
  extraction_method: string;
  confidence_level: string;
  confidence_score: number;
  facts: Fact[];
  unsupported_facts: Array<{ reason: string }>;
}

export interface Claim extends Fact {
  id: number;
  claim_type: string;
  section_id: number;
  evidence: string;
  supported: boolean;
  extraction_method: string;
}

export interface Plan {
  plan_id: number;
  status: "READY" | "BLOCKED";
  objective: string;
  blockers: string[];
  missing_information: string[];
  differences: Array<{
    dimension: string;
    original_setup: string;
    observed_setup: string;
    significance: string;
    observed_run_path: string;
  }>;
  reproduction_setup: Record<string, unknown>;
}

export interface Experiment {
  id: string;
  paper_id: number | null;
  plan_id: number | null;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  error: string | null;
  config_path: string | null;
  metrics_path: string | null;
  artifact_directory: string | null;
}

export interface LogItem {
  id: number;
  stream: "stdout" | "stderr";
  message: string;
  created_at: string;
}

export interface PipelineStatus {
  papers: number;
  experiments_by_status: Record<string, number>;
  latest_paper_id: number | null;
  latest_experiment_id: string | null;
  latest_plan_id: number | null;
  latest_comparison_id: number | null;
  latest_pipeline_status: string;
}

export interface Comparison {
  status: string;
  score: number | null;
  rationale: string;
  blockers: string[];
  metrics: Record<string, unknown>;
  verification: Record<string, string[]>;
}

export interface PaperReport {
  comparison: Comparison | null;
}

export function listPapers(): Promise<{ papers: Paper[] }> {
  return request("/papers");
}

export function getPaper(paperId: number): Promise<PaperDetail> {
  return request(`/papers/${paperId}`);
}

export function getUnderstanding(paperId: number): Promise<Understanding> {
  return request(`/papers/${paperId}/understanding`);
}

export function getClaims(paperId: number): Promise<{ claims: Claim[] }> {
  return request(`/papers/${paperId}/claims`);
}

export function getPlan(paperId: number): Promise<Plan> {
  return request(`/papers/${paperId}/plan`);
}

export function getPaperReport(paperId: number): Promise<PaperReport> {
  return request(`/papers/${paperId}/report`);
}

export function listExperiments(paperId?: number): Promise<{ experiments: Experiment[] }> {
  const query = paperId ? `?paper_id=${paperId}` : "";
  return request(`/experiments${query}`);
}

export function getExperiment(experimentId: string): Promise<Experiment> {
  return request(`/experiments/${encodeURIComponent(experimentId)}`);
}

export function getLogs(
  experimentId: string,
  afterId = 0,
): Promise<{ logs: LogItem[]; next_after_id: number | null }> {
  return request(
    `/experiments/${encodeURIComponent(experimentId)}/logs?after_id=${afterId}`,
  );
}

export function getPipelineStatus(): Promise<PipelineStatus> {
  return request("/pipeline/status");
}

export function checkHealth(): Promise<{ status: string }> {
  return request("/health");
}

export function ingestSource(
  sourceType: "local" | "url" | "arxiv",
  sourceRef: string,
): Promise<Paper> {
  return post("/papers/ingest", { source_type: sourceType, source_ref: sourceRef });
}

export function uploadPaper(file: File): Promise<Paper> {
  const body = new FormData();
  body.append("file", file);
  return request("/papers/upload", { method: "POST", body });
}

export function analyzePaper(paperId: number): Promise<Understanding> {
  return post(`/papers/${paperId}/analyze`, {});
}

export function createPlan(paperId: number): Promise<Plan> {
  return post(`/papers/${paperId}/plan`, {});
}

export function startExperiment(
  paperId: number,
  planId: number | undefined,
  options: { epochs: number; batch_size: number; learning_rate: number; device: string },
): Promise<Experiment> {
  return post("/experiments", {
    paper_id: paperId,
    plan_id: planId,
    ...options,
  });
}

export function comparePlan(planId: number): Promise<Comparison> {
  return post(`/plans/${planId}/compare`, {});
}
