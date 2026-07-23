// Talks to the FastAPI backend (server/app.py) on :8000.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// Agent actions are typed artifacts the UI renders (plan card, data card, scorecard…).
// Loosely typed because each type carries a different payload.
export type AgentAction = {
  type: string;
  job_id?: string;
  run_name?: string;
  project_id?: string;
  dataset_id?: string;
  name?: string;
  plan?: PlanData;
  stats?: DataStats;
  preview?: { input: string; target: string }[];
  n?: number;
  card?: ScoreCard;
  info?: DeployInfo;
};

export type PlanData = {
  task_type: string;
  base_model_label: string;
  method: string;
  est_train_gb?: number | null;
  n_examples: number;
  needs_colab: boolean;
  est_time_min: string;
};

export type DataStats = {
  n: number;
  n_classes?: number;
  classes?: { label: string; count: number }[];
  split?: { train: number; val: number; test: number };
  balance_warning?: string | null;
};

export type ScoreCard = {
  correct: number;
  total: number;
  score: number;
  verdict: string;
  results: { input: string; expected?: string; got: string; verdict: string }[];
};

export type DeployInfo = {
  run_name: string;
  base_model: string;
  size_mb: number;
  files: string[];
  lm_studio_steps: string[];
  gguf_script: string;
  inference_snippet: string;
};

export type AgentResponse = { reply: string; actions?: AgentAction[]; error?: string; project_id?: string | null };

export async function callAgent(
  messages: { role: string; content: string }[],
  projectId?: string | null,
): Promise<AgentResponse> {
  const res = await fetch(`${API_BASE}/api/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, project_id: projectId ?? null }),
  });
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${API_BASE}/api/train/${jobId}/cancel`, { method: "POST" });
}

export type ChatResponse = { reply?: string; task?: string; error?: string };

export async function chatWithModel(
  runName: string,
  message: string,
  useAdapter: boolean,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_name: runName, message, use_adapter: useAdapter }),
  });
  return res.json();
}

export type SystemInfo = { badge: string; recommended_label: string; gpu: { cuda: boolean; total_gb: number } };

export async function getSystem(): Promise<SystemInfo | null> {
  try {
    const res = await fetch(`${API_BASE}/api/system`);
    return res.json();
  } catch {
    return null;
  }
}

// --- Admin: which Gemini "teacher" model powers the agent + data generation ---
export type Settings = { gemini_model: string; default_model: string; key_set: boolean };
export type GeminiModel = { id: string; label: string; description: string; recommended: boolean };
export type GeminiModelList = { models: GeminiModel[]; source: "live" | "fallback"; error?: string };

export async function getSettings(): Promise<Settings | null> {
  try {
    const res = await fetch(`${API_BASE}/api/settings`);
    return res.json();
  } catch {
    return null;
  }
}

export async function updateSettings(geminiModel: string): Promise<Settings> {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gemini_model: geminiModel }),
  });
  return res.json();
}

export async function getGeminiModels(): Promise<GeminiModelList> {
  try {
    const res = await fetch(`${API_BASE}/api/gemini/models`);
    return res.json();
  } catch {
    return { models: [], source: "fallback", error: "Couldn't reach the backend." };
  }
}

export function wsUrl(jobId: string): string {
  return `${API_BASE.replace(/^http/, "ws")}/ws/${jobId}`;
}

// --- Use-case templates (the "what do you want to build?" gallery) ---
export type Template = {
  key: string;
  title: string;
  tagline: string;
  task_type: string;
  what_it_does: string;
  honest_note: string;
  example_labels: string[];
  example_rows: { input: string; target: string }[];
  suggested_n: number;
  min_n: number;
  needs_bigger_model: boolean;
  local_model: string;
  colab_model: string;
  data_hint: string;
};

export async function getTemplates(): Promise<Template[]> {
  try {
    const res = await fetch(`${API_BASE}/api/templates`);
    return (await res.json()).templates ?? [];
  } catch {
    return [];
  }
}

// --- Dataset viewing + editing (the spreadsheet-style editor) ---
export type DatasetRow = { input: string; target: string };

export async function getDatasetFull(
  pid: string,
  did: string,
): Promise<{ rows: DatasetRow[]; meta: DataStats; n: number } | null> {
  try {
    const res = await fetch(`${API_BASE}/api/projects/${pid}/data/${did}?full=1`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function saveDataset(
  pid: string,
  did: string,
  rows: DatasetRow[],
): Promise<{ n: number; stats: DataStats } | { error: string }> {
  const res = await fetch(`${API_BASE}/api/projects/${pid}/data/${did}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows }),
  });
  return res.json();
}

// Colab notebook download (one-click bigger-model training). Returns a URL to hit.
export function colabDownloadUrl(pid: string): string {
  return `${API_BASE}/api/projects/${pid}/colab`;
}

export async function getProject(pid: string): Promise<{ dataset_id?: string | null } | null> {
  try {
    const res = await fetch(`${API_BASE}/api/projects/${pid}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// --- Projects panel: reopen past projects, run history, retroactive judging ---
export type ProjectSummary = {
  id: string;
  name: string;
  task_type: string;
  base_model?: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  active_run?: string | null;
};

export type RunRecord = {
  run_name: string;
  version?: number;
  task?: string;
  before?: number | null;
  after?: number | null;
  delta?: number | null;
  created_at?: string;
};

export type ProjectDetail = {
  id: string;
  name: string;
  goal?: string;
  task_type: string;
  status: string;
  dataset_id?: string | null;
  active_run?: string | null;
  runs?: RunRecord[];
};

export async function listProjects(): Promise<ProjectSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/api/projects`);
    return (await res.json()).projects ?? [];
  } catch {
    return [];
  }
}

export async function getProjectDetail(pid: string): Promise<ProjectDetail | null> {
  try {
    const res = await fetch(`${API_BASE}/api/projects/${pid}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export type JudgeResult = {
  run_name?: string;
  before_avg?: number | null;
  after_avg?: number | null;
  n?: number;
  error?: string;
};

export async function judgeRun(runName: string): Promise<JudgeResult> {
  try {
    const res = await fetch(`${API_BASE}/api/runs/${runName}/judge`, { method: "POST" });
    return res.json();
  } catch {
    return { error: "backend unreachable" };
  }
}
