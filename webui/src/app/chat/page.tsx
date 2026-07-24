"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Settings, Download, Pencil, Plus, Trash2, X, Cloud, Upload, FolderOpen, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  callAgent,
  cancelJob,
  chatWithModel,
  getSystem,
  getTemplates,
  getDatasetFull,
  saveDataset,
  colabDownloadUrl,
  getProject,
  listProjects,
  getProjectDetail,
  judgeRun,
  compareRuns,
  runRobustness,
  pairwiseJudge,
  wsUrl,
} from "@/lib/api";
import type {
  PlanData,
  DataStats,
  ScoreCard,
  DeployInfo,
  SystemInfo,
  Template,
  DatasetRow,
  ProjectSummary,
  ProjectDetail,
  JudgeResult,
  RobustnessResult,
  PairwiseResult,
} from "@/lib/api";

type Msg = { kind: "msg"; role: "user" | "assistant"; content: string };
type Training = { kind: "training"; jobId: string };
type PlanItem = { kind: "plan"; plan: PlanData };
type DataItem = {
  kind: "data";
  stats: DataStats;
  preview: { input: string; target: string }[];
  projectId?: string;
  datasetId?: string;
};
type ScoreItem = { kind: "score"; card: ScoreCard };
type DeployItem = { kind: "deploy"; info: DeployInfo };
type Item = Msg | Training | PlanItem | DataItem | ScoreItem | DeployItem;

type Sample = {
  input: string;
  gold?: string;
  base: string;
  tuned: string;
  base_ok?: boolean;
  tuned_ok?: boolean;
};

type PerClass = { label: string; support: number; base_acc: number; tuned_acc: number };

type CI = { ci_lo?: number | null; ci_hi?: number | null };
type Report = {
  task: string;
  before: { accuracy?: number; macro_f1?: number; avg_judge_score?: number | null; n?: number } & CI;
  after: { accuracy?: number; macro_f1?: number; avg_judge_score?: number | null; n?: number } & CI;
  prompted?: { avg_judge_score?: number | null; n?: number } | null;
  delta_accuracy?: number;
  delta_judge?: number | null;
  mcnemar_base_vs_tuned?: { p_value?: number; b?: number; c?: number; n_discordant?: number } | null;
  eval_set_id?: string | null;
  samples?: Sample[];
  per_class?: PerClass[];
  labels?: string[];
};

type TrainSample = { input: string; label: string };

type Status = "running" | "done" | "error" | "stopped";

const EXAMPLES = [
  "Build a customer-support bot that sorts messages by intent",
  "Make a tone rewriter that turns blunt text polite",
  "Create a data extractor that pulls fields into JSON",
];

export default function Home() {
  const [items, setItems] = useState<Item[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [templates, setTemplates] = useState<Template[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items, busy]);

  useEffect(() => {
    getSystem().then(setSys);
    getTemplates().then(setTemplates);
  }, []);

  // Landing-page deep link: /chat?t=<templateKey> starts that use case immediately.
  const deepLinked = useRef(false);
  useEffect(() => {
    if (deepLinked.current || templates.length === 0 || items.length > 0) return;
    const key = new URLSearchParams(window.location.search).get("t");
    const t = templates.find((x) => x.key === key);
    if (t) {
      deepLinked.current = true;
      pickTemplate(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [templates]);

  function pickTemplate(t: Template) {
    send(
      `I want to build a ${t.title}. ${t.what_it_does} Guide me step by step, and make sure we collect enough good training data for a model that actually works.`,
    );
  }

  const [showProjects, setShowProjects] = useState(false);

  async function openProject(p: ProjectDetail) {
    setShowProjects(false);
    setProjectId(p.id);
    const add: Item[] = [];
    let dataNote = "";
    if (p.dataset_id) {
      const d = await getDatasetFull(p.id, p.dataset_id);
      if (d) {
        dataNote = ` ${d.n} training examples are loaded — open **View & edit all** on the data card to review or change them.`;
        add.push({
          kind: "data",
          stats: d.meta,
          preview: (d.rows ?? []).slice(0, 4),
          projectId: p.id,
          datasetId: p.dataset_id,
        });
      }
    }
    const runNote = p.active_run
      ? ` Latest trained version: \`${p.active_run}\`.`
      : " No trained version yet.";
    add.unshift({
      kind: "msg",
      role: "assistant",
      content: `Resumed **${p.name}** (${p.task_type}).${dataNote}${runNote} Tell me what you'd like to do next — generate more data, retrain, test it, or deploy.`,
    });
    setItems(add);
  }

  async function send(text?: string) {
    const content = (text ?? input).trim();
    if (!content || busy) return;
    setInput("");
    const history = items
      .filter((i): i is Msg => i.kind === "msg")
      .map((i) => ({ role: i.role, content: i.content }));
    setItems((x) => [...x, { kind: "msg", role: "user", content }]);
    setBusy(true);
    try {
      const res = await callAgent([...history, { role: "user", content }], projectId);
      if (res.error) {
        setItems((x) => [...x, { kind: "msg", role: "assistant", content: "⚠ " + res.error }]);
      } else {
        if (res.project_id) setProjectId(res.project_id);
        const add: Item[] = [{ kind: "msg", role: "assistant", content: res.reply }];
        for (const a of res.actions ?? []) {
          if (a.type === "training_started" && a.job_id) add.push({ kind: "training", jobId: a.job_id });
          else if (a.type === "plan" && a.plan) add.push({ kind: "plan", plan: a.plan });
          else if (a.type === "data_generated" && a.stats)
            add.push({
              kind: "data",
              stats: a.stats,
              preview: a.preview ?? [],
              projectId: a.project_id,
              datasetId: a.dataset_id,
            });
          else if (a.type === "autotest" && a.card) add.push({ kind: "score", card: a.card });
          else if (a.type === "deploy" && a.info) add.push({ kind: "deploy", info: a.info });
        }
        // A fresh plan replaces any earlier plan card (no stacked duplicates).
        const hasPlan = add.some((it) => it.kind === "plan");
        setItems((x) => [...(hasPlan ? x.filter((it) => it.kind !== "plan") : x), ...add]);
      }
    } catch {
      setItems((x) => [
        ...x,
        { kind: "msg", role: "assistant", content: "⚠ Couldn't reach the backend. Is it running on :8000?" },
      ]);
    } finally {
      setBusy(false);
    }
  }

  const empty = items.length === 0;

  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b">
        <div className="mx-auto flex max-w-2xl items-center justify-between px-4 py-4">
          <Link href="/" className="flex items-center gap-2">
            <div className="h-5 w-5 rounded-sm bg-foreground" />
            <span className="text-sm font-semibold tracking-tight">Fine-Tune Studio</span>
          </Link>
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              {sys?.gpu?.cuda ? `${sys.gpu.total_gb.toFixed(0)}gb · ${sys.recommended_label}` : "local · qwen · qlora"}
            </span>
            <button
              onClick={() => setShowProjects(true)}
              aria-label="Your projects"
              title="Your projects"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              <FolderOpen className="size-4" />
            </button>
            <Link
              href="/admin"
              aria-label="Admin settings"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              <Settings className="size-4" />
            </Link>
          </div>
        </div>
      </header>

      {showProjects && <ProjectsPanel onClose={() => setShowProjects(false)} onOpen={openProject} />}

      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col px-4">
        <div className="flex-1 space-y-5 py-8">
          {empty && (
            <div className="mt-12 flex flex-col items-center text-center">
              <h1 className="text-2xl font-semibold tracking-tight">What kind of model do you want?</h1>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                Pick the shape of the thing you&apos;re building. I&apos;ll help you collect good data, train
                it, and prove — with real held-out examples — whether it actually got better.
              </p>
              <p className="mt-2 max-w-md text-xs text-muted-foreground/80">
                Reality check: a small local model won&apos;t be a general genius like Gemini. The win is that it{" "}
                <b>beats the big models on your one specific job</b> — free, private, and fast.
              </p>
              {sys?.badge && (
                <div className="mt-4 inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs text-muted-foreground">
                  <span className={`h-1.5 w-1.5 rounded-full ${sys.gpu.cuda ? "bg-green-500" : "bg-amber-500"}`} />
                  {sys.badge}
                  {!sys.gpu.cuda && " · use the Colab path for training"}
                </div>
              )}
              {templates.length > 0 ? (
                <TemplateGallery templates={templates} onPick={pickTemplate} />
              ) : (
                <div className="mt-6 flex w-full max-w-sm flex-col gap-2">
                  {EXAMPLES.map((e) => (
                    <button
                      key={e}
                      onClick={() => send(e)}
                      className="rounded-md border px-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      {e}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {items.map((it, i) => {
            if (it.kind === "msg") return <MessageBubble key={i} role={it.role} content={it.content} />;
            if (it.kind === "training") return <TrainingPanel key={i} jobId={it.jobId} />;
            if (it.kind === "plan")
              return <PlanCard key={i} plan={it.plan} onApprove={() => send("Yes — approve the plan and continue.")} />;
            if (it.kind === "data")
              return (
                <DataCard
                  key={i}
                  stats={it.stats}
                  preview={it.preview}
                  projectId={it.projectId}
                  datasetId={it.datasetId}
                />
              );
            if (it.kind === "score") return <ScoreCardView key={i} card={it.card} />;
            if (it.kind === "deploy") return <DeployCard key={i} info={it.info} />;
            return null;
          })}

          {busy && <div className="font-mono text-xs text-muted-foreground">thinking…</div>}
          <div ref={endRef} />
        </div>

        <div className="sticky bottom-0 bg-background pb-6 pt-2">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
            className="flex items-center gap-2 rounded-lg border p-1.5"
          >
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me to train a model…"
              className="border-0 shadow-none focus-visible:ring-0"
            />
            <Button type="submit" disabled={busy || !input.trim()} size="sm">
              Send
            </Button>
          </form>
          <p className="mt-2 text-center font-mono text-[10px] text-muted-foreground">
            training runs on your GPU · a few minutes per run
          </p>
        </div>
      </main>
    </div>
  );
}

function MessageBubble({ role, content }: { role: string; content: string }) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-foreground px-4 py-2 text-sm text-background">
          {content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex gap-3">
      <div className="mt-1 h-5 w-5 shrink-0 rounded-sm border bg-muted" />
      <div className="min-w-0 flex-1">
        <Markdown text={content} />
      </div>
    </div>
  );
}

function TrainingPanel({ jobId }: { jobId: string }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState<{
    step: number;
    max: number;
    loss?: number;
    eta?: number | null;
    vram?: { used_gb: number; total_gb: number } | null;
  }>({ step: 0, max: 0 });
  const [report, setReport] = useState<Report | null>(null);
  const [status, setStatus] = useState<Status>("running");
  const [stopping, setStopping] = useState(false);
  const [trainSamples, setTrainSamples] = useState<TrainSample[]>([]);
  const [runName, setRunName] = useState<string | null>(null);

  useEffect(() => {
    // One socket per subscription. React StrictMode (dev) mounts the effect twice;
    // the throwaway socket's close must NOT read as a training failure — so we only
    // ever flip to "error" on an explicit {type:"error"} event, never on socket close.
    let closed = false;
    setLogs([]);
    setProgress({ step: 0, max: 0 });
    const ws = new WebSocket(wsUrl(jobId));
    ws.onmessage = (e) => {
      if (closed) return;
      const ev = JSON.parse(e.data);
      if (ev.type === "log") setLogs((l) => [...l.slice(-40), ev.msg]);
      else if (ev.type === "progress")
        setProgress({ step: ev.step, max: ev.max_steps, loss: ev.loss, eta: ev.eta_sec, vram: ev.vram });
      else if (ev.type === "train_samples") setTrainSamples(ev.samples ?? []);
      else if (ev.type === "result") {
        setReport(ev.report);
        if (ev.run_name) setRunName(ev.run_name);
      } else if (ev.type === "done") {
        setStatus("done");
        if (ev.run_name) setRunName(ev.run_name);
      } else if (ev.type === "cancelled") {
        setStatus("stopped");
        setLogs((l) => [...l, ev.msg]);
      } else if (ev.type === "error") {
        setStatus("error");
        setLogs((l) => [...l, "Error: " + ev.msg]);
      }
    };
    return () => {
      closed = true;
      ws.close();
    };
  }, [jobId]);

  async function stop() {
    setStopping(true);
    try {
      await cancelJob(jobId);
    } catch {
      /* the socket reports the real outcome; nothing to do here */
    }
  }

  const pct = progress.max ? Math.round((progress.step / progress.max) * 100) : status === "done" ? 100 : 0;
  const last = logs[logs.length - 1];
  const label =
    status === "running" ? "Training" : status === "done" ? "Trained" : status === "stopped" ? "Stopped" : "Error";

  return (
    <div className="rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
        <div className="flex items-center gap-3">
          {status === "running" && (
            <button
              onClick={stop}
              disabled={stopping}
              className="rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
            >
              {stopping ? "stopping…" : "stop"}
            </button>
          )}
          <span
            className={`h-2 w-2 rounded-full ${
              status === "running"
                ? "animate-pulse bg-foreground"
                : status === "done"
                  ? "bg-foreground"
                  : status === "stopped"
                    ? "bg-muted-foreground"
                    : "bg-destructive"
            }`}
          />
        </div>
      </div>

      <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-foreground transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[11px] text-muted-foreground">
        <span>
          {progress.max ? `step ${progress.step}/${progress.max}` : "starting…"}
          {progress.eta != null && status === "running" ? ` · ~${fmtEta(progress.eta)} left` : ""}
        </span>
        <span>{progress.loss != null ? `loss ${progress.loss.toFixed(3)}` : ""}</span>
      </div>

      {progress.vram && status === "running" && (
        <div className="mt-2">
          <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-muted-foreground transition-all"
              style={{ width: `${Math.min(100, Math.round((progress.vram.used_gb / progress.vram.total_gb) * 100))}%` }}
            />
          </div>
          <div className="mt-1 text-right font-mono text-[10px] text-muted-foreground">
            VRAM {progress.vram.used_gb.toFixed(1)} / {progress.vram.total_gb.toFixed(1)} GB
          </div>
        </div>
      )}

      {last && !report && <div className="mt-2 truncate font-mono text-[11px] text-muted-foreground">› {last}</div>}

      {!report && trainSamples.length > 0 && <TrainSamples samples={trainSamples} />}

      {report && (
        <>
          <ResultCard report={report} />
          {report.samples && report.samples.length > 0 && <ExamplesCard report={report} />}
          {report.per_class && report.per_class.length > 0 && <PerClassCard rows={report.per_class} />}
          {runName && <Playground runName={runName} task={report.task} />}
        </>
      )}
    </div>
  );
}

function TrainSamples({ samples }: { samples: TrainSample[] }) {
  return (
    <div className="mt-3 rounded-md border border-dashed p-3">
      <div className="mb-1.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        Learning from examples like
      </div>
      <div className="space-y-1.5">
        {samples.map((s, i) => (
          <div key={i} className="text-[13px] leading-snug">
            <span className="text-muted-foreground">“{s.input}”</span>
            <span className="text-muted-foreground"> → </span>
            <span className="font-mono text-xs">{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const TEST_SUGGESTIONS = ["where is my order?", "I want to cancel my subscription", "how do I get a refund?"];

function ResultCard({ report }: { report: Report }) {
  if (report.task === "classification") {
    const before = Math.round((report.before.accuracy ?? 0) * 100);
    const after = Math.round((report.after.accuracy ?? 0) * 100);
    const delta = after - before;
    const ciLo = report.after.ci_lo != null ? Math.round(report.after.ci_lo * 100) : null;
    const ciHi = report.after.ci_hi != null ? Math.round(report.after.ci_hi * 100) : null;
    const mp = report.mcnemar_base_vs_tuned?.p_value;
    return (
      <div className="mt-4 border-t pt-4">
        <div className="flex items-end justify-center gap-6">
          <Stat label="Before" value={`${before}%`} muted />
          <div className="pb-2 text-muted-foreground">→</div>
          <Stat label="After" value={`${after}%`} />
          <div className="pb-2 font-mono text-xs text-muted-foreground">
            {delta >= 0 ? "+" : ""}
            {delta} pts
          </div>
        </div>
        <div className="mt-2 text-center font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          accuracy on {report.after.n ?? "?"} held-out messages
          {report.after.macro_f1 != null ? ` · macro-F1 ${report.after.macro_f1.toFixed(2)}` : ""}
        </div>
        {ciLo != null && (
          <div className="mt-1 text-center font-mono text-[10px] text-muted-foreground">
            95% CI {ciLo}–{ciHi}%
            {mp != null && (
              <> · vs base p={mp < 0.001 ? "<0.001" : mp.toFixed(3)} {mp < 0.05 ? "✓ significant" : "· not significant"}</>
            )}
          </div>
        )}
      </div>
    );
  }

  const b = report.before.avg_judge_score;
  const a = report.after.avg_judge_score;
  if (a == null) {
    return (
      <div className="mt-4 border-t pt-4 text-center text-sm text-muted-foreground">
        Trained. Auto-scoring wasn&apos;t available this run (judge unreachable or rate-limited) — test it
        yourself below.
      </div>
    );
  }
  const p = report.prompted?.avg_judge_score;
  return (
    <div className="mt-4 border-t pt-4">
      <div className="flex items-end justify-center gap-6">
        <Stat label="Before" value={(b ?? 0).toFixed(1)} muted />
        <div className="pb-2 text-muted-foreground">→</div>
        {p != null && (
          <>
            <Stat label="Good prompt" value={p.toFixed(1)} muted />
            <div className="pb-2 text-muted-foreground">→</div>
          </>
        )}
        <Stat label="Fine-tuned" value={a.toFixed(1)} />
        <div className="pb-2 font-mono text-xs text-muted-foreground">/ 10</div>
      </div>
      <div className="mt-2 text-center font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        LLM-judged reply quality{p != null ? " · tuned must beat the good prompt too" : ""}
      </div>
      {report.after.ci_lo != null && (
        <div className="mt-1 text-center font-mono text-[10px] text-muted-foreground">
          95% CI {report.after.ci_lo.toFixed(1)}–{(report.after.ci_hi ?? 0).toFixed(1)} · n={report.after.n ?? "?"}
        </div>
      )}
    </div>
  );
}

function ExamplesCard({ report }: { report: Report }) {
  const samples = (report.samples ?? []).slice(0, 6);
  const isCls = report.task === "classification";
  return (
    <div className="mt-4 border-t pt-4">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        Real held-out examples
      </div>
      <div className="space-y-2">
        {samples.map((s, i) => (
          <div key={i} className="rounded-md border p-2.5">
            <div className="text-sm">{s.input}</div>
            {isCls ? (
              <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]">
                <span className="text-muted-foreground">
                  before:{" "}
                  <span className={s.base_ok ? "text-foreground" : "text-destructive line-through"}>{s.base}</span>
                </span>
                <span className="text-muted-foreground">
                  now: <span className={s.tuned_ok ? "text-foreground" : "text-destructive"}>{s.tuned}</span>
                </span>
                {s.gold && <span className="text-muted-foreground">correct: {s.gold}</span>}
                <span className="ml-auto">{s.tuned_ok && !s.base_ok ? "✓ fixed" : s.tuned_ok ? "✓" : "✗"}</span>
              </div>
            ) : (
              <div className="mt-1.5 space-y-1 text-[13px]">
                <div className="text-muted-foreground">before: {s.base}</div>
                <div>now: {s.tuned}</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function PerClassCard({ rows }: { rows: PerClass[] }) {
  return (
    <div className="mt-4 border-t pt-4">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        By topic (before → after)
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const before = Math.round(r.base_acc * 100);
          const after = Math.round(r.tuned_acc * 100);
          return (
            <div key={r.label} className="flex items-center gap-2 text-[11px]">
              <span className="w-36 shrink-0 truncate font-mono" title={r.label}>
                {r.label}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                <div className="h-full bg-foreground transition-all" style={{ width: `${after}%` }} />
              </div>
              <span className="w-20 shrink-0 text-right font-mono text-muted-foreground">
                {before}%→{after}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Playground({ runName, task }: { runName: string; task: string }) {
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<{ q: string; base: string; tuned: string }[]>([]);

  async function run(text?: string) {
    const q = (text ?? msg).trim();
    if (!q || busy) return;
    setMsg("");
    setBusy(true);
    try {
      // Sequential on a 4GB GPU: one cached model, adapter toggled on then off.
      const tuned = await chatWithModel(runName, q, true);
      const base = await chatWithModel(runName, q, false);
      setRows((r) => [
        ...r,
        { q, base: base.reply ?? base.error ?? "—", tuned: tuned.reply ?? tuned.error ?? "—" },
      ]);
    } catch {
      setRows((r) => [...r, { q, base: "—", tuned: "(couldn't reach the backend)" }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-4 border-t pt-4">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Test it yourself</div>
      {rows.length === 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {TEST_SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => run(s)}
              disabled={busy}
              className="rounded-full border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="space-y-2">
        {rows.map((r, i) => (
          <div key={i} className="rounded-md border p-2.5">
            <div className="text-sm">{r.q}</div>
            <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]">
              <span className="text-muted-foreground">before: {r.base}</span>
              <span>→</span>
              <span>now: {r.tuned}</span>
            </div>
          </div>
        ))}
        {busy && (
          <div className="font-mono text-[11px] text-muted-foreground">running your message through both models…</div>
        )}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
        className="mt-2 flex items-center gap-2 rounded-lg border p-1.5"
      >
        <Input
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          placeholder={task === "classification" ? "Type a customer message to classify…" : "Type a customer message…"}
          className="border-0 shadow-none focus-visible:ring-0"
        />
        <Button type="submit" size="sm" disabled={busy || !msg.trim()}>
          {busy ? "…" : "Test"}
        </Button>
      </form>
    </div>
  );
}

function Stat({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="text-center">
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className={`text-3xl font-semibold tabular-nums ${muted ? "text-muted-foreground" : ""}`}>{value}</div>
    </div>
  );
}

function fmtEta(sec: number): string {
  if (sec < 60) return `${Math.max(1, Math.round(sec))}s`;
  const m = Math.round(sec / 60);
  return `${m} min`;
}

function CardShell({ tag, children }: { tag: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border p-4">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{tag}</div>
      {children}
    </div>
  );
}

function PlanCard({ plan, onApprove }: { plan: PlanData; onApprove: () => void }) {
  return (
    <CardShell tag="Plan — approve to continue">
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
        <Field k="Task" v={plan.task_type} />
        <Field k="Model" v={plan.base_model_label} />
        <Field k="Method" v={plan.method} />
        <Field k="Examples" v={plan.n_examples ? String(plan.n_examples) : "to be generated"} />
        <Field k="Est. time" v={plan.est_time_min} />
        <Field k="Runs on" v={plan.needs_colab ? "Colab (too big for 4 GB)" : "your GPU"} />
      </div>
      <div className="mt-3 flex justify-end">
        <Button size="sm" onClick={onApprove}>
          Approve &amp; continue
        </Button>
      </div>
    </CardShell>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{k}</div>
      <div>{v}</div>
    </div>
  );
}

function DataCard({
  stats: initialStats,
  preview,
  projectId,
  datasetId,
}: {
  stats: DataStats;
  preview: { input: string; target: string }[];
  projectId?: string;
  datasetId?: string;
}) {
  const [stats, setStats] = useState<DataStats>(initialStats);
  const [editing, setEditing] = useState(false);
  const [did, setDid] = useState<string | undefined>(datasetId);
  const labels = stats.classes?.map((c) => c.label) ?? [];
  const canEdit = Boolean(projectId && did);

  // Some actions omit dataset_id — recover it from the project so editing always works.
  useEffect(() => {
    if (!did && projectId)
      getProject(projectId).then((p) => {
        if (p?.dataset_id) setDid(p.dataset_id);
      });
  }, [did, projectId]);

  return (
    <CardShell tag="Data ready">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span>
          <b className="tabular-nums">{stats.n}</b> examples
        </span>
        {stats.n_classes != null && (
          <span>
            <b className="tabular-nums">{stats.n_classes}</b> classes
          </span>
        )}
        {stats.split && (
          <span className="text-muted-foreground">
            split {stats.split.train}/{stats.split.val}/{stats.split.test}
          </span>
        )}
        {canEdit && (
          <button
            onClick={() => setEditing(true)}
            className="ml-auto inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <Pencil className="size-3" /> View &amp; edit all {stats.n}
          </button>
        )}
      </div>
      {stats.balance_warning && (
        <div className="mt-1.5 text-xs text-amber-600 dark:text-amber-500">⚠ {stats.balance_warning}</div>
      )}
      {(stats.n ?? 0) < 120 && (
        <div className="mt-1.5 text-xs text-amber-600 dark:text-amber-500">
          ⚠ This is a small set — good for a quick test, but add more (aim for 250+) for a bot you&apos;d actually
          ship. You can generate more or add your own real examples in the editor.
        </div>
      )}
      {preview.length > 0 && (
        <div className="mt-2 space-y-1">
          {preview.slice(0, 4).map((r, i) => (
            <div key={i} className="text-[13px] leading-snug">
              <span className="text-muted-foreground">“{r.input}”</span>
              <span className="text-muted-foreground"> → </span>
              <span className="font-mono text-xs">{r.target}</span>
            </div>
          ))}
        </div>
      )}
      {projectId && <ScaleUp projectId={projectId} />}
      {editing && projectId && did && (
        <DatasetEditor
          projectId={projectId}
          datasetId={did}
          labels={labels}
          onClose={() => setEditing(false)}
          onSaved={(s) => setStats((prev) => ({ ...prev, ...s }))}
        />
      )}
    </CardShell>
  );
}

function ScoreCardView({ card }: { card: ScoreCard }) {
  const pill = (v: string) =>
    v === "correct" ? "text-foreground" : v === "partial" ? "text-amber-600 dark:text-amber-500" : "text-destructive";
  return (
    <CardShell tag="Auto-test — 10 hard cases">
      <div className="flex items-baseline gap-3">
        <div className="text-2xl font-semibold tabular-nums">
          {card.correct}/{card.total}
        </div>
        <div className="text-sm text-muted-foreground">{card.verdict}</div>
      </div>
      <div className="mt-2 space-y-1.5">
        {card.results.slice(0, 6).map((r, i) => (
          <div key={i} className="rounded-md border p-2 text-[13px]">
            <div>{r.input}</div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 font-mono text-[11px]">
              <span className={pill(r.verdict)}>{r.verdict}</span>
              {r.expected && <span className="text-muted-foreground">expected: {r.expected}</span>}
              <span className="text-muted-foreground">got: {r.got}</span>
            </div>
          </div>
        ))}
      </div>
    </CardShell>
  );
}

function DeployCard({ info }: { info: DeployInfo }) {
  const [tab, setTab] = useState<"lmstudio" | "api" | "gguf">("lmstudio");
  return (
    <CardShell tag="Deploy your model">
      <div className="text-sm">
        Adapter <b>{info.size_mb} MB</b> · base {info.base_model.split("/").pop()}
      </div>
      <div className="mt-1 flex flex-wrap gap-2">
        <a
          className="rounded border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          href={`${apiBase()}/api/runs/${info.run_name}/download`}
        >
          ↓ Download adapter (.zip)
        </a>
        {(["lmstudio", "api", "gguf"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded border px-2.5 py-1 text-xs transition-colors ${
              tab === t ? "bg-foreground text-background" : "text-muted-foreground hover:bg-muted"
            }`}
          >
            {t === "lmstudio" ? "LM Studio" : t === "api" ? "API" : "GGUF"}
          </button>
        ))}
      </div>
      <div className="mt-2">
        {tab === "lmstudio" && (
          <ol className="list-inside list-decimal space-y-0.5 text-[13px] text-muted-foreground">
            {info.lm_studio_steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        )}
        {tab === "api" && <Code text={info.inference_snippet} />}
        {tab === "gguf" && <Code text={info.gguf_script} />}
      </div>
    </CardShell>
  );
}

function Code({ text }: { text: string }) {
  return (
    <pre className="overflow-x-auto rounded-md border bg-muted/40 p-2 font-mono text-[11px] leading-relaxed">
      {text}
    </pre>
  );
}

function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}

// ------------------------------------------------------ lightweight markdown
// Renders the subset the agent actually emits (**bold**, *italic*, `code`,
// bullet/numbered lists, links) as real React nodes — no raw HTML, so it's safe.
function safeUrl(u: string): string {
  return /^(https?:|mailto:)/i.test(u.trim()) ? u.trim() : "#";
}

function mdInline(text: string, key: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const re = /\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*([^*]+)\*|_([^_]+)_/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) out.push(<strong key={`${key}-${i}`}>{m[1]}</strong>);
    else if (m[2] !== undefined)
      out.push(
        <code key={`${key}-${i}`} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">
          {m[2]}
        </code>,
      );
    else if (m[3] !== undefined)
      out.push(
        <a
          key={`${key}-${i}`}
          href={safeUrl(m[4])}
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2 hover:text-foreground"
        >
          {m[3]}
        </a>,
      );
    else if (m[5] !== undefined) out.push(<em key={`${key}-${i}`}>{m[5]}</em>);
    else if (m[6] !== undefined) out.push(<em key={`${key}-${i}`}>{m[6]}</em>);
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function Markdown({ text }: { text: string }) {
  const lines = (text ?? "").replace(/\r/g, "").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  const bullet = /^\s*[-*]\s+/;
  const ordered = /^\s*\d+\.\s+/;
  while (i < lines.length) {
    if (!lines[i].trim()) {
      i++;
      continue;
    }
    if (bullet.test(lines[i])) {
      const items: string[] = [];
      while (i < lines.length && bullet.test(lines[i])) items.push(lines[i++].replace(bullet, ""));
      blocks.push(
        <ul key={`b${i}`} className="ml-4 list-disc space-y-0.5">
          {items.map((it, j) => (
            <li key={j}>{mdInline(it, `b${i}-${j}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }
    if (ordered.test(lines[i])) {
      const items: string[] = [];
      while (i < lines.length && ordered.test(lines[i])) items.push(lines[i++].replace(ordered, ""));
      blocks.push(
        <ol key={`o${i}`} className="ml-4 list-decimal space-y-0.5">
          {items.map((it, j) => (
            <li key={j}>{mdInline(it, `o${i}-${j}`)}</li>
          ))}
        </ol>,
      );
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !bullet.test(lines[i]) && !ordered.test(lines[i]))
      para.push(lines[i++]);
    blocks.push(
      <p key={`p${i}`}>
        {para.map((ln, j) => (
          <span key={j}>
            {mdInline(ln, `p${i}-${j}`)}
            {j < para.length - 1 ? <br /> : null}
          </span>
        ))}
      </p>,
    );
  }
  return <div className="space-y-2 text-sm leading-relaxed">{blocks}</div>;
}

// --------------------------------------------------------- template gallery
function TemplateGallery({ templates, onPick }: { templates: Template[]; onPick: (t: Template) => void }) {
  return (
    <div className="mt-6 grid w-full gap-2 sm:grid-cols-2">
      {templates.map((t) => (
        <button
          key={t.key}
          onClick={() => onPick(t)}
          className="flex flex-col rounded-lg border p-3 text-left transition-colors hover:bg-muted"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">{t.title}</span>
            {t.needs_bigger_model && (
              <span className="shrink-0 rounded-full border px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-muted-foreground">
                bigger model
              </span>
            )}
          </div>
          <span className="mt-1 text-xs text-muted-foreground">{t.tagline}</span>
        </button>
      ))}
    </div>
  );
}

// --------------------------------------------------------- scale-up (Colab)
function ScaleUp({ projectId }: { projectId: string }) {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t pt-3">
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        Want a bigger model?
      </span>
      <a
        href={colabDownloadUrl(projectId)}
        title="Downloads a ready-to-run notebook — open it in Colab, set Runtime → T4 GPU, then Run all."
        className="inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <Download className="size-3" /> Colab notebook (Qwen 3B+)
      </a>
      <button
        disabled
        title="Coming soon — one-click managed cloud GPU training."
        className="inline-flex cursor-not-allowed items-center gap-1 rounded border px-2 py-0.5 text-xs text-muted-foreground/40"
      >
        <Cloud className="size-3" /> Cloud GPU · soon
      </button>
    </div>
  );
}

// --------------------------------------------------------- dataset editor
// A real spreadsheet grid: row numbers, column headers, and DYNAMIC columns.
// `input` and `target` are locked (they're what trains the model); every other
// column is free — added, renamed, deleted, imported — and is saved with the
// dataset but ignored by training.
const LOCKED_COLS = ["input", "target"];

function DatasetEditor({
  projectId,
  datasetId,
  labels,
  onClose,
  onSaved,
}: {
  projectId: string;
  datasetId: string;
  labels: string[];
  onClose: () => void;
  onSaved: (stats: DataStats) => void;
}) {
  const [rows, setRows] = useState<DatasetRow[] | null>(null);
  const [columns, setColumns] = useState<string[]>(LOCKED_COLS);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const isClassification = labels.length > 0;

  useEffect(() => {
    getDatasetFull(projectId, datasetId).then((d) => {
      const rs = (d?.rows ?? []) as DatasetRow[];
      setRows(rs);
      const extras: string[] = [];
      for (const r of rs)
        for (const k of Object.keys(r)) if (!LOCKED_COLS.includes(k) && !extras.includes(k)) extras.push(k);
      setColumns([...LOCKED_COLS, ...extras]);
    });
  }, [projectId, datasetId]);

  const update = (i: number, field: string, val: string) =>
    setRows((rs) => (rs ? rs.map((r, j) => (j === i ? ({ ...r, [field]: val } as DatasetRow) : r)) : rs));
  const delRow = (i: number) => setRows((rs) => (rs ? rs.filter((_, j) => j !== i) : rs));
  const addRow = () =>
    setRows((rs) => {
      const blank = {} as DatasetRow;
      for (const c of columns) blank[c] = "";
      blank.target = labels[0] ?? "";
      return [...(rs ?? []), blank];
    });

  function addColumn() {
    const name = window.prompt("New column name (extra columns are saved with your data, but not trained on):");
    const key = (name ?? "").trim();
    if (!key || columns.includes(key)) return;
    setColumns((c) => [...c, key]);
    setRows((rs) => (rs ? rs.map((r) => ({ ...r, [key]: r[key] ?? "" }) as DatasetRow) : rs));
  }

  function renameColumn(c: string) {
    if (LOCKED_COLS.includes(c)) return;
    const name = window.prompt("Rename column:", c);
    const key = (name ?? "").trim();
    if (!key || key === c || columns.includes(key)) return;
    setColumns((cols) => cols.map((x) => (x === c ? key : x)));
    setRows((rs) =>
      rs
        ? rs.map((r) => {
            const { [c]: v, ...rest } = r as Record<string, string>;
            return { ...rest, [key]: v ?? "" } as DatasetRow;
          })
        : rs,
    );
  }

  function delColumn(c: string) {
    if (LOCKED_COLS.includes(c)) return;
    setColumns((cols) => cols.filter((x) => x !== c));
    setRows((rs) =>
      rs
        ? rs.map((r) => {
            const rest = { ...(r as Record<string, string>) };
            delete rest[c];
            return rest as DatasetRow;
          })
        : rs,
    );
  }

  function exportCsv() {
    if (!rows) return;
    const esc = (s: string) => `"${(s ?? "").replace(/"/g, '""')}"`;
    const csv = [
      columns.map(esc).join(","),
      ...rows.map((r) => columns.map((c) => esc(r[c] ?? "")).join(",")),
    ].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "training_data.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function importCsv(file: File) {
    const text = await file.text();
    const parsed = parseCsv(text);
    if (parsed.rows.length === 0) {
      setErr("Couldn't read any rows — needs at least an input column.");
      return;
    }
    // If the grid already has rows, let the user REPLACE them with the file (clean
    // "bring your own data") or ADD to them. Empty grid -> just load the file.
    const hasRows = (rows?.length ?? 0) > 0;
    const replace =
      !hasRows ||
      window.confirm(
        `Import ${parsed.rows.length} rows from the file.\n\n` +
          `OK = replace all current rows with the file\nCancel = add them to the current rows`,
      );
    const fileCols = [...LOCKED_COLS, ...parsed.columns.filter((c) => !LOCKED_COLS.includes(c))];
    setColumns((cols) => (replace ? fileCols : [...cols, ...parsed.columns.filter((c) => !cols.includes(c))]));
    setRows((rs) => (replace ? parsed.rows : [...(rs ?? []), ...parsed.rows]));
    setErr(null);
    setNote(`${replace ? "Replaced with" : "Added"} ${parsed.rows.length} rows — review, then Save.`);
  }

  async function save() {
    if (!rows) return;
    setSaving(true);
    setErr(null);
    const res = await saveDataset(
      projectId,
      datasetId,
      rows.filter((r) => (r.input ?? "").trim()),
    );
    setSaving(false);
    if ("error" in res) {
      setErr(res.error);
      return;
    }
    onSaved(res.stats);
    onClose();
  }

  const wide = (c: string) => c === "input" || (c === "target" && !isClassification);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="flex max-h-[88vh] w-full max-w-4xl flex-col rounded-lg border bg-background shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b p-3">
          <div>
            <div className="text-sm font-medium">Edit training data</div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              {rows?.length ?? "…"} rows · {columns.length} columns · <b>input</b> &amp; <b>target</b> train the
              model — extra columns are kept, not trained
            </div>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="size-4" />
          </button>
        </div>

        <div className="flex-1 overflow-auto">
          {rows === null ? (
            <div className="p-3 font-mono text-xs text-muted-foreground">loading all rows…</div>
          ) : (
            <table className="w-full min-w-max border-collapse text-[13px]">
              <thead className="sticky top-0 z-10 bg-background shadow-[0_1px_0_0_var(--border)]">
                <tr>
                  <th className="w-10 border-b px-2 py-2 text-right font-mono text-[10px] font-normal text-muted-foreground">
                    #
                  </th>
                  {columns.map((c) => (
                    <th key={c} className={`border-b border-l px-2 py-1.5 text-left ${wide(c) ? "min-w-[280px]" : "min-w-[150px]"}`}>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs font-semibold">{c}</span>
                        {LOCKED_COLS.includes(c) ? (
                          <span className="rounded-full border px-1.5 py-px font-mono text-[9px] font-normal uppercase tracking-wide text-muted-foreground">
                            trains
                          </span>
                        ) : (
                          <span className="flex items-center gap-0.5">
                            <button
                              onClick={() => renameColumn(c)}
                              title="Rename column"
                              className="text-muted-foreground/60 transition-colors hover:text-foreground"
                            >
                              <Pencil className="size-3" />
                            </button>
                            <button
                              onClick={() => delColumn(c)}
                              title="Delete column"
                              className="text-muted-foreground/60 transition-colors hover:text-destructive"
                            >
                              <X className="size-3" />
                            </button>
                          </span>
                        )}
                      </div>
                    </th>
                  ))}
                  <th className="border-b border-l px-1.5 py-1.5">
                    <button
                      onClick={addColumn}
                      title="Add a column"
                      className="inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] font-normal uppercase tracking-wide text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      <Plus className="size-3" /> col
                    </button>
                  </th>
                  <th className="w-8 border-b" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="group hover:bg-muted/30">
                    <td className="border-b px-2 py-1.5 text-right align-top font-mono text-[10px] text-muted-foreground">
                      {i + 1}
                    </td>
                    {columns.map((c) => (
                      <td key={c} className="border-b border-l p-0 align-top">
                        {c === "target" && isClassification ? (
                          <input
                            list={`labels-${datasetId}`}
                            value={r[c] ?? ""}
                            onChange={(e) => update(i, c, e.target.value)}
                            className="h-8 w-full bg-transparent px-2 font-mono text-xs outline-none focus:bg-muted/60"
                            placeholder="label"
                          />
                        ) : (
                          <textarea
                            rows={1}
                            value={r[c] ?? ""}
                            onChange={(e) => update(i, c, e.target.value)}
                            className="block min-h-8 w-full resize-none bg-transparent px-2 py-1.5 leading-snug outline-none focus:bg-muted/60"
                            placeholder={c === "input" ? "input message" : c === "target" ? "ideal answer" : ""}
                          />
                        )}
                      </td>
                    ))}
                    <td className="border-b border-l" />
                    <td className="border-b px-1.5 align-top">
                      <button
                        onClick={() => delRow(i)}
                        aria-label="delete row"
                        className="pt-1.5 text-muted-foreground/0 transition-colors hover:!text-destructive group-hover:text-muted-foreground"
                      >
                        <Trash2 className="size-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <datalist id={`labels-${datasetId}`}>
            {labels.map((l) => (
              <option key={l} value={l} />
            ))}
          </datalist>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t p-3">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={addRow}
              className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Plus className="size-3" /> Add row
            </button>
            <button
              onClick={() => fileRef.current?.click()}
              title="Append rows from a CSV — all columns are kept; input/target-like headers map automatically"
              className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Upload className="size-3" /> Import CSV
            </button>
            <button
              onClick={exportCsv}
              className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Download className="size-3" /> Export CSV
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) importCsv(f);
                e.target.value = "";
              }}
            />
            {note && <span className="text-xs text-muted-foreground">{note}</span>}
          </div>
          <div className="flex items-center gap-2">
            {err && <span className="text-xs text-destructive">{err}</span>}
            <Button size="sm" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button size="sm" onClick={save} disabled={saving || !rows}>
              {saving ? "Saving…" : "Save changes"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Minimal RFC-4180-ish CSV parser (quoted fields, "" escapes, CRLF). Keeps EVERY
// column; maps input/target-like headers onto the locked training columns.
function parseCsv(text: string): { columns: string[]; rows: DatasetRow[] } {
  const grid: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQ = false;
  const endField = () => {
    row.push(field);
    field = "";
  };
  const endRow = () => {
    grid.push(row);
    row = [];
  };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else inQ = false;
      } else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ",") endField();
    else if (c === "\n") {
      endField();
      endRow();
    } else if (c !== "\r") field += c;
  }
  if (field.length > 0 || row.length > 0) {
    endField();
    endRow();
  }
  if (grid.length === 0) return { columns: [], rows: [] };

  const INPUT_NAMES = ["input", "text", "message", "question", "instruction", "prompt"];
  const TARGET_NAMES = ["target", "label", "output", "response", "answer", "completion"];
  const head = grid[0].map((h) => h.trim());
  const lower = head.map((h) => h.toLowerCase());
  const hasHeader = lower.some((h) => INPUT_NAMES.includes(h) || TARGET_NAMES.includes(h));

  let cols: string[];
  let start = 0;
  if (hasHeader) {
    cols = head.map((h, i) =>
      INPUT_NAMES.includes(lower[i]) ? "input" : TARGET_NAMES.includes(lower[i]) ? "target" : h || `col${i + 1}`,
    );
    start = 1;
  } else {
    cols = grid[0].map((_, i) => (i === 0 ? "input" : i === 1 ? "target" : `col${i + 1}`));
  }
  // de-dupe column names (e.g. two target-ish headers): first one wins the name
  const seen = new Set<string>();
  cols = cols.map((c) => {
    let n = c;
    let k = 2;
    while (seen.has(n)) n = `${c}_${k++}`;
    seen.add(n);
    return n;
  });
  if (!cols.includes("target")) cols = [...cols, "target"];

  const rows: DatasetRow[] = [];
  for (let r = start; r < grid.length; r++) {
    const rec = {} as DatasetRow;
    cols.forEach((c, i) => {
      rec[c] = (grid[r][i] ?? "").trim();
    });
    if ((rec.input ?? "").trim()) rows.push(rec);
  }
  return { columns: cols, rows };
}

// --------------------------------------------------------- projects panel
function fmtRunScore(v: number | null | undefined, task?: string): string {
  if (v == null) return "—";
  if (task === "classification" || v <= 1) return `${Math.round(v * 100)}%`;
  return `${v}`;
}

function ProjectsPanel({ onClose, onOpen }: { onClose: () => void; onOpen: (p: ProjectDetail) => void }) {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [detail, setDetail] = useState<Record<string, ProjectDetail>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [judging, setJudging] = useState<string | null>(null);
  const [judged, setJudged] = useState<Record<string, JudgeResult>>({});
  const [cmp, setCmp] = useState<Record<string, string>>({});
  const [rob, setRob] = useState<Record<string, RobustnessResult>>({});
  const [robBusy, setRobBusy] = useState<string | null>(null);
  const [pw, setPw] = useState<Record<string, PairwiseResult>>({});
  const [pwBusy, setPwBusy] = useState<string | null>(null);

  useEffect(() => {
    listProjects().then(setProjects);
  }, []);

  async function compareLatest(d: ProjectDetail) {
    const runs = d.runs ?? [];
    if (runs.length < 2) return;
    const a = runs[runs.length - 2].run_name;
    const b = runs[runs.length - 1].run_name;
    setCmp((m) => ({ ...m, [d.id]: "comparing…" }));
    const res = await compareRuns(a, b);
    const metric = (side?: { report?: { after?: { accuracy?: number | null; avg_judge_score?: number | null } } | null }) => {
      const af = side?.report?.after;
      if (!af) return "—";
      if (af.accuracy != null) return `${Math.round(af.accuracy * 100)}%`;
      return af.avg_judge_score != null ? `${af.avg_judge_score.toFixed(1)}/10` : "unscored";
    };
    if (!res) {
      setCmp((m) => ({ ...m, [d.id]: "couldn't compare" }));
      return;
    }
    let text = `${a} → ${metric(res.a)}   vs   ${b} → ${metric(res.b)}`;
    if (res.comparable === false) {
      text += " · ⚠ not comparable (different eval sets)";
    } else if (res.mcnemar?.p_value != null) {
      const mp = res.mcnemar.p_value;
      text += ` · McNemar p=${mp < 0.001 ? "<0.001" : mp.toFixed(3)}${mp < 0.05 ? " ✓ significant" : " · not significant"}`;
    }
    setCmp((m) => ({ ...m, [d.id]: text }));
  }

  async function runRob(d: ProjectDetail) {
    setRobBusy(d.id);
    const res = await runRobustness(d.id);
    setRob((m) => ({ ...m, [d.id]: res }));
    setRobBusy(null);
  }

  async function runPw(d: ProjectDetail) {
    if (!d.active_run) return;
    setPwBusy(d.id);
    const res = await pairwiseJudge(d.active_run);
    setPw((m) => ({ ...m, [d.id]: res }));
    setPwBusy(null);
  }

  async function toggle(id: string) {
    if (expanded === id) {
      setExpanded(null);
      return;
    }
    setExpanded(id);
    if (!detail[id]) {
      const d = await getProjectDetail(id);
      if (d) setDetail((m) => ({ ...m, [id]: d }));
    }
  }

  async function score(runName: string) {
    setJudging(runName);
    const res = await judgeRun(runName);
    setJudged((m) => ({ ...m, [runName]: res }));
    setJudging(null);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-lg border bg-background shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b p-3">
          <div>
            <div className="text-sm font-medium">Your projects</div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              reopen a project · see versions · score old runs
            </div>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="size-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          {projects === null ? (
            <div className="font-mono text-xs text-muted-foreground">loading…</div>
          ) : projects.length === 0 ? (
            <div className="text-sm text-muted-foreground">No projects yet — describe one in the chat to start.</div>
          ) : (
            <div className="space-y-2">
              {projects.map((p) => {
                const d = detail[p.id];
                const open = expanded === p.id;
                return (
                  <div key={p.id} className="rounded-lg border">
                    <button onClick={() => toggle(p.id)} className="flex w-full items-center gap-2 p-3 text-left">
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{p.name}</span>
                        <span className="block font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                          {p.task_type} · {p.status}
                          {p.updated_at ? ` · ${p.updated_at.slice(0, 10)}` : ""}
                        </span>
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">{open ? "▾" : "▸"}</span>
                    </button>
                    {open && (
                      <div className="border-t p-3">
                        {!d ? (
                          <div className="font-mono text-xs text-muted-foreground">loading…</div>
                        ) : (
                          <>
                            {d.goal && <div className="mb-2 text-xs text-muted-foreground">{d.goal}</div>}
                            {(d.runs ?? []).length > 0 && (
                              <div className="mb-2 space-y-1">
                                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                                  Versions
                                </div>
                                {(d.runs ?? []).map((r) => {
                                  const j = judged[r.run_name];
                                  return (
                                    <div key={r.run_name} className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
                                      <span className="font-mono">v{r.version ?? "?"}</span>
                                      <span className="font-mono text-muted-foreground">{r.run_name}</span>
                                      <span className="text-muted-foreground">
                                        {fmtRunScore(r.before, r.task)} → {fmtRunScore(r.after, r.task)}
                                      </span>
                                      {r.task === "classification" && r.after_ci && r.after_ci[0] != null && (
                                        <span className="font-mono text-[10px] text-muted-foreground">
                                          ±CI {Math.round((r.after_ci[0] as number) * 100)}–
                                          {Math.round((r.after_ci[1] as number) * 100)}%{r.n ? ` n=${r.n}` : ""}
                                        </span>
                                      )}
                                      {r.task !== "classification" && (
                                        <button
                                          onClick={() => score(r.run_name)}
                                          disabled={judging === r.run_name}
                                          className="rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                                        >
                                          {judging === r.run_name ? "scoring…" : "score replies"}
                                        </button>
                                      )}
                                      {j &&
                                        (j.error ? (
                                          <span className="text-destructive">{j.error}</span>
                                        ) : (
                                          <span>
                                            judged: <b>{j.before_avg ?? "—"}</b> → <b>{j.after_avg ?? "—"}</b> /10 (n=
                                            {j.n})
                                            {j.after_ci && j.after_ci[0] != null && (
                                              <span className="ml-1 font-mono text-[10px] text-muted-foreground">
                                                ±CI {(j.after_ci[0] as number).toFixed(1)}–{(j.after_ci[1] as number).toFixed(1)}
                                              </span>
                                            )}
                                          </span>
                                        ))}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                            <div className="mt-2 flex flex-wrap items-center gap-2">
                              <Button size="sm" onClick={() => onOpen(d)}>
                                Open in chat
                              </Button>
                              {(d.runs ?? []).length >= 2 && (
                                <Button size="sm" variant="outline" onClick={() => compareLatest(d)}>
                                  Compare last two
                                </Button>
                              )}
                              {d.active_run && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => runRob(d)}
                                  disabled={robBusy === d.id}
                                >
                                  {robBusy === d.id ? "checking…" : "Robustness check"}
                                </Button>
                              )}
                              {d.task_type === "generation" && d.active_run && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => runPw(d)}
                                  disabled={pwBusy === d.id}
                                >
                                  {pwBusy === d.id ? "judging…" : "Pairwise judge"}
                                </Button>
                              )}
                              {judging && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
                            </div>
                            {cmp[d.id] && (
                              <div className="mt-1.5 font-mono text-[11px] text-muted-foreground">{cmp[d.id]}</div>
                            )}
                            {rob[d.id] &&
                              (rob[d.id].error ? (
                                <div className="mt-1.5 font-mono text-[11px] text-destructive">{rob[d.id].error}</div>
                              ) : (
                                <div className="mt-1.5 rounded border p-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
                                  robustness: {Math.round((rob[d.id].pass_rate ?? 0) * 100)}% pass ·{" "}
                                  {rob[d.id].n_cases} perturbations · {rob[d.id].severity_1} confidently-wrong · ECE{" "}
                                  {rob[d.id].calibration?.ece != null
                                    ? (rob[d.id].calibration!.ece as number).toFixed(3)
                                    : "—"}{" "}
                                  · bank {rob[d.id].bank_size} (+{rob[d.id].newly_banked} new, {rob[d.id].bank_regressed}{" "}
                                  regressed)
                                </div>
                              ))}
                            {pw[d.id] &&
                              (pw[d.id].error ? (
                                <div className="mt-1.5 font-mono text-[11px] text-destructive">{pw[d.id].error}</div>
                              ) : (
                                <div className="mt-1.5 rounded border p-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
                                  pairwise tuned vs base: win-rate{" "}
                                  {pw[d.id].vs_base?.a_win_rate != null
                                    ? `${Math.round(pw[d.id].vs_base!.a_win_rate! * 100)}%`
                                    : "—"}
                                  {pw[d.id].vs_base?.lo != null
                                    ? ` (CI ${Math.round(pw[d.id].vs_base!.lo! * 100)}–${Math.round(
                                        pw[d.id].vs_base!.hi! * 100,
                                      )}%)`
                                    : ""}{" "}
                                  · swap-consistency{" "}
                                  {pw[d.id].vs_base?.consistency != null
                                    ? `${Math.round(pw[d.id].vs_base!.consistency! * 100)}%`
                                    : "—"}
                                  {pw[d.id].vs_base?.verbosity_warning ? " · ⚠ verbosity" : ""} · anchors{" "}
                                  {pw[d.id].anchors?.great_wins}/{pw[d.id].anchors?.n}
                                  {pw[d.id].anchors?.unstable ? " ⚠ unstable" : " ✓"}
                                </div>
                              ))}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
