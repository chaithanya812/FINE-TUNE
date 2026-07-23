"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Check, Loader2, Sparkles, TriangleAlert } from "lucide-react";
import { getGeminiModels, getSettings, updateSettings } from "@/lib/api";
import type { GeminiModel, Settings } from "@/lib/api";

// Plain-language hints for the models a non-technical user is most likely to pick.
// The 429 that sent the user here came from gemini-2.5-flash's 20/day cap — say so.
const HINTS: Record<string, string> = {
  "gemini-3.1-flash-lite": "Newest lite model — fast, with a generous free tier.",
  "gemini-flash-lite-latest": "Always tracks Google's latest lite model.",
  "gemini-2.5-flash-lite": "Reliable and cheap, with lots of free-tier headroom.",
  "gemini-2.0-flash-lite": "Older, but very high free-tier limits.",
  "gemini-2.0-flash": "Older flagship Flash — solid quality.",
  "gemini-2.5-flash": "Capable, but a small free tier (20 requests/day) — this is the one that hit the limit.",
};

export default function AdminPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [models, setModels] = useState<GeminiModel[]>([]);
  const [source, setSource] = useState<"live" | "fallback" | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    Promise.all([getSettings(), getGeminiModels()]).then(([s, list]) => {
      setSettings(s);
      setModels(list.models ?? []);
      setSource(list.source ?? null);
      setListError(list.error ?? null);
      setLoading(false);
    });
  }, []);

  async function choose(id: string) {
    if (saving || id === settings?.gemini_model) return;
    setSaving(id);
    try {
      const next = await updateSettings(id);
      setSettings(next);
      setSavedId(id);
      setTimeout(() => setSavedId((cur) => (cur === id ? null : cur)), 1600);
    } catch {
      /* keep the old selection; nothing persisted */
    } finally {
      setSaving(null);
    }
  }

  const current = settings?.gemini_model ?? null;
  const recommended = models.filter((m) => m.recommended);
  const others = models.filter((m) => !m.recommended);
  // Make sure the active model is always visible even if it's not "recommended".
  const activeInRecommended = recommended.some((m) => m.id === current);

  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b">
        <div className="mx-auto flex max-w-2xl items-center justify-between px-4 py-4">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="size-3.5" />
            Fine-Tune Studio
          </Link>
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            admin · teacher model
          </span>
        </div>
      </header>

      <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-8">
        <h1 className="text-2xl font-semibold tracking-tight">Teacher model</h1>
        <p className="mt-2 max-w-xl text-sm text-muted-foreground">
          This is the Gemini model that plans your project, writes training examples, and grades the
          result. It&apos;s <b>not</b> the model you&apos;re training — it&apos;s the &quot;teacher&quot; that
          builds the lesson. Pick one with a bigger free tier if you keep hitting quota limits.
        </p>

        {settings && !settings.key_set && (
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-500">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" />
            <span>
              No <code className="font-mono text-xs">GEMINI_API_KEY</code> found. Add it to{" "}
              <code className="font-mono text-xs">.env</code> and restart the backend, or the teacher
              can&apos;t run.
            </span>
          </div>
        )}

        {loading ? (
          <div className="mt-8 flex items-center gap-2 font-mono text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" /> loading models…
          </div>
        ) : (
          <>
            <div className="mt-6 rounded-lg border bg-muted/30 p-4">
              <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                Currently using
              </div>
              <div className="mt-1 font-mono text-lg font-semibold">{current ?? "—"}</div>
              {current && HINTS[current] && (
                <div className="mt-1 text-xs text-muted-foreground">{HINTS[current]}</div>
              )}
            </div>

            <div className="mt-6">
              <div className="mb-2 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                <Sparkles className="size-3" /> Recommended
              </div>
              <div className="space-y-2">
                {recommended.map((m) => (
                  <ModelRow
                    key={m.id}
                    model={m}
                    active={m.id === current}
                    saving={saving === m.id}
                    saved={savedId === m.id}
                    onClick={() => choose(m.id)}
                  />
                ))}
              </div>
            </div>

            {others.length > 0 && (
              <div className="mt-6">
                <button
                  onClick={() => setShowAll((v) => !v)}
                  className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground transition-colors hover:text-foreground"
                >
                  {showAll ? "▾ hide" : "▸ show"} all {models.length} available models
                </button>
                {showAll && (
                  <div className="mt-3 space-y-2">
                    {!activeInRecommended && current && (
                      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                        Active
                      </div>
                    )}
                    {others.map((m) => (
                      <ModelRow
                        key={m.id}
                        model={m}
                        active={m.id === current}
                        saving={saving === m.id}
                        saved={savedId === m.id}
                        onClick={() => choose(m.id)}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}

            <p className="mt-6 font-mono text-[10px] text-muted-foreground">
              {source === "live"
                ? "list fetched live from your Gemini key"
                : "showing a fallback list — couldn't fetch live models"}
              {listError ? ` · ${listError}` : ""}
            </p>
            <p className="mt-1 font-mono text-[10px] text-muted-foreground">
              changes apply immediately — no restart needed
            </p>
          </>
        )}
      </main>
    </div>
  );
}

function ModelRow({
  model,
  active,
  saving,
  saved,
  onClick,
}: {
  model: GeminiModel;
  active: boolean;
  saving: boolean;
  saved: boolean;
  onClick: () => void;
}) {
  const hint = HINTS[model.id] ?? model.description;
  return (
    <button
      onClick={onClick}
      disabled={active || saving}
      className={`flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors ${
        active ? "border-foreground/40 bg-muted/40" : "hover:bg-muted disabled:opacity-100"
      } disabled:cursor-default`}
    >
      <span
        className={`flex size-4 shrink-0 items-center justify-center rounded-full border ${
          active ? "border-foreground bg-foreground text-background" : "border-muted-foreground/40"
        }`}
      >
        {active && <Check className="size-3" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-sm font-medium">{model.id}</span>
        {hint && <span className="mt-0.5 block truncate text-xs text-muted-foreground">{hint}</span>}
      </span>
      {saving ? (
        <Loader2 className="size-3.5 shrink-0 animate-spin text-muted-foreground" />
      ) : saved ? (
        <span className="shrink-0 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          saved
        </span>
      ) : active ? (
        <span className="shrink-0 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          active
        </span>
      ) : null}
    </button>
  );
}
