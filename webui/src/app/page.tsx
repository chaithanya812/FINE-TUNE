import Link from "next/link";
import {
  Split,
  MessageCircleHeart,
  Braces,
  Feather,
  Sigma,
  Table2,
  Activity,
  ShieldCheck,
  Scale,
  NotebookPen,
  GitBranch,
  Settings2,
  Star,
  ArrowRight,
  Terminal,
} from "lucide-react";

// Fine-Tune Studio landing. Identity: monochrome + emerald marker, terminal
// accents, bento cards — the marketing face of the black/white chat app at /chat.

const USE_CASES = [
  { icon: Split, t: "router", title: "Router / Classifier", blurb: "Sort every message into your buckets — intents, topics, moderation." },
  { icon: MessageCircleHeart, t: "assistant", title: "Assistant in your voice", blurb: "Free-form replies with your tone, format, and do's & don'ts." },
  { icon: Braces, t: "extractor", title: "Data extractor", blurb: "Messy emails and notes in, clean structured JSON out." },
  { icon: Feather, t: "rewriter", title: "Tone rewriter", blurb: "Same meaning, your house style — blunt → polite, formal → human." },
  { icon: Sigma, t: "reasoner", title: "Domain reasoner", blurb: "Numbers, tables, multi-step logic over your domain." },
];

const CAPS = [
  { icon: Table2, label: "Spreadsheet data editor" },
  { icon: Activity, label: "Live training telemetry" },
  { icon: ShieldCheck, label: "Adversarial auto-test" },
  { icon: Scale, label: "LLM-as-judge scoring" },
  { icon: NotebookPen, label: "One-click Colab export" },
  { icon: GitBranch, label: "Version history" },
  { icon: Settings2, label: "Teacher-model admin" },
];

export default function Landing() {
  return (
    <div className="flex flex-1 flex-col bg-background">
      {/* ---------------------------------------------------------- nav */}
      <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Link href="/" className="flex items-center gap-2">
            <div className="h-5 w-5 rounded-sm bg-foreground" />
            <span className="text-sm font-semibold tracking-tight">Fine-Tune Studio</span>
          </Link>
          <nav className="hidden items-center gap-6 text-sm text-muted-foreground sm:flex">
            <a href="#build" className="transition-colors hover:text-foreground">What you can build</a>
            <a href="#why" className="transition-colors hover:text-foreground">Why specialists</a>
            <a href="#how" className="transition-colors hover:text-foreground">How it works</a>
            <a
              href="https://github.com/chaithanya812/FINE-TUNE"
              target="_blank"
              rel="noreferrer"
              className="transition-colors hover:text-foreground"
            >
              GitHub
            </a>
          </nav>
          <div className="flex items-center gap-3">
            <Link href="/admin" className="hidden text-sm text-muted-foreground transition-colors hover:text-foreground sm:block">
              Admin
            </Link>
            <Link
              href="/chat"
              className="rounded-md bg-foreground px-3.5 py-2 text-sm font-medium text-background transition-opacity hover:opacity-85"
            >
              Start building
            </Link>
          </div>
        </div>
      </header>

      {/* ---------------------------------------------------------- hero */}
      <section className="mx-auto w-full max-w-5xl px-4 pb-20 pt-16 text-center sm:pt-24">
        <h1 className="pop-in text-4xl font-semibold leading-[1.1] tracking-tight sm:text-6xl">
          Your own AI model,
          <br />
          trained on <span className="hl-swipe">your laptop</span>.
        </h1>
        <p className="pop-in mx-auto mt-5 max-w-xl text-base text-muted-foreground sm:text-lg" style={{ animationDelay: "120ms" }}>
          Describe the model you want in plain English. An agent builds the data with you, fine-tunes a small open
          model on your GPU, and <span className="font-medium text-foreground">proves it got better</span> on data it
          never saw.
        </p>

        <div className="pop-in mt-9 flex flex-wrap items-center justify-center gap-3" style={{ animationDelay: "220ms" }}>
          <Link
            href="/chat"
            className="rounded-md bg-foreground px-6 py-3 text-sm font-semibold text-background shadow-sm transition-opacity hover:opacity-85"
          >
            Start now — it&apos;s free
          </Link>
          <a
            href="#how"
            className="rounded-md border px-6 py-3 text-sm font-medium text-foreground/80 transition-colors hover:bg-muted"
          >
            See how it works
          </a>
        </div>

        {/* terminal chip — our version of a price tag */}
        <div className="pop-in mx-auto mt-10 max-w-md rounded-lg border bg-muted/40 px-4 py-3 text-left font-mono text-xs text-muted-foreground" style={{ animationDelay: "320ms" }}>
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-widest">
            <Terminal className="size-3" /> the deal
          </div>
          <div>cost: <span className="text-foreground">₹0 / month</span> · hardware: <span className="text-foreground">your&nbsp;gpu</span> (4&nbsp;GB is enough)</div>
          <div>privacy: <span className="text-foreground">100% local</span> · bigger models: <span className="text-foreground">free Colab T4</span></div>
        </div>
      </section>

      {/* ---------------------------------------------------------- build */}
      <section id="build" className="border-t bg-muted/40 py-16">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            One small model for every job<span className="text-emerald-500">.</span>
          </h2>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Pick the shape of the thing you need — each card drops you into the chat with the right plan.
          </p>

          <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {USE_CASES.map((u, i) => {
              const Icon = u.icon;
              return (
                <Link
                  key={u.t}
                  href={`/chat?t=${u.t}`}
                  className="group pop-in rounded-xl border bg-background p-5 transition-all hover:-translate-y-0.5 hover:shadow-md"
                  style={{ animationDelay: `${i * 70}ms` }}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg border bg-muted/60">
                      <Icon className="size-4.5 text-emerald-600 dark:text-emerald-400" />
                    </div>
                    <span className="font-medium">{u.title}</span>
                  </div>
                  <p className="mt-2.5 text-sm text-muted-foreground">{u.blurb}</p>
                  <span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-emerald-600 opacity-0 transition-opacity group-hover:opacity-100 dark:text-emerald-400">
                    build this <ArrowRight className="size-3" />
                  </span>
                </Link>
              );
            })}
            <div className="pop-in flex items-center justify-center rounded-xl border border-dashed p-5 text-center text-sm text-muted-foreground" style={{ animationDelay: "350ms" }}>
              Something else?
              <Link href="/chat" className="ml-1.5 font-medium text-foreground underline-offset-2 hover:underline">
                Just describe it →
              </Link>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {CAPS.map((c) => {
              const Icon = c.icon;
              return (
                <span key={c.label} className="inline-flex items-center gap-1.5 rounded-full border bg-background px-3 py-1.5 text-xs text-muted-foreground">
                  <Icon className="size-3.5" /> {c.label}
                </span>
              );
            })}
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- why */}
      <section id="why" className="mx-auto w-full max-w-5xl px-4 py-20">
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Why teams train <span className="ul-swipe">specialists</span>
        </h2>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Frontier APIs are brilliant generalists. But routing tickets, extracting invoices, or answering on-brand
          doesn&apos;t need a genius — it needs a cheap, fast, private model that nails one task.
        </p>
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <div className="pop-in rounded-xl border p-6">
            <div className="text-4xl font-semibold tabular-nums">+67<span className="text-xl text-muted-foreground">pts</span></div>
            <div className="mt-2 text-sm text-muted-foreground">
              Measured here: held-out intent accuracy <b className="text-foreground">22% → 89%</b> after one ~15-minute
              QLoRA run on a 4&nbsp;GB laptop GPU.
            </div>
          </div>
          <div className="pop-in rounded-xl border p-6" style={{ animationDelay: "100ms" }}>
            <div className="text-4xl font-semibold tabular-nums">~1<span className="text-xl text-muted-foreground">%</span></div>
            <div className="mt-2 text-sm text-muted-foreground">
              Rough serving cost vs a frontier API — and tuned small models keep matching far larger generalists on the
              narrow task they were trained for.
            </div>
          </div>
          <div className="pop-in rounded-xl border p-6" style={{ animationDelay: "200ms" }}>
            <div className="text-4xl font-semibold tabular-nums">100<span className="text-xl text-muted-foreground">%</span></div>
            <div className="mt-2 text-sm text-muted-foreground">
              Private. Training and inference never leave the machine — support logs, contracts, and customer data stay
              yours.
            </div>
          </div>
        </div>
        <blockquote className="mx-auto mt-10 max-w-xl border-l-2 border-emerald-500 pl-4 text-sm italic text-muted-foreground">
          &ldquo;A specialist beats a generalist — at the one job you actually need done.&rdquo;
          <span className="mt-1 block not-italic font-mono text-[11px]">— the whole point of fine-tuning</span>
        </blockquote>
        <p className="mt-8 text-center text-xs text-muted-foreground">
          Honesty built in: fine-tuning teaches <b>style, format, and behaviour — not facts</b>. Every run must beat the
          base model <i>and</i> a good prompt on held-out data, or we tell you not to bother.
        </p>
      </section>

      {/* ---------------------------------------------------------- how */}
      <section id="how" className="border-t bg-muted/40 py-20">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Proof, not vibes<span className="text-emerald-500">.</span>
          </h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-4">
            {[
              ["01", "Describe", "Say what you want in plain English. The agent clarifies route vs answer vs extract vs rewrite — before building anything."],
              ["02", "Own the data", "A teacher AI drafts hundreds of examples. You see every row in a spreadsheet — fix, delete, add columns, import CSV."],
              ["03", "Watch it train", "QLoRA on your GPU with live loss, ETA and VRAM. Stop, resume, or export a Colab notebook for a 3–7B model."],
              ["04", "Get proof", "Base vs good-prompt vs fine-tuned on held-out data, per-topic breakdowns, adversarial tests, an LLM judge for open replies."],
            ].map(([n, t, d], i) => (
              <div key={n} className="pop-in rounded-xl border bg-background p-5" style={{ animationDelay: `${i * 90}ms` }}>
                <div className="font-mono text-xs text-emerald-600 dark:text-emerald-400">{n}</div>
                <div className="mt-1 font-semibold">{t}</div>
                <p className="mt-2 text-sm text-muted-foreground">{d}</p>
              </div>
            ))}
          </div>
          <div className="mt-8 flex items-center justify-center gap-3 font-mono text-sm">
            <span className="rounded-md border bg-background px-3 py-1 text-muted-foreground">before&nbsp;22%</span>
            <ArrowRight className="size-4 text-muted-foreground" />
            <span className="rounded-md border border-emerald-500/50 bg-background px-3 py-1 font-semibold">after&nbsp;89%</span>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- guarantees */}
      <section className="mx-auto w-full max-w-5xl px-4 py-20">
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">The boring guarantees</h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {[
            ["Open source", "The whole studio is on GitHub — engine, agent, UI. Read it, fork it, point at it in an interview.", true],
            ["No lock-in", "Plain files everywhere: downloadable adapters, GGUF export for LM Studio/Ollama, datasets as JSONL/CSV.", false],
            ["Honest evals", "Held-out test sets, per-class breakdowns, a good-prompt baseline, regressions shown — never just a happy number.", false],
            ["Free means free", "₹0 locally on your GPU. Bigger models ride Google Colab's free T4 via a generated Unsloth notebook.", false],
          ].map(([t, d, g]) => (
            <div key={t as string} className="rounded-xl border p-6">
              <div className="text-base font-semibold">{t}</div>
              <p className="mt-2 text-sm text-muted-foreground">{d}</p>
              {g === true && (
                <a
                  href="https://github.com/chaithanya812/FINE-TUNE"
                  target="_blank"
                  rel="noreferrer"
                  className="mt-4 inline-flex items-center gap-2 rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-85"
                >
                  <Star className="size-4" /> Star on GitHub
                </a>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ---------------------------------------------------------- final CTA */}
      <section className="border-t bg-muted/40 py-16 text-center">
        <h2 className="text-3xl font-semibold tracking-tight">
          Ready to train <span className="hl-swipe">yours</span>?
        </h2>
        <Link
          href="/chat"
          className="mt-6 inline-block rounded-md bg-foreground px-7 py-3 text-sm font-semibold text-background transition-opacity hover:opacity-85"
        >
          Start now — it&apos;s free
        </Link>
        <p className="mt-3 font-mono text-[11px] text-muted-foreground">
          needs a free Gemini API key · trains on a 4 GB GPU, or free Colab for bigger models
        </p>
      </section>

      <footer className="border-t py-6 text-center font-mono text-[11px] text-muted-foreground">
        Fine-Tune Studio · local-first fine-tuning ·{" "}
        <a className="underline-offset-2 hover:underline" href="https://github.com/chaithanya812/FINE-TUNE" target="_blank" rel="noreferrer">
          github.com/chaithanya812/FINE-TUNE
        </a>
      </footer>
    </div>
  );
}
