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
} from "lucide-react";

// Odoo-inspired marketing landing. The actual app lives at /chat.
const PLUM = "#714B67";

const TILES = [
  { icon: Split, label: "Router", color: "text-rose-500", t: "router" },
  { icon: MessageCircleHeart, label: "Assistant", color: "text-purple-500", t: "assistant" },
  { icon: Braces, label: "Extractor", color: "text-emerald-500", t: "extractor" },
  { icon: Feather, label: "Rewriter", color: "text-sky-500", t: "rewriter" },
  { icon: Sigma, label: "Reasoner", color: "text-amber-500", t: "reasoner" },
  { icon: Table2, label: "Data editor", color: "text-teal-500" },
  { icon: Activity, label: "Live training", color: "text-orange-500" },
  { icon: ShieldCheck, label: "Auto-test", color: "text-green-600" },
  { icon: Scale, label: "LLM judge", color: "text-indigo-500" },
  { icon: NotebookPen, label: "Colab export", color: "text-yellow-600" },
  { icon: GitBranch, label: "Versions", color: "text-fuchsia-500" },
  { icon: Settings2, label: "Admin", color: "text-slate-500" },
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
            <a href="#apps" className="transition-colors hover:text-foreground">Apps</a>
            <a href="#why" className="transition-colors hover:text-foreground">Why fine-tune</a>
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
              className="rounded-md px-3.5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90"
              style={{ backgroundColor: PLUM }}
            >
              Try it free
            </Link>
          </div>
        </div>
      </header>

      {/* ---------------------------------------------------------- hero */}
      <section className="relative overflow-hidden">
        <div className="mx-auto max-w-5xl px-4 pb-24 pt-16 text-center sm:pt-24">
          <h1 className="font-hand pop-in text-5xl font-bold leading-tight tracking-tight sm:text-7xl">
            Your own AI model, trained on <span className="hl-swipe">your laptop</span>.
          </h1>
          <p className="font-hand pop-in mt-3 text-3xl text-foreground/90 sm:text-4xl" style={{ animationDelay: "120ms" }}>
            Simple, private, yet <span className="ul-swipe">free</span>!
          </p>

          <div className="pop-in mt-9 flex flex-wrap items-center justify-center gap-3" style={{ animationDelay: "220ms" }}>
            <Link
              href="/chat"
              className="rounded-md px-6 py-3 text-sm font-semibold text-white shadow-sm transition-opacity hover:opacity-90"
              style={{ backgroundColor: PLUM }}
            >
              Start now — It&apos;s free
            </Link>
            <a
              href="#how"
              className="rounded-md border bg-muted/50 px-6 py-3 text-sm font-medium text-foreground/80 transition-colors hover:bg-muted"
            >
              See how it works
            </a>
          </div>

          {/* handwritten annotation + arrow, Odoo-style */}
          <div className="pop-in mx-auto mt-6 flex max-w-xl items-start justify-end gap-1 pr-2" style={{ animationDelay: "320ms" }}>
            <svg viewBox="0 0 60 40" className="mt-1 h-8 w-10 -scale-x-100 text-foreground/70" fill="none">
              <path d="M55 4 C 30 8, 14 16, 8 34" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              <path d="M4 24 L 8 35 L 18 32" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
            </svg>
            <div className="font-hand rotate-[-3deg] text-2xl leading-tight text-foreground/85">
              ₹0 / month —<br />runs on YOUR gpu
            </div>
          </div>
        </div>
        {/* curved divider */}
        <div className="relative h-20">
          <div className="absolute left-1/2 top-0 h-[600px] w-[160%] -translate-x-1/2 rounded-[100%] bg-muted/60" />
        </div>
      </section>

      {/* ---------------------------------------------------------- app grid */}
      <section id="apps" className="bg-muted/60 pb-16 pt-4">
        <div className="mx-auto max-w-4xl px-4">
          <div className="grid grid-cols-3 gap-x-4 gap-y-8 sm:grid-cols-6">
            {TILES.map((tile, i) => {
              const Icon = tile.icon;
              const card = (
                <>
                  <div className="float-slow mx-auto flex h-16 w-16 items-center justify-center rounded-xl border bg-background shadow-sm transition-transform group-hover:scale-105"
                    style={{ animationDelay: `${(i % 6) * 350}ms` }}
                  >
                    <Icon className={`size-7 ${tile.color}`} />
                  </div>
                  <div className="mt-2 text-center text-xs font-medium">{tile.label}</div>
                </>
              );
              return tile.t ? (
                <Link key={tile.label} href={`/chat?t=${tile.t}`} className="group pop-in" style={{ animationDelay: `${i * 60}ms` }}>
                  {card}
                </Link>
              ) : (
                <Link key={tile.label} href="/chat" className="group pop-in" style={{ animationDelay: `${i * 60}ms` }}>
                  {card}
                </Link>
              );
            })}
          </div>

          <div className="mt-12 text-center">
            <p className="text-lg font-semibold">Imagine one small model for every job.</p>
            <p className="mt-1.5 text-sm text-muted-foreground">
              Got a repetitive task? There&apos;s a specialist for that. No ML degree, no cloud bill — describe it in chat
              and train it in minutes.
            </p>
            <Link href="/chat" className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium" style={{ color: PLUM }}>
              Open the studio <ArrowRight className="size-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- quote */}
      <section className="mx-auto w-full max-w-3xl px-4 py-14">
        <div className="relative mx-auto max-w-xl rounded-xl border bg-background p-5 shadow-sm">
          <div className="absolute -left-2 -top-3 rotate-[-6deg] rounded bg-amber-400/90 px-6 py-2" aria-hidden />
          <p className="relative text-sm italic">&ldquo;A specialist beats a generalist — at the one job you actually need done.&rdquo;</p>
          <p className="relative mt-1 text-xs text-muted-foreground">— the whole point of fine-tuning</p>
        </div>
      </section>

      {/* ---------------------------------------------------------- why */}
      <section id="why" className="mx-auto w-full max-w-5xl px-4 pb-20">
        <h2 className="font-hand text-center text-4xl font-bold sm:text-5xl">
          <span className="hl-swipe">Level up</span> your models
        </h2>
        <p className="mx-auto mt-3 max-w-2xl text-center text-sm text-muted-foreground">
          Why teams fine-tune: frontier APIs are brilliant generalists — but routing tickets, extracting invoices, or
          answering on-brand doesn&apos;t need a genius. It needs a cheap, fast, private specialist that nails one task.
        </p>
        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          <div className="pop-in rounded-xl border p-6 text-center">
            <div className="text-4xl font-bold tabular-nums">+67<span className="text-xl">pts</span></div>
            <div className="mt-2 text-sm text-muted-foreground">
              Measured on this project: held-out intent accuracy went <b>22% → 89%</b> after one ~15-minute QLoRA run on a
              4&nbsp;GB laptop GPU.
            </div>
          </div>
          <div className="pop-in rounded-xl border p-6 text-center" style={{ animationDelay: "100ms" }}>
            <div className="text-4xl font-bold tabular-nums">~1<span className="text-xl">%</span></div>
            <div className="mt-2 text-sm text-muted-foreground">
              Rough serving cost of a small specialist vs a frontier API — and tuned single-digit-B models keep matching
              far larger generalists <i>on the narrow task they were trained for</i>.
            </div>
          </div>
          <div className="pop-in rounded-xl border p-6 text-center" style={{ animationDelay: "200ms" }}>
            <div className="text-4xl font-bold tabular-nums">100<span className="text-xl">%</span></div>
            <div className="mt-2 text-sm text-muted-foreground">
              Private. Training and inference never leave the machine — your support logs, contracts, and customer data
              stay yours.
            </div>
          </div>
        </div>
        <p className="mt-6 text-center text-xs text-muted-foreground">
          Honesty built in: fine-tuning teaches <b>style, format, and behaviour — not facts</b>. Every run must beat the
          base model <i>and</i> a good prompt on held-out data, or we tell you not to bother.
        </p>
      </section>

      {/* ---------------------------------------------------------- how */}
      <section id="how" className="border-t bg-muted/40 py-20">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="font-hand text-center text-4xl font-bold sm:text-5xl">
            Optimized for <span className="ul-swipe">proof</span>
          </h2>
          <div className="mt-10 grid gap-4 sm:grid-cols-4">
            {[
              ["1 · Describe", "Tell the agent what you want in plain English. It clarifies whether you need to route, answer, extract, or rewrite — before building anything."],
              ["2 · Own the data", "A teacher AI drafts hundreds of training examples. You see every row in a spreadsheet editor — fix, delete, add your own, import CSV."],
              ["3 · Watch it train", "QLoRA on your GPU with live loss, ETA and VRAM. Stop, resume, or export a free Colab notebook for a 3–7B model."],
              ["4 · Get proof", "Base vs good-prompt vs fine-tuned on held-out data, per-topic breakdowns, adversarial auto-tests, and an LLM judge for open-ended replies."],
            ].map(([t, d], i) => (
              <div key={t} className="pop-in rounded-xl border bg-background p-5" style={{ animationDelay: `${i * 90}ms` }}>
                <div className="font-hand text-2xl font-semibold">{t}</div>
                <p className="mt-2 text-sm text-muted-foreground">{d}</p>
              </div>
            ))}
          </div>
          <div className="mt-8 flex items-center justify-center gap-4 font-mono text-sm">
            <span className="rounded-full border bg-background px-3 py-1 text-muted-foreground">before 22%</span>
            <ArrowRight className="size-4 text-muted-foreground" />
            <span className="rounded-full border bg-background px-3 py-1 font-semibold">after 89%</span>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- done right */}
      <section className="mx-auto w-full max-w-5xl px-4 py-20">
        <h2 className="font-hand text-4xl font-bold sm:text-5xl">
          Fine-tuning <span className="ul-swipe">done right</span>.
        </h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {[
            ["Open source", "The whole studio is on GitHub — engine, agent, UI. Read it, fork it, point at it in an interview.", "github"],
            ["No lock-in", "Plain files everywhere: adapters you can download, GGUF export for LM Studio/Ollama, datasets as JSONL/CSV.", null],
            ["Honest evals", "Held-out test sets, per-class breakdowns, a good-prompt baseline, and regressions shown — never just a happy number.", null],
            ["Fair pricing", "₹0 locally on your GPU. Bigger models ride Google Colab's free T4 via a generated Unsloth notebook.", null],
          ].map(([t, d, g]) => (
            <div key={t as string} className="rounded-xl border p-6">
              <div className="text-base font-semibold">{t}</div>
              <p className="mt-2 text-sm text-muted-foreground">{d}</p>
              {g && (
                <a
                  href="https://github.com/chaithanya812/FINE-TUNE"
                  target="_blank"
                  rel="noreferrer"
                  className="mt-4 inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90"
                  style={{ backgroundColor: PLUM }}
                >
                  <Star className="size-4" /> Star on GitHub
                </a>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ---------------------------------------------------------- final CTA */}
      <section className="border-t bg-muted/60 py-16 text-center">
        <h2 className="font-hand text-5xl font-bold">Ready to train <span className="hl-swipe">yours</span>?</h2>
        <Link
          href="/chat"
          className="mt-6 inline-block rounded-md px-7 py-3 text-sm font-semibold text-white transition-opacity hover:opacity-90"
          style={{ backgroundColor: PLUM }}
        >
          Start now — It&apos;s free
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
