"use strict";

const $ = (id) => document.getElementById(id);
const RUN_NAME = "ui_run";
let losses = [];
let useAdapter = true;
let currentTask = "classification";

// ---- show/hide the label-granularity row when task isn't classification ----
$("task").addEventListener("change", () => {
  $("labelFieldRow").style.display = $("task").value === "classification" ? "block" : "none";
});

// ---------------------------------------------------------------- start job
$("startBtn").addEventListener("click", startTraining);

async function startTraining() {
  resetUI();
  const body = {
    run_name: RUN_NAME,
    model_id: $("model_id").value,
    task: $("task").value,
    label_field: $("label_field").value,
    mode: $("mode").value,
    dataset: $("dataset").value,
    epochs: parseFloat($("epochs").value),
    n_train: parseInt($("n_train").value),
    n_eval: parseInt($("n_eval").value),
  };
  currentTask = body.task;
  setStatus("running");
  $("startBtn").disabled = true;

  let res;
  try {
    res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
  } catch (e) {
    return fail("Could not reach the server. Is it running?");
  }
  if (res.error) return fail(res.error);
  connectWS(res.job_id);
}

// ---------------------------------------------------------------- websocket
function connectWS(jobId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${jobId}`);
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onerror = () => fail("WebSocket error.");
}

function handleEvent(ev) {
  switch (ev.type) {
    case "log":
      addLog(ev.msg);
      break;
    case "progress":
      onProgress(ev);
      break;
    case "result":
      renderResults(ev.report);
      break;
    case "done":
      setStatus("done");
      $("startBtn").disabled = false;
      revealChat();
      break;
    case "error":
      fail(ev.msg);
      break;
  }
}

function onProgress(ev) {
  const pct = ev.max_steps ? Math.round((ev.step / ev.max_steps) * 100) : 0;
  $("bar").style.width = pct + "%";
  $("stepLabel").textContent = `step ${ev.step} / ${ev.max_steps}`;
  if (ev.loss != null) {
    $("lossLabel").textContent = `loss ${Number(ev.loss).toFixed(3)}`;
    losses.push(Number(ev.loss));
    drawSpark();
  }
}

// ---------------------------------------------------------------- loss chart
function drawSpark() {
  if (losses.length < 2) return;
  const w = 300, h = 44, pad = 3;
  const max = Math.max(...losses), min = Math.min(...losses);
  const rng = max - min || 1;
  const pts = losses.map((l, i) => {
    const x = pad + (i / (losses.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (l - min) / rng) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  $("sparkLine").setAttribute("points", pts.join(" "));
}

// ---------------------------------------------------------------- results
function renderResults(rep) {
  $("results").classList.remove("hidden");
  if (rep.task === "classification") {
    const b = rep.before.accuracy * 100, a = rep.after.accuracy * 100;
    const d = a - b;
    $("beforeVal").textContent = b.toFixed(0) + "%";
    $("afterVal").textContent = a.toFixed(0) + "%";
    setDelta(d, "pts");
    $("resumeLine").textContent =
      `Fine-tuned ${$("model_id").value.split("/").pop()} (${$("mode").value.toUpperCase()}) for support ` +
      `intent classification, improving held-out accuracy ${b.toFixed(0)}% → ${a.toFixed(0)}% ` +
      `(${rep.after.n}-example test set) on a 4GB laptop GPU.`;
  } else {
    const b = rep.before.avg_judge_score, a = rep.after.avg_judge_score;
    if (a == null) {
      $("beforeVal").textContent = "—"; $("afterVal").textContent = "—";
      $("deltaBadge").textContent = "Set JUDGE_API_KEY to score generation";
      $("resumeLine").textContent = "";
      return;
    }
    $("beforeVal").textContent = b.toFixed(1); $("afterVal").textContent = a.toFixed(1);
    setDelta(a - b, "pts (1-10 judge)");
    $("resumeLine").textContent =
      `Fine-tuned ${$("model_id").value.split("/").pop()} for on-brand support replies, ` +
      `raising LLM-judge quality ${b.toFixed(1)} → ${a.toFixed(1)} / 10.`;
  }
}

function setDelta(d, unit) {
  const badge = $("deltaBadge");
  const up = d >= 0;
  badge.textContent = `${up ? "▲" : "▼"} ${up ? "+" : ""}${d.toFixed(0)} ${unit}`;
  badge.style.background = up ? "rgba(52,211,153,.14)" : "rgba(248,113,113,.14)";
  badge.style.color = up ? "var(--good)" : "var(--bad)";
  $("deltaArrow").textContent = up ? "↑" : "↓";
}

// ---------------------------------------------------------------- chat
function revealChat() {
  $("chatCard").classList.remove("hidden");
}
$("segTuned").addEventListener("click", () => setSeg(true));
$("segBase").addEventListener("click", () => setSeg(false));
function setSeg(tuned) {
  useAdapter = tuned;
  $("segTuned").classList.toggle("active", tuned);
  $("segBase").classList.toggle("active", !tuned);
}
$("sendBtn").addEventListener("click", sendChat);
$("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

async function sendChat() {
  const input = $("chatInput");
  const text = input.value.trim();
  if (!text) return;
  addBubble(text, "user");
  input.value = "";
  const thinking = addBubble("…", "bot");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_name: RUN_NAME, message: text, use_adapter: useAdapter }),
    }).then((r) => r.json());
    thinking.remove();
    if (res.error) return addBubble("⚠ " + res.error, "bot");
    const prefix = res.task === "classification" ? "intent → " : "";
    addBubble(prefix + res.reply, "bot", res.task === "classification");
  } catch (e) {
    thinking.remove();
    addBubble("⚠ request failed", "bot");
  }
}

function addBubble(text, who, tag = false) {
  const div = document.createElement("div");
  div.className = "msg " + who + (tag ? " tag" : "");
  div.textContent = text;
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
  return div;
}

// ---------------------------------------------------------------- helpers
function addLog(msg, cls = "") {
  const log = $("log");
  const empty = log.querySelector(".log-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "line " + cls;
  div.textContent = "› " + msg;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function setStatus(s) {
  const pill = $("statusPill");
  pill.className = "pill " + s;
  pill.textContent = s;
}

function resetUI() {
  losses = [];
  $("log").innerHTML = "";
  $("results").classList.add("hidden");
  $("bar").style.width = "0%";
  $("stepLabel").textContent = "step 0 / 0";
  $("lossLabel").textContent = "loss —";
  $("sparkLine").setAttribute("points", "");
}

function fail(msg) {
  setStatus("error");
  addLog(msg, "err");
  $("startBtn").disabled = false;
}
