/* TerraLingua dashboard client.
   Plain ES modules-free JS on purpose: no build step, no npm, no CDN, so the
   dashboard works offline next to the logs it reads. */

const $ = (sel) => document.querySelector(sel);
const api = (path) => fetch(path).then((r) => {
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
});

/* Categorical slots in fixed order. Green is deliberately absent: it is the food
   ramp's hue, and a being must never read as the ground it stands on. */
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* Above this many beings, identity by hue stops working, so everyone shares one
   color and identity is carried by labels and the inspector instead. Cycling a
   palette past its slots would invent meaning that is not there. */
const MAX_IDENTITY_COLORS = 6;

const state = {
  run: null,
  meta: null,
  step: 0,
  world: null,
  selected: null,
  playing: false,
  timer: null,
  speed: 250,
  series: null,
  viral: null,
  stream: null,
  trail: null,
  // While true, new steps from the live stream pull the view forward.
  following: false,
  cache: new Map(),
};

/* ---------------- run selection ---------------- */

async function loadRuns() {
  const { runs } = await api("/api/runs");
  const box = $("#runs");
  box.innerHTML = "";
  $("#runs-empty").hidden = runs.length > 0;
  for (const r of runs) {
    const btn = document.createElement("button");
    btn.className = "run-card";
    const pct = r.max_ts ? Math.round((100 * (r.last_step + 1)) / r.max_ts) : 0;
    btn.innerHTML = `
      <span class="badge ${r.status}"><span class="dot"></span>${r.status}</span>
      <span style="flex:1">
        <span style="font-weight:600">${esc(r.name)}</span>
        <span class="subtitle"> ${esc(r.description || "")}</span><br>
        <span class="subtitle">${esc(r.model)} · ${r.grid_size}×${r.grid_size} · step ${r.last_step} of ${r.max_ts ?? "?"} (${pct}%)</span>
      </span>
      ${r.has_viral ? '<span class="badge">☣ viral</span>' : ""}`;
    btn.onclick = () => { $("#run-overlay").classList.add("hidden"); openRun(r.name); };
    box.appendChild(btn);
  }
  return runs;
}

async function openRun(name) {
  if (state.stream) { state.stream.close(); state.stream = null; }
  state.run = name;
  state.cache.clear();
  state.meta = await api(`/api/runs/${name}/meta`);
  state.selected = state.meta.agents[0] || null;

  $("#run-name").textContent = name;
  $("#run-sub").textContent =
    `${state.meta.description || "no description"} — ${state.meta.model}`;
  $("#grid-dims").textContent = `${state.meta.grid_size}×${state.meta.grid_size}`;
  const recon = $("#recon-badge");
  recon.classList.toggle("hidden", state.meta.provenance !== "reconstructed");
  // Say exactly how trustworthy this run is: it predates the per-step world log,
  // so positions are inferred and the food layer is only what beings have seen.
  const agree = state.meta.sighting_agreement;
  recon.title =
    "This run predates the per-step world log, so it was reconstructed from what " +
    "the beings observed.\n" +
    (agree ? `Cross-checks: ${agree[0]} of ${agree[1]} sightings agree.\n` : "") +
    "Paths of beings nobody could see may be approximate, and the food map shows " +
    "only cells somebody has visited.";
  $("#legend-unknown").hidden = state.meta.food_source !== "observed";
  $("#legend-infected").hidden = !state.meta.has_viral;

  setStatus(state.meta.status);
  state.following = state.meta.status === "live";
  $("#scrub").max = Math.max(0, state.meta.last_step);
  await refreshSeries();
  await goto(state.meta.last_decision_step ?? state.meta.last_step);
  if (state.meta.status === "live") startStream();
}

function setStatus(status) {
  const el = $("#status-badge");
  el.className = `badge ${status}`;
  el.lastElementChild.textContent = status;
}

/* ---------------- time ---------------- */

async function fetchStep(t) {
  if (state.cache.has(t)) return state.cache.get(t);
  const data = await api(`/api/runs/${state.run}/step/${t}`);
  if (state.cache.size > 400) state.cache.clear();
  state.cache.set(t, data);
  return data;
}

const TRAIL_STEPS = 40;

async function goto(t) {
  t = Math.max(0, Math.min(t, state.meta.last_step));
  state.step = t;
  try {
    state.world = await fetchStep(t);
  } catch {
    return;
  }
  $("#scrub").value = t;
  $("#step-readout").textContent = `${t} / ${state.meta.max_ts ?? "?"}`;
  render();
  refreshTrail();
}

async function refreshTrail() {
  const tag = state.selected;
  if (!tag) return (state.trail = null);
  const start = Math.max(0, state.step - TRAIL_STEPS);
  const cur = state.trail;
  // Only refetch when the selection changed or we scrubbed outside the window.
  if (cur && cur.tag === tag && start >= cur.start && state.step <= cur.end) return;
  try {
    const data = await api(
      `/api/runs/${state.run}/trail/${tag}?start=${start}&end=${state.meta.last_step}`);
    state.trail = { tag, start, end: state.meta.last_step, points: data.points };
    drawMap();
  } catch { /* trail is decoration; a failure must not blank the map */ }
}

function play() {
  if (state.playing) return stop();
  state.playing = true;
  $("#play").textContent = "❚❚";
  state.timer = setInterval(async () => {
    if (state.step >= state.meta.last_step) {
      if (state.meta.status !== "live") return stop();
      return;
    }
    await goto(state.step + 1);
  }, state.speed);
}

function stop() {
  state.playing = false;
  $("#play").textContent = "▶";
  clearInterval(state.timer);
}

function startStream() {
  const es = new EventSource(`/api/runs/${state.run}/stream?since=${state.meta.last_step}`);
  state.stream = es;
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    state.meta.last_step = msg.last_step;
    $("#scrub").max = msg.last_step;
    if (msg.status) { state.meta.status = msg.status; setStatus(msg.status); }
    if (msg.series) { state.series = { ...state.series, ...msg.series }; drawCharts(); }
    // Follow the live edge until the viewer scrubs away. Comparing the current
    // step against the newest one is not enough: the world log runs a step ahead
    // of the agent decisions, so the view is legitimately never "at the tip".
    if (state.following) goto(msg.last_decision_step ?? msg.last_step);
    if (msg.status === "finished") es.close();
  };
  // EventSource retries on its own; a hard failure just leaves replay working.
  es.onerror = () => setStatus(state.meta.status === "live" ? "stalled" : state.meta.status);
}

/* ---------------- rendering ---------------- */

function agentColor(tag) {
  const agents = state.meta.agents;
  if (agents.length > MAX_IDENTITY_COLORS) {
    return tag === state.selected ? cssVar("--s2") : cssVar("--s1");
  }
  const i = agents.indexOf(tag);
  return cssVar(SERIES[Math.max(0, i) % SERIES.length]);
}

function foodColor(v, max) {
  const steps = ["--food-0", "--food-1", "--food-2", "--food-3", "--food-4"];
  const idx = Math.min(steps.length - 1, Math.max(0, Math.round((v / max) * (steps.length - 1))));
  return cssVar(steps[idx]);
}

function render() {
  drawMap();
  drawAgentList();
  drawAgentDetail();
  drawThought();
  drawChat();
  drawCharts();
}

function mapGeometry() {
  const canvas = $("#map");
  const wrap = $("#map-wrap");
  const dpr = window.devicePixelRatio || 1;
  const n = state.meta.grid_size;
  // The map lives in a content-sized column, so its own width is a consequence
  // of this calculation -- measuring it here would be circular. Height comes
  // from the row and is independent, so drive off that and cap the share of the
  // window the map may take.
  const size = Math.max(120, Math.min(wrap.clientHeight, document.body.clientWidth * 0.5));
  const cell = Math.max(2, Math.floor(size / n));
  const px = cell * n;
  canvas.width = px * dpr;
  canvas.height = px * dpr;
  canvas.style.width = px + "px";
  canvas.style.height = px + "px";
  return { ctx: canvas.getContext("2d"), cell, n, dpr };
}

function drawMap() {
  if (!state.world) return;
  const { ctx, cell, n, dpr } = mapGeometry();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Backdrop. On a reconstructed run the food layer is only what beings have
  // seen, so the base reads as "unknown" and observed-but-empty is lighter.
  const unknown = state.meta.food_source === "observed";
  ctx.fillStyle = unknown ? cssVar("--plane") : cssVar("--surface-2");
  ctx.fillRect(0, 0, cell * n, cell * n);

  for (const [x, y, v] of state.world.food) {
    ctx.fillStyle = foodColor(v, state.meta.max_food_value || 10);
    ctx.fillRect(y * cell, x * cell, cell, cell);
  }

  if (cell >= 6) {
    ctx.strokeStyle = cssVar("--grid-line");
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 1; i < n; i++) {
      ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, n * cell);
      ctx.moveTo(0, i * cell); ctx.lineTo(n * cell, i * cell);
    }
    ctx.stroke();
  }

  // Vision radius of the selected being, drawn under everything else.
  const sel = state.selected && state.world.agents[state.selected];
  if (sel && state.meta.vision_radius) {
    const r = state.meta.vision_radius;
    ctx.fillStyle = cssVar("--s2");
    // Needs to read over bright food as well as over unobserved black, so it
    // cannot be as faint as a wash on a neutral background would be.
    ctx.globalAlpha = 0.22;
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        const x = mod(sel[0] + dx, n), y = mod(sel[1] + dy, n);
        ctx.fillRect(y * cell, x * cell, cell, cell);
      }
    }
    ctx.globalAlpha = 1;
  }

  // Artifacts: a diamond, so shape carries them even where color cannot.
  ctx.fillStyle = cssVar("--s6");
  for (const [x, y] of state.world.artifacts) {
    const cx = y * cell + cell / 2, cy = x * cell + cell / 2, r = Math.max(2, cell * 0.32);
    ctx.beginPath();
    ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy);
    ctx.lineTo(cx, cy + r); ctx.lineTo(cx - r, cy);
    ctx.closePath(); ctx.fill();
  }

  drawTrail(ctx, cell, n);

  for (const [tag, a] of Object.entries(state.world.agents)) {
    const [x, y, , , , nViral] = a;
    const cx = y * cell + cell / 2, cy = x * cell + cell / 2;
    const r = Math.max(1.5, cell * 0.36);
    ctx.fillStyle = agentColor(tag);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();

    // A 2px surface ring keeps overlapping marks legible.
    ctx.strokeStyle = cssVar("--surface");
    ctx.lineWidth = Math.min(2, cell * 0.12);
    ctx.stroke();

    if (nViral > 0) {   // status color plus a glyph, never color alone
      ctx.strokeStyle = cssVar("--status-critical");
      ctx.lineWidth = Math.max(1, cell * 0.14);
      ctx.beginPath(); ctx.arc(cx, cy, r + ctx.lineWidth, 0, Math.PI * 2); ctx.stroke();
    }
    if (tag === state.selected) {
      ctx.strokeStyle = cssVar("--s2");
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, r + 3, 0, Math.PI * 2); ctx.stroke();
    }
  }
}

/* The world is a torus, so a trail that leaves one edge reappears on the other.
   Drawing straight between those two points would streak a line across the whole
   map, so segments that wrap are simply not drawn. */
function drawTrail(ctx, cell, n) {
  const trail = state.trail;
  if (!state.selected || !trail || trail.tag !== state.selected) return;
  const pts = trail.points.filter((p) => p[0] <= state.step).map((p) => [p[1], p[2]]);
  if (pts.length < 2) return;
  ctx.strokeStyle = cssVar("--s2");
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = Math.max(1, cell * 0.14);
  ctx.lineCap = "round";
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
    if (Math.abs(x1 - x0) > n / 2 || Math.abs(y1 - y0) > n / 2) continue; // wrapped
    ctx.beginPath();
    ctx.moveTo(y0 * cell + cell / 2, x0 * cell + cell / 2);
    ctx.lineTo(y1 * cell + cell / 2, x1 * cell + cell / 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

const mod = (a, n) => ((a % n) + n) % n;

function selectAgent(tag) {
  state.selected = tag;
  state.trail = null;
  render();
  refreshTrail();
}

function drawAgentList() {
  const box = $("#agent-list");
  box.innerHTML = "";
  const alive = state.world ? state.world.agents : {};
  for (const tag of state.meta.agents) {
    const here = alive[tag];
    const pill = document.createElement("button");
    pill.className = "agent-pill" + (tag === state.selected ? " sel" : "");
    pill.style.opacity = here ? 1 : 0.35;
    pill.title = here ? "" : "not present at this step";
    pill.innerHTML =
      `<span class="chip" style="background:${agentColor(tag)}"></span>${esc(tag)}` +
      (here && here[5] > 0 ? " ☣" : "");
    pill.onclick = () => selectAgent(tag);
    box.appendChild(pill);
  }
}

function drawAgentDetail() {
  const box = $("#agent-detail");
  const tag = state.selected;
  const a = tag && state.world && state.world.agents[tag];
  if (!a) {
    box.innerHTML = `<p class="empty">${tag ? esc(tag) + " is not in the world at this step." : "No being selected."}</p>`;
    return;
  }
  const [x, y, energy, age, nInv, nViral] = a;
  const maxAge = state.meta.params?.env?.agent_lifespan || 100;
  const initE = state.meta.params?.env?.init_agent_energy || 100;
  const genome = state.meta.genomes[tag] || {};

  const traits = Object.entries(genome).map(([k, v]) => {
    const pctW = Math.abs(v) * 50;
    const left = v >= 0 ? 50 : 50 - pctW;
    // Diverging: warm for positive, cool for negative, neutral zero in the middle.
    const color = v >= 0 ? cssVar("--s2") : cssVar("--s1");
    return `<div class="trait">
      <span class="label">${esc(k)}</span>
      <span class="track"><span class="fill" style="left:${left}%;width:${pctW}%;background:${color}"></span></span>
      <span class="val">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>
    </div>`;
  }).join("");

  box.innerHTML = `
    <div class="agent-head">
      <span class="agent-chip" style="background:${agentColor(tag)}"></span>
      <span class="agent-name">${esc(tag)}</span>
      ${nViral > 0 ? '<span class="badge" style="color:var(--status-critical)">☣ infected</span>' : ""}
    </div>
    ${meter("Energy", energy, initE * 2, "--s3", energy == null ? "—" : Math.round(energy))}
    ${age == null
      ? meter("Age", 0, maxAge, "--s4", "—")
      : meter("Age", maxAge - age, maxAge, "--s4", `${maxAge - age} / ${maxAge}`)}
    <div class="stat-row" style="margin-top:10px"><span>Position</span><b>${x}:${y}</b></div>
    <div class="stat-row"><span>Inventory</span><b>${nInv}</b></div>
    ${traits ? `<h2 style="margin-top:14px">Genome</h2>${traits}` : ""}`;
}

function meter(label, value, max, colorVar, text) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="stat">
    <div class="stat-row"><span>${label}</span><b>${text}</b></div>
    <div class="meter"><i style="width:${pct}%;background:${cssVar(colorVar)}"></i></div>
  </div>`;
}

function drawThought() {
  const body = $("#thought-body");
  const tick = state.world?.ticks?.[state.selected];
  if (!tick) {
    body.innerHTML = `<p class="empty">No decision recorded for this being at step ${state.step}.</p>`;
    return;
  }
  // A long artifact payload would crowd out the internal memory, which is the
  // one thing this panel shows that appears nowhere else -- the chat feed
  // already carries what the being said out loud.
  const params = Object.entries(tick.params || {})
    .map(([k, v]) => {
      const s = typeof v === "string" ? v : JSON.stringify(v);
      return `${k}=${s.length > 44 ? s.slice(0, 44) + "…" : s}`;
    })
    .join("  ");
  body.innerHTML = `
    <div class="action-line">
      <span class="tag">Action</span>
      <span class="action-name">${esc(tick.action || "—")}</span>
    </div>
    ${params ? `<div class="subtitle" style="margin:-4px 0 8px 60px">${esc(params)}</div>` : ""}
    <div class="action-line" style="align-items:flex-start">
      <span class="tag">Thought</span>
      <span class="thought">${esc(tick.internal_memory || "(nothing recorded)")}</span>
    </div>`;
}

function drawChat() {
  const box = $("#chat-body");
  const msgs = state.world?.chat || [];
  if (!msgs.length) {
    box.innerHTML = `<p class="empty">Nobody spoke at step ${state.step}.</p>`;
    return;
  }
  box.innerHTML = msgs.map((m) => `
    <div class="msg">
      <span class="who" style="color:${agentColor(m.agent_tag)}" data-tag="${esc(m.agent_tag)}">${esc(m.agent_name)}</span>
      <span class="body">${esc(m.message)}</span>
    </div>`).join("");
  box.querySelectorAll(".who").forEach((el) => {
    el.onclick = () => selectAgent(el.dataset.tag);
  });
}

/* ---------------- charts ---------------- */

async function refreshSeries() {
  state.series = await api(`/api/runs/${state.run}/series`);
  if (state.meta.has_viral) {
    try { state.viral = await api(`/api/runs/${state.run}/viral`); } catch { state.viral = null; }
  }
}

function drawCharts() {
  if (!state.series) return;
  const s = state.series;
  const box = $("#charts");
  box.innerHTML = "";

  box.appendChild(chartCard({
    title: "Food in the world",
    series: [{ name: "food", points: zip(s.t, s.food_total), color: cssVar("--s3") }],
    format: (v) => Math.round(v).toLocaleString(),
  }));

  const pop = [{ name: "beings", points: zip(s.t, s.n_agents), color: cssVar("--s1") }];
  if (state.meta.has_viral) {
    // Same unit (a count of beings), so both share one axis. Never two scales.
    pop.push({ name: "infected", points: zip(s.t, s.n_infected), color: cssVar("--s2") });
  }
  box.appendChild(chartCard({ title: "Population", series: pop, format: (v) => v }));

  const artT = (s.artifacts || []).map((a) => a.t).sort((a, b) => a - b);
  const cum = artT.map((t, i) => [t, i + 1]);
  box.appendChild(chartCard({
    title: "Artifacts created",
    series: [{ name: "artifacts", points: cum.length ? cum : [[0, 0]], color: cssVar("--s6") }],
    format: (v) => v,
    step: true,
  }));

  const tok = s.tokens || [];
  box.appendChild(chartCard({
    title: "LLM tokens",
    series: [{ name: "cumulative", points: tok.map((r) => [r.t, r.cum_input + r.cum_output]), color: cssVar("--s4") }],
    format: (v) => v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "k" : v,
  }));

  if (state.viral?.generations?.length) {
    box.appendChild(r0Card(state.viral));
  }
}

const zip = (a, b) => a.map((t, i) => [t, b[i]]);

function chartCard({ title, series, format, step }) {
  const card = document.createElement("section");
  card.className = "panel chart";
  const latest = series[0].points.length ? series[0].points[series[0].points.length - 1][1] : 0;

  const W = 300, H = 92, PAD = 4;
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs, 0), x1 = Math.max(...xs, 1);
  const y0 = 0, y1 = Math.max(...ys, 1);
  const sx = (x) => PAD + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * PAD);
  const sy = (y) => H - PAD - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * PAD);

  const paths = series.map((s) => {
    const d = s.points.map((p, i) => {
      const cmd = i === 0 ? "M" : step ? `L${sx(p[0])},${sy(s.points[i - 1][1])}L` : "L";
      return `${cmd}${sx(p[0])},${sy(p[1])}`;
    }).join("");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join("");

  const cursorX = sx(state.step);
  card.innerHTML = `
    <h2>${esc(title)}</h2>
    <div class="hero">${format(latest)} <small>at step ${state.step}</small></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(title)}">
      <line class="grid-line" x1="${PAD}" y1="${sy(y1)}" x2="${W - PAD}" y2="${sy(y1)}"/>
      <line class="axis-line" x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}"/>
      <line class="cursor-line" x1="${cursorX}" y1="0" x2="${cursorX}" y2="${H}"/>
      ${paths}
    </svg>
    ${series.length > 1 ? `<div class="chart-legend">${series.map((s) =>
      `<span><span class="swatch" style="background:${s.color}"></span> ${esc(s.name)}</span>`).join("")}</div>` : ""}`;
  return card;
}

function r0Card(viral) {
  const card = document.createElement("section");
  card.className = "panel chart";
  const rows = viral.generations.map((g) => `
    <div class="stat-row"><span>gen ${g.generation} · ${g.cases} case${g.cases === 1 ? "" : "s"}</span>
    <b>R = ${g.mean_secondary.toFixed(2)}</b></div>`).join("");
  card.innerHTML = `
    <h2>☣ Transmission</h2>
    <div class="hero">${viral.r0 == null ? "—" : viral.r0.toFixed(2)} <small>R₀ (generations 0–1)</small></div>
    ${rows}
    ${viral.censored ? `<div class="subtitle">${viral.censored} still infectious, excluded</div>` : ""}`;
  return card;
}

/* ---------------- misc ---------------- */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function hookTooltip() {
  const canvas = $("#map"), tip = $("#tooltip");
  canvas.addEventListener("mousemove", (e) => {
    if (!state.world) return;
    const rect = canvas.getBoundingClientRect();
    const n = state.meta.grid_size;
    const cell = rect.width / n;
    const y = Math.floor((e.clientX - rect.left) / cell);
    const x = Math.floor((e.clientY - rect.top) / cell);
    if (x < 0 || y < 0 || x >= n || y >= n) return (tip.style.opacity = 0);

    const lines = [`${x}:${y}`];
    const food = state.world.food.find((f) => f[0] === x && f[1] === y);
    if (food) lines.push(`food ${food[2]}`);
    else if (state.meta.food_source === "observed") lines.push("never observed");
    for (const [tag, a] of Object.entries(state.world.agents)) {
      if (a[0] === x && a[1] === y) lines.push(`${tag}${a[5] > 0 ? " ☣" : ""}  energy ${a[2] ?? "∞"}`);
    }
    for (const [ax, ay, name] of state.world.artifacts) {
      if (ax === x && ay === y) lines.push(`◆ ${name}`);
    }
    tip.textContent = lines.join("\n");
    tip.style.opacity = 1;
    tip.style.left = e.clientX - rect.left + canvas.offsetLeft + 14 + "px";
    tip.style.top = e.clientY - rect.top + canvas.offsetTop + 14 + "px";
  });
  canvas.addEventListener("mouseleave", () => ($("#tooltip").style.opacity = 0));
  canvas.addEventListener("click", (e) => {
    if (!state.world) return;
    const rect = canvas.getBoundingClientRect();
    const n = state.meta.grid_size, cell = rect.width / n;
    const y = Math.floor((e.clientX - rect.left) / cell);
    const x = Math.floor((e.clientY - rect.top) / cell);
    for (const [tag, a] of Object.entries(state.world.agents)) {
      if (a[0] === x && a[1] === y) { selectAgent(tag); return; }
    }
  });
}

function hookControls() {
  $("#play").onclick = play;
  $("#scrub").oninput = (e) => {
    stop();
    const t = +e.target.value;
    state.following = state.meta.status === "live" && t >= state.meta.last_step;
    goto(t);
  };
  $("#speed").onchange = (e) => {
    state.speed = +e.target.value;
    if (state.playing) { stop(); play(); }
  };
  $("#pick-run").onclick = async () => {
    await loadRuns();
    $("#run-overlay").classList.remove("hidden");
  };
  $("#theme").onclick = () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    render();
  };
  $("#chat-all").onclick = async () => {
    const { messages } = await api(`/api/runs/${state.run}/chat`);
    $("#chat-step").textContent = `0–${state.meta.last_step}`;
    $("#chat-full").innerHTML = messages.map((m) => `
      <div class="msg"><span class="subtitle">${m.t}</span>
      <span class="who" style="color:${agentColor(m.agent_tag)}">${esc(m.agent_name)}</span>
      <span class="body">${esc(m.message)}</span></div>`).join("") ||
      '<p class="empty">Nothing was said in this run.</p>';
    $("#chat-overlay").classList.remove("hidden");
  };
  for (const ov of document.querySelectorAll(".overlay")) {
    ov.onclick = (e) => { if (e.target === ov) ov.classList.add("hidden"); };
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === " ") { e.preventDefault(); play(); }
    else if (e.key === "ArrowLeft") { stop(); state.following = false; goto(state.step - (e.shiftKey ? 10 : 1)); }
    else if (e.key === "ArrowRight") { stop(); goto(state.step + (e.shiftKey ? 10 : 1)); }
    else if (e.key === "Home") { stop(); state.following = false; goto(0); }
    else if (e.key === "End") { stop(); state.following = state.meta.status === "live"; goto(state.meta.last_step); }
    else if (e.key === "Escape") document.querySelectorAll(".overlay").forEach((o) => o.classList.add("hidden"));
  });

  // The map is sized from its container, which keeps changing as the charts and
  // panels below it fill in. Watching the element beats listening for `resize`,
  // which never fires for those internal reflows.
  new ResizeObserver(() => drawMap()).observe($("#map-wrap"));
}

async function init() {
  hookControls();
  hookTooltip();
  const runs = await loadRuns();
  const wanted = new URLSearchParams(location.search).get("run");
  const first = runs.find((r) => r.name === wanted) || runs.find((r) => r.status === "live") || runs[0];
  if (first) await openRun(first.name);
  else $("#run-overlay").classList.remove("hidden");
}

init().catch((e) => {
  $("#run-sub").textContent = "failed to load: " + e.message;
});
