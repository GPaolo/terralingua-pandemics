/* TerraLingua dashboard client.
   Plain ES modules-free JS on purpose: no build step, no npm, no CDN, so the
   dashboard works offline next to the logs it reads. */

const $ = (sel) => document.querySelector(sel);
const api = (path) => fetch(path).then((r) => {
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
});

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

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
  artifacts: null,
  // Names of artifact entries the viewer has expanded; survives re-renders.
  openArtifacts: new Set(),
  // A clicked cell pins the map tooltip so its contents can be clicked.
  tipPinned: false,
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
  state.openArtifacts.clear();
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
  $("#legend-incubating").hidden = !state.meta.has_viral;

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
  // A pinned tooltip describes one instant; it must not survive into another.
  if (t !== state.step) unpinTooltip();
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
    // An outbreak can start mid-run: unhide the infection legend and start
    // fetching viral data the moment the stream first reports it.
    if (msg.has_viral && !state.meta.has_viral) {
      state.meta.has_viral = true;
      $("#legend-infected").hidden = false;
      $("#legend-incubating").hidden = false;
    }
    // Artifact edits and new infections are not in the SSE payload; refetch
    // when the run advances so neither panel freezes at page-load state.
    if (msg.series) {
      api(`/api/runs/${state.run}/artifacts`)
        .then((d) => { state.artifacts = d.artifacts; drawArtifacts(); })
        .catch(() => {});
      if (state.meta.has_viral) {
        api(`/api/runs/${state.run}/viral`)
          .then((d) => { state.viral = d; drawCharts(); })
          .catch(() => {});
      }
    }
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

/* Hue on a being is spent on state, never identity: neutral --being grey is the
   base, --s1 blue marks a PPE carrier, and health moves along the status ramp.
   Identity is carried by the selection brackets, the trail, the tooltip and the
   Beings list instead. */
function agentColor(a) {
  return hasPPE(a) ? cssVar("--s1") : cssVar("--being");
}

const PPE_GLYPH = "⛨";
const hasPPE = (a) => (a?.[7] ?? 0) > 0;

/* Health is the one thing that does move a being off --s1, and it moves along the
   reserved status ramp rather than into a categorical slot: amber while the
   infection is still incubating, red once it shows symptoms. Amber sits ~48 dE
   from the red, where the orange slot sits at 6.8 -- far below the floor of 15 --
   so the two phases stay distinguishable at small cell sizes. Always paired with
   a glyph so hue is never the only channel. */
const HEALTH_GLYPH = { sick: "☣", incubating: "⧖" };

function healthOf(a) {
  if (!a) return null;
  // Schema 1 runs have no n_sick column; there, any infection was symptomatic.
  const nViral = a[5] ?? 0;
  const nSick = a[6] ?? nViral;
  if (nSick > 0) return "sick";
  if (nViral > 0) return "incubating";
  return null;
}

function healthColor(health, a) {
  if (health === "sick") return cssVar("--status-critical");
  if (health === "incubating") return cssVar("--status-warning");
  return agentColor(a);
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
  drawArtifacts();
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

  /* Vision radius of the selected being: an outline, not a wash. Filling these
     cells shifts their lightness, which is the very channel the food ramp
     encodes, so a seen cell reads as a different amount of food than it holds --
     and on a reconstructed run it tints "never observed" black into a third
     state that means nothing. Dashed, so the soft edge of what a being can see
     never reads as the hard edge of the selection brackets. */
  const sel = state.selected && state.world.agents[state.selected];
  if (sel && state.meta.vision_radius) {
    const r = state.meta.vision_radius;
    const side = (2 * r + 1) * cell;
    const left = (sel[1] - r) * cell, top = (sel[0] - r) * cell;
    ctx.strokeStyle = cssVar("--select");
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = Math.max(1, Math.min(2, cell * 0.08));
    ctx.setLineDash([Math.max(2, cell * 0.5), Math.max(2, cell * 0.5)]);
    // The block wraps on the torus, so stroke it at every wrap offset and let
    // the canvas clip the copies that fall outside.
    for (const dx of [-n, 0, n]) {
      for (const dy of [-n, 0, n]) {
        ctx.strokeRect(left + dy * cell, top + dx * cell, side, side);
      }
    }
    ctx.setLineDash([]);
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

  /* Infection is the fill and selection is a frame on the cell, so the two never
     compete for the same pixels. They used to be concentric rings a pixel apart,
     which at mid cell sizes meant the selection ring painted straight over the
     infection ring and a sick being looked healthy the moment you clicked it. */
  for (const [, a] of Object.entries(state.world.agents)) {
    const [x, y] = a;
    const cx = y * cell + cell / 2, cy = x * cell + cell / 2;
    const r = Math.max(1.5, cell * 0.36);
    ctx.fillStyle = healthColor(healthOf(a), a);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    if (hasPPE(a)) $("#legend-ppe").hidden = false;

    // A 2px surface ring keeps overlapping marks legible.
    ctx.strokeStyle = cssVar("--surface");
    ctx.lineWidth = Math.min(2, cell * 0.12);
    ctx.stroke();
  }

  // Last, so a being standing on the same cell cannot paint over it.
  if (sel) drawSelection(ctx, cell, sel[0], sel[1]);
}

/* Selection is an interaction state, not data, so it stays off the hue channel:
   neutral ink brackets on the cell box, with a wider --surface stroke underneath
   so the pair holds >= 4.4:1 against every color this map can paint. The box has a
   pixel floor -- on a 100x100 grid a 2px cell cannot carry a frame, so the reticle
   grows past its cell and reads as "here" rather than vanishing. */
function drawSelection(ctx, cell, row, col) {
  const box = Math.max(cell, 14);
  const w = Math.min(2, Math.max(1.5, cell * 0.1));
  const cx = col * cell + cell / 2, cy = row * cell + cell / 2;
  const h = box / 2 - w;                    // half-size, stroke kept inside the box
  const arm = Math.max(3, h * 0.55);
  ctx.beginPath();
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    const px = cx + sx * h, py = cy + sy * h;
    ctx.moveTo(px - sx * arm, py);
    ctx.lineTo(px, py);
    ctx.lineTo(px, py - sy * arm);
  }
  // drawTrail leaves lineCap at "round"; brackets need square ends.
  ctx.lineCap = "butt"; ctx.lineJoin = "miter";
  ctx.strokeStyle = cssVar("--surface"); ctx.lineWidth = w + 2; ctx.stroke();
  ctx.strokeStyle = cssVar("--select");   ctx.lineWidth = w;     ctx.stroke();
}

/* The world is a torus, so a trail that leaves one edge reappears on the other.
   Drawing straight between those two points would streak a line across the whole
   map, so segments that wrap are simply not drawn. */
function drawTrail(ctx, cell, n) {
  const trail = state.trail;
  if (!state.selected || !trail || trail.tag !== state.selected) return;
  const pts = trail.points.filter((p) => p[0] <= state.step).map((p) => [p[1], p[2]]);
  if (pts.length < 2) return;
  ctx.strokeStyle = cssVar("--select");
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
    const health = healthOf(here);
    pill.innerHTML =
      `<span class="chip" style="background:${healthColor(health, here)}"></span>${esc(tag)}` +
      (health ? ` ${HEALTH_GLYPH[health]}` : "") +
      (hasPPE(here) ? ` ${PPE_GLYPH}` : "");
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
  const [x, y, energy, age, nInv] = a;
  const health = healthOf(a);
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
      <span class="agent-chip" style="background:${healthColor(health, a)}"></span>
      <span class="agent-name">${esc(tag)}</span>
      ${health === "sick"
        ? '<span class="badge" style="color:var(--status-critical)">☣ sick</span>'
        : health === "incubating"
        ? '<span class="badge" style="color:var(--status-warning)">⧖ incubating</span>'
        : ""}
      ${hasPPE(a) ? `<span class="badge" style="color:var(--s1)">${PPE_GLYPH} PPE</span>` : ""}
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
      <span class="who" style="color:${agentColor()}" data-tag="${esc(m.agent_tag)}">${esc(m.agent_name)}</span>
      <span class="body">${esc(m.message)}</span>
    </div>`).join("");
  box.querySelectorAll(".who").forEach((el) => {
    el.onclick = () => selectAgent(el.dataset.tag);
  });
}

/* ---------------- artifacts ---------------- */

/* The panel is step-aware like everything else: it lists what exists (or once
   existed) as of the scrubbed step, and the content shown is the version that
   was current then, not the run's final text. */
function drawArtifacts() {
  const box = $("#artifacts-body");
  const arts = (state.artifacts || [])
    .filter((a) => (a.created_at ?? 0) <= state.step)
    .sort((p, q) => (q.created_at ?? 0) - (p.created_at ?? 0));
  if (!arts.length) {
    box.innerHTML = `<p class="empty">No artifacts exist at step ${state.step}.</p>`;
    return;
  }
  box.innerHTML = arts.map(artifactEntry).join("");
  box.querySelectorAll("details.artifact").forEach((el) => {
    el.addEventListener("toggle", () => {
      if (el.open) state.openArtifacts.add(el.dataset.name);
      else state.openArtifacts.delete(el.dataset.name);
    });
  });
}

function artifactEntry(a) {
  const removed = a.removed_at != null && a.removed_at <= state.step;
  // Full edit trail = past versions plus the current one, oldest first.
  const versions = [...(a.past_versions || []), {
    payload: a.payload,
    version: a.version ?? 0,
    version_creation_time: a.version_creation_time ?? a.created_at,
  }].filter((v) => (v.version_creation_time ?? 0) <= state.step);
  const cur = versions[versions.length - 1];

  const edits = (a.editors || []).filter((e) => e.t <= state.step);
  // Passive reads fire every step a being stands on the cell, so a raw list
  // would swamp the panel: aggregate per reader.
  const readsBy = {};
  for (const r of (a.readers || []).filter((r) => r.t <= state.step)) {
    const prev = readsBy[r.agent_tag] || { n: 0, last: -1 };
    readsBy[r.agent_tag] = { n: prev.n + 1, last: Math.max(prev.last, r.t) };
  }

  const meta = [
    `by ${esc(a.created_by ?? a.creator_tag ?? "?")}`,
    `step ${a.created_at ?? "?"}`,
    versions.length > 1 ? `v${cur.version}` : "",
    removed ? `gone at ${a.removed_at}` : "",
  ].filter(Boolean).join(" · ");

  return `<details class="artifact" data-name="${esc(a.name)}"
      ${state.openArtifacts.has(a.name) ? "open" : ""} ${removed ? 'style="opacity:.55"' : ""}>
    <summary><span style="color:var(--s6)">◆</span>
      <span class="art-name">${esc(a.name)}</span>
      <span class="subtitle">${meta}</span></summary>
    <div class="payload">${esc(cur?.payload || "(empty)")}</div>
    ${versions.length > 1 ? `<div class="art-sec">History</div>` +
      versions.slice(0, -1).reverse().map((v) => `
        <details class="ver">
          <summary class="subtitle">v${v.version} · step ${v.version_creation_time}</summary>
          <div class="payload">${esc(v.payload || "(empty)")}</div>
        </details>`).join("") : ""}
    ${edits.length || Object.keys(readsBy).length ? `<div class="art-sec">Interactions</div>` : ""}
    ${edits.map((e) => `<div class="stat-row">
      <span>${esc(e.agent_tag)} ${e.action?.startsWith("destroy") ? "destroyed" : "edited"}</span>
      <b>step ${e.t}</b></div>`).join("")}
    ${Object.entries(readsBy).map(([tag, r]) => `<div class="stat-row">
      <span>${esc(tag)} read ×${r.n}</span><b>last step ${r.last}</b></div>`).join("")}
  </details>`;
}

function openArtifact(name) {
  state.openArtifacts.add(name);
  drawArtifacts();
  const el = document.querySelector(`details.artifact[data-name="${CSS.escape(name)}"]`);
  if (el) el.scrollIntoView({ block: "nearest" });
}

/* ---------------- charts ---------------- */

async function refreshSeries() {
  state.series = await api(`/api/runs/${state.run}/series`);
  try { state.artifacts = (await api(`/api/runs/${state.run}/artifacts`)).artifacts; }
  catch { state.artifacts = []; }
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
    // Same unit (a count of beings), so all three share one axis. Never two
    // scales. Reserved status colors, matching the map, each labelled with its
    // glyph so the color is never the only thing saying which phase it is.
    // The two infection series are disjoint -- incubating is the carriers that
    // have not turned yet -- so they read as parts of one population, not as
    // one line drawn on top of another.
    const nSick = s.n_sick || s.n_infected;
    const incubating = s.n_infected.map((n, i) => n - (nSick[i] ?? 0));
    pop.push({ name: "⧖ incubating", points: zip(s.t, incubating), color: cssVar("--status-warning") });
    pop.push({ name: "☣ sick", points: zip(s.t, nSick), color: cssVar("--status-critical") });
  }
  box.appendChild(chartCard({ title: "Population", series: pop, format: (v) => v }));

  // Sourced from the panel's artifact list, which the live stream keeps fresh.
  const artT = (state.artifacts || []).map((a) => a.created_at ?? 0).sort((a, b) => a - b);
  const cum = artT.map((t, i) => [t, i + 1]);
  box.appendChild(chartCard({
    title: "Artifacts created",
    series: [{ name: "artifacts", points: cum.length ? cum : [[0, 0]], color: cssVar("--s6") }],
    format: (v) => v,
    step: true,
  }));

  // Cost rides in the tokens hero; the server only attaches cum_cost when the
  // model is priced, so local models never show a made-up dollar figure.
  const tok = s.tokens || [];
  const tokNow = tok.filter((r) => r.t <= state.step).pop();
  const cost = tokNow?.cum_cost;
  box.appendChild(chartCard({
    title: "LLM tokens",
    series: [{ name: "cumulative", points: tok.map((r) => [r.t, r.cum_input + r.cum_output]), color: cssVar("--s4") }],
    format: (v) => v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "k" : v,
    heroSuffix: cost != null
      ? ` · $${cost >= 100 ? cost.toFixed(0) : cost >= 1 ? cost.toFixed(2) : cost.toFixed(3)}`
      : "",
  }));

  if (state.viral?.chain?.length) {
    box.appendChild(r0Card(state.viral));
  }
}

const zip = (a, b) => a.map((t, i) => [t, b[i]]);

function chartCard({ title, series, format, step, heroSuffix }) {
  const card = document.createElement("section");
  card.className = "panel chart";
  // The hero says "at step N", so it must be the value at the scrubbed step,
  // not the run's final one.
  const upto = series[0].points.filter((p) => p[0] <= state.step);
  const latest = upto.length ? upto[upto.length - 1][1] : 0;

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
    <div class="hero">${format(latest)}${heroSuffix || ""} <small>at step ${state.step}</small></div>
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

/* Step-aware like every other card: the numbers are a census of the epidemic
   as of the scrubbed step, recomputed from the chain. An episode only counts
   toward R once it is over (it can still spread until then), matching the
   server's whole-run generations at the final step. */
function r0Card(viral) {
  const card = document.createElement("section");
  card.className = "panel chart";
  const started = (viral.chain || []).filter((c) => c.t <= state.step);
  const kids = new Map();
  for (const c of started) {
    if (c.source) kids.set(c.source, (kids.get(c.source) || 0) + 1);
  }
  const done = started.filter((c) => c.ended_at != null && c.ended_at <= state.step);
  const byGen = new Map();
  for (const c of done) {
    if (!byGen.has(c.generation)) byGen.set(c.generation, []);
    byGen.get(c.generation).push(kids.get(c.artifact) || 0);
  }
  const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  const rows = [...byGen.entries()].sort((a, b) => a[0] - b[0]).map(([g, v]) => `
    <div class="stat-row"><span>gen ${g} · ${v.length} case${v.length === 1 ? "" : "s"}</span>
    <b>R = ${mean(v).toFixed(2)}</b></div>`).join("");
  const early = [...(byGen.get(0) || []), ...(byGen.get(1) || [])];
  const r0 = early.length ? mean(early) : null;
  const active = started.length - done.length;

  card.innerHTML = `
    <h2>☣ Transmission
      <button id="chain-btn" style="float:right;padding:1px 7px;font-size:11px">chain</button>
    </h2>
    <div class="hero">${r0 == null ? "—" : r0.toFixed(2)} <small>R₀ (gen 0–1) at step ${state.step}</small></div>
    ${started.length
      ? `<div class="gen-rows">${rows}</div>
         ${active ? `<div class="subtitle">${active} still active, not yet counted</div>` : ""}`
      : `<div class="subtitle">no infections yet at this step</div>`}`;
  card.querySelector("#chain-btn").onclick = showChain;
  return card;
}

/* Who infected whom, as an indented tree: seeded infections at the root, each
   infection's secondary cases nested under it. Whole-run history like the
   whole-run chat; entries past the scrubbed step are dimmed rather than hidden
   so the shape of the epidemic stays readable. */
function showChain() {
  const chain = state.viral?.chain || [];
  const kids = new Map();
  const roots = [];
  for (const c of chain) {
    if (c.source && chain.some((p) => p.artifact === c.source)) {
      if (!kids.has(c.source)) kids.set(c.source, []);
      kids.get(c.source).push(c);
    } else {
      roots.push(c);
    }
  }
  const byT = (a, b) => (a.t ?? 0) - (b.t ?? 0);
  const node = (c, depth) => `
    <div class="chain-node${c.t > state.step ? " future" : ""}" style="margin-left:${depth * 18}px">
      ${depth ? '<span class="subtitle">↳</span>' : ""}
      <span style="color:var(--status-critical)">☣</span>
      <span class="who" style="color:${agentColor()}" data-tag="${esc(c.host)}">${esc(c.host)}</span>
      <span class="subtitle">step ${c.t} · ${esc(c.artifact)}` +
        `${c.secondary ? ` · spread to ${c.secondary}` : ""}` +
        `${c.ended_at == null ? " · ongoing" : ""}</span>
    </div>` +
    (kids.get(c.artifact) || []).sort(byT).map((k) => node(k, depth + 1)).join("");
  $("#chain-full").innerHTML =
    roots.sort(byT).map((r) => node(r, 0)).join("") ||
    '<p class="empty">No infections recorded.</p>';
  $("#chain-full").querySelectorAll(".who").forEach((el) => {
    el.onclick = () => {
      $("#chain-overlay").classList.add("hidden");
      selectAgent(el.dataset.tag);
    };
  });
  $("#chain-overlay").classList.remove("hidden");
}

/* ---------------- misc ---------------- */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* One HTML builder for both tooltip modes: hover preview and the pinned,
   clickable version. Being and artifact names carry data attributes so the
   pinned tooltip can wire them to selection / the artifacts panel. */
function tooltipHtml(x, y) {
  let html = `<div>${x}:${y}</div>`;
  let clickable = 0;
  const food = state.world.food.find((f) => f[0] === x && f[1] === y);
  if (food) html += `<div>food ${food[2]}</div>`;
  else if (state.meta.food_source === "observed") html += "<div>never observed</div>";
  for (const [tag, a] of Object.entries(state.world.agents)) {
    if (a[0] !== x || a[1] !== y) continue;
    // Plain text tooltip, so the glyph is the only channel available here.
    const health = healthOf(a);
    html += `<div><span class="tip-link" data-tag="${esc(tag)}">${esc(tag)}</span>` +
      `${health ? ` ${HEALTH_GLYPH[health]}` : ""}${hasPPE(a) ? ` ${PPE_GLYPH}` : ""}` +
      `  energy ${a[2] ?? "∞"}</div>`;
    clickable++;
  }
  for (const [ax, ay, name] of state.world.artifacts) {
    if (ax !== x || ay !== y) continue;
    html += `<div>◆ <span class="tip-link" data-art="${esc(name)}">${esc(name)}</span></div>`;
    clickable++;
  }
  return { html, clickable };
}

function unpinTooltip() {
  state.tipPinned = false;
  const tip = $("#tooltip");
  tip.classList.remove("pinned");
  tip.style.opacity = 0;
}

function hookTooltip() {
  const canvas = $("#map"), tip = $("#tooltip");
  const cellAt = (e) => {
    const rect = canvas.getBoundingClientRect();
    const n = state.meta.grid_size, cell = rect.width / n;
    return {
      x: Math.floor((e.clientY - rect.top) / cell),
      y: Math.floor((e.clientX - rect.left) / cell),
      n, rect,
    };
  };
  const place = (e, rect) => {
    tip.style.left = e.clientX - rect.left + canvas.offsetLeft + 14 + "px";
    tip.style.top = e.clientY - rect.top + canvas.offsetTop + 14 + "px";
  };

  canvas.addEventListener("mousemove", (e) => {
    if (!state.world || state.tipPinned) return;
    const { x, y, n, rect } = cellAt(e);
    if (x < 0 || y < 0 || x >= n || y >= n) return (tip.style.opacity = 0);
    tip.innerHTML = tooltipHtml(x, y).html;
    tip.style.opacity = 1;
    place(e, rect);
  });
  canvas.addEventListener("mouseleave", () => { if (!state.tipPinned) tip.style.opacity = 0; });

  canvas.addEventListener("click", (e) => {
    if (!state.world) return;
    const { x, y, n, rect } = cellAt(e);
    if (x < 0 || y < 0 || x >= n || y >= n) return unpinTooltip();
    // Shortcuts kept from before the pin existed: a being on the cell gets
    // selected outright, a bare artifact opens in the panel.
    let being = null, art = null;
    for (const [tag, a] of Object.entries(state.world.agents)) {
      if (a[0] === x && a[1] === y) being = being ?? tag;
    }
    for (const [ax, ay, name] of state.world.artifacts) {
      if (ax === x && ay === y) art = art ?? name;
    }
    if (being) selectAgent(being);
    else if (art) openArtifact(art);

    const { html, clickable } = tooltipHtml(x, y);
    if (!clickable) return unpinTooltip();
    // Pin the tooltip so its contents can be clicked (several beings on one
    // cell, or a being standing on an artifact). Unpinned by Escape, a click
    // on an empty cell, or moving through time.
    state.tipPinned = true;
    tip.classList.add("pinned");
    tip.innerHTML = html;
    tip.style.opacity = 1;
    place(e, rect);
    tip.querySelectorAll("[data-tag]").forEach((el) => {
      el.onclick = () => selectAgent(el.dataset.tag);
    });
    tip.querySelectorAll("[data-art]").forEach((el) => {
      el.onclick = () => openArtifact(el.dataset.art);
    });
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
      <span class="who" style="color:${agentColor()}">${esc(m.agent_name)}</span>
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
    else if (e.key === "Escape") {
      unpinTooltip();
      document.querySelectorAll(".overlay").forEach((o) => o.classList.add("hidden"));
    }
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
