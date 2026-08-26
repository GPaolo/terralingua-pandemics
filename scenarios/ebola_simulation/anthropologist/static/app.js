"use strict";

const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const fmt = {
  pct: (v) => (v == null ? "—" : `${Math.round(v * 100)}%`),
  num: (v) => (v == null ? "—" : `${Math.round(v * 100) / 100}`),
  mult: (v) => (v == null ? "—" : `×${Math.round(v * 100) / 100}`),
};

let activeRun = null;
let allRuns = [];

// ------------------------------------------------------------------ run list

async function loadRuns() {
  const r = await api("/api/runs");
  allRuns = r.runs;
  activeRun = r.initial;
  $("runsel").innerHTML = allRuns.map((n) =>
    `<option value="${escapeHtml(n)}" ${n === activeRun ? "selected" : ""}>${escapeHtml(n)}</option>`
  ).join("");
  $("runchecks").innerHTML = allRuns.map((n) =>
    `<label><input type="checkbox" value="${escapeHtml(n)}" ${n === activeRun ? "checked" : ""} /> ${escapeHtml(n)}</label>`
  ).join("");
}

$("runsel").addEventListener("change", (e) => {
  activeRun = e.target.value;
  lastRender = "";
  seenDone.clear();
  loadState();
  loadFiles();
  poll();
});

// ---------------------------------------------------------------- left pane

function tile(value, label, sub = "") {
  return `<div class="tile"><div class="value">${value}</div>` +
         `<div class="label">${label}</div>` +
         (sub ? `<div class="sub">${sub}</div>` : "") + `</div>`;
}

async function loadState() {
  if (!activeRun) return;
  const s = await api(`/api/state?run=${encodeURIComponent(activeRun)}`);
  $("no-report").hidden = s.metrics !== null;

  if (s.metrics) {
    const m = s.metrics, o = m.outbreak, p = m.population, eff = m.ppe.efficiency;
    $("tiles").innerHTML =
      tile(o.infections, "infections", `${o.unique_hosts} hosts, ${o.index_cases} index`) +
      tile(fmt.pct(o.attack_rate), "attack rate", `of ${p.ever_alive} beings`) +
      tile(o.peak_sick, "peak sick", o.peak_sick_t == null ? "" : `day ${o.peak_sick_t}`) +
      tile(fmt.num(m.r0.empirical_r0), "R0 (gens 0–1)",
           `mean R ${fmt.num(m.r0.overall_mean_r)}, ${m.r0.censored} active`) +
      tile(p.deaths, "deaths", `${p.deaths_while_infected} while infected`) +
      tile(fmt.mult(eff.protection_realized), "PPE protection",
           eff.protection_configured == null ? "" : `configured ×${eff.protection_configured}`);
  } else {
    $("tiles").innerHTML = "";
  }

  const bust = Date.now(); // plots are regenerated in place
  $("plots").innerHTML = s.plots.map((url) => {
    const name = url.split("/").pop().replace(".png", "").replaceAll("_", " ");
    return `<figure><img src="${url}?v=${bust}" alt="${name}" loading="lazy" />` +
           `<figcaption>${name}</figcaption></figure>`;
  }).join("");
}

$("regen").addEventListener("click", async () => {
  $("regen").disabled = true;
  $("regen").textContent = "Computing…";
  try { await api("/api/report", { run: activeRun }); await loadState(); }
  catch (e) { alert(e.message); }
  $("regen").disabled = false;
  $("regen").textContent = "Regenerate report";
});

// ------------------------------------------------------------------- compare

$("cmp-toggle").addEventListener("click", () => {
  $("compare").hidden = !$("compare").hidden;
});

$("docompare").addEventListener("click", async () => {
  const runs = [...document.querySelectorAll("#runchecks input:checked")]
    .map((c) => c.value);
  const mode = document.querySelector("input[name=cmode]:checked").value;
  $("docompare").disabled = true;
  $("docompare").textContent = "Computing…";
  try {
    const r = await api("/api/compare", { runs, mode });
    const bust = Date.now();
    const head = ["run", "steps", "infections", "attack", "peak sick", "R0",
                  "mean R", "deaths", "PPE prot."];
    const rows = r.table.map((row) =>
      `<tr><td>${escapeHtml(row.run)}</td><td>${row.steps}</td>` +
      `<td>${row.infections}</td><td>${fmt.pct(row.attack_rate)}</td>` +
      `<td>${row.peak_sick}</td><td>${fmt.num(row.r0)}</td>` +
      `<td>${fmt.num(row.mean_r)}</td><td>${row.deaths}</td>` +
      `<td>${fmt.mult(row.ppe_protection)}</td></tr>`).join("");
    $("cmp-out").innerHTML =
      `<table class="cmp-table"><thead><tr>` +
      head.map((h) => `<th>${h}</th>`).join("") +
      `</tr></thead><tbody>${rows}</tbody></table>` +
      `<div class="plots">` +
      r.plots.map((u) => `<figure><img src="${u}?v=${bust}" /></figure>`).join("") +
      `</div>`;
  } catch (e) { alert(e.message); }
  $("docompare").disabled = false;
  $("docompare").textContent = "Compare";
});

// ---------------------------------------------------------------- chat pane

let lastRender = "";
const seenDone = new Set();

function codeCard(ev, autoRun) {
  const chip = `<span class="chip ${ev.status}">${ev.status}</span>`;
  const approve = ev.status === "pending" && !autoRun
    ? `<div class="approve-row">
         <button onclick="decide(${ev.id}, true)">Run</button>
         <button onclick="decide(${ev.id}, false)">Deny</button>
       </div>` : "";
  const out = ev.output
    ? `<pre class="out">${escapeHtml(ev.output)}</pre>` : "";
  return `<div class="msg code"><div class="head">python ${chip}</div>` +
         `<pre>${escapeHtml(ev.code)}</pre>${approve}${out}</div>`;
}

window.decide = async (id, approved) => {
  try { await api("/api/approve", { run: activeRun, id, approved }); }
  catch (e) { /* raced */ }
  poll();
};

function render(data) {
  const html = data.events.map((ev) => {
    if (ev.type === "user") return `<div class="msg user">${escapeHtml(ev.text)}</div>`;
    if (ev.type === "assistant") return `<div class="msg assistant">${escapeHtml(ev.text)}</div>`;
    if (ev.type === "error") return `<div class="msg error">${escapeHtml(ev.text)}</div>`;
    if (ev.type === "tool") return `<div class="msg tool">⌕ ${escapeHtml(ev.text)}</div>`;
    if (ev.type === "notice") return `<div class="msg notice">${escapeHtml(ev.text)}</div>`;
    if (ev.type === "code") return codeCard(ev, data.auto_run);
    if (ev.type === "plots") {
      return `<div class="msg plots">` +
             ev.urls.map((u) => `<img src="${u}" alt="chat plot" />`).join("") + `</div>`;
    }
    return "";
  }).join("") + (data.busy ? `<div class="thinking">working</div>` : "");

  if (html !== lastRender) {
    const t = $("transcript");
    const pinned = t.scrollTop + t.clientHeight >= t.scrollHeight - 40;
    t.innerHTML = html;
    if (pinned) t.scrollTop = t.scrollHeight;
    lastRender = html;
  }
  $("send").disabled = data.busy;
  $("stop").hidden = !data.busy;
  $("autorun").checked = data.auto_run;
  const donePlots = data.events.some(
    (ev) => ev.type === "done" && !seenDone.has(ev.id) && seenDone.add(ev.id));
  if (donePlots) { loadState(); loadFiles(); } // a turn may add files/plots
}

async function poll() {
  if (!activeRun) return;
  try { render(await api(`/api/events?run=${encodeURIComponent(activeRun)}`)); }
  catch (e) { /* server restarting */ }
}

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  try { await api("/api/chat", { run: activeRun, question: q }); }
  catch (err) { alert(err.message); }
  poll();
});

$("autorun").addEventListener("change", async (e) => {
  await api("/api/autorun", { run: activeRun, enabled: e.target.checked });
});

$("stop").addEventListener("click", () =>
  api("/api/stop", { run: activeRun }).then(poll));

$("reset").addEventListener("click", async () => {
  if (!confirm("Clear the conversation and the sandbox session?")) return;
  try { await api("/api/reset", { run: activeRun }); }
  catch (e) { alert(e.message); return; }
  lastRender = "";
  seenDone.clear();
  poll();
});

// ------------------------------------------------------------- @file mentions

let fileList = [];
async function loadFiles() {
  if (!activeRun) return;
  try {
    fileList = (await api(`/api/files?run=${encodeURIComponent(activeRun)}`)).files;
  } catch (e) { /* offline */ }
}

const qInput = $("question"), mbox = $("mentions");
let mSel = 0, mItems = [];

function currentToken() {
  const upto = qInput.value.slice(0, qInput.selectionStart);
  const m = upto.match(/@([\w./-]*)$/);
  return m ? { start: upto.length - m[0].length, query: m[1] } : null;
}

function updateMentions() {
  const tok = currentToken();
  mItems = tok
    ? fileList.filter((f) => f.toLowerCase().includes(tok.query.toLowerCase())).slice(0, 8)
    : [];
  if (!mItems.length) { mbox.hidden = true; return; }
  mSel = Math.min(mSel, mItems.length - 1);
  mbox.innerHTML = mItems.map((f, i) =>
    `<div class="mention ${i === mSel ? "sel" : ""}" data-f="${escapeHtml(f)}">${escapeHtml(f)}</div>`
  ).join("");
  mbox.hidden = false;
}

function insertMention(f) {
  const tok = currentToken();
  if (!tok) return;
  const tail = qInput.value.slice(qInput.selectionStart);
  qInput.value = qInput.value.slice(0, tok.start) + "@" + f + " " + tail;
  mbox.hidden = true;
  qInput.focus();
}

qInput.addEventListener("input", () => { mSel = 0; updateMentions(); });
qInput.addEventListener("blur", () => setTimeout(() => { mbox.hidden = true; }, 150));
qInput.addEventListener("keydown", (e) => {
  if (mbox.hidden) return;
  if (e.key === "ArrowDown") { e.preventDefault(); mSel = (mSel + 1) % mItems.length; updateMentions(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); mSel = (mSel + mItems.length - 1) % mItems.length; updateMentions(); }
  else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); insertMention(mItems[mSel]); }
  else if (e.key === "Escape") { mbox.hidden = true; }
});
mbox.addEventListener("mousedown", (e) => {
  const f = e.target.closest(".mention")?.dataset.f;
  if (f) { e.preventDefault(); insertMention(f); }
});

// --------------------------------------------------------------------- init

(async function init() {
  await loadRuns();
  if (location.hash === "#compare") $("compare").hidden = false;
  loadState();
  loadFiles();
  poll();
  setInterval(poll, 900);
})();
