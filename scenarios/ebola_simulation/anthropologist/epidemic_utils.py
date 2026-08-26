"""Loaders and epidemic metrics for a viral run under logs/<exp_name>.

Reads only files written as the run goes, so everything works mid-run.
Timing rule used throughout: the world log runs one step ahead of the events,
so frame t is paired with events stamped t-1.
"""

import json
from collections import defaultdict
from pathlib import Path

AGENT_FIELD_DEFAULTS = {"n_viral": 0, "n_sick": 0, "n_ppe": 0, "n_recovered": 0}


def load_params(run_dir) -> dict:
    """params.json as a nested {agent, env, run} dict ({} if absent)."""
    path = Path(run_dir) / "params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_frames(run_dir):
    """world_state.jsonl as (meta, frames), artifact deltas replayed to
    (row, col, name) sets, missing schema fields defaulted to 0."""
    path = Path(run_dir) / "world_state.jsonl"
    meta, frames = None, []
    artifacts: set = set()
    fields = ["row", "col", "energy", "time", "n_inv"]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "meta":
                meta = row
                fields = meta.get("agent_fields", fields)
                continue
            agents = {}
            for tag, values in row["agents"].items():
                agent = dict(AGENT_FIELD_DEFAULTS)
                agent.update({
                    name: values[i]
                    for i, name in enumerate(fields) if i < len(values)
                })
                agents[tag] = agent
            art = row.get("artifacts", {})
            if "set" in art:
                artifacts = {tuple(a) for a in art["set"]}
            else:
                artifacts |= {tuple(a) for a in art.get("add", [])}
                artifacts -= {tuple(a) for a in art.get("del", [])}
            frames.append({
                "t": row["t"],
                "agents": agents,
                "artifacts": set(artifacts),
                "food_total": row.get("food_total", 0.0),
            })
    return meta, frames


def load_events(run_dir):
    """open_gridworld.log as a list of event dicts, in log order."""
    path = Path(run_dir) / "open_gridworld.log"
    events = []
    if not path.exists():
        return events
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in entry:
                events.append(entry)
    return events


def infection_records(events):
    """One record per host-infection episode, with generation, secondary
    count, removed_at (None = still active, i.e. censored) and outcome
    (died / recovered / active — the host's fate, not the artifact's:
    a corpse's remains can stay infectious long after the host died)."""
    removed_at = {}
    deaths_by_tag = defaultdict(list)
    for e in events:
        art = e.get("artifact") or {}
        if e.get("event") == "ARTIFACT_REMOVED" and art.get("art_type") == "viral":
            removed_at[art.get("name")] = e.get("timestamp")
        elif e.get("event") == "AGENT_DIED":
            deaths_by_tag[e.get("agent_tag")].append(e.get("timestamp"))

    records, by_name = [], {}
    for e in events:
        if e.get("event") != "VIRAL_INFECTION":
            continue
        art = e["artifact"]
        rec = {
            "artifact": art["name"],
            "strain": e.get("strain", "virus"),
            "host_tag": e.get("agent_tag"),
            "host_name": e.get("agent_name"),
            "t": e.get("timestamp"),
            "incubation": e.get("incubation", 0),
            "source_artifact": e.get("source_artifact"),
            "source_tag": e.get("source_tag"),
            "source_name": e.get("source_name"),
            "removed_at": removed_at.get(art["name"]),
            "secondary": 0,
        }
        records.append(rec)
        by_name[rec["artifact"]] = rec

    for rec in records:
        parent = by_name.get(rec["source_artifact"])
        if parent is not None:
            parent["secondary"] += 1
    for rec in records:
        gen, cur = 0, rec
        while cur is not None and cur["source_artifact"] is not None:
            cur = by_name.get(cur["source_artifact"])
            gen += 1
        rec["generation"] = gen
        # Death inside the episode window; tags can be reused by respawns,
        # so a death stamped before the infection is a different being.
        died = any(
            t is not None and rec["t"] is not None and t >= rec["t"]
            and (rec["removed_at"] is None or t <= rec["removed_at"])
            for t in deaths_by_tag.get(rec["host_tag"], [])
        )
        rec["outcome"] = ("died" if died
                          else "recovered" if rec["removed_at"] is not None
                          else "active")
    return records


def death_records(events):
    return [
        {"tag": e.get("agent_tag"), "name": e.get("agent_name"),
         "t": e.get("timestamp"), "reason": e.get("reason")}
        for e in events if e.get("event") == "AGENT_DIED"
    ]


def ppe_names(events):
    """Names of every PPE artifact the run created (ARTIFACT_ADDED, art_type ppe)."""
    return {
        (e.get("artifact") or {}).get("name")
        for e in events
        if e.get("event") == "ARTIFACT_ADDED"
        and (e.get("artifact") or {}).get("art_type") == "ppe"
    }


def ppe_transfers(events):
    """Successful PPE pickups, drops and gifts. Transfer events carry only a
    name and are logged on failure too, hence the name-set and status filters."""
    names = ppe_names(events)
    out = []
    for e in events:
        kind = e.get("event")
        if kind not in ("ARTIFACT_PICKUP", "ARTIFACT_DROP", "GIVE_ARTIFACT"):
            continue
        if e.get("artifact_name") not in names or e.get("status") != "Success":
            continue
        out.append({
            "t": e.get("timestamp"), "kind": kind,
            "tag": e.get("agent_tag"), "name": e.get("agent_name"),
            "target_tag": e.get("target_tag"), "target_name": e.get("target_name"),
            "artifact": e.get("artifact_name"),
        })
    return out


def status_series(frames, infections=(), deaths=()):
    """Per-frame population counts; susceptible, incubating, sick and
    recovered are disjoint (recovery is permanent immunity). Virus deaths
    are the ones the env stamps reason "sickness"."""
    new_by_t, dead_by_t = defaultdict(int), defaultdict(int)
    virus_dead_by_t = defaultdict(int)
    for r in infections:
        new_by_t[r["t"]] += 1
    for d in deaths:
        dead_by_t[d["t"]] += 1
        if d["reason"] == "sickness":
            virus_dead_by_t[d["t"]] += 1

    series, cum_inf, cum_dead, cum_dead_virus = [], 0, 0, 0
    for fr in frames:
        agents = list(fr["agents"].values())
        sick = sum(1 for a in agents if a["n_sick"] > 0)
        viral = sum(1 for a in agents if a["n_viral"] > 0)
        recovered = sum(
            1 for a in agents if a["n_recovered"] > 0 and a["n_viral"] == 0
        )
        cum_inf += new_by_t.get(fr["t"] - 1, 0)
        cum_dead += dead_by_t.get(fr["t"] - 1, 0)
        cum_dead_virus += virus_dead_by_t.get(fr["t"] - 1, 0)
        series.append({
            "t": fr["t"],
            "alive": len(agents),
            "susceptible": len(agents) - viral - recovered,
            "incubating": viral - sick,
            "sick": sick,
            "recovered": recovered,
            "ppe_carriers": sum(1 for a in agents if a["n_ppe"] > 0),
            "new_infections": new_by_t.get(fr["t"] - 1, 0),
            "cum_infections": cum_inf,
            "cum_deaths": cum_dead,
            "cum_deaths_virus": cum_dead_virus,
            "cum_deaths_other": cum_dead - cum_dead_virus,
            "food_total": fr["food_total"],
        })
    return series


def exposure_records(frames, infections, grid_size, radius=1):
    """Exposure-steps reconstructed from positions (the env does not log
    failed transmissions): one record per step per susceptible being with a
    symptomatic source in range. Assumes a single strain."""
    viral_names = {r["artifact"] for r in infections}
    newly_by_t = defaultdict(set)
    for r in infections:
        newly_by_t[r["t"]].add(r["host_tag"])

    def dist(a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(min(dr, grid_size - dr), min(dc, grid_size - dc))

    records = []
    for fr in frames:
        tau = fr["t"] - 1
        if tau < 0:
            continue
        newly = newly_by_t.get(tau, set())
        # Beings infected this very step were not yet sources at spread time.
        sources = [
            (tag, (a["row"], a["col"]))
            for tag, a in fr["agents"].items()
            if a["n_sick"] > 0 and tag not in newly
        ]
        # (row, col, name) before schema 5, (row, col, name, kind) after.
        sources += [(None, (a[0], a[1])) for a in fr["artifacts"]
                    if a[2] in viral_names]
        if not sources:
            continue
        for tag, a in fr["agents"].items():
            if a["n_viral"] > 0 and tag not in newly:
                continue  # already hosting: immune to the strain
            pos = (a["row"], a["col"])
            contacts = sum(
                1 for s_tag, s_pos in sources
                if s_tag != tag and dist(pos, s_pos) <= radius
            )
            if contacts > 0 or tag in newly:
                records.append({
                    "t": tau, "tag": tag, "ppe": a["n_ppe"] > 0,
                    "contacts": contacts, "infected": tag in newly,
                })
    return records


def ppe_efficiency(exposures, configured_protection=None):
    """Transmission rates split by PPE; protection_realized is the with/without
    rate ratio, the empirical counterpart of the ppe_protection multiplier."""
    groups = {
        True: {"exposure_steps": 0, "contacts": 0, "infections": 0},
        False: {"exposure_steps": 0, "contacts": 0, "infections": 0},
    }
    for e in exposures:
        g = groups[e["ppe"]]
        g["exposure_steps"] += 1
        g["contacts"] += e["contacts"]
        g["infections"] += int(e["infected"])

    def rates(g):
        return {
            **g,
            "rate_per_contact": (
                g["infections"] / g["contacts"] if g["contacts"] else None
            ),
            "rate_per_exposure_step": (
                g["infections"] / g["exposure_steps"] if g["exposure_steps"] else None
            ),
        }

    with_ppe, without_ppe = rates(groups[True]), rates(groups[False])
    r_ppe, r_no = with_ppe["rate_per_contact"], without_ppe["rate_per_contact"]
    realized = r_ppe / r_no if r_ppe is not None and r_no else None
    averted = (
        groups[True]["contacts"] * r_no - groups[True]["infections"]
        if r_no is not None and groups[True]["contacts"] else None
    )
    return {
        "with_ppe": with_ppe,
        "without_ppe": without_ppe,
        "protection_realized": realized,
        "protection_configured": configured_protection,
        "infections_averted": averted,
    }


def r0_table(infections):
    """Mean secondary infections per completed episode by generation — the
    compute_r0.py estimator (censored excluded, gens 0-1 give R0)."""
    by_gen, censored = defaultdict(list), 0
    for r in infections:
        if r["removed_at"] is None:
            censored += 1
        else:
            by_gen[r["generation"]].append(r["secondary"])
    rows = [
        {"generation": g, "cases": len(v),
         "mean_secondary": sum(v) / len(v), "max_secondary": max(v)}
        for g, v in sorted(by_gen.items())
    ]
    everything = [s for v in by_gen.values() for s in v]
    early = by_gen.get(0, []) + by_gen.get(1, [])
    return {
        "per_generation": rows,
        "total_infections": len(infections),
        "completed": len(everything),
        "censored": censored,
        "overall_mean_r": sum(everything) / len(everything) if everything else None,
        "empirical_r0": sum(early) / len(early) if early else None,
    }


def serial_intervals(infections):
    """Days between an infection and each infection it caused."""
    at = {r["artifact"]: r["t"] for r in infections}
    return [
        r["t"] - at[r["source_artifact"]]
        for r in infections
        if r["source_artifact"] in at and r["t"] is not None
    ]


def compute_all(run_dir):
    """Everything at once: (metrics, series, infections, exposures);
    metrics is the JSON-safe dict report.py writes out."""
    run_dir = Path(run_dir)
    params = load_params(run_dir)
    env_p = params.get("env", {})
    meta, frames = load_frames(run_dir)
    events = load_events(run_dir)
    infections = infection_records(events)
    deaths = death_records(events)
    series = status_series(frames, infections, deaths)

    grid = (meta or {}).get("grid_size") or env_p.get("grid_size")
    radius = env_p.get("viral_infection_radius", 1)
    exposures = exposure_records(frames, infections, grid, radius) if grid else []

    hosts = {r["host_tag"] for r in infections}
    first_at = {}
    for r in infections:
        if r["host_tag"] not in first_at or r["t"] < first_at[r["host_tag"]]:
            first_at[r["host_tag"]] = r["t"]
    ever_alive = set()
    for fr in frames:
        ever_alive |= fr["agents"].keys()

    peak = max(series, key=lambda s: s["sick"], default=None)
    active = [s["t"] for s in series if s["incubating"] + s["sick"] > 0]
    still_active = bool(series) and series[-1]["incubating"] + series[-1]["sick"] > 0
    intervals = serial_intervals(infections)
    incubations = [r["incubation"] for r in infections]
    reasons = defaultdict(int)
    for d in deaths:
        reasons[d["reason"]] += 1

    metrics = {
        "run": run_dir.name,
        "steps": frames[-1]["t"] if frames else 0,
        "population": {
            "ever_alive": len(ever_alive),
            "final_alive": series[-1]["alive"] if series else 0,
            "final_recovered": series[-1]["recovered"] if series else 0,
            "deaths": len(deaths),
            "deaths_by_reason": dict(reasons),
            "deaths_while_infected": sum(
                1 for d in deaths
                if d["tag"] in first_at and first_at[d["tag"]] <= (d["t"] or 0)
            ),
        },
        "outbreak": {
            "index_cases": sum(1 for r in infections if r["source_artifact"] is None),
            "infections": len(infections),
            "unique_hosts": len(hosts),
            "attack_rate": len(hosts) / len(ever_alive) if ever_alive else None,
            "peak_sick": peak["sick"] if peak else 0,
            "peak_sick_t": peak["t"] if peak else None,
            "last_active_t": active[-1] if active else None,
            "still_active": still_active,
            "incubation_mean": (
                sum(incubations) / len(incubations) if incubations else None
            ),
            "serial_interval_mean": (
                sum(intervals) / len(intervals) if intervals else None
            ),
        },
        "r0": r0_table(infections),
        "ppe": {
            "initial_carriers": series[0]["ppe_carriers"] if series else 0,
            "peak_carriers": max((s["ppe_carriers"] for s in series), default=0),
            "final_carriers": series[-1]["ppe_carriers"] if series else 0,
            "artifacts_created": len(ppe_names(events) - {None}),
            "transfers": {
                k: sum(1 for tr in ppe_transfers(events) if tr["kind"] == k)
                for k in ("ARTIFACT_PICKUP", "ARTIFACT_DROP", "GIVE_ARTIFACT")
            },
            "efficiency": ppe_efficiency(exposures, env_p.get("ppe_protection")),
        },
    }
    return metrics, series, infections, exposures
