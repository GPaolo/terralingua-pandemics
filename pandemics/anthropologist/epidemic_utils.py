"""Loaders and epidemic metrics for a run folder of the pandemics scenario.

Everything reads files the run writes as it goes, so it works on a run still
in progress: ``params.json``, ``world_state.jsonl`` (one world line per step,
written by the mechanic) and ``open_gridworld.log`` (the event log).

Timing rule. The world line for step t is written during step t, after that
step's infections and before its deaths. So the infections of frame t are the
``VIRAL_INFECTION`` events stamped t, and a being that dies at step t is still
in frame t and gone from frame t+1.
"""

import json
from collections import defaultdict
from pathlib import Path

from pandemics.epidemic import EpidemicOptions

AGENT_FIELD_DEFAULTS = {
    "n_viral": 0, "n_sick": 0, "n_ppe": 0, "n_recovered": 0, "n_bedridden": 0,
}


def load_params(run_dir) -> dict:
    """params.json as a nested {agent, env, run} dict ({} if absent)."""
    path = Path(run_dir) / "params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def scenario_options(params: dict) -> EpidemicOptions:
    """The sickness settings of the run, with the defaults filled in."""
    return EpidemicOptions.model_validate(
        params.get("run", {}).get("scenario_options", {})
    )


def load_frames(run_dir):
    """world_state.jsonl as (meta, frames). Each frame has ``t``, ``agents``
    (tag -> dict of the agent fields), ``artifacts`` as a set of
    (row, col, name, kind) and ``food_total``. Changes are replayed, so every
    frame holds the full set of artifacts on the map. A resumed run repeats
    the steps after its checkpoint; the latest line of a step wins."""
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
    by_t = {frame["t"]: frame for frame in frames}
    return meta, [by_t[t] for t in sorted(by_t)]


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
    """One record per infection, from the ``VIRAL_INFECTION`` events.

    ``parent`` is the infection it came from (None for an outbreak case),
    ``generation`` the number of parent links up to an outbreak case,
    ``secondary`` the number of infections it caused. ``removed_at`` is the
    step of the ``VIRAL_HEALED`` event or of the host's death; None means the
    infection is still active. ``outcome`` is died, recovered or active.
    """
    healed_at = {}
    deaths_by_tag = defaultdict(list)
    for e in events:
        if e.get("event") == "VIRAL_HEALED":
            healed_at[e.get("infection_id")] = e.get("timestamp")
        elif e.get("event") == "AGENT_DIED":
            deaths_by_tag[e.get("agent_tag")].append(e.get("timestamp"))

    records, by_id = [], {}
    for e in events:
        if e.get("event") != "VIRAL_INFECTION":
            continue
        rec = {
            "infection": e.get("infection_id"),
            "strain": e.get("strain", "virus"),
            "host_tag": e.get("agent_tag"),
            "host_name": e.get("agent_name"),
            "t": e.get("timestamp"),
            "incubation": e.get("incubation", 0),
            "parent": e.get("source_infection_id"),
            "source_kind": e.get("source_kind"),
            "source_tag": e.get("source_tag"),
            "source_name": e.get("source_name"),
            "secondary": 0,
        }
        died_at = min(
            (t for t in deaths_by_tag.get(rec["host_tag"], []) if t >= rec["t"]),
            default=None,
        )
        healed = healed_at.get(rec["infection"])
        if died_at is not None and (healed is None or died_at <= healed):
            rec["removed_at"], rec["outcome"] = died_at, "died"
        elif healed is not None:
            rec["removed_at"], rec["outcome"] = healed, "recovered"
        else:
            rec["removed_at"], rec["outcome"] = None, "active"
        records.append(rec)
        by_id[rec["infection"]] = rec

    for rec in records:
        parent = by_id.get(rec["parent"])
        if parent is not None:
            parent["secondary"] += 1
    for rec in records:
        gen, cur = 0, rec
        while cur is not None and cur["parent"] is not None:
            cur = by_id.get(cur["parent"])
            gen += 1
        rec["generation"] = gen
    return records


def death_records(events):
    return [
        {"tag": e.get("agent_tag"), "name": e.get("agent_name"),
         "t": e.get("timestamp"), "reason": e.get("reason")}
        for e in events if e.get("event") == "AGENT_DIED"
    ]


def burial_records(events):
    """One record per burial: who dug, when, whose remains, and whether the
    grave infected the digger."""
    return [
        {"t": e.get("timestamp"), "tag": e.get("agent_tag"),
         "name": e.get("agent_name"), "infected": bool(e.get("infected")),
         "remains": e.get("remains")}
        for e in events if e.get("event") == "BURIAL"
    ]


def health_centers(events):
    """Every seeded health center's name, cell, care radius and care parameters."""
    return [
        {"name": (e.get("artifact") or {}).get("name"),
         "pose": tuple(e.get("position") or ()),
         "radius": int((e.get("artifact") or {}).get("radius", 1)),
         "hazard_multiplier": float((e.get("artifact") or {}).get("hazard_multiplier", 0.5)),
         "heal_probability": float((e.get("artifact") or {}).get("heal_probability", 0.0))}
        for e in events
        if e.get("event") == "ARTIFACT_ADDED"
        and (e.get("artifact") or {}).get("art_type") == "health_center"
    ]


def torus_distance(a, b, grid_size):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(min(dr, grid_size - dr), min(dc, grid_size - dc))


def care_coverage(centers, grid_size):
    """Cells inside at least one center's care radius."""
    return sum(
        1 for r in range(grid_size) for c in range(grid_size)
        if any(torus_distance((r, c), ctr["pose"], grid_size) <= ctr["radius"]
               for ctr in centers)
    )


def care_series(frames, centers, grid_size):
    """Per-frame health-center reach: beings inside any center's care radius,
    split by sickness."""
    if not centers or not grid_size:
        return []
    series = []
    for fr in frames:
        agents = list(fr["agents"].values())
        in_care = [
            a for a in agents
            if any(torus_distance((a["row"], a["col"]), c["pose"], grid_size) <= c["radius"]
                   for c in centers)
        ]
        series.append({
            "t": fr["t"],
            "in_care": len(in_care),
            "sick_in_care": sum(1 for a in in_care if a["n_sick"] > 0),
            "sick_total": sum(1 for a in agents if a["n_sick"] > 0),
        })
    return series


def ppe_names(events):
    """Names of every protective equipment artifact the run created."""
    return {
        (e.get("artifact") or {}).get("name")
        for e in events
        if e.get("event") == "ARTIFACT_ADDED"
        and (e.get("artifact") or {}).get("art_type") == "ppe"
    }


def ppe_transfers(events):
    """Successful pickups, drops and gifts of protective equipment."""
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
    """Per-frame population counts. Susceptible, incubating, sick and
    recovered are disjoint, since recovery gives immunity. Virus deaths are
    the deaths with reason "sickness"."""
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
        cum_inf += new_by_t.get(fr["t"], 0)
        cum_dead += dead_by_t.get(fr["t"] - 1, 0)
        cum_dead_virus += virus_dead_by_t.get(fr["t"] - 1, 0)
        series.append({
            "t": fr["t"],
            "alive": len(agents),
            "susceptible": len(agents) - viral - recovered,
            "incubating": viral - sick,
            "sick": sick,
            "pct_sick": 100.0 * sick / len(agents) if agents else 0.0,
            "recovered": recovered,
            "ppe_carriers": sum(1 for a in agents if a["n_ppe"] > 0),
            "new_infections": new_by_t.get(fr["t"], 0),
            "cum_infections": cum_inf,
            "cum_deaths": cum_dead,
            "cum_deaths_virus": cum_dead_virus,
            "cum_deaths_other": cum_dead - cum_dead_virus,
            "food_total": fr["food_total"],
        })
    return series


def exposure_records(events):
    """One record per chance to catch the sickness, from the
    ``VIRAL_EXPOSURE`` events. ``ppe`` is True when protective equipment
    lowered the chance."""
    return [
        {"t": e.get("timestamp"), "tag": e.get("agent_tag"),
         "source_kind": e.get("source_kind"),
         "probability": e.get("probability"), "protection": e.get("protection"),
         "ppe": (e.get("protection") or 1.0) < 1.0,
         "infected": bool(e.get("infected"))}
        for e in events if e.get("event") == "VIRAL_EXPOSURE"
    ]


def ppe_efficiency(exposures, configured_protection=None):
    """Transmission rates split by protective equipment. ``protection_realized``
    is the with/without rate ratio, the measured counterpart of the
    ``ppe_protection`` setting. ``exposure_steps`` counts distinct
    (step, being) pairs, ``contacts`` counts single chances."""
    groups = {
        True: {"exposure_steps": set(), "contacts": 0, "infections": 0},
        False: {"exposure_steps": set(), "contacts": 0, "infections": 0},
    }
    for e in exposures:
        g = groups[e["ppe"]]
        g["exposure_steps"].add((e["t"], e["tag"]))
        g["contacts"] += 1
        g["infections"] += int(e["infected"])

    def rates(g):
        steps = len(g["exposure_steps"])
        return {
            "exposure_steps": steps,
            "contacts": g["contacts"],
            "infections": g["infections"],
            "rate_per_contact": (
                g["infections"] / g["contacts"] if g["contacts"] else None
            ),
            "rate_per_exposure_step": g["infections"] / steps if steps else None,
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
    """Mean secondary infections per completed infection, by generation.
    Active infections are left out; generations 0 and 1 give R0."""
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
    """Steps between an infection and each infection it caused."""
    at = {r["infection"]: r["t"] for r in infections}
    return [
        r["t"] - at[r["parent"]]
        for r in infections
        if r["parent"] in at and r["t"] is not None
    ]


def compute_all(run_dir):
    """Everything at once: (metrics, series, infections, exposures).
    ``metrics`` is the JSON-safe dict report.py writes out."""
    run_dir = Path(run_dir)
    params = load_params(run_dir)
    options = scenario_options(params)
    meta, frames = load_frames(run_dir)
    events = load_events(run_dir)
    infections = infection_records(events)
    deaths = death_records(events)
    series = status_series(frames, infections, deaths)
    exposures = exposure_records(events)

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
            "index_cases": sum(1 for r in infections if r["source_kind"] == "outbreak"),
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
            "efficiency": ppe_efficiency(exposures, options.ppe_protection),
        },
    }
    return metrics, series, infections, exposures
