"""Epidemic report for a viral run: metrics + plots.

Usage:
    python scenarios/ebola_simulation/anthropologist/report.py logs/<exp_name>

Writes to logs/<exp_name>/epidemic_analysis/ (override with --out):
    metrics.json, timeseries.csv, epidemic_curves.png, infections.png,
    transmission_tree.png, secondary_cases.png, ppe.png
and prints a text summary. Works on a run still in progress.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epidemic_utils as eu

# Dashboard light-theme palette (viz/static/style.css); amber sits below 3:1
# on white, so amber series are always direct-labeled.
INK, INK2 = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BEING, AMBER, RED, BLUE = "#898781", "#fab219", "#d03b3b", "#2a78d6"
CYAN = "#0a9ac0"  # --recovered, dashed like the dashboard's recovered line


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
        "axes.titlesize": 12, "lines.linewidth": 2.0,
        "legend.frameon": False, "figure.dpi": 150,
    })


def label_end(ax, xs, ys, text):
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK)


def label_ends(ax, entries):
    """Direct labels at each line's end, nudged apart when finals coincide."""
    if not entries:
        return
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.045
    placed = []
    for xs, ys, text in sorted(entries, key=lambda e: e[1][-1]):
        y = ys[-1]
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(text, (xs[-1], y), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK)


def top_legend(ax, ncols):
    """Legend in the title pad, where it can never sit on data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=ncols,
              fontsize=8, borderaxespad=0, handlelength=1.6)


def save(fig, out_dir, name):
    fig.tight_layout()
    path = Path(out_dir) / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_epidemic_curves(series, out_dir):
    ts = [s["t"] for s in series]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    entries = []
    for key, color, label, dashed in [
        ("susceptible", BEING, "susceptible", False),
        ("incubating", AMBER, "incubating", False),
        ("sick", RED, "sick", False),
        ("recovered", CYAN, "recovered (immune)", True),
    ]:
        ys = [s[key] for s in series]
        ax1.plot(ts, ys, color=color, linestyle="--" if dashed else "-", label=label)
        entries.append((ts, ys, label))
    label_ends(ax1, entries)
    ax1.set_ylabel("beings")
    ax1.set_title("Epidemic curves (the four states are disjoint)", pad=26)
    top_legend(ax1, ncols=4)
    ax1.margins(x=0.10)

    # Red↔ink sits in the CVD 6–8 band, so cause is never hue-alone:
    # line style and the end labels carry it too.
    entries = []
    for key, color, label, dashed in [
        ("cum_deaths_virus", RED, "virus", False),
        ("cum_deaths_other", INK2, "other causes", True),
    ]:
        ys = [s[key] for s in series]
        ax2.plot(ts, ys, color=color, linestyle="--" if dashed else "-", label=label)
        entries.append((ts, ys, label))
    label_ends(ax2, entries)
    ax2.set_xlabel("day")
    ax2.set_ylabel("cumulative deaths")
    ax2.set_title("Deaths by cause", pad=26)
    top_legend(ax2, ncols=2)
    ax2.margins(x=0.10)
    return save(fig, out_dir, "epidemic_curves.png")


def plot_infections(series, out_dir):
    ts = [s["t"] for s in series]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.bar(ts, [s["new_infections"] for s in series], color=RED, width=1.0)
    ax1.set_ylabel("new infections / day")
    ax1.set_title("Incidence")
    ys = [s["cum_infections"] for s in series]
    ax2.plot(ts, ys, color=RED)
    label_end(ax2, ts, ys, "total")
    ax2.set_ylabel("cumulative infections")
    ax2.set_xlabel("day")
    ax2.margins(x=0.08)
    return save(fig, out_dir, "infections.png")


def plot_transmission_tree(infections, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    at = {r["artifact"]: r for r in infections}
    # Stack episodes that share (t, generation) so nodes never overlap.
    seen = {}
    pos = {}
    for r in sorted(infections, key=lambda r: (r["t"], r["artifact"])):
        key = (r["t"], r["generation"])
        off = seen.get(key, 0)
        seen[key] = off + 1
        pos[r["artifact"]] = (r["t"], r["generation"] + off * 0.12)
    for r in infections:
        parent = at.get(r["source_artifact"])
        if parent is not None:
            (x0, y0), (x1, y1) = pos[parent["artifact"]], pos[r["artifact"]]
            ax.plot([x0, x1], [y0, y1], color=AXIS, linewidth=1.0, zorder=1)
    # Recovered wears the dashboard's cyan ring, never a fill.
    states = [
        ("died", RED, SURFACE, 1.2, "died"),
        ("recovered", SURFACE, CYAN, 1.6, "recovered (immune)"),
        ("active", AMBER, SURFACE, 1.2, "still active"),
    ]
    for outcome, face, edge, lw, label in states:
        pts = [pos[r["artifact"]] for r in infections if r["outcome"] == outcome]
        if pts:
            ax.scatter(*zip(*pts), s=45, facecolor=face, edgecolors=edge,
                       linewidths=lw, zorder=2, label=label)
    for r in infections:
        x, y = pos[r["artifact"]]
        ax.annotate(r["host_name"] or r["host_tag"], (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("day of infection")
    ax.set_ylabel("generation")
    ax.set_title("Transmission tree (index cases at generation 0)")
    if infections:
        ax.set_yticks(sorted({r["generation"] for r in infections}))
        ax.legend(loc="upper left", fontsize=9)
    return save(fig, out_dir, "transmission_tree.png")


def plot_secondary_cases(infections, r0, out_dir):
    completed = [r["secondary"] for r in infections if r["removed_at"] is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    if completed:
        bins = range(0, max(completed) + 2)
        ax1.hist(completed, bins=bins, color=BEING, rwidth=0.9, align="left")
        if r0["overall_mean_r"] is not None:
            ax1.axvline(r0["overall_mean_r"], color=INK, linewidth=1.2, linestyle="--")
            ax1.annotate(f"mean R = {r0['overall_mean_r']:.2f}",
                         (r0["overall_mean_r"], ax1.get_ylim()[1] * 0.95),
                         xytext=(5, 0), textcoords="offset points",
                         fontsize=9, color=INK)
    ax1.set_xlabel("secondary infections per completed episode")
    ax1.set_ylabel("episodes")
    ax1.set_title("Secondary case distribution")
    rows = r0["per_generation"]
    if rows:
        gens = [row["generation"] for row in rows]
        ax2.bar(gens, [row["mean_secondary"] for row in rows], color=BEING, width=0.7)
        for row in rows:
            ax2.annotate(f"{row['mean_secondary']:.2f} (n={row['cases']})",
                         (row["generation"], row["mean_secondary"]),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, color=INK)
        ax2.set_xticks(gens)
    ax2.set_xlabel("generation")
    ax2.set_ylabel("mean secondary infections")
    ax2.set_title("R by generation")
    ax2.margins(y=0.15)
    return save(fig, out_dir, "secondary_cases.png")


def plot_ppe(series, ppe_metrics, out_dir):
    ts = [s["t"] for s in series]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    entries = []
    for key, color, label in [
        ("alive", BEING, "alive"),
        ("ppe_carriers", BLUE, "PPE carriers"),
        ("sick", RED, "sick"),
    ]:
        ys = [s[key] for s in series]
        ax1.plot(ts, ys, color=color, label=label)
        entries.append((ts, ys, label))
    label_ends(ax1, entries)
    ax1.set_xlabel("day")
    ax1.set_ylabel("beings")
    ax1.set_title("PPE coverage", pad=26)
    top_legend(ax1, ncols=3)
    ax1.margins(x=0.16)

    eff = ppe_metrics["efficiency"]
    pairs = [("without PPE", eff["without_ppe"], BEING),
             ("with PPE", eff["with_ppe"], BLUE)]
    rates = [(g["rate_per_contact"] or 0.0) for _, g, _ in pairs]
    ax2.bar(range(2), rates, color=[c for _, _, c in pairs], width=0.6)
    for i, (_label, g, _color) in enumerate(pairs):
        note = f"{g['infections']}/{g['contacts']} contacts"
        ax2.annotate(f"{rates[i]:.3f}\n{note}", (i, rates[i]), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels([name for name, _, _ in pairs])
    ax2.set_ylabel("transmission per contact-step")
    title = "PPE efficiency"
    if eff["protection_realized"] is not None:
        title += (f"\nrealized ×{eff['protection_realized']:.2f}"
                  f" (configured ×{eff['protection_configured']})")
    ax2.set_title(title)
    ax2.margins(y=0.2)
    return save(fig, out_dir, "ppe.png")


def write_timeseries(series, out_dir):
    path = Path(out_dir) / "timeseries.csv"
    if not series:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    return path


def print_summary(m):
    o, p, r = m["outbreak"], m["population"], m["r0"]
    print(f"\n=== Epidemic report: {m['run']} ({m['steps']} steps) ===")
    print(f"Population: {p['ever_alive']} beings ever alive, "
          f"{p['final_alive']} at the end "
          f"({p['final_recovered']} recovered/immune), "
          f"{p['deaths']} deaths ({p['deaths_by_reason']}), "
          f"{p['deaths_while_infected']} of them while infected")
    print(f"Outbreak:   {o['index_cases']} index case(s), "
          f"{o['infections']} infections in "
          f"{o['unique_hosts']} hosts (attack rate "
          f"{o['attack_rate']:.0%})"
          if o["attack_rate"] is not None else "Outbreak:   none")
    if o["infections"]:
        print(f"            peak {o['peak_sick']} sick at day {o['peak_sick_t']}, "
              + ("still active at the end of the log"
                 if o["still_active"] else f"over by day {o['last_active_t']}"))
        if o["incubation_mean"] is not None:
            print(f"            mean incubation {o['incubation_mean']:.1f} days"
                  + (f", mean serial interval {o['serial_interval_mean']:.1f} days"
                     if o["serial_interval_mean"] is not None else ""))
        print(f"R0:         empirical {r['empirical_r0']}"
              f" (gens 0-1), overall mean R {r['overall_mean_r']}, "
              f"{r['censored']} censored episode(s)")
    ppe = m["ppe"]
    eff = ppe["efficiency"]
    print(f"PPE:        {ppe['artifacts_created']} artifacts, carriers "
          f"{ppe['initial_carriers']} → {ppe['final_carriers']} "
          f"(peak {ppe['peak_carriers']}), "
          f"transfers {ppe['transfers']}")
    if eff["protection_realized"] is not None:
        print(f"            per-contact transmission "
              f"{eff['without_ppe']['rate_per_contact']:.3f} without PPE vs "
              f"{eff['with_ppe']['rate_per_contact']:.3f} with — realized protection "
              f"×{eff['protection_realized']:.2f} "
              f"(configured ×{eff['protection_configured']}), "
              f"≈{eff['infections_averted']:.1f} infections averted")
    elif eff["with_ppe"]["contacts"]:
        print(f"            {eff['with_ppe']['contacts']} PPE contact-steps, "
              f"{eff['with_ppe']['infections']} infections (no unprotected baseline)")


def generate(run_dir, out_dir=None):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "epidemic_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, series, infections, _ = eu.compute_all(run_dir)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_timeseries(series, out_dir)

    style()
    paths = [plot_epidemic_curves(series, out_dir), plot_infections(series, out_dir)]
    if infections:
        paths.append(plot_transmission_tree(infections, out_dir))
        paths.append(plot_secondary_cases(infections, metrics["r0"], out_dir))
    paths.append(plot_ppe(series, metrics["ppe"], out_dir))

    print_summary(metrics)
    print(f"\nWritten to {out_dir}/: metrics.json, timeseries.csv, "
          + ", ".join(p.name for p in paths))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="logs/<exp_name>")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: <run_dir>/epidemic_analysis)")
    args = parser.parse_args()
    generate(args.run_dir, args.out)


if __name__ == "__main__":
    main()
