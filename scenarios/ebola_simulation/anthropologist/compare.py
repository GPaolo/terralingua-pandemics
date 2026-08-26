"""Cross-run comparison: seed-averaged bands or side-by-side overlays.

Usage:
    python scenarios/ebola_simulation/anthropologist/compare.py logs/run_a logs/run_b \
        [--mode average|sidebyside] [--out DIR]

average treats the runs as seeds of one configuration (mean line, min-max
band); sidebyside overlays them as distinct sets (max 6 — the categorical
palette has 6 slots). Runs of different lengths are averaged over whichever
runs reach each day.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epidemic_utils as eu
import report
from report import AMBER, BEING, CYAN, RED, label_ends, save, style, top_legend

# viz light-theme categorical slots, fixed order, validated on #fcfcfb.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
MAX_SIDE_BY_SIDE = len(SLOTS)


def load_runs(run_dirs):
    data = []
    for run_dir in run_dirs:
        metrics, series, _, _ = eu.compute_all(Path(run_dir))
        data.append({"name": Path(run_dir).name, "metrics": metrics,
                     "series": series})
    return data


def _mean_band(data, key):
    """Per-day (mean, min, max) over the runs that reach that day."""
    length = max(len(d["series"]) for d in data)
    mean, lo, hi = [], [], []
    for t in range(length):
        vals = [d["series"][t][key] for d in data if t < len(d["series"])]
        mean.append(sum(vals) / len(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    return list(range(length)), mean, lo, hi


def plot_average_curves(data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    entries = []
    for key, color, label, dashed in [
        ("susceptible", BEING, "susceptible", False),
        ("incubating", AMBER, "incubating", False),
        ("sick", RED, "sick", False),
        ("recovered", CYAN, "recovered", True),
    ]:
        ts, mean, lo, hi = _mean_band(data, key)
        ax.fill_between(ts, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(ts, mean, color=color, linestyle="--" if dashed else "-",
                label=label)
        entries.append((ts, mean, label))
    label_ends(ax, entries)
    ax.set_xlabel("day")
    ax.set_ylabel("beings (mean, min–max band)")
    ax.set_title(f"Epidemic curves averaged over {len(data)} seeds", pad=26)
    top_legend(ax, ncols=4)
    ax.margins(x=0.10)
    return save(fig, out_dir, "avg_epidemic_curves.png")


def plot_average_infections(data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ts, mean, lo, hi = _mean_band(data, "cum_infections")
    ax.fill_between(ts, lo, hi, color=RED, alpha=0.18, linewidth=0)
    ax.plot(ts, mean, color=RED)
    label_ends(ax, [(ts, mean, "mean")])
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative infections")
    ax.set_title(f"Cumulative infections over {len(data)} seeds "
                 "(mean, min–max band)")
    ax.margins(x=0.08)
    return save(fig, out_dir, "avg_infections.png")


def plot_side_by_side_curves(data, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for ax, key, ylabel in [(ax1, "sick", "sick beings"),
                            (ax2, "cum_infections", "cumulative infections")]:
        entries = []
        for d, color in zip(data, SLOTS):
            ts = [s["t"] for s in d["series"]]
            ys = [s[key] for s in d["series"]]
            ax.plot(ts, ys, color=color, label=d["name"])
            entries.append((ts, ys, d["name"]))
        label_ends(ax, entries)
        ax.set_ylabel(ylabel)
        ax.margins(x=0.14)
    ax1.set_title("Runs side by side", pad=26)
    top_legend(ax1, ncols=min(len(data), 3))
    ax2.set_xlabel("day")
    return save(fig, out_dir, "cmp_curves.png")


def plot_side_by_side_metrics(data, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    xs = range(len(data))
    colors = SLOTS[:len(data)]
    for ax, get, title, fmt in [
        (ax1, lambda m: m["outbreak"]["attack_rate"] or 0, "Attack rate", "{:.0%}"),
        (ax2, lambda m: m["r0"]["empirical_r0"], "Empirical R0 (gens 0–1)", "{:.2f}"),
    ]:
        vals = [get(d["metrics"]) for d in data]
        ax.bar(xs, [v or 0 for v in vals], color=colors, width=0.6)
        for i, v in enumerate(vals):
            ax.annotate("n/a" if v is None else fmt.format(v),
                        (i, v or 0), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=report.INK)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([d["name"] for d in data], rotation=20, ha="right")
        ax.set_title(title)
        ax.margins(y=0.2)
    return save(fig, out_dir, "cmp_metrics.png")


def summary_table(data):
    rows = []
    for d in data:
        m = d["metrics"]
        eff = m["ppe"]["efficiency"]
        rows.append({
            "run": d["name"],
            "steps": m["steps"],
            "infections": m["outbreak"]["infections"],
            "attack_rate": m["outbreak"]["attack_rate"],
            "peak_sick": m["outbreak"]["peak_sick"],
            "r0": m["r0"]["empirical_r0"],
            "mean_r": m["r0"]["overall_mean_r"],
            "deaths": m["population"]["deaths"],
            "ppe_protection": eff["protection_realized"],
        })
    return rows


def compare(run_dirs, mode, out_dir):
    if mode not in ("average", "sidebyside"):
        raise ValueError(f"unknown mode {mode}")
    if mode == "sidebyside" and len(run_dirs) > MAX_SIDE_BY_SIDE:
        raise ValueError(f"side-by-side is limited to {MAX_SIDE_BY_SIDE} runs "
                         "(one categorical slot each) — use average instead")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_runs(run_dirs)
    style()
    if mode == "average":
        plots = [plot_average_curves(data, out_dir),
                 plot_average_infections(data, out_dir)]
    else:
        plots = [plot_side_by_side_curves(data, out_dir),
                 plot_side_by_side_metrics(data, out_dir)]
    result = {"mode": mode, "runs": [d["name"] for d in data],
              "plots": [p.name for p in plots], "table": summary_table(data)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="logs/<exp_name> ...")
    parser.add_argument("--mode", choices=["average", "sidebyside"],
                        default="average")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: <logs root>/_comparisons)")
    args = parser.parse_args()
    out = args.out or args.runs[0].resolve().parent / "_comparisons"
    result = compare(args.runs, args.mode, out)
    for row in result["table"]:
        print(row)
    print(f"Written to {out}/: summary.json, " + ", ".join(result["plots"]))


if __name__ == "__main__":
    main()
