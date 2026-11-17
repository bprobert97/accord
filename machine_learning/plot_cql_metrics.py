"""
plot_cql_metrics.py

Publication-quality plots for:
  (a) the harvested offline dataset produced by harvest_dag_dataset.py
  (b) the CQL training run itself, read from a CSV metrics log.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_publication_style(font_size: int = 9) -> None:
    """Apply a compact, serif, high-DPI style suited to two-column conference templates."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": font_size,
        "font.family": "serif",
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "figure.figsize": (3.6, 2.6),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average; returns an array the same length as x."""
    if window <= 1 or len(x) == 0:
        return x.astype(float)
    cumsum = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty_like(x, dtype=float)
    half = window // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def save(fig: plt.Figure, outdir: Path, name: str, fmt: str) -> None:
    """Save the figure to the specified directory."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """Load the harvested .npz archive into a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def _group_by_step(
    data: Dict[str, np.ndarray], values: np.ndarray, max_steps: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average values across episodes at each within-episode step_idx."""
    step_idx = data["step_idx"]
    n_steps = int(step_idx.max()) + 1 if max_steps is None else max_steps
    means = np.full(n_steps, np.nan)
    stds = np.full(n_steps, np.nan)
    for t in range(n_steps):
        mask = step_idx == t
        if mask.any():
            means[t] = values[mask].mean()
            stds[t] = values[mask].std()
    steps = np.arange(n_steps)
    valid = ~np.isnan(means)
    return steps[valid], means[valid], stds[valid]


def plot_score_and_acceptance(data: Dict[str, np.ndarray], outdir: Path, fmt: str, window: int = 25) -> None:
    """Consensus score, correctness score, and acceptance fraction vs. attack step."""
    steps_s, mean_s, _ = _group_by_step(data, data["scores"])

    fig, ax1 = plt.subplots()
    ax1.plot(steps_s, moving_average(mean_s, window), color="C3", label="consensus score")

    if "correctness_scores" in data:
        _, mean_c, _ = _group_by_step(data, data["correctness_scores"])
        ax1.plot(steps_s, moving_average(mean_c, window), color="C4", label="correctness score")

    ax1.set_xlabel("attack step")
    ax1.set_ylabel("score", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")

    if "accepted_frac" in data:
        steps_a, mean_a, _ = _group_by_step(data, data["accepted_frac"])
        ax2 = ax1.twinx()
        ax2.plot(steps_a, moving_average(mean_a, window), color="C5", label="acceptance fraction", linestyle="--")
        ax2.set_ylabel("acceptance fraction", color="C5")
        ax2.tick_params(axis="y", labelcolor="C5")

    ax1.set_title("Consensus, Correctness, and Acceptance")
    ax1.legend(loc="upper left")
    save(fig, outdir, "score_and_correctness", fmt)


def plot_reputation_over_time(
    data: Dict[str, np.ndarray], outdir: Path, fmt: str, window: int = 25, reputation_floor: float = 0.5
) -> None:
    """Mean compromised-node reputation vs. attack step."""
    steps, mean, std = _group_by_step(data, data["reps"])
    fig, ax = plt.subplots()
    ax.plot(steps, moving_average(mean, window), color="C2")
    ax.fill_between(steps, mean - std, mean + std, color="C2", alpha=0.15, linewidth=0)
    ax.axhline(reputation_floor, color="firebrick", linewidth=1.0, linestyle="--", label="reputation floor")
    ax.set_xlabel("attack step")
    ax.set_ylabel("mean reputation")
    ax.set_title("Node reputation over the attack episode")
    ax.legend(loc="best")
    save(fig, outdir, "reputation_over_time", fmt)


def load_training_log(path: str) -> List[Dict[str, float]]:
    """Load a metrics CSV written during CQLTrainer.train()."""
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: float(v) for k, v in raw.items() if v not in (None, "")}
            if row:
                rows.append(row)
    return rows


def _plot_single_series(
    rows: List[Dict[str, float]], x_key: str, y_key: str, ylabel: str, title: str,
    outdir: Path, fmt: str, name: str, window: int = 1, color: str = "C0"
) -> None:
    """Plot a single time series from the training logs."""
    pairs = [(r[x_key], r[y_key]) for r in rows if x_key in r and y_key in r]
    if not pairs:
        print(f"Skipping {name}: no rows have both '{x_key}' and '{y_key}'")
        return
    pairs.sort(key=lambda p: p[0])
    x = np.asarray([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])
    fig, ax = plt.subplots()
    ax.plot(x, moving_average(y, window), color=color)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save(fig, outdir, name, fmt)


def make_all_training_plots(log_path: str, outdir: Path, fmt: str, window: int = 50) -> None:
    """Generate all diagnostic plots from the CQL training run."""
    log = load_training_log(log_path)
    _plot_single_series(log, "step", "critic_loss", "critic loss", "Critic loss", outdir, fmt, "critic_loss", window, "C0")
    _plot_single_series(log, "step", "actor_loss", "actor loss", "Actor loss", outdir, fmt, "actor_loss", window, "C1")
    _plot_single_series(log, "step", "cql_gap", "CQL gap", "Conservative Q-learning gap", outdir, fmt, "cql_gap", window, "C2")
    _plot_single_series(log, "step", "alpha", r"$\alpha$", "CQL penalty weight", outdir, fmt, "cql_alpha", window, "C3")
    _plot_single_series(log, "val_step", "val_q_gap", "Q(policy) - Q(data)", "Validation extrapolation gap", outdir, fmt, "val_q_gap", 1, "C5")
    _plot_single_series(log, "val_step", "val_td_error", "TD error", "Validation TD error", outdir, fmt, "val_td_error", 1, "C6")



def plot_attack_profile(
    trajectory_path: str, outdir: Path, fmt: str, window: int = 9, attack_start: int = 90
) -> None:
    """Paper Fig. 3: trained-policy reputation, consensus score, and drift vs. attack step.

    Reads a live_eval_trajectory.csv written by evaluate_cql_live.py (one row per admitted
    attack-phase transaction). ApSTL-compliant: no title, line styles not colour, units on axes,
    panels lettered a)/b). `attack_start` is filter_burn_in_steps + honest_steps (the absolute
    simulation step at which the attack phase begins), used to re-base the x-axis to 0.
    """
    rows = []
    with open(trajectory_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    step = np.array([int(r["step"]) for r in rows])
    rep = np.array([float(r["reputation"]) for r in rows])
    score = np.array([float(r["consensus_score"]) for r in rows])
    drift = np.array([float(r["drift"]) for r in rows])
    astep = step - attack_start

    # mean per attack step, then smooth
    unique = np.unique(astep)
    rep_m = moving_average(np.array([rep[astep == s].mean() for s in unique]), window)
    score_m = moving_average(np.array([score[astep == s].mean() for s in unique]), window)
    drift_m = moving_average(np.array([drift[astep == s].mean() for s in unique]), window)

    _vir = plt.get_cmap("viridis")
    c_rep, c_score, c_drift = _vir(0.15), _vir(0.6), _vir(0.15)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 4.4), sharex=True)
    fig.subplots_adjust(hspace=0.30)
    ax1.plot(unique, rep_m, linestyle="-", color=c_rep, label="reputation")
    ax1.plot(unique, score_m, linestyle="--", color=c_score, label="consensus score")
    ax1.axhline(0.5, linestyle=":", color="0.45", linewidth=1.0)
    ax1.axhline(0.3, linestyle=(0, (1, 1)), color="0.45", linewidth=1.0)
    ax1.set_ylabel("trust factor")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    _bbox = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)
    ax1.text(0.99, 0.52, "reputation floor", transform=ax1.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=7, color="black", bbox=_bbox)
    ax1.text(0.99, 0.32, "score floor", transform=ax1.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=7, color="black", bbox=_bbox)

    ax2.plot(unique, drift_m, linestyle="-", color=c_drift)
    ax2.set_ylabel(r"injected drift, $\sigma$")
    ax2.set_xlabel("attack step")
    ax2.set_xlim(0, unique.max())
    fig.align_ylabels([ax1, ax2])

    # Panel letters below and to the left of each part, outside the plot box,
    # per AIAA multi-part figure guidance.
    ax1.text(0.0, -0.13, "a)", transform=ax1.transAxes, ha="left", va="top")
    ax2.text(0.0, -0.28, "b)", transform=ax2.transAxes, ha="left", va="top")
    save(fig, outdir, "attack_profile", fmt)


def plot_drift_coverage(data: Dict[str, np.ndarray], outdir: Path, fmt: str, num_bins: int = 22) -> None:
    """Paper Fig. 4: histogram of cumulative injected drift across the harvested dataset.

    Drift magnitude is the Euclidean norm of the first two state dimensions (cum_bias_x,
    cum_bias_y). ApSTL-compliant: no title, units on axes, single colourblind-safe grey series.
    """
    mag = np.linalg.norm(data["states"][:, :2], axis=1)
    upper = float(np.ceil(mag.max()))
    fig, ax = plt.subplots()
    ax.hist(mag, bins=num_bins, range=(0, upper),
            color=plt.get_cmap("viridis")(0.3), edgecolor="black", linewidth=0.5)
    ax.set_xlabel(r"cumulative injected drift, $\sigma$")
    ax.set_ylabel("number of transitions")
    ax.set_xlim(0, upper)
    save(fig, outdir, "drift_coverage", fmt)


def main() -> None:
    """Parse CLI arguments and generate the requested plots."""
    parser = argparse.ArgumentParser(description="Plot CQL / DAG-harvester metrics for publication")
    parser.add_argument("--dataset", default=None, help="Path to dag_harvested_dataset.npz")
    parser.add_argument("--train-log", default=None, help="Path to a training metrics CSV")
    parser.add_argument("--trajectory", default=None,
                        help="Path to a live_eval_trajectory.csv (enables the attack-profile figure)")
    parser.add_argument("--attack-start", type=int, default=90,
                        help="filter_burn_in_steps + honest_steps; re-bases the attack-step axis to 0")
    parser.add_argument("--outdir", default="machine_learning/figures", help="Directory to write figures into")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Figure file format")
    parser.add_argument("--window", type=int, default=25, help="Smoothing window (in samples)")
    parser.add_argument("--font-size", type=int, default=9)
    args = parser.parse_args()

    set_publication_style(args.font_size)
    outdir = Path(args.outdir)

    if args.dataset:
        data = load_dataset(args.dataset)
        plot_score_and_acceptance(data, outdir, args.format, args.window)
        if "reps" in data:
            plot_reputation_over_time(data, outdir, args.format, args.window)
        if "states" in data:
            plot_drift_coverage(data, outdir, args.format)  # paper Fig. 4

    if args.trajectory:
        plot_attack_profile(args.trajectory, outdir, args.format,
                            window=max(args.window // 3, 1), attack_start=args.attack_start)  # paper Fig. 3

    if args.train_log:
        make_all_training_plots(args.train_log, outdir, args.format, max(args.window, 1))

    if not args.dataset and not args.train_log and not args.trajectory:
        parser.error("Pass at least one of --dataset, --train-log, or --trajectory")


if __name__ == "__main__":
    main()
