"""Publication-quality plotting for the DAG-harvester dataset and CQL training run.

This module renders the figures used in the paper and the diagnostic plots used
during development:

* the harvested offline dataset produced by ``harvest_dag_dataset.py`` (consensus
  score, compromised-node reputation, and the distribution of injected drift);
* the Conservative Q-Learning training run, read from a metrics CSV; and
* the trained policy's live-evaluation trajectory produced by
  ``evaluate_cql_live.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
# matplotlib.use("Agg") must run before pyplot is imported, which forces these
# submodule imports below it.
# pylint: disable=wrong-import-position,ungrouped-imports
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
# pylint: enable=wrong-import-position,ungrouped-imports


def set_publication_style(font_size: int = 9) -> None:
    """Apply a compact serif, high-resolution style for two-column templates.

    Args:
        font_size: Base font size in points; axis, tick, and legend sizes are
            derived from it.

    Returns:
        None. The global Matplotlib run-time configuration is modified in place.
    """
    plt.rcParams.update(
        {
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
        }
    )


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a centred moving average of the same length as the input.

    Args:
        x: One-dimensional array of samples.
        window: Number of samples in the averaging window. A value of one or
            less returns the input unchanged (cast to float).

    Returns:
        An array of the same length as ``x`` holding the centred moving average.
    """
    if window <= 1 or len(x) == 0:
        return x.astype(float)
    # A prefix sum makes each window average an O(1) difference of two cumulative
    # sums, rather than re-adding the whole window at every position.
    cumsum = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty_like(x, dtype=float)
    half = window // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def save(fig: Figure, outdir: Path, name: str, fmt: str) -> None:
    """Write a figure to disk and close it.

    Args:
        fig: The figure to save.
        outdir: Directory to write into; created if it does not exist.
        name: File stem, without extension.
        fmt: File extension and format, for example ``"pdf"`` or ``"png"``.

    Returns:
        None.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """Load a harvested dataset archive into a dictionary of arrays.

    Args:
        path: Path to a ``.npz`` archive produced by the harvester.

    Returns:
        A dictionary mapping each stored array name to its array.
    """
    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def _group_by_step(
    data: Dict[str, np.ndarray], values: np.ndarray, max_steps: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a value array across episodes at each within-episode step index.

    Args:
        data: Dataset dictionary; must contain a ``"step_idx"`` array.
        values: Per-transition values aligned with ``data["step_idx"]``.
        max_steps: Optional upper bound on the number of steps to consider. When
            omitted, the maximum observed step index is used.

    Returns:
        A tuple ``(steps, means, stds)`` holding the step indices that have at
        least one sample, the mean of ``values`` at each such step, and the
        corresponding standard deviation.
    """
    # Collapse the per-episode transitions onto a common within-episode step axis by
    # averaging every episode that has a sample at each step.
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


def plot_score_and_acceptance(
    data: Dict[str, np.ndarray], outdir: Path, fmt: str, window: int = 25
) -> None:
    """Plot consensus score, correctness score, and acceptance against step.

    Args:
        data: Dataset dictionary; must contain ``"scores"`` and may contain
            ``"correctness_scores"`` and ``"accepted_frac"``.
        outdir: Directory to write the figure into.
        fmt: Figure file format.
        window: Smoothing window, in samples.

    Returns:
        None.
    """
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
        ax2.plot(
            steps_a, moving_average(mean_a, window), color="C5",
            label="acceptance fraction", linestyle="--",
        )
        ax2.set_ylabel("acceptance fraction", color="C5")
        ax2.tick_params(axis="y", labelcolor="C5")

    ax1.legend(loc="upper left")
    save(fig, outdir, "score_and_correctness", fmt)


def plot_reputation_over_time(
    data: Dict[str, np.ndarray], outdir: Path, fmt: str,
    window: int = 25, reputation_floor: float = 0.5,
) -> None:
    """Plot mean compromised-node reputation against step, with its floor.

    Args:
        data: Dataset dictionary; must contain ``"reps"``.
        outdir: Directory to write the figure into.
        fmt: Figure file format.
        window: Smoothing window, in samples.
        reputation_floor: Reputation threshold drawn as a reference line.

    Returns:
        None.
    """
    steps, mean, std = _group_by_step(data, data["reps"])
    fig, ax = plt.subplots()
    ax.plot(steps, moving_average(mean, window), color="C2")
    ax.fill_between(steps, mean - std, mean + std, color="C2", alpha=0.15, linewidth=0)
    ax.axhline(
        reputation_floor, color="0.45", linewidth=1.0, linestyle="--", label="reputation floor"
    )
    ax.set_xlabel("attack step")
    ax.set_ylabel("mean reputation")
    ax.legend(loc="best")
    save(fig, outdir, "reputation_over_time", fmt)


def load_training_log(path: str) -> List[Dict[str, float]]:
    """Load a metrics CSV written during training into a list of rows.

    Args:
        path: Path to the metrics CSV.

    Returns:
        A list of dictionaries, one per non-empty row, mapping each populated
        column name to its float value. Empty cells are omitted from each row.
    """
    rows: List[Dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {key: float(value) for key, value in raw.items() if value not in (None, "")}
            if row:
                rows.append(row)
    return rows


def _plot_single_series(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rows: List[Dict[str, float]], x_key: str, y_key: str, ylabel: str,
    outdir: Path, fmt: str, name: str, window: int = 1, color: str = "C0",
) -> None:
    """Plot one time series extracted from training-log rows.

    Args:
        rows: Training-log rows, as returned by :func:`load_training_log`.
        x_key: Column name for the horizontal axis.
        y_key: Column name for the vertical axis.
        ylabel: Vertical-axis label.
        outdir: Directory to write the figure into.
        fmt: Figure file format.
        name: File stem for the saved figure.
        window: Smoothing window, in samples.
        color: Line colour.

    Returns:
        None. If no row contains both keys, the figure is skipped.
    """
    pairs = [(row[x_key], row[y_key]) for row in rows if x_key in row and y_key in row]
    if not pairs:
        print(f"Skipping {name}: no rows have both '{x_key}' and '{y_key}'")
        return
    pairs.sort(key=lambda pair: pair[0])
    x = np.asarray([pair[0] for pair in pairs])
    y = np.asarray([pair[1] for pair in pairs])
    fig, ax = plt.subplots()
    ax.plot(x, moving_average(y, window), color=color)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    save(fig, outdir, name, fmt)


def make_all_training_plots(log_path: str, outdir: Path, fmt: str, window: int = 50) -> None:
    """Generate the full set of diagnostic plots from a training run.

    Args:
        log_path: Path to the training metrics CSV.
        outdir: Directory to write the figures into.
        fmt: Figure file format.
        window: Smoothing window, in samples, for the per-step series.

    Returns:
        None.
    """
    log = load_training_log(log_path)
    _plot_single_series(
        log, "step", "critic_loss", "critic loss", outdir, fmt, "critic_loss", window, "C0"
    )
    _plot_single_series(
        log, "step", "actor_loss", "actor loss", outdir, fmt, "actor_loss", window, "C1"
    )
    _plot_single_series(log, "step", "cql_gap", "CQL gap", outdir, fmt, "cql_gap", window, "C2")
    _plot_single_series(log, "step", "alpha", r"$\alpha$", outdir, fmt, "cql_alpha", window, "C3")
    _plot_single_series(
        log, "val_step", "val_q_gap", "Q(policy) - Q(data)", outdir, fmt, "val_q_gap", 1, "C5"
    )
    _plot_single_series(
        log, "val_step", "val_td_error", "TD error", outdir, fmt, "val_td_error", 1, "C6"
    )


def plot_attack_profile(  # pylint: disable=too-many-locals
    trajectory_path: str, outdir: Path, fmt: str, window: int = 9, attack_start: int = 90
) -> None:
    """Plot trained-policy reputation, consensus score, and drift against step.

    The figure has two parts: part a) shows reputation and consensus score with
    their admission floors, and part b) shows the injected drift. Values are
    averaged across compromised satellites at each attack step and smoothed.

    Args:
        trajectory_path: Path to a ``live_eval_trajectory.csv`` written by
            ``evaluate_cql_live.py`` (one row per admitted attack-phase
            transaction).
        outdir: Directory to write the figure into.
        fmt: Figure file format.
        window: Smoothing window, in samples.
        attack_start: Absolute simulation step at which the attack phase begins
            (``filter_burn_in_steps + honest_steps``), used to re-base the
            horizontal axis so that it starts at zero.

    Returns:
        None.
    """
    rows: List[Dict[str, str]] = []
    with open(trajectory_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    step = np.array([int(row["step"]) for row in rows])
    reputation = np.array([float(row["reputation"]) for row in rows])
    score = np.array([float(row["consensus_score"]) for row in rows])
    drift = np.array([float(row["drift"]) for row in rows])
    attack_step = step - attack_start

    # Average the per-transaction values at each attack step and then smooth them, so
    # the trend stays legible despite many satellites reporting within each step.
    unique = np.unique(attack_step)
    reputation_mean = moving_average(
        np.array([reputation[attack_step == s].mean() for s in unique]), window
    )
    score_mean = moving_average(
        np.array([score[attack_step == s].mean() for s in unique]), window
    )
    drift_mean = moving_average(
        np.array([drift[attack_step == s].mean() for s in unique]), window
    )

    # Reputation and drift share the dark viridis tone; the consensus score takes a
    # contrasting green. Line styles differ too, so the figure also reads in greyscale.
    viridis = plt.get_cmap("viridis")
    colour_reputation, colour_score, colour_drift = viridis(0.15), viridis(0.6), viridis(0.15)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 4.4), sharex=True)
    fig.subplots_adjust(hspace=0.30)

    ax1.plot(unique, reputation_mean, linestyle="-", color=colour_reputation, label="reputation")
    ax1.plot(unique, score_mean, linestyle="--", color=colour_score, label="consensus score")
    ax1.axhline(0.5, linestyle=":", color="0.45", linewidth=1.0)
    ax1.axhline(0.3, linestyle=(0, (1, 1)), color="0.45", linewidth=1.0)
    ax1.set_ylabel("trust factor")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    label_bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0}
    ax1.text(
        0.99, 0.52, "reputation floor", transform=ax1.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="black", bbox=label_bbox,
    )
    ax1.text(
        0.99, 0.32, "score floor", transform=ax1.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="black", bbox=label_bbox,
    )

    ax2.plot(unique, drift_mean, linestyle="-", color=colour_drift)
    ax2.set_ylabel(r"injected drift, $\sigma$")
    ax2.set_xlabel("attack step")
    ax2.set_xlim(0, float(unique.max()))
    fig.align_ylabels([ax1, ax2])

    # Part letters sit below and to the left of each part, outside the plot box.
    ax1.text(0.0, -0.13, "a)", transform=ax1.transAxes, ha="left", va="top")
    ax2.text(0.0, -0.28, "b)", transform=ax2.transAxes, ha="left", va="top")
    save(fig, outdir, "attack_profile", fmt)


def plot_drift_coverage(
    data: Dict[str, np.ndarray], outdir: Path, fmt: str, num_bins: int = 22
) -> None:
    """Plot the distribution of cumulative injected drift across the dataset.

    The drift magnitude of each transition is the Euclidean norm of the first
    two state dimensions (the two components of the cumulative injected bias).

    Args:
        data: Dataset dictionary; must contain ``"states"``.
        outdir: Directory to write the figure into.
        fmt: Figure file format.
        num_bins: Number of histogram bins spanning zero to the observed
            maximum drift.

    Returns:
        None.
    """
    magnitude = np.linalg.norm(data["states"][:, :2], axis=1)
    upper = float(np.ceil(magnitude.max()))
    fig, ax = plt.subplots()
    ax.hist(
        magnitude, bins=num_bins, range=(0.0, upper),
        color=plt.get_cmap("viridis")(0.3), edgecolor="black", linewidth=0.5,
    )
    ax.set_xlabel(r"cumulative injected drift, $\sigma$")
    ax.set_ylabel("number of transitions")
    ax.set_xlim(0, upper)
    save(fig, outdir, "drift_coverage", fmt)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot CQL and DAG-harvester metrics for publication"
    )
    parser.add_argument("--dataset", default=None, help="Path to dag_harvested_dataset.npz")
    parser.add_argument("--train-log", default=None, help="Path to a training metrics CSV")
    parser.add_argument(
        "--trajectory", default=None,
        help="Path to a live_eval_trajectory.csv (enables the attack-profile figure)",
    )
    parser.add_argument(
        "--attack-start", type=int, default=90,
        help="filter_burn_in_steps + honest_steps; re-bases the attack-step axis to zero",
    )
    parser.add_argument(
        "--outdir", default="machine_learning/figures", help="Directory to write figures into"
    )
    parser.add_argument(
        "--format", default="pdf", choices=["pdf", "png", "svg"], help="Figure file format",
    )
    parser.add_argument("--window", type=int, default=25, help="Smoothing window, in samples")
    parser.add_argument("--font-size", type=int, default=9, help="Base font size, in points")
    return parser.parse_args()


def main() -> None:
    """Parse arguments and generate the requested figures.

    Returns:
        None.

    Raises:
        SystemExit: If none of ``--dataset``, ``--train-log``, or
            ``--trajectory`` is supplied.
    """
    args = parse_args()
    set_publication_style(args.font_size)
    outdir = Path(args.outdir)

    if args.dataset:
        data = load_dataset(args.dataset)
        plot_score_and_acceptance(data, outdir, args.format, args.window)
        if "reps" in data:
            plot_reputation_over_time(data, outdir, args.format, args.window)
        if "states" in data:
            plot_drift_coverage(data, outdir, args.format)

    if args.trajectory:
        plot_attack_profile(
            args.trajectory, outdir, args.format,
            window=max(args.window // 3, 1), attack_start=args.attack_start,
        )

    if args.train_log:
        make_all_training_plots(args.train_log, outdir, args.format, max(args.window, 1))

    if not args.dataset and not args.train_log and not args.trajectory:
        raise SystemExit("Pass at least one of --dataset, --train-log, or --trajectory")


if __name__ == "__main__":
    main()
