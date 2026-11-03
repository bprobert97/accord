"""
plot_cql_metrics.py

Publication-quality plots for the harvested offline dataset and CQL training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_publication_style(font_size: int = 9) -> None:
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
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")

def load_dataset(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}

def _group_by_step(data: Dict[str, np.ndarray], values: np.ndarray, max_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    steps_s, mean_s, _ = _group_by_step(data, data["scores"])

    fig, ax1 = plt.subplots()
    ax1.plot(steps_s, moving_average(mean_s, window), color="C3", label="consensus score")

    # Plot correctness_scores to evaluate DOF weighting
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

def make_all_dataset_plots(dataset_path: str, outdir: Path, fmt: str, window: int) -> None:
    data = load_dataset(dataset_path)
    plot_score_and_acceptance(data, outdir, fmt, window)
    # Additional plots can be added here
