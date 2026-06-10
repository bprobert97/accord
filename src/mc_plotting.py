"""
The Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework.
Author: Beth Probert
Email: beth.probert@strath.ac.uk

Copyright (C) 2025 Applied Space Technology Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from src.plotting import plot_mc_nis_boxplot

DATA_DIR = os.path.join("sim_data", "mc_results")

def plot_undetected_reputations(all_kpis: List[Dict[str, Any]],
                                threshold: float = 0.5,
                                start_step: int = 0) -> None:
    """
    Plot the full reputation history of every undetected faulty satellite across all runs,
    colour-coded by satellite ID.

    Args:
    - all_kpis: List of KPI dictionaries from MC runs.
    - threshold: The detection threshold used.
    - start_step: The step to start plotting from.

    Returns:
    - None. Plots the reputations using MatPlotLib.
    """
    plt.figure(figsize=(12, 7))

    # Identify all unique IDs that went undetected
    unique_ids = sorted(list(set(
        sid for kpi in all_kpis
        for sid in kpi.get("undetected_faulty_ids", [])
    )))

    if not unique_ids:
        plt.text(0.5, 0.5, "No undetected faulty satellites found",
                 ha="center", va="center", transform=plt.gca().transAxes)
    else:
        # Delegate the plotting logic to isolate local variables
        _plot_undetected_lines(all_kpis, unique_ids, threshold, start_step)

    plt.xlabel("Step")
    plt.ylabel("Satellite Credibility Assessment")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(DATA_DIR, "mc_undetected_reps.png"))
    plt.show()


def _plot_undetected_lines(all_kpis: List[Dict[str, Any]],
                           unique_ids: list[int],
                           threshold: float,
                           start_step: int) -> None:
    """
    Helper to plot individual histories and manage the Matplotlib legend.

    Args:
    - all_kpis: List of KPI dictionaries from MC runs.
    - unique_ids: IDs of faulty satellites that went undetected.
    - threshold: The detection threshold used.
    - start_step: The step to start plotting from.

    Returns:
    - None. Plots the lines using MatPlotLib.
    """
    # Inline the cmap call directly into the dictionary comprehension
    id_to_color = {sid: plt.get_cmap("tab20")(i % 20) for i, sid in enumerate(unique_ids)}
    plotted_legend_ids = set()

    for kpi in all_kpis:
        ids = kpi.get("undetected_faulty_ids", [])
        reps = kpi.get("undetected_faulty_reps", [])

        for sid, history in zip(ids, reps):
            # Only add to legend once per unique satellite ID
            label = f"Sat {sid}" if sid not in plotted_legend_ids else None

            # Plot from start_step onwards
            steps = np.arange(start_step, len(history))
            plt.plot(steps, history[start_step:], color=id_to_color[sid],
                     alpha=0.6, linewidth=1.5, label=label)

            plotted_legend_ids.add(sid)

    plt.axhline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold})")

    # Adjust legend position and columns based on the number of items
    num_items = len(plotted_legend_ids)
    if num_items > 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        plt.tight_layout(rect=(0, 0, 0.85, 1))
    elif num_items > 0:
        plt.legend(loc='best', fontsize='small')


def plot_mc_results(all_kpis_raw: List[Optional[Dict[str, Any]]],
                    start_step: int = 0) -> None:
    """
    Aggregate results from all Monte Carlo runs and generate summary plots.

    Generates summary plots for reputation spread, KPI distributions, and
    undetected faulty nodes.

    Args:
    - all_kpis_raw: A list of KPI dictionaries from multiple simulation runs.
    - start_step: The step to start plotting from.

    Returns:
    - None. Saves plots to disk and displays them.
    """
    all_kpis: List[Dict[str, Any]] = [k for k in all_kpis_raw if k is not None]
    if not all_kpis:
        print("No successful runs to plot.")
        return

    plot_undetected_reputations(all_kpis, start_step=start_step)

    _plot_reputation_histories(all_kpis, start_step)
    _plot_kpi_distributions(all_kpis)
    _print_mc_summary(all_kpis)

    plot_mc_nis_boxplot(all_kpis)


def _plot_reputation_histories(all_kpis: List[Dict[str, Any]], start_step: int) -> None:
    """
    Plot the mean reputation history for honest and faulty satellites across all runs,
    with shaded areas representing the standard deviation spread.

    Args:
    - all_kpis: A list of KPI dictionaries from multiple simulation runs.
    - start_step: The step to start plotting from.

    Returns:
    - None. Saves the plot to disk and displays it.
    """
    honest_means_list = [np.mean(kpi["honest_matrix"], axis=0) for kpi in all_kpis]
    faulty_means_list = [np.mean(kpi["faulty_matrix"], axis=0) for kpi in all_kpis]

    all_honest_means = np.array(honest_means_list)[:, start_step:]
    all_faulty_means = np.array(faulty_means_list)[:, start_step:]
    steps = np.arange(start_step, start_step + all_honest_means.shape[1])

    plt.figure(figsize=(10, 6))

    h_mean = np.mean(all_honest_means, axis=0)
    h_std = np.std(all_honest_means, axis=0)
    plt.plot(steps, h_mean, color="green", label="Honest (MC Mean)")
    plt.fill_between(steps, h_mean - h_std, h_mean + h_std, color="green",
                     alpha=0.2, label="Honest Pop. 1 Std. Dev. Spread")

    f_mean = np.mean(all_faulty_means, axis=0)
    f_std = np.std(all_faulty_means, axis=0)
    plt.plot(steps, f_mean, color="red", label="Faulty (MC Mean)")
    plt.fill_between(steps, f_mean - f_std, f_mean + f_std, color="red",
                     alpha=0.2, label="Faulty Pop. 1 Std. Dev. Spread")

    plt.axhline(0.5, color="gray", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Reputation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DATA_DIR, "mc_reputation.png"))
    plt.show()


def _plot_kpi_distributions(all_kpis: List[Dict[str, Any]]) -> None:
    """
    Plot histograms for Time to Detection (TTD) and False Positive Rate (FPR),
    and a scatter plot for Recall vs Precision across all Monte Carlo runs.

    Args:
    - all_kpis: A list of KPI dictionaries from multiple simulation runs.

    Returns:
    - None. Saves the plot to disk and displays it.
    """
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TTD Histogram
    ttds = [float(k["avg_ttd"]) for k in all_kpis if k.get("avg_ttd") is not None]
    if ttds:
        axes[0].hist(ttds, bins=10, color='skyblue', edgecolor='black')
        axes[0].set_title("Time to Detection (Steps)")
        axes[0].axvline(float(np.mean(ttds)), color='red', linestyle='dashed',
                        label=f'Mean: {np.mean(ttds):.1f}')
        axes[0].legend()

    # FPR Histogram
    fprs = [float(k.get("fpr", 0)) for k in all_kpis]
    axes[1].hist(fprs, bins=10, color='salmon', edgecolor='black')
    axes[1].set_title("False Positive Rate (%)")
    axes[1].axvline(float(np.mean(fprs)), color='red', linestyle='dashed',
                    label=f'Mean: {np.mean(fprs):.1f}%')
    axes[1].legend()

    # Recall/Precision Scatter
    recalls = [float(k.get("recall", 0)) for k in all_kpis]
    precisions = [float(k.get("precision", 0)) for k in all_kpis]
    axes[2].scatter(recalls, precisions, color='purple', alpha=0.5)
    axes[2].set_xlabel("Recall (%)")
    axes[2].set_ylabel("Precision (%)")
    axes[2].set_title("Detection Reliability")
    axes[2].set_xlim(-5, 105)
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "mc_kpis.png"))
    plt.show()


def _print_mc_summary(all_kpis: List[Dict[str, Any]]) -> None:
    """
    Print a summary of the Monte Carlo results, including mean recall, precision, FPR, TTD,
    and other relevant KPIs.

    Args:
    - all_kpis: A list of KPI dictionaries from multiple simulation runs.

    Returns:
    - None. Prints the summary to the console.
    """
    recalls = [float(k.get("recall", 0)) for k in all_kpis]
    precisions = [float(k.get("precision", 0)) for k in all_kpis]
    fprs = [float(k.get("fpr", 0)) for k in all_kpis]
    ttds = [float(k["avg_ttd"]) for k in all_kpis if k.get("avg_ttd") is not None]

    print("--- Monte Carlo Summary ---")
    print(f"Total Runs: {len(all_kpis)}")
    print(f"Mean Recall: {np.mean(recalls):.2f}%")
    print(f"Mean Precision: {np.mean(precisions):.2f}%")
    print(f"Mean FPR: {np.mean(fprs):.2f}%")

    if ttds:
        print(f"Mean TTD: {np.mean(ttds):.2f} steps")
        worst_ttds = [float(k['worst_ttd']) for k in all_kpis if k.get('worst_ttd') is not None]
        if worst_ttds:
            print(f"Worst-Case TTD: {np.max(worst_ttds):.2f} steps")

    print(f"Avg Detection Margin: {np.mean([float(k.get('detection_margin', 0)) \
                                            for k in all_kpis]):.4f}")
    print(f"Avg Honest Spread: {np.mean([float(k.get('honest_spread', 0)) \
                                         for k in all_kpis]):.4f}")
    print(f"Avg Stability (Total Flips): {np.mean([float(k.get('flips', 0)) \
                                                   for k in all_kpis]):.2f}")
    print(f"Avg Final Honest Rep: {np.mean([float(k.get('final_honest_rep', 0)) \
                                            for k in all_kpis]):.4f}")
    print(f"Avg Final Faulty Rep: {np.mean([float(k.get('final_faulty_rep', 0)) \
                                            for k in all_kpis]):.4f}")
