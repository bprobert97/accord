"""
The Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework.
Author: Beth Probert
Email: beth.probert@strath.ac.uk

Copyright (C) 2025 Applied Space Technology Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
import os
import argparse
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from src.plotting.mc_plotting import get_aggregated_reps

# Paths
PATH_EKF = "sim_data/mc_results/sim_1000km/mc_results_1000.0km.npz"
PATH_UKF = "sim_data/mc_results/sim_ukf/mc_results_1000.0km.npz"
OUTPUT_DIR = "sim_data/comparison_ekf_ukf"

def load_results(path: str) -> List[Dict[str, Any]]:
    """
    Loads Monte Carlo results from a compressed .npz file.

    Args:
        path: The file path to the .npz results.

    Returns:
        A list of result dictionaries, excluding any failed runs (None).
    """
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return []
    try:
        with np.load(path, allow_pickle=True) as data:
            results = list(data['results'])
            return [res for res in results if res is not None]

    except (OSError, ValueError) as e:
        # Catches file read errors or corrupted/invalid .npz formats
        print(f"Data error loading {path}: {e}")
        return []

    except KeyError as e:
        # Catches the specific case where the file loaded, but 'results' is missing
        print(f"Missing key in {path}: {e}")
        return []


def plot_reputation_comparison(results_ekf: List[Dict[str, Any]],
                               results_ukf: List[Dict[str, Any]],
                               start_step_ekf: int = 0,
                               start_step_ukf: int = 0) -> None:
    """
    Plots reputation history for both datasets on the same graph.

    Args:
        results_ekf: EKF Dataset
        results_ukf: UKF Dataset
        start_step_ekf: Step to start plotting from for EKF.
        start_step_ukf: Step to start plotting from for UKF.

    Returns:
        None. Saves MatPlotLib figures in OUTPUT_DIR.
    """
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('plasma')

    # Inline the color selection to save local variables
    datasets = [
        (results_ekf, start_step_ekf, cmap(0.25), "EKF"),
        (results_ukf, start_step_ukf, cmap(0.85), "UKF")
    ]

    for res, start_step, color, label in datasets:
        if res:
            _plot_single_dataset(res, start_step, color, label)

    plt.axhline(0.5, color="gray", linestyle=":", label="Neutral Reputation")
    plt.xlabel("Timestep [-]", fontsize=18)
    plt.ylabel("Reputation [-]", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reputation_comparison.png"))
    plt.show()

def _plot_single_dataset(res: List[Dict[str, Any]],
                         start_step: int,
                         colour: Any,
                         label: str) -> None:
    """
    Helper function to calculate stats and plot lines for a single dataset.

    Args:
    - res: A list of result dictionaries containing the simulation data to aggregate.
    - start_step: The integer step index from which to begin plotting the data.
    - colour: The Matplotlib color identifier (e.g., an RGBA tuple) used for the
             plot lines and the shaded standard deviation regions.
    - label: A string descriptor (e.g., "EKF") used to label the honest and
             compromised lines in the plot's legend.

    Returns:
    - None. The function adds plotted lines and fill_between regions directly
      to the active Matplotlib figure.
    """
    h, f = get_aggregated_reps(res)
    h, f = h[:, start_step:], f[:, start_step:]
    steps = np.arange(start_step, start_step + h.shape[1])

    # Plot Honest
    h_mean, h_std = np.mean(h, axis=0), np.std(h, axis=0)
    plt.plot(steps, h_mean, color=colour, label=f"Honest ({label})", linewidth=3.5)
    plt.fill_between(steps, h_mean - h_std, h_mean + h_std, color=colour, alpha=0.1)

    # Plot Compromised
    f_mean, f_std = np.mean(f, axis=0), np.std(f, axis=0)
    plt.plot(steps, f_mean, color=colour, linestyle="--", label=f"Compromised ({label})",
             linewidth=3.5)
    plt.fill_between(steps, f_mean - f_std, f_mean + f_std, color=colour, alpha=0.1)

def plot_kpi_comparison(results_ekf: List[Dict[str, Any]],
                        results_ukf: List[Dict[str, Any]]) -> None:
    """
    Plots a bar chart comparison of key KPIs, including TTD as a percentage
    of runtime.

    Args:
    - results_ekf: EKF Dataset
    - results_ukf: UKF Dataset

    Returns:
    - None. Saves MatPlotLib figures in OUTPUT_DIR.
    """
    _plot_kpi_bar_chart(results_ekf, results_ukf)
    _plot_ttd_boxplot(results_ekf, results_ukf)

def _plot_kpi_bar_chart(results_ekf: List[Dict[str, Any]],
                        results_ukf: List[Dict[str, Any]]) -> None:
    """
    Plots a bar chart comparison of recall, precision, FPR, and TTD percentage.

    Args:
    - results_ekf: EKF Dataset
    - results_ukf: UKF Dataset

    Returns:
    - None. Saves a MatPlotLib figure in OUTPUT_DIR.
    """
    metrics = ["recall", "precision", "fpr"]
    labels = ["Recall", "Precision", "False Positive Rate", "Normalised TTD"]

    means_ekf = [np.mean([k[m] for k in results_ekf]) for m in metrics]
    means_ukf = [np.mean([k[m] for k in results_ukf]) for m in metrics]

    def get_ttd_percent(results):
        ttd_pcts = [
            (k["avg_ttd"] / k["honest_matrix"].shape[1]) * 100
            for k in results if k.get("avg_ttd") is not None
        ]
        return np.mean(ttd_pcts) if ttd_pcts else 0

    means_ekf.append(get_ttd_percent(results_ekf))
    means_ukf.append(get_ttd_percent(results_ukf))

    x = np.arange(len(labels))
    width = 0.35
    _, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('plasma')

    ax.bar(x - width/2, means_ekf, width, label='EKF', color=cmap(0.25))
    ax.bar(x + width/2, means_ukf, width, label='UKF', color=cmap(0.85))

    ax.set_ylabel('Percentage [%]', fontsize=18)
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(axis='y', alpha=0.1)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kpi_percentage_comparison.png"))
    plt.show()

def _plot_ttd_boxplot(results_ekf: List[Dict[str, Any]],
                      results_ukf: List[Dict[str, Any]]) -> None:
    """
    Plots a boxplot comparison of Time to Detection (TTD) in timesteps for both datasets.
    Only includes runs where TTD is available (not None).

    Args:
    - results_ekf: EKF Dataset
    - results_ukf: UKF Dataset

    Returns:
    - None. Saves a MatPlotLib figure in OUTPUT_DIR.
    """
    ttds_ekf = [float(k.get("avg_ttd", 0)) for k in results_ekf if k.get("avg_ttd") is not None]
    ttds_ukf = [float(k.get("avg_ttd", 0)) for k in results_ukf if k.get("avg_ttd") is not None]

    if not ttds_ekf and not ttds_ukf:
        return

    plt.figure(figsize=(6, 6))
    data_to_plot, labels_ttd = [], []

    if ttds_ekf:
        data_to_plot.append(ttds_ekf)
        labels_ttd.append("EKF")
    if ttds_ukf:
        data_to_plot.append(ttds_ukf)
        labels_ttd.append("UKF")

    cmap = plt.get_cmap('plasma')
    colors = [cmap(0.25), cmap(0.85)]
    bp = plt.boxplot(data_to_plot, tick_labels=labels_ttd, positions=[1, 1.5][:len(data_to_plot)],
                     widths=0.35, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.xlim(0.5, 2.0)
    plt.ylabel("Time to Detection  [Timesteps]")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ttd_comparison.png"))
    plt.show()

def main():
    """
    Main entry point for the comparison script.
    Loads results, generates comparison plots,
    and prints a summary to the console.
    """
    parser = argparse.ArgumentParser(description="Compare EKF and UKF Monte Carlo \
                                     results at 1000km.")
    parser.add_argument("--start-step-ekf", type=int, default=0,
                        help="Step to start plotting from for the EKF dataset.")
    parser.add_argument("--start-step-ukf", type=int, default=0,
                        help="Step to start plotting from for the UKF dataset.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading EKF results (1000km)...")
    res_ekf = load_results(PATH_EKF)
    print("Loading UKF results (1000km)...")
    res_ukf = load_results(PATH_UKF)

    if not res_ekf and not res_ukf:
        print("No results found to compare.")
        return

    print(f"Comparing {len(res_ekf)} EKF runs vs {len(res_ukf)} UKF runs")

    plot_reputation_comparison(res_ekf, res_ukf,
                               start_step_ekf=args.start_step_ekf,
                               start_step_ukf=args.start_step_ukf)
    plot_kpi_comparison(res_ekf, res_ukf)

    # Also print summary
    def print_summary(label, results):
        if not results:
            return
        print(f"\n--- {label} Summary ---")
        print(f"Mean Recall: {np.mean([k['recall'] for k in results]):.2f}%")
        print(f"Mean Precision: {np.mean([k['precision'] for k in results]):.2f}%")
        print(f"Mean FPR: {np.mean([k['fpr'] for k in results]):.2f}%")
        ttds = [float(k.get("avg_ttd", 0)) for k in results if k.get("avg_ttd") is not None]
        if ttds:
            print(f"Mean TTD: {np.mean(ttds):.2f} steps")
        print(f"Avg Detection Margin: {np.mean([k.get('detection_margin', 0) \
                                                for k in results]):.4f}")

    print_summary("EKF", res_ekf)
    print_summary("UKF", res_ukf)

if __name__ == "__main__":
    main()
