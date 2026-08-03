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
import argparse
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from src.plotting.mc_plotting import get_aggregated_reps

# Paths
PATH_A = "sim_data/mc_results/sim_ekf/mc_results_1000.0km.npz"
PATH_B = "sim_data/mc_results/sim_ekf/mc_results_2000.0km.npz"
OUTPUT_DIR = "sim_data/comparison"

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

def plot_reputation_comparison(results_a: List[Dict[str, Any]],
                               results_b: List[Dict[str, Any]],
                               start_step_a: int = 0,
                               start_step_b: int = 0) -> None:
    """
    Plots reputation history for both datasets on the same graph.

    Args:
        results_a: Dataset A
        results_b: Dataset B
        start_step_a: Step to start plotting from for dataset A.
        start_step_b: Step to start plotting from for dataset B.

    Returns:
        None. Saves MatPlotLib figures in OUTPUT_DIR.
    """
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('viridis')

    # Inline the color selection to save local variables
    datasets = [
        (results_a, start_step_a, cmap(0.25), "1000km"),
        (results_b, start_step_b, cmap(0.85), "2000km")
    ]

    for res, start_step, color, label in datasets:
        if res:
            _plot_single_dataset(res, start_step, color, label)

    plt.axhline(0.5, color="gray", linestyle=":", label="Neutral Reputation")
    plt.xlabel("Timestep [-]", fontsize=20)
    plt.ylabel("Reputation [-]", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc='lower right', fontsize=16)
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
    - label: A string descriptor (e.g., "1000km") used to label the honest and
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
    plt.plot(steps, h_mean, color=colour, label=f"Honest ({label} ISL)", linewidth=3.5)
    plt.fill_between(steps, h_mean - h_std, h_mean + h_std, color=colour, alpha=0.1)

    # Plot Compromised
    f_mean, f_std = np.mean(f, axis=0), np.std(f, axis=0)
    plt.plot(steps, f_mean, color=colour, linestyle="--", label=f"Compromised ({label} ISL)",
             linewidth=3.5)
    plt.fill_between(steps, f_mean - f_std, f_mean + f_std, color=colour, alpha=0.1)


def plot_kpi_comparison(results_a: List[Dict[str, Any]],
                        results_b: List[Dict[str, Any]]) -> None:
    """
    Plots a bar chart comparison of key KPIs, including TTD as a percentage
    of runtime.

    Args:
    - results_a: Dataset A
    - results_b: Dataset B

    Returns:
    - None. Saves MatPlotLib figures in OUTPUT_DIR.
    """
    _plot_kpi_bar_chart(results_a, results_b)
    _plot_ttd_boxplot(results_a, results_b)


def _plot_kpi_bar_chart(results_a: List[Dict[str, Any]],
                        results_b: List[Dict[str, Any]]) -> None:
    """
    Plots a bar chart comparison of recall, precision, FPR, and TTD percentage,
    with error bars representing standard deviation.

    Args:
    - results_a: Dataset A
    - results_b: Dataset B

    Returns:
    - None. Saves a MatPlotLib figure in OUTPUT_DIR.
    """
    metrics = ["recall", "precision", "fpr"]
    labels = ["Recall", "Precision", "False Positive Rate", "Normalised TTD"]

    # 1. Calculate means
    means_a = [np.mean([k[m] for k in results_a]) for m in metrics]
    means_b = [np.mean([k[m] for k in results_b]) for m in metrics]

    # 2. Calculate standard deviations for the error bars
    stds_a = [np.std([k[m] for k in results_a]) for m in metrics]
    stds_b = [np.std([k[m] for k in results_b]) for m in metrics]

    # 3. Helper function updated to return both mean and std for TTD
    def get_ttd_stats(results):
        ttd_pcts = [
            (k["avg_ttd"] / k["honest_matrix"].shape[1]) * 100
            for k in results if k.get("avg_ttd") is not None
        ]
        if not ttd_pcts:
            return 0.0, 0.0
        return np.mean(ttd_pcts), np.std(ttd_pcts)

    # 4. Append TTD stats to the lists
    ttd_mean_a, ttd_std_a = get_ttd_stats(results_a)
    ttd_mean_b, ttd_std_b = get_ttd_stats(results_b)

    means_a.append(ttd_mean_a)
    means_b.append(ttd_mean_b)

    stds_a.append(ttd_std_a)
    stds_b.append(ttd_std_b)

    x = np.arange(len(labels))
    width = 0.35
    _, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('viridis')

    # 5. Plot bars with yerr (error bars), capsize, and alpha for better contrast
    ax.bar(x - width/2, means_a, width, yerr=stds_a, capsize=5, alpha=0.9,
           label='1000km ISL', color=cmap(0.25))
    ax.bar(x + width/2, means_b, width, yerr=stds_b, capsize=5, alpha=0.9,
           label='2000km ISL', color=cmap(0.85))

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


def _plot_ttd_boxplot(results_a: List[Dict[str, Any]],
                      results_b: List[Dict[str, Any]]) -> None:
    """
    Plots a boxplot comparison of Time to Detection (TTD) in timesteps for both datasets.
    Only includes runs where TTD is available (not None).

    Args:
    - results_a: Dataset A
    - results_b: Dataset B

    Returns:
    - None. Saves a MatPlotLib figure in OUTPUT_DIR.
    """
    ttds_a = [float(k.get("avg_ttd", 0)) for k in results_a if k.get("avg_ttd") is not None]
    ttds_b = [float(k.get("avg_ttd", 0)) for k in results_b if k.get("avg_ttd") is not None]

    if not ttds_a and not ttds_b:
        return

    plt.figure(figsize=(6, 6))
    data_to_plot, labels_ttd = [], []

    if ttds_a:
        data_to_plot.append(ttds_a)
        labels_ttd.append("1000km ISL")
    if ttds_b:
        data_to_plot.append(ttds_b)
        labels_ttd.append("2000km ISL")

    cmap = plt.get_cmap('viridis')
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
    Loads results generates comparison plots,
    and prints a summary to the console.

    To run this file in a terminal, execute:
    python src/mc_comparison.py
    """
    parser = argparse.ArgumentParser(description="Compare Monte Carlo results.")
    parser.add_argument("--start-step-a", type=int, default=0,
                        help="Step to start plotting from for dataset A (1000km).")
    parser.add_argument("--start-step-b", type=int, default=0,
                        help="Step to start plotting from for dataset B (2000km).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading 1000km results...")
    res_a = load_results(PATH_A)
    print("Loading 2000km results...")
    res_b = load_results(PATH_B)

    if not res_a and not res_b:
        print("No results found to compare.")
        return

    print(f"Comparing {len(res_a)} runs (1000km) vs {len(res_b)} runs (2000km)")

    plot_reputation_comparison(res_a, res_b,
                               start_step_a=args.start_step_a,
                               start_step_b=args.start_step_b)
    plot_kpi_comparison(res_a, res_b)

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

    print_summary("1000km ISL", res_a)
    print_summary("2000km ISL", res_b)

if __name__ == "__main__":
    main()
