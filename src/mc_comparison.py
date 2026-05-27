# pylint: disable=protected-access, too-many-locals, too-many-statements, too-many-arguments, too-many-positional-arguments, broad-exception-caught, duplicate-code
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

# Paths
PATH_A = "sim_data/mc_results/sim_1000km/mc_results_1000.0km.npz"
PATH_B = "sim_data/mc_results/sim_2000km/mc_results_2000.0km.npz"
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
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def get_aggregated_reps(results: List[Dict[str, Any]]) -> tuple[np.ndarray,
                                                                np.ndarray]:
    """
    Collects aggregated reputations across different satellite populations.

    Args:
        results: The results to extract aggregated reputation values from.

    Returns:
        Two arrays: One for the mean reputation values of the honest nodes,
        and another for the faulty nodes.
    """
    honest_means = []
    faulty_means = []
    for kpi in results:
        honest_means.append(np.mean(kpi["honest_matrix"], axis=0))
        faulty_means.append(np.mean(kpi["faulty_matrix"], axis=0))
    return np.array(honest_means), np.array(faulty_means)


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
    color_a = cmap(0.25)  # 1000km
    color_b = cmap(0.85) # 2000km

    if results_a:
        h_a, f_a = get_aggregated_reps(results_a)
        
        # Slice results based on start_step_a
        h_a = h_a[:, start_step_a:]
        f_a = f_a[:, start_step_a:]
        steps_a = np.arange(start_step_a, start_step_a + h_a.shape[1])

        h_mean_a = np.mean(h_a, axis=0)
        h_std_a = np.std(h_a, axis=0)
        plt.plot(steps_a, h_mean_a, color=color_a, label="Honest (1000km ISL)")
        plt.fill_between(steps_a, h_mean_a - h_std_a,
                         h_mean_a + h_std_a, color=color_a, alpha=0.1)

        f_mean_a = np.mean(f_a, axis=0)
        f_std_a = np.std(f_a, axis=0)
        plt.plot(steps_a, f_mean_a, color=color_a, linestyle="--",
                 label="Faulty (1000km ISL)")
        plt.fill_between(steps_a, f_mean_a - f_std_a,
                         f_mean_a + f_std_a, color=color_a, alpha=0.1)

    if results_b:
        h_b, f_b = get_aggregated_reps(results_b)
        
        # Slice results based on start_step_b
        h_b = h_b[:, start_step_b:]
        f_b = f_b[:, start_step_b:]
        steps_b = np.arange(start_step_b, start_step_b + h_b.shape[1])

        h_mean_b = np.mean(h_b, axis=0)
        h_std_b = np.std(h_b, axis=0)
        plt.plot(steps_b, h_mean_b, color=color_b,
                 label="Honest (2000km ISL)")
        plt.fill_between(steps_b, h_mean_b - h_std_b,
                         h_mean_b + h_std_b, color=color_b, alpha=0.1)

        f_mean_b = np.mean(f_b, axis=0)
        f_std_b = np.std(f_b, axis=0)
        plt.plot(steps_b, f_mean_b, color=color_b, linestyle="--",
                 label="Faulty (2000km ISL)")
        plt.fill_between(steps_b, f_mean_b - f_std_b,
                         f_mean_b + f_std_b, color=color_b, alpha=0.1)

    plt.axhline(0.5, color="gray", linestyle=":", label="Neutral")
    plt.xlabel("Timestep [-]", fontsize=14)
    plt.ylabel("Reputation [-]", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reputation_comparison.png"))
    plt.show()


def plot_kpi_comparison(results_a: List[Dict[str, Any]],
                        results_b: List[Dict[str, Any]]) -> None:
    """
    Plots a bar chart comparison of key KPIs, including TTD as a percentage of runtime.

    Args:
        results_a: Dataset A
        results_b: Dataset B

    Returns:
        None. Saves MatPlotLib figures in OUTPUT_DIR.
    """
    metrics = ["recall", "precision", "fpr"]
    labels = ["Recall", "Precision", "False Positive Rate", "Normalised TTD"]

    # Calculate means for standard metrics
    means_a = [np.mean([k[m] for k in results_a]) for m in metrics]
    means_b = [np.mean([k[m] for k in results_b]) for m in metrics]

    # Calculate Normalised TTD (as % of total steps)
    def get_ttd_percent(results):
        ttd_pcts = []
        for k in results:
            if k.get("avg_ttd") is not None:
                total_steps = k["honest_matrix"].shape[1]
                ttd_pcts.append((k["avg_ttd"] / total_steps) * 100)
        return np.mean(ttd_pcts) if ttd_pcts else 0

    means_a.append(get_ttd_percent(results_a))
    means_b.append(get_ttd_percent(results_b))

    x = np.arange(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 6))
    
    # Use viridis for color-blind friendly comparisons
    cmap = plt.get_cmap('viridis')
    color_a = cmap(0.25) # Deep teal/blue
    color_b = cmap(0.85) # Bright yellow/green

    ax.bar(x - width/2, means_a, width, label='1000km ISL', color=color_a)
    ax.bar(x + width/2, means_b, width, label='2000km ISL', color=color_b)

    ax.set_ylabel('Percentage [%]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.1)
    ax.set_ylim(0, 105) # Metrics are percentages

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kpi_percentage_comparison.png"))
    plt.show()

    # TTD Comparison
    ttds_a = [float(k.get("avg_ttd", 0)) for k in results_a if k.get("avg_ttd") is not None]
    ttds_b = [float(k.get("avg_ttd", 0)) for k in results_b if k.get("avg_ttd") is not None]

    if ttds_a or ttds_b:
        plt.figure(figsize=(6, 6))
        data_to_plot = []
        labels_ttd = []
        if ttds_a:
            data_to_plot.append(ttds_a)
            labels_ttd.append("1000km ISL")
        if ttds_b:
            data_to_plot.append(ttds_b)
            labels_ttd.append("2000km ISL")

        # Reduce gap by setting positions closer and increasing widths
        positions = [1, 1.5]

        cmap = plt.get_cmap('viridis')
        colors = [cmap(0.25), cmap(0.85)]

        bp = plt.boxplot(data_to_plot, tick_labels=labels_ttd,
                    positions=positions[:len(data_to_plot)], widths=0.35,
                    patch_artist=True)

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
    parser.add_argument("--start-step-a", type=int, default=210,
                        help="Step to start plotting from for dataset A (1000km).")
    parser.add_argument("--start-step-b", type=int, default=125,
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
