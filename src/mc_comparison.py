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
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

# Paths
PATH_100KM = "sim_data/mc_results/mc_results.npz"
PATH_2000KM = "sim_data/mc_results_2000km/mc_results.npz"
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

def plot_reputation_comparison(results_100km: List[Dict[str, Any]],
                                results_2000km: List[Dict[str, Any]]) -> None:
    """
    Plots reputation history for both datasets on the same graph.
    """
    plt.figure(figsize=(12, 7))

    def get_aggregated_reps(results):
        honest_means = []
        faulty_means = []
        for kpi in results:
            honest_means.append(np.mean(kpi["honest_matrix"], axis=0))
            faulty_means.append(np.mean(kpi["faulty_matrix"], axis=0))
        return np.array(honest_means), np.array(faulty_means)

    if results_100km:
        h_100, f_100 = get_aggregated_reps(results_100km)
        steps_100 = np.arange(h_100.shape[1])

        h_mean_100 = np.mean(h_100, axis=0)
        h_std_100 = np.std(h_100, axis=0)
        plt.plot(steps_100, h_mean_100, color="green", label="Honest (100km ISL)")
        plt.fill_between(steps_100, h_mean_100 - h_std_100,
                         h_mean_100 + h_std_100, color="green", alpha=0.1)

        f_mean_100 = np.mean(f_100, axis=0)
        f_std_100 = np.std(f_100, axis=0)
        plt.plot(steps_100, f_mean_100, color="red", label="Faulty (100km ISL)")
        plt.fill_between(steps_100, f_mean_100 - f_std_100,
                         f_mean_100 + f_std_100, color="red", alpha=0.1)

    if results_2000km:
        h_2000, f_2000 = get_aggregated_reps(results_2000km)
        steps_2000 = np.arange(h_2000.shape[1])

        h_mean_2000 = np.mean(h_2000, axis=0)
        h_std_2000 = np.std(h_2000, axis=0)
        plt.plot(steps_2000, h_mean_2000, color="green",
                 linestyle="--", label="Honest (2000km ISL)")
        plt.fill_between(steps_2000, h_mean_2000 - h_std_2000,
                         h_mean_2000 + h_std_2000, color="green", alpha=0.1)

        f_mean_2000 = np.mean(f_2000, axis=0)
        f_std_2000 = np.std(f_2000, axis=0)
        plt.plot(steps_2000, f_mean_2000, color="red", linestyle="--",
                 label="Faulty (2000km ISL)")
        plt.fill_between(steps_2000, f_mean_2000 - f_std_2000,
                         f_mean_2000 + f_std_2000, color="red", alpha=0.1)

    plt.axhline(0.5, color="gray", linestyle=":", label="Neutral")
    plt.xlabel("Timestep [-]", fontsize=14)
    plt.ylabel("Reputation [-]", fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reputation_comparison.png"))
    plt.show()

def plot_kpi_comparison(results_100km: List[Dict[str, Any]],
                        results_2000km: List[Dict[str, Any]]) -> None:
    """
    Plots a bar chart comparison of key KPIs, including TTD as a percentage of runtime.
    """
    metrics = ["recall", "precision", "fpr"]
    labels = ["Recall", "Precision", "False Positive Rate", "Normalised TTD"]

    # Calculate means for standard metrics
    means_100 = [np.mean([k[m] for k in results_100km]) for m in metrics]
    means_2000 = [np.mean([k[m] for k in results_2000km]) for m in metrics]

    # Calculate Normalised TTD (as % of total steps)
    def get_ttd_percent(results):
        ttd_pcts = []
        for k in results:
            if k.get("avg_ttd") is not None:
                total_steps = k["honest_matrix"].shape[1]
                ttd_pcts.append((k["avg_ttd"] / total_steps) * 100)
        return np.mean(ttd_pcts) if ttd_pcts else 0

    means_100.append(get_ttd_percent(results_100km))
    means_2000.append(get_ttd_percent(results_2000km))

    x = np.arange(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, means_100, width, label='100km ISL', color='skyblue')
    ax.bar(x + width/2, means_2000, width, label='2000km ISL', color='salmon')

    ax.set_ylabel('Percentage [%]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105) # Metrics are percentages

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kpi_percentage_comparison.png"))
    plt.show()

    # TTD Comparison
    ttds_100 = [float(k.get("avg_ttd", 0)) for k in results_100km if k.get("avg_ttd") is not None]
    ttds_2000 = [float(k.get("avg_ttd", 0)) for k in results_2000km if k.get("avg_ttd") is not None]

    if ttds_100 or ttds_2000:
        plt.figure(figsize=(6, 6))
        data_to_plot = []
        labels_ttd = []
        if ttds_100:
            data_to_plot.append(ttds_100)
            labels_ttd.append("100km ISL")
        if ttds_2000:
            data_to_plot.append(ttds_2000)
            labels_ttd.append("2000km ISL")

        # Reduce gap by setting positions closer and increasing widths
        positions = [1, 1.5]
        plt.boxplot(data_to_plot, tick_labels=labels_ttd,
                    positions=positions[:len(data_to_plot)], widths=0.35)
        plt.xlim(0.5, 2.0)
        plt.ylabel("Time to Detection  [Timesteps]")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "ttd_comparison.png"))
        plt.show()

def main():
    """
    Main entry point for the comparison script.
    Loads results for both 100km and 2000km ISL ranges, generates
    comparison plots, and prints a summary to the console.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading 100km results...")
    res_100 = load_results(PATH_100KM)
    print("Loading 2000km results...")
    res_2000 = load_results(PATH_2000KM)

    if not res_100 and not res_2000:
        print("No results found to compare.")
        return

    print(f"Comparing {len(res_100)} runs (100km) vs {len(res_2000)} runs (2000km)")

    plot_reputation_comparison(res_100, res_2000)
    plot_kpi_comparison(res_100, res_2000)

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

    print_summary("100km ISL", res_100)
    print_summary("2000km ISL", res_2000)

if __name__ == "__main__":
    main()
