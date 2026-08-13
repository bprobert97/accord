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

import json
from typing import Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def plot_nis_boxplot_wd(dag: Any, compromised_ids: set[int],
                     convergence_index: Optional[int] = None) -> None:
    """
    Generates a grouped box plot for NIS values, separating honest and compromised satellites,
    overlaid with a jittered scatter plot using the viridis colormap.

    Note: This is a legacy version of the plot_nis_boxplot function
    from v3.1 of the ACCORD framework, retained for backward compatibility.

    Args:
        dag (DAG): The DAG object containing transaction data.
        compromised_ids (set[int]): A set of IDs for compromised satellites.
        convergence_index (int): Optional index to only plot data
                                 after filter convergence.

    Returns:
        None: Displays a matplotlib plot.
    """
    honest_nis = []
    compromised_nis = []
    start_index = convergence_index if convergence_index is not None else 0

    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "nis"):
                continue

            try:
                tx_data = json.loads(tx.tx_data)
            except Exception:
                continue

            sid = tx_data.get("observer")
            nis = getattr(tx.metadata, "nis", None)

            if sid is None or nis is None:
                continue

            if int(sid) in compromised_ids:
                compromised_nis.append(nis)
            else:
                honest_nis.append(nis)

    if not honest_nis and not compromised_nis:
        print("No NIS data available to create a box plot.")
        return

    honest_nis = honest_nis[start_index:]
    compromised_nis = compromised_nis[start_index:]

    plot_data = []
    labels = []

    if honest_nis:
        plot_data.append(honest_nis)
        labels.append("Honest Satellites")
    if compromised_nis:
        plot_data.append(compromised_nis)
        labels.append("Compromised Satellites")

    _, ax = plt.subplots(figsize=(10, 6))

    # Create box plot
    # Setting zorder to 3 so the boxplot lines render above the scatter points
    parts = ax.boxplot(plot_data, labels=labels, zorder=3)  # type: ignore[call-arg]

    # Style the box plot lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'boxes', 'whiskers', 'caps'):
        if partname in parts:
            for item in parts[partname]:
                item.set_color('black')
                item.set_linewidth(1.5)

    # 2. Setup the viridis colors
    cmap = plt.get_cmap('viridis')
    # Use linspace between 0.2 and 0.8 to avoid the invisible yellow/black extremes of the colormap
    scatter_colors = [cmap(val) for val in np.linspace(0.95, 0.85, len(plot_data))]

    # 3. Plot the raw scatter points underneath (zorder=2)
    for i, data in enumerate(plot_data):
        x_jitter = np.random.normal(loc=i + 1, scale=0.02, size=len(data))
        # Assign the fixed viridis color for this specific dataset
        ax.scatter(x_jitter, data, alpha=0.1, color=scatter_colors[i], s=20, zorder=2)

    # Add expected median (assuming DOF=2)
    expected_median = 1.386

    # Compute chi-square 95% confidence bounds
    chi2_lower = float(chi2.ppf((1 - 0.95) / 2, df=2))
    chi2_upper = float(chi2.ppf((1 + 0.95) / 2, df=2))

    cmap = plt.get_cmap('viridis')
    color_bound = cmap(0.1) # Dark Purple for bounds

    # Plot the horizontal lines for the confidence interval bounds
    ax.axhline(chi2_lower, color=color_bound, linestyle='--', alpha=0.7,
               label='95% Confidence Interval Bounds', zorder=1)
    ax.axhline(chi2_upper, color=color_bound, linestyle='--', alpha=0.7, zorder=1)

    ax.axhline(expected_median, color='black', linestyle=':',
               label='Expected Median (1.386)', zorder=4)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Normalised Innovation Squared [-]", fontsize=20)

    # Set y-axis to logarithmic scale as in the original
    ax.set_yscale("log")
    ax.tick_params(axis='y', labelsize=20)

    ax.legend(fontsize=16, loc="upper center")
    ax.grid(True, linestyle=":", alpha=0.7, zorder=0)

    plt.tight_layout()
    plt.show()