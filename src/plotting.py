# pylint: disable=too-many-locals, too-many-statements, protected-access, broad-exception-caught, too-many-branches, too-many-lines
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

import base64
import json
import os
import re
from typing import Optional, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2
import seaborn as sns
from src.simulation import generate_random_keplerian_elements
from src.dag import DAG
from src.reputation import MAX_REPUTATION, ReputationManager

# === Configuration ===
DATA_DIR = "sim_data"
FILENAME = "sim_data/app.log"  # your log file path
THRESHOLD = 0.5                # consensus threshold
CMAP = "viridis"               # color map for correctness
REP_MGR = ReputationManager()

def plot_nis_vs_consensus(df: pd.DataFrame) -> None:
    """
    Plots Normalised Innovation Squared (NIS) vs. consensus score.

    Args:
        df (pd.DataFrame): DataFrame containing 'nis', 'consensus_score', and 'correctness' columns.

    Returns:
        None: Displays a matplotlib plot.
    """

    fig, ax = plt.subplots(figsize=(10, 7))

    # Main plot
    scatter = ax.scatter(
        df["nis"],
        df["consensus_score"],
        c=df["correctness"],
        cmap=CMAP,
        s=20,
        alpha=0.8,
        edgecolors='none'
    )

    ax.axhline(THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold = {THRESHOLD}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Correctness [-]", fontsize=20)

    ax.set_xlabel("Normalised Innovation Squared [-]", fontsize=20)
    ax.set_ylabel("Consensus Score [-]", fontsize=20)
    ax.set_xscale('symlog')

    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    # Adjust legend to handle the transparency gracefully
    leg = ax.legend(fontsize=20)
    for lh in leg.legend_handles:
        lh.set_alpha(1)  # type: ignore [union-attr]

    ax.grid(True, linestyle=":", alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_constellation(truth: np.ndarray, n: int) -> None:
    """
    Plots the 3D orbits of a satellite constellation around the Earth.

    Args:
        truth (np.ndarray): The history of true stacked state vectors, with shape (steps, 6*N).
        n (int): The number of satellites.

    Returns:
        None: Displays a matplotlib plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    r_e = 6378e3  # Earth radius in meters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = r_e * np.outer(np.cos(u), np.sin(v))
    y_earth = r_e * np.outer(np.sin(u), np.sin(v))
    z_earth = r_e * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, # type: ignore [attr-defined]
                    color='blue', alpha=0.3,
                    rstride=4, cstride=4)  # type: ignore [attr-defined]

    # Plot satellite orbits
    for i in range(n):
        # Extract position history for satellite i
        pos_hist = truth[:, i*6:i*6+3]

        # Plot orbit path
        ax.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], label=f'Sat {i}')

        # Plot final position
        ax.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2], s=30) # type: ignore [misc]

    # Set plot labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)") # type: ignore [attr-defined]

    # Make axes equal to avoid distortion
    max_range_temp = np.array([ax.get_xlim(), ax.get_ylim(),
                               ax.get_zlim()]) # type: ignore [attr-defined]
    max_range = np.ptp(max_range_temp).max() / 2.0
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim()) # type: ignore [attr-defined]
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range) # type: ignore [attr-defined]

    ax.legend()
    plt.show()


def plot_reputation(rep_history: dict[str, list[float]]) -> None:
    """
    Plot the reputation history of satellite nodes.

    Args:
        rep_history (dict[str, list[float]]): A dictionary where keys are node IDs and
                                              values are lists of reputation scores over time.

    Returns:
        None: Displays a plot of reputation over time for each node.
    """
    neutral_level: float = MAX_REPUTATION / 2
    plt.figure(figsize=(8, 5))

    max_len = max((len(h) for h in rep_history.values()), default=0)
    steps = list(range(max_len))

    # Plot reputation histories
    for node_id, history in rep_history.items():
        plt.plot(range(len(history)), history, marker="o", \
                 markersize=2, label=f"Sat_{node_id} Reputation")

    # Plot target curve ONCE (using max length)
    if max_len > 0:
        # Simulate the max reputation trajectory (all positives, with decay)
        exp_pos = 0
        rep = MAX_REPUTATION / 2
        target_curve = []
        for _ in steps:
            rep = REP_MGR.decay(rep)
            gompertz_target = REP_MGR._gompertz_target(exp_pos)
            rep = rep + REP_MGR.alpha * (gompertz_target - rep)
            target_curve.append(rep)
            exp_pos += 1
        target_curve = np.array(target_curve) # type: ignore [assignment]

        plt.plot(steps, target_curve, linestyle="--",
                 color="orange", linewidth=2, label="Target curve")

    # Neutral line
    plt.axhline(neutral_level, color="gray", linestyle=":", label=f"Neutral ({neutral_level})")

    plt.xlabel("Chronological Transaction Index [-]", fontsize=14)
    plt.ylabel("Reputation Score [-]", fontsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.,
               fontsize=14)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()


def plot_nis_consistency_by_satellite(dag: DAG, confidence: float = 0.95) -> None:
    """
    Plots Normalised Innovation Squared (NIS) values for each satellite individually,
    comparing them to expected chi-squared consistency bounds. Each satellite
    is displayed in a separate plot window.

    Args:
        dag (DAG): The final DAG object containing transactions (with NIS + DOF metadata).
        confidence (float): Confidence level for chi-square bounds (default=0.95).

    Returns:
        None: Displays NIS plots with statistical consistency regions for each satellite.
    """
    # Collect data by satellite
    data_by_sat: dict[str, list] = {}
    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "nis") or not hasattr(tx.metadata, "dof"):
                continue

            try:
                tx_data = json.loads(tx.tx_data)
            except Exception:
                continue

            sid = tx_data.get("observer")
            if sid is None:
                continue

            nis = getattr(tx.metadata, "nis", None)
            dof = getattr(tx.metadata, "dof", None)
            if nis is None or dof is None:
                continue

            data_by_sat.setdefault(str(sid), []).append({
                "nis": nis,
                "dof": dof,
            })

    # Filter out satellites with no data
    data_by_sat = {sid: vals for sid, vals in data_by_sat.items() if vals}
    if not data_by_sat:
        print("No NIS/DOF data available to plot.")
        return

    # Sort by satellite ID for consistent plot order
    sorted_sats = sorted(data_by_sat.items(), key=lambda item: int(item[0]))

    for sid, records in sorted_sats:
        # Create a new figure for each satellite
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        nis_vals = np.array([r["nis"] for r in records])
        dof_vals = np.array([r["dof"] for r in records])

        if len(nis_vals) == 0:
            continue

        mean_dof = np.mean(dof_vals)

        # Compute chi-square confidence bounds
        chi2_lower = chi2.ppf((1 - confidence) / 2, df=mean_dof)
        chi2_upper = chi2.ppf((1 + confidence) / 2, df=mean_dof)
        expected_mean = mean_dof

        # Plot NIS sequence
        steps = np.arange(len(nis_vals))
        ax.plot(steps, nis_vals, "o", color="black",
                label=f"NIS (Sat_{sid})")

        # Expected mean and confidence region
        ax.axhline(expected_mean, color="blue", linestyle="--",
                    label=f"Expected mean (DOF={mean_dof:.1f})")
        ax.fill_between(
            steps,
            chi2_lower,
            chi2_upper,
            color="green",
            alpha=0.1,
            label=f"{int(confidence*100)}% confidence region"
        )

        ax.set_ylabel("Normalised Innovation Squared", fontsize=24)
        ax.set_yscale("symlog")
        ax.grid(True, linestyle=":")
        ax.legend(loc="upper right", fontsize=20)
        ax.set_xlabel("Transaction Index", fontsize=24)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        plt.tight_layout()

    plt.show()


def plot_nis_boxplot(dag: DAG, faulty_ids: set[int],
                    convergence_index: Optional[int] = None) -> None:
    """
    Generates a grouped box plot for NIS values, separating honest and faulty satellites.

    Args:
        dag (DAG): The DAG object containing transaction data.
        faulty_ids (set[int]): A set of IDs for faulty satellites.
        convergence_index (int): Optional index to only plot data
                                 after filter convergence.

    Returns:
        None: Displays a matplotlib plot.
    """
    honest_nis = []
    faulty_nis = []
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

            if int(sid) in faulty_ids:
                faulty_nis.append(nis)
            else:
                honest_nis.append(nis)

    if not honest_nis and not faulty_nis:
        print("No NIS data available to create a box plot.")
        return

    honest_nis = honest_nis[start_index:]
    faulty_nis = faulty_nis[start_index:]

    plot_data = []
    labels = []

    if honest_nis:
        plot_data.append(honest_nis)
        labels.append("Honest Satellites")
    if faulty_nis:
        plot_data.append(faulty_nis)
        labels.append("Faulty Satellites")

    _, ax = plt.subplots(figsize=(10, 6))

    # Create box plot
    parts = ax.boxplot(plot_data, labels=labels) # type: ignore [call-arg]

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.5)

    # Add expected median (assuming DOF=2)
    expected_median = 1.386

    # Compute chi-square 95% confidence bounds
    chi2_lower = chi2.ppf((1 - 0.95) / 2, df=2)
    chi2_upper = chi2.ppf((1 + 0.95) / 2, df=2)

    # Plot the horizontal lines for the confidence interval bounds
    ax.axhline(chi2_lower, color='red', linestyle='--', alpha=0.7, \
        label='95% Confidence Interval Bounds')
    ax.axhline(chi2_upper, color='red', linestyle='--', alpha=0.7)

    ax.axhline(expected_median, color='black', linestyle=':', label='Expected Median (1.386)')

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Normalised Innovation Squared [-]", fontsize=20)
    ax.set_yscale("log")
    ax.tick_params(axis='y', labelsize=20)

    ax.legend(fontsize=16, loc="upper center")
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.show()


def calculate_median_percentiles(dof: int = 2) -> None:
    """
    Calculates the chi-squared CDF percentiles for given median values
    and determines their absolute distance from the ideal 50th percentile.
    """
    median_values: list[float] = [1.386, 1.707, 2.678]
    print(f"--- Chi-Squared CDF Percentiles (DOF={dof}) ---")
    print(f"{'Median Value':<15} | {'Percentile (CDF)':<20} | {'Distance from 0.5':<20}")
    print("-" * 60)

    for val in median_values:
        # Calculate the cumulative probability (percentile)
        percentile = chi2.cdf(val, df=dof)

        # Calculate how far it deviates from the ideal 0.5 (50%) mark
        distance_from_ideal = abs(percentile - 0.5)

        print(f"{val:<15.3f} | {percentile:<20.4f} | {distance_from_ideal:<20.4f}")


def check_consensus_outcomes(dag: DAG, consensus_threshold: float = 0.5) -> bool:
    """
    Checks if transaction consensus outcomes (confirmed/rejected) are consistent
    with their consensus scores and reports any discrepancies.

    This function iterates through all transactions in the DAG that have a consensus
    score and verifies that:
    1. Transactions with a score >= threshold are marked as 'confirmed'.
    2. Transactions with a score < threshold are marked as 'rejected'.

    Args:
        dag (DAG): The DAG containing transaction data.
        consensus_threshold (float): The consensus threshold used in the simulation.

    Returns:
        True if all outcomes are consistent, False otherwise.
    """
    inconsistencies = []
    counter = 0
    for tx_hash, tx_list in dag.ledger.items():
        for tx in tx_list:
            # Skip genesis transactions or transactions without a score
            if not hasattr(tx.metadata, "consensus_score"):
                continue

            score = tx.metadata.consensus_score
            is_confirmed = getattr(tx.metadata, "is_confirmed", False)
            is_rejected = getattr(tx.metadata, "is_rejected", False)

            # Expected outcome based on the score
            should_be_confirmed = score >= consensus_threshold

            # Check for inconsistencies
            if should_be_confirmed:
                # Skip first 2 genesis transactions
                if not is_confirmed:
                    inconsistencies.append(
                        f"TX {tx_hash[:8]}: score {score:.3f} >= {consensus_threshold} "
                        f"but was NOT confirmed."
                    )
                if is_rejected:
                    inconsistencies.append(
                        f"TX {tx_hash[:8]}: score {score:.3f} >= {consensus_threshold} "
                        f"but was REJECTED."
                    )
            else:  # Should be rejected
                if is_confirmed and "Genesis" not in tx_hash:
                    inconsistencies.append(
                        f"TX {tx_hash[:8]}: score {score:.3f} < {consensus_threshold} "
                        f"but was CONFIRMED."
                    )
                    # Skip 2 genesis transactions and 3 real transactions
                    # needed for BFT quorum
                elif not is_rejected and counter >= 5:
                    inconsistencies.append(
                        f"TX {tx_hash[:8]}: score {score:.3f} < {consensus_threshold} "
                        f"but was NOT rejected."
                    )
            counter += 1

    if not inconsistencies:
        print("✅ Consensus outcomes are consistent with scores.")
        return True

    print("❌ Found inconsistencies in consensus outcomes:")
    for issue in inconsistencies:
        print("- %s", issue)
    return False


def calculate_convergence_index(
    rep_history: dict[str, list[float]],
    faulty_ids: set[int],
    threshold: float = 0.5
) -> int:
    """
    Heuristically identifies the convergence index based on when the mean
    reputation of honest satellites starts to rise significantly above neutral.

    Args:
        rep_history (dict[str, list[float]]): Dictionary of reputation histories.
        faulty_ids (set[int]): Set of faulty satellite IDs.
        threshold (float): Reputation threshold to consider "converged".

    Returns:
        int: The index of the first step where convergence is detected.
    """
    honest_sids = [sid for sid in rep_history.keys() if int(sid) not in faulty_ids]
    if not honest_sids:
        return 0

    honest_histories = [rep_history[sid] for sid in honest_sids]
    max_len = max(len(h) for h in honest_histories)

    # Pad histories for mean calculation
    padded = [h + [h[-1]] * (max_len - len(h)) for h in honest_histories]
    honest_mean = np.mean(padded, axis=0)

    # Find first index where honest mean exceeds threshold
    indices = np.where(honest_mean > threshold)[0]
    return int(indices[0]) if indices.size > 0 else 0


def calculate_nis_convergence_index(
    dag: DAG,
    faulty_ids: set[int],
    confidence: float = 0.95,
    window_size: int = 5
) -> int:
    """
    Identifies the convergence index based on when the NIS values of honest
    satellites enter and stay within the expected chi-squared consistency bounds.

    Args:
        dag (DAG): The DAG object containing transactions with NIS metadata.
        faulty_ids (set[int]): Set of faulty satellite IDs to exclude.
        confidence (float): Confidence level for chi-square bounds (default=0.95).
        window_size (int): Number of consecutive steps NIS must be within bounds.

    Returns:
        int: The first step where convergence is detected.
    """
    # 1. Collect NIS values for honest satellites, grouped by step
    step_nis_data: dict[int, list[float]] = {}
    step_dof_data: dict[int, list[int]] = {}

    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "nis") or not hasattr(tx.metadata, "dof"):
                continue

            try:
                tx_data = json.loads(tx.tx_data)
            except (json.JSONDecodeError, TypeError):
                continue

            sid = tx_data.get("observer")
            step = tx_data.get("step")
            if sid is None or step is None or int(sid) in faulty_ids:
                continue

            nis = getattr(tx.metadata, "nis", None)
            dof = getattr(tx.metadata, "dof", None)
            if nis is None or dof is None:
                continue

            step_nis_data.setdefault(int(step), []).append(float(nis))
            step_dof_data.setdefault(int(step), []).append(int(dof))

    if not step_nis_data:
        return 0

    sorted_steps = sorted(step_nis_data.keys())

    # 2. Check per step if mean NIS is within chi-square upper bound
    is_converged = []
    for step in sorted_steps:
        mean_nis = np.mean(step_nis_data[step])
        mean_dof = np.mean(step_dof_data[step])

        # Upper bound for chi-square
        chi2_upper = chi2.ppf((1 + confidence) / 2, df=mean_dof)

        is_converged.append(mean_nis <= chi2_upper)

    # 3. Find first step where we have a window of consecutive converged steps
    for i in range(len(is_converged) - window_size + 1):
        if all(is_converged[i : i + window_size]):
            return int(sorted_steps[i])

    return 0


def plot_aggregated_reputation(
    rep_history: dict[str, list[float]],
    faulty_ids: set[int],
    start_at_full_constellation: bool = False,
    convergence_index: Optional[int] = None
) -> None:
    """
    Plots the aggregated median reputation over time for honest vs. faulty satellites,
    with shaded regions indicating the 10th to 90th percentile spread.

    Args:
        rep_history (dict[str, list[float]]): A dictionary of reputation histories
                                              for each satellite.
        faulty_ids (set[int]): A set of IDs for faulty satellites.
        start_at_full_constellation (bool): If True, starts plotting only after
                                            a number of transactions equal to the
                                            number of satellites has passed,
                                            assuming this is when all nodes have
                                            had a chance to submit data.
        convergence_index (int): Optional index to plot a vertical dashed line
                                 indicating filter convergence.

    Returns:
        None: Displays a matplotlib plot.
    """
    if not rep_history:
        print("No reputation data to plot.")
        return

    max_len = max(len(h) for h in rep_history.values())
    honest_matrix = []
    faulty_matrix = []

    # Pad histories to the same length for numpy operations
    for sid, history in rep_history.items():
        padded_history = history + [history[-1]] * (max_len - len(history))
        if int(sid) in faulty_ids:
            faulty_matrix.append(padded_history)
        else:
            honest_matrix.append(padded_history)

    honest_matrix = np.array(honest_matrix)  # type: ignore [assignment]
    faulty_matrix = np.array(faulty_matrix)  # type: ignore [assignment]

    start_index = 0
    if start_at_full_constellation:
        # Assuming the constellation is fully formed after 60% of the transactions.
        start_index = convergence_index if convergence_index is not None else int(0.6 * max_len)

    if start_index >= max_len:
        print("Not enough data to plot with 'start_at_full_constellation'=True. Plotting all data.")
        start_index = 0

    # Slice data for plotting
    steps = np.arange(max_len)[start_index:]
    if len(honest_matrix) > 0:
        honest_matrix = honest_matrix[:, start_index:]  # type: ignore [call-overload]
    if len(faulty_matrix) > 0:
        faulty_matrix = faulty_matrix[:, start_index:]  # type: ignore [call-overload]

    if not steps.size:
        print("No data points to plot after filtering.")
        return

    plt.figure(figsize=(10, 6))

    # Plot Honest Satellites
    if len(honest_matrix) > 0:
        honest_mean = np.mean(honest_matrix, axis=0)
        honest_std = np.std(honest_matrix, axis=0)

        plt.plot(steps, honest_mean, color="green", linewidth=2, label="Honest Mean")
        plt.fill_between(
            steps,
            honest_mean - honest_std,
            honest_mean + honest_std,
            color="green",
            alpha=0.2,
            label="Honest Spread (1 std. dev.)",
        )

    # Plot Faulty Satellites
    if len(faulty_matrix) > 0:
        faulty_mean = np.mean(faulty_matrix, axis=0)
        faulty_std = np.std(faulty_matrix, axis=0)

        plt.plot(steps, faulty_mean, color="red", linewidth=2, label="Faulty Mean")
        plt.fill_between(
            steps,
            faulty_mean - faulty_std,
            faulty_mean + faulty_std,
            color="red",
            alpha=0.2,
            label="Faulty Spread (1 std. dev.)",
        )

    # Formatting
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=2, label="Neutral (0.5)")

    if convergence_index is not None and not start_at_full_constellation:
        plt.axvline(x=convergence_index, color="black", linestyle="--",\
            linewidth=1, label="Filter Convergence")

    plt.xlabel("Chronological Transaction Index [-]", fontsize=20)
    plt.ylabel("Reputation Score [-]", fontsize=20)

    plt.tick_params(axis='both', labelsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_ground_tracks(truth: np.ndarray, n: int) -> None:
    """
    Plots a 2D ground track map using a static Earth image background.

    Args:
        truth (np.ndarray): The history of true stacked state vectors, with shape (steps, 6*N).
        n (int): The number of satellites.

    Returns:
        None: Displays a matplotlib plot.
    """
    _, ax = plt.subplots(figsize=(14, 8))

    # --- PART 1: The "Poor Man's" Map Background ---

    img_path = "images/1024px-Land_ocean_ice_2048.jpg"

    # Display the image with the correct extent [-180, 180, -90, 90]
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img, extent=(-180.0, 180.0, -90.0, 90.0), aspect='auto', alpha=0.2)
    else:
        # Fallback if download fails
        ax.set_facecolor('lightgray')

    # --- PART 2: Plotting the Data ---

    # Set limits explicitly to match the image extent
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    for i in range(n):
        pos_hist = truth[:, i*6:i*6+3]

        # Convert Cartesian X,Y,Z to Lat, Lon
        r = np.linalg.norm(pos_hist, axis=1)
        lat = np.degrees(np.arcsin(np.clip(pos_hist[:, 2] / r, -1, 1)))
        lon = np.degrees(np.arctan2(pos_hist[:, 1], pos_hist[:, 0]))

        # Handle wraparound
        lon_diff = np.abs(np.diff(lon))
        wrap_idx = np.where(lon_diff > 180)[0]

        lon_plot = np.insert(lon, wrap_idx + 1, np.nan)
        lat_plot = np.insert(lat, wrap_idx + 1, np.nan)

        # Formatting
        color = 'black'
        alpha = 0.1
        zorder = 5
        lw = 1.2

        ax.plot(lon_plot, lat_plot, color=color, alpha=alpha,
                lw=lw, zorder=zorder)

        # Plot current/final position
        ax.scatter(lon[-1], lat[-1], color=color, s=30,
                   edgecolor='white', linewidth=0.5, zorder=zorder+1)

    # --- PART 3: Styling ---

    # Custom legend
    handles = [
        Line2D([0], [0], color='black', lw=2, label='Simulated Satellite Orbits')
    ]
    leg = ax.legend(handles=handles, loc='upper right', framealpha=0.7, facecolor='white',
                    fontsize=16)
    leg.set_zorder(10)

    ax.set_xlabel("Longitude [Degrees]", fontsize=20)
    ax.set_ylabel("Latitude [Degrees]", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    # White grid looks better on dark maps
    ax.grid(True, linestyle=":", alpha=0.4, color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "orbit_map.png"))

def plot_ground_tracks_plotly(truth: np.ndarray, n: int) -> go.Figure:
    """
    Plots an interactive 2D ground track map using Plotly with an Earth background.
    
    Args:
        truth (np.ndarray): The history of true stacked state vectors.
        n (int): The number of satellites.
        
    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    # --- PART 1: Background Image ---
    img_path = "images/1024px-Land_ocean_ice_2048.jpg"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        img_source = f"data:image/jpeg;base64,{encoded_string}"

        fig.add_layout_image(
            {
                "source": img_source,
                "xref": "x",
                "yref": "y",
                "x": -180,
                "y": 90,
                "sizex": 360,
                "sizey": 180,
                "sizing": "stretch",
                "opacity": 0.2,
                "layer": "below"
            }
        )

    # --- PART 2: Data ---
    for i in range(n):
        pos_hist = truth[:, i*6:i*6+3]

        # Convert Cartesian X,Y,Z to Lat, Lon
        r = np.linalg.norm(pos_hist, axis=1)
        lat = np.degrees(np.arcsin(np.clip(pos_hist[:, 2] / r, -1, 1)))
        lon = np.degrees(np.arctan2(pos_hist[:, 1], pos_hist[:, 0]))

        # Handle wraparound (insert NaNs)
        lon_diff = np.abs(np.diff(lon))
        wrap_idx = np.where(lon_diff > 180)[0]
        lon_plot = np.insert(lon, wrap_idx + 1, np.nan)
        lat_plot = np.insert(lat, wrap_idx + 1, np.nan)

        # Track line
        fig.add_trace(go.Scatter(
            x=lon_plot, y=lat_plot,
            mode='lines',
            line={"width": 1.2, "color": "black"},
            opacity=0.1,
            name=f"Sat {i} Track",
            showlegend=False,
            hoverinfo='skip'
        ))

        # Current position
        fig.add_trace(go.Scatter(
            x=[lon[-1]], y=[lat[-1]],
            mode='markers',
            marker={
                "size": 10,
                "color": "black",
                "line": {"width": 0.5, "color": "white"}
            },
            name=f"Sat {i}",
            text=f"Satellite {i}<br>Lat: {lat[-1]:.2f}<br>Lon: {lon[-1]:.2f}",
            hoverinfo='text',
            showlegend=False
        ))

    # --- PART 3: Custom Legend ---
    # Add a dummy trace to represent the legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line={"width": 2, "color": "black"},
        name='Simulated Satellite Orbits'
    ))

    # --- PART 4: Styling ---
    fig.update_xaxes(
        range=[-180, 180],
        title={"text": "Longitude [Degrees]", "font": {"size": 20}},
        gridcolor='rgba(255,255,255,0.4)',
        gridwidth=1,
        zeroline=False,
        tickfont={"size": 16}
    )
    fig.update_yaxes(
        range=[-90, 90],
        title={"text": "Latitude [Degrees]", "font": {"size": 20}},
        gridcolor='rgba(255,255,255,0.4)',
        gridwidth=1,
        zeroline=False,
        tickfont={"size": 16}
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin={"l": 60, "r": 40, "t": 40, "b": 60},
        height=500,
        showlegend=True,
        legend={
            "x": 0.98, "y": 0.98,
            "xanchor": "right", "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.7)",
            "bordercolor": "rgba(0,0,0,0.1)",
            "borderwidth": 1,
            "font": {"size": 14, "color": "black"}
        }
    )

    # Force axis labels and ticks to be black for visibility on white background
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont={"color": "black"},
                     title_font={"color": "black"})
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont={"color": "black"},
                     title_font={"color": "black"})

    return fig

def plot_mc_nis_boxplot(all_kpis: List[Dict[str, Any]]) -> None:
    """
    Plots the distribution of median NIS values across multiple Monte Carlo runs,
    separated by honest and faulty populations.

    Args:
        all_kpis (List[Dict[str, Any]]): A list of KPI dictionaries, each containing
                                         'honest_nis_stats' and 'faulty_nis_stats' dicts.
    """

    all_honest_medians = []
    all_faulty_medians = []

    for kpi in all_kpis:
        h_stats = kpi.get("honest_nis_stats", {})
        f_stats = kpi.get("faulty_nis_stats", {})

        # Extract the median directly
        if "median" in h_stats:
            all_honest_medians.append(h_stats["median"])
        if "median" in f_stats:
            all_faulty_medians.append(f_stats["median"])

    if not all_honest_medians and not all_faulty_medians:
        print("No MC NIS data available to plot.")
        return

    plot_data = []
    labels = []
    if all_honest_medians:
        plot_data.append(all_honest_medians)
        labels.append("Honest Medians")
    if all_faulty_medians:
        plot_data.append(all_faulty_medians)
        labels.append("Faulty Medians")

    _, ax = plt.subplots(figsize=(10, 6))

    # Create box plot for the medians
    ax.boxplot(plot_data, label=labels, patch_artist=True,
               boxprops={"facecolor": 'lightblue', "alpha": 0.5},
               medianprops={"color": 'black', "linewidth": 2})

    # Reference lines (assuming DOF=2)
    dof = 2
    expected_median = chi2.ppf(0.5, df=dof)

    ax.axhline(expected_median, color='black', linestyle=':',
                label=f'Expected Median ({expected_median:.3f})')

    ax.set_ylabel("Median NIS per Run [-]", fontsize=20)
    ax.set_yscale("log")
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

def generate_constellation_df(num_sats: int, seed: int) -> pd.DataFrame:
    """
    Generates valid Keplerian elements for a LEO constellation using vectorized RNG.
    Returns a pandas DataFrame formatted for easy plotting.
    """
    elements = []
    for n in range(num_sats):
        a, e, i, raan, argp, ta = generate_random_keplerian_elements(seed=seed + n)
        elements.append((a, e, i, raan, argp, ta))

    # Create a DataFrame
    df = pd.DataFrame({
        'Semi-Major Axis\n[km]': [elem[0] for elem in elements],
        'Eccentricity\n[-]': [elem[1] for elem in elements],
        'Inclination\n[deg]': [elem[2] for elem in elements],
        'RAAN\n[deg]': [elem[3] for elem in elements],
        'Arg of Perigee\n[deg]': [elem[4] for elem in elements],
        'True Anomaly\n[deg]': [elem[5] for elem in elements]
    })

    return df

def generate_corner_plot() -> None:
    """
    Generates a corner plot of the Keplerian elements for a LEO constellation.
    """
    # 1. Generate the 400 satellites
    df_sats = generate_constellation_df(num_sats=400, seed=42)

    # 2. Set up the visual style (reduced font_scale slightly for better fit)
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    # 3. Create the Corner Plot
    # Use 'height' to set the size of EACH subplot (in inches).
    # height=2.2 makes a nice, large figure that gives the labels breathing room.
    g = sns.PairGrid(df_sats, corner=True, diag_sharey=False, height=2.2)

    g.map_diag(sns.histplot, kde=True, color="steelblue", element="step")
    g.map_lower(sns.scatterplot, s=10, alpha=0.5, color="darkblue")

    # 4. Fix overlapping labels and ticks
    for ax in g.axes.flatten():
        if ax is not None:
            # Rotate x-axis numbers by 45 degrees so they don't crash into each other
            ax.tick_params(axis='x', rotation=45)

            # Explicitly control the axis label font size and padding
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=5)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=5)

    # Align labels across the grid to ensure they are even
    g.figure.align_labels()

    # 5. Force matplotlib to respect the margins
    # top=0.92 leaves space for the suptitle. wspace/hspace control gaps between plots.
    plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.15, hspace=0.15)

    # Save and show
    plt.savefig("images/orbital_elements_corner_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    """Main function to parse log and generate plots."""
    # === Step 1: Parse the log file ===
    pattern = re.compile(
        r"NIS=([0-9.]+), DOF=([0-9]+), correctness=([0-9.]+), consensus_score=([0-9.]+),\s*reputation=([0-9.]+)" # pylint: disable=line-too-long
    )

    data = []
    try:
        with open(FILENAME, "r", encoding="utf-8") as f:
            content = f.read()
            for match in pattern.finditer(content):
                data.append(tuple(map(float, match.groups())))
    except FileNotFoundError:
        print(f"Error: Log file not found at '{FILENAME}'. \
              Make sure the path is correct.")
        return

    if not data:
        print("No data found in log file matching the pattern.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["nis", "dof", "correctness",
                                     "consensus_score", "reputation"])

    # === Step 2: Generate plots ===
    plot_nis_vs_consensus(df)


if __name__ == "__main__":
    main()
