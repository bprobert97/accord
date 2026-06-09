# mypy: disable-error-code="attr-defined"
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
import os
import re
from typing import Optional, List, Dict, Any, Iterator, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import chi2
import seaborn as sns
from src.simulation import generate_random_keplerian_elements
from src.dag import DAG, MockDAG
from src.reputation import ReputationManager, MAX_REPUTATION

# === Configuration ===
DATA_DIR = "sim_data"
FILENAME = "sim_data/app.log"  # your log file path
THRESHOLD = 0.5                # consensus threshold
CMAP = "viridis"               # color map for correctness
REP_MGR = ReputationManager()

def extract_nis_transactions(dag: Any) -> Iterator[Tuple[Any, dict]]:
    """
    Generator that iterates through a DAG ledger and yields
    transactions (and their parsed JSON) that contain NIS metadata.
    """
    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "nis"):
                continue

            try:
                tx_data = json.loads(tx.tx_data)
                yield tx, tx_data
            except (json.JSONDecodeError, TypeError):
                continue

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
        if lh is not None:
            lh.set_alpha(1)

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
    ax.plot_surface(x_earth, y_earth, z_earth,
                    color='blue', alpha=0.3,
                    rstride=4, cstride=4)

    # Plot satellite orbits
    for i in range(n):
        # Extract position history for satellite i
        pos_hist = truth[:, i*6:i*6+3]

        # Plot orbit path
        ax.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], color='black', alpha=0.3)

        # Plot final position
        ax.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2],
                   color='black', s=10)  # type: ignore[misc]

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='black', markersize=8),
        Line2D([0], [0], color='black', lw=1.5, label='Simulated Orbit')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set plot labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Make axes equal to avoid distortion
    max_range_temp = np.array([ax.get_xlim(), ax.get_ylim(),
                               ax.get_zlim()])
    max_range = np.ptp(max_range_temp).max() / 2.0
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def plot_nis_boxplot(dag: DAG | MockDAG, faulty_ids: set[int],
                    convergence_index: Optional[int] = None) -> None:
    """
    Generates a grouped box plot for NIS values, separating honest and faulty satellites.

    Args:
        dag (DAG | MockDAG): The DAG or mock DAG object containing transaction data.
        faulty_ids (set[int]): A set of IDs for faulty satellites.
        convergence_index (int): Optional index to only plot data
                                 after filter convergence.

    Returns:
        None: Displays a matplotlib plot.
    """
    honest_nis = []
    faulty_nis = []
    start_index = convergence_index if convergence_index is not None else 0

    for tx, tx_data in extract_nis_transactions(dag):
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
    parts = ax.boxplot(plot_data)

    # Apply the labels manually
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.5)

    # Add expected median (assuming DOF=2)
    expected_median = 1.386

    # Compute chi-square 95% confidence bounds
    chi2_lower = chi2.ppf((1 - 0.95) / 2, df=2)
    chi2_upper = chi2.ppf((1 + 0.95) / 2, df=2)

    cmap = plt.get_cmap('viridis')
    color_bound = cmap(0.1) # Dark Purple for bounds

    # Plot the horizontal lines for the confidence interval bounds
    ax.axhline(chi2_lower, color=color_bound, linestyle='--', alpha=0.7, \
        label='95% Confidence Interval Bounds')
    ax.axhline(chi2_upper, color=color_bound, linestyle='--', alpha=0.7)

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
    median_values: list[float] = [1.386, 1.703, 1.836, 1.447, 1.330]
    print(f"--- Chi-Squared CDF Percentiles (DOF={dof}) ---")
    print(f"{'Median Value':<15} | {'Percentile (CDF)':<20} | {'Distance from 0.5':<20}")
    print("-" * 60)

    for val in median_values:
        # Calculate the cumulative probability (percentile)
        percentile = chi2.cdf(val, df=dof)

        # Calculate how far it deviates from the ideal 0.5 (50%) mark
        distance_from_ideal = abs(percentile - 0.5)

        print(f"{val:<15.3f} | {percentile:<20.4f} | {distance_from_ideal:<20.4f}")


def check_consensus_outcomes(dag: DAG | MockDAG, consensus_threshold: float = 0.5) -> bool:
    """
    Checks if transaction consensus outcomes (confirmed/rejected) are consistent
    with their consensus scores and reports any discrepancies.

    This function iterates through all transactions in the DAG that have a consensus
    score and verifies that:
    1. Transactions with a score >= threshold are marked as 'confirmed'.
    2. Transactions with a score < threshold are marked as 'rejected'.

    Args:
        dag (DAG | MockDAG): The DAG or mock DAG containing transaction data.
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
    dag: DAG | MockDAG,
    faulty_ids: set[int],
    confidence: float = 0.95,
    window_size: int = 5
) -> int:
    """
    Identifies the convergence index based on when the NIS values of honest
    satellites enter and stay within the expected chi-squared consistency bounds.

    Args:
        dag (DAG | MockDAG): The DAG or mock DAG object containing transactions with NIS metadata.
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
    Plots the aggregated median reputation over time for honest vs. faulty satellites.

    Args:
    - rep_history: Dictionary mapping satellite IDs to their reputation history lists.
    - faulty_ids: Set of satellite IDs that are considered faulty.
    - start_at_full_constellation: If True, starts the plot at the convergence index or
      60% of the data if convergence index is not provided. If False, plots from
      the beginning.
    - convergence_index: Optional index to indicate filter convergence point on the plot.

    Returns:
    - None: Displays a matplotlib plot.
    """
    if not rep_history:
        print("No reputation data to plot.")
        return

    honest_arr, faulty_arr, max_len = _prepare_reputation_matrices(rep_history, faulty_ids)

    start_index = 0
    if start_at_full_constellation:
        start_index = convergence_index if convergence_index is not None else int(0.6 * max_len)
        if start_index >= max_len:
            print("Not enough data to plot with 'start_at_full_constellation'=True. \
                  Plotting all data.")
            start_index = 0

    steps = np.arange(max_len)[start_index:]
    if not steps.size:
        print("No data points to plot after filtering.")
        return

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')

    _plot_reputation_spread(steps, honest_arr[:, start_index:] if honest_arr.size \
                            else [], cmap(0.5), "Honest")
    _plot_reputation_spread(steps, faulty_arr[:, start_index:] if faulty_arr.size \
                            else [], cmap(0.05), "Faulty")

    plt.axhline(MAX_REPUTATION/2, color="gray", linestyle=":", linewidth=2, label="Neutral (0.5)")
    if convergence_index is not None and not start_at_full_constellation:
        plt.axvline(x=convergence_index, color="black", linestyle="--",
                    linewidth=1, label="Filter Convergence")

    plt.xlabel("Chronological Transaction Index [-]", fontsize=20)
    plt.ylabel("Reputation Score [-]", fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()


def _prepare_reputation_matrices(rep_history: dict[str, list[float]],
                                 faulty_ids: set[int])-> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepares separate matrices for honest and faulty satellite reputations, padding
    shorter histories with their last known value to allow for mean and std calculations.

    Args:
    - rep_history: Dictionary mapping satellite IDs to their reputation history lists.
    - faulty_ids: Set of satellite IDs that are considered faulty.

    Returns:
    - Tuple containing:
        - honest_matrix: 2D numpy array of shape (num_honest, max_len)
        with padded reputations.
        - faulty_matrix: 2D numpy array of shape (num_faulty, max_len)
        with padded reputations.
        - max_len: The maximum length of the reputation histories (after padding).
    """
    max_len = max(len(h) for h in rep_history.values())
    honest_matrix, faulty_matrix = [], []

    for sid, history in rep_history.items():
        padded_history = history + [history[-1]] * (max_len - len(history))
        if int(sid) in faulty_ids:
            faulty_matrix.append(padded_history)
        else:
            honest_matrix.append(padded_history)

    return np.array(honest_matrix), np.array(faulty_matrix), max_len


def _plot_reputation_spread(steps, data_matrix, colour, label_prefix) -> None:
    """
    Plots the mean reputation over time with a shaded area representing one standard deviation
    around the mean.

    Args:
    - steps: 1D array of step indices corresponding to the reputation data.
    - data_matrix: 2D array where each row is a satellite's reputation history.
    - colour: Colour for the plot line and shaded area.
    - label_prefix: String prefix for the legend label (e.g., "Honest" or "Faulty").

    Returns:
    - None: Adds the plot to the current matplotlib axes.
    """

    if len(data_matrix) > 0:
        mean_vals = np.mean(data_matrix, axis=0)
        std_vals = np.std(data_matrix, axis=0)

        plt.plot(steps, mean_vals, color=colour, linewidth=2, label=f"{label_prefix} Mean")
        plt.fill_between(
            steps, mean_vals - std_vals, mean_vals + std_vals,
            color=colour, alpha=0.2, label=f"{label_prefix} Spread (1 std. dev.)",
        )

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
        ax.scatter(lon[-1], lat[-1], color=color, s=20,
                   edgecolor='white', linewidth=0.5, zorder=zorder+1)

    # --- PART 3: Styling ---

    # Custom legend
    handles = [
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='black', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='Simulated Orbit')
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
        h_stats = kpi.get("honest_nis", {})
        f_stats = kpi.get("faulty_nis", {})

        if h_stats:
            all_honest_medians.append(np.median(h_stats))
        if f_stats:
            all_faulty_medians.append(np.median(f_stats))

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
    ax.boxplot(plot_data)

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
        kep_elements = generate_random_keplerian_elements(seed=seed + n)
        a, e, i, raan, argp, ta = kep_elements.a, kep_elements.e, kep_elements.i, \
                                    kep_elements.raan, kep_elements.argp, kep_elements.ta
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

def generate_corner_plot(num_sats_per_run: int = 400, num_runs: int = 40,
                         base_seed: int = 42) -> None:
    """
    Generates a corner plot of the Keplerian elements aggregated across multiple
    Monte Carlo runs to show the full distribution of the sampled space.

    Args:
        num_sats_per_run: Number of satellites in each simulation run.
        num_runs: Total number of Monte Carlo iterations.
        base_seed: The starting seed used in the simulation.
    """
    # 1. Aggregate satellites across all runs
    all_dfs = []
    print(f"Aggregating distributions for {num_runs} runs...")

    for run_idx in range(num_runs):
        # Match mc_demo.py seeding logic: config.seed += run_idx
        run_seed = base_seed + run_idx
        df_run = generate_constellation_df(num_sats=num_sats_per_run, seed=run_seed)
        all_dfs.append(df_run)

    df_sats = pd.concat(all_dfs, ignore_index=True)
    print(f"Total satellites in distribution: {len(df_sats)}")

    # 2. Set up the visual style
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    # 3. Create the Corner Plot
    cmap = plt.get_cmap('viridis')
    color_main = cmap(0.3)
    color_scatter = cmap(0.1)

    g = sns.PairGrid(df_sats, corner=True, diag_sharey=False, height=2.2)

    g.map_diag(sns.histplot, kde=True, color=color_main, element="step")
    # Higher alpha (lower transparency) for 16,000 points to show density
    g.map_lower(sns.scatterplot, s=1, alpha=0.1, color=color_scatter)

    # 4. Fix overlapping labels and ticks
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='x', rotation=45)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=5)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=5)

    g.figure.align_labels()
    plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.15, hspace=0.15)

    # Save and show
    plt.savefig("images/orbital_elements_corner_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    """Main function to parse log and generate plots."""
    # === Step 1: Parse the log file ===
    pattern = re.compile(
    r"NIS=([0-9.]+), DOF=([0-9]+), correctness=([0-9.]+), "
    r"consensus_score=([0-9.]+),\s*reputation=([0-9.]+)"
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
