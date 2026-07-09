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
from typing import Optional, Any, Iterator, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import chi2
import seaborn as sns
from src.simulation import generate_random_keplerian_elements
from src.dag import DAG, MockDAG
from src.reputation import ReputationManager, MAX_REPUTATION, \
    ReputationParams

# === Configuration ===
DATA_DIR = "sim_data"
FILENAME = "sim_data/app.log"  # your log file path
THRESHOLD = 0.5                # consensus threshold
CMAP = "viridis"               # colour map for correctness
REP_PARAMS = ReputationParams()
REP_MGR = ReputationManager(REP_PARAMS)


def _get_ledger_items(dag: Any) -> Any:
    """
    Helper to extract ledger items regardless of whether dag is a single DAG/MockDAG
    object, a unified ledger dictionary, or a decentralised dictionary of node DAGs.

    Args:
    - dag (Any): The ledger container structure to unpack.

    Returns:
    - Any: An iterable view of ledger items (hash, transaction_list).
    """
    if hasattr(dag, "ledger"):
        return dag.ledger.items()

    if isinstance(dag, dict):
        # Check if it's a decentralised map of satellite IDs -> DAG objects
        if dag and hasattr(next(iter(dag.values())), "ledger"):
            unified_ledger = {}
            for local_dag in dag.values():
                unified_ledger.update(local_dag.ledger)
            return unified_ledger.items()

        # Otherwise, assume it is already a raw unified ledger dictionary
        return dag.items()

    raise TypeError(f"Object of type {type(dag)} cannot be parsed for ledger data.")


def is_state_evaluated(state: dict) -> bool:
    """
    Determines if a given consensus state dictionary has been evaluated
    by clearing the BFT quorum threshold or POISE scoring gates.

    Args:
    - state (dict): The node-local consensus state dictionary to inspect.

    Returns:
    - bool: True if the transaction has been evaluated, False if it is a pending placeholder.
    """
    return bool(
        state.get("is_confirmed", False) or
        state.get("is_rejected", False) or
        state.get("consensus_score", 0.0) > 0.0
    )


def _get_local_consensus_states(dag: Any) -> dict:
    """
    Helper to extract local consensus states regardless of whether dag is a single
    DAG/MockDAG object, a unified dictionary, or a decentralised dictionary of node DAGs.
    Implements strict conflict resolution to prevent pending states from overwriting evaluated ones.

    Args:
    - dag (Any): The consensus state container structure to unpack.

    Returns:
    - dict: A dictionary mapping transaction hashes to their local consensus state metrics.
    """
    if hasattr(dag, "local_consensus_states"):
        return dag.local_consensus_states

    if not isinstance(dag, dict):
        return {}

    # Guard Clause: Handle raw unified dictionaries immediately to reduce indentation nesting
    first_node = next(iter(dag.values()), None)
    if not (first_node and hasattr(first_node, "local_consensus_states")):
        return dag

    unified_states: dict[str, dict] = {}
    for local_dag in dag.values():
        for tx_hash, current_state in local_dag.local_consensus_states.items():
            if tx_hash not in unified_states:
                unified_states[tx_hash] = current_state
                continue

            # Check if state has been evaluated.
            # This performs conflict resolution for overall end-of-sim
            # statistics, giving us a snapshot of the state of the ledger
            # globally (even though a true global state may not exist
            # among all satellites, as some might not have synced recently)
            if is_state_evaluated(current_state) and not \
                is_state_evaluated(unified_states[tx_hash]):
                unified_states[tx_hash] = current_state

    return unified_states


def extract_nis_transactions(dag: Any) -> Iterator[Tuple[Any, dict]]:
    """
    Generator that iterates through a DAG ledger and yields
    transactions (and their parsed JSON) that contain NIS metadata.

    Args:
    - dag (DAG): The DAG object containing transaction data.

    Returns:
    - Iterator[Tuple[Any, dict]]: An iterator yielding the transaction object and parsed JSON data.
    """
    states = _get_local_consensus_states(dag)
    for tx_hash, tx_list in _get_ledger_items(dag):
        for tx in tx_list:
            state = states.get(tx_hash, {})
            if "nis" not in state or state["nis"] is None:
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
    - df (pd.DataFrame): DataFrame containing 'nis', 'consensus_score', and 'correctness' columns.

    Returns:
    - None: Displays a matplotlib plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

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
    - truth (np.ndarray): The history of true stacked state vectors, with shape (steps, 6*N).
    - n (int): The number of satellites.

    Returns:
    - None: Displays a matplotlib plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    _plot_earth_surface(ax)

    for i in range(n):
        pos_hist = truth[:, i*6:i*6+3]
        ax.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], color='black', alpha=0.3)
        ax.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2],
                   color='black', s=10) # type: ignore[misc]

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='black', markersize=8),
        Line2D([0], [0], color='black', lw=1.5, label='Simulated Orbit')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    _set_axes_equal(ax)
    plt.show()


def _plot_earth_surface(ax: Any) -> None:
    """
    Helper to generate and plot the Earth's surface as a 3D sphere.

    Args:
    - ax (Any): The Axes of the subplot.

    Returns:
    - None: Plots the earth's surface on the subplot.
    """
    r_e = 6378e3
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x_earth = r_e * np.outer(np.cos(u), np.sin(v))
    y_earth = r_e * np.outer(np.sin(u), np.sin(v))
    z_earth = r_e * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3, rstride=4, cstride=4)


def _set_axes_equal(ax: Any) -> None:
    """
    Helper to scale the 3D plot axes equally to prevent orbital distortion.

    Args:
    - ax (Any): The Axes of the subplot.

    Returns:
    - None: Scales the axes of the plot.
    """
    max_range_temp = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    max_range = np.ptp(max_range_temp).max() / 2.0

    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def _draw_scatter_underneath(ax: Any, plot_data: list) -> None:
    """
    Helper to draw raw scatter points underneath box plots with viridis colors.

    Args:
    - ax (Any): The Axes of the subplot.
    - plot_data (list): The lists of values to plot.

    Returns:
    - None: Modifies the plot in-place.
    """
    scatter_colors = [
        plt.get_cmap('viridis')(val) for val in np.linspace(0.8, 0.6, len(plot_data))
    ]
    for i, data in enumerate(plot_data):
        ax.scatter(np.random.normal(loc=i + 1, scale=0.02, size=len(data)),
                   data, alpha=0.3, color=scatter_colors[i], s=20, zorder=2)


def plot_nis_boxplot(dag: Any,
                     compromised_ids: set[int],
                     convergence_index: Optional[int] = None) -> None:
    """
    Generates a grouped box plot for NIS values, separating honest and compromised satellites.

    Args:
    - dag (DAG): The DAG object containing transaction data.
    - compromised_ids (set[int]): A set of IDs for compromised satellites.
    - convergence_index (Optional[int]): Optional index to only plot data after filter convergence.

    Returns:
    - None: Displays a matplotlib plot.
    """
    raw_data = extract_nis_data(
        dag, compromised_ids, convergence_index if convergence_index is not None else 0
    )

    if not raw_data[0] and not raw_data[1]:
        print("No NIS data available to create a box plot.")
        return

    plot_data = [d for d in raw_data if d]
    labels = [l for d, l in zip(raw_data, ["Honest Satellites", "Compromised Satellites"]) if d]

    ax = plt.subplots(figsize=(10, 6))[1]
    parts = ax.boxplot(plot_data, showfliers=False, zorder=3)

    # 2. Plot raw scatter points underneath using helper
    _draw_scatter_underneath(ax, plot_data)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(1.5)

    ax.axhline(
        chi2.ppf((1 - 0.95) / 2, df=2),
        color=plt.get_cmap('viridis')(0.1),
        linestyle='--',
        alpha=0.7,
        label='95% Confidence Interval Bounds'
    )
    ax.axhline(
        chi2.ppf((1 + 0.95) / 2, df=2),
        color=plt.get_cmap('viridis')(0.1),
        linestyle='--',
        alpha=0.7
    )
    ax.axhline(1.386, color='black', linestyle=':', label='Expected Median (1.386)')

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Normalised Innovation Squared [-]", fontsize=20)
    ax.set_yscale("log")
    ax.tick_params(axis='y', labelsize=20)

    ax.legend(fontsize=16, loc="upper center")
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.show()


def extract_nis_data(dag: Any,
                     compromised_ids: Optional[set[int]] = None,
                     start_index: int = 0) -> tuple[list[float], list[float]]:
    """
    Parses the DAG and extracts honest and compromised NIS values.

    Args:
    - dag (DAG): The DAG object containing transaction data.
    - compromised_ids (Optional[set[int]]): A set of IDs for compromised satellites.
    - start_index (int): Index to start plotting or extracting data from.

    Returns:
    - tuple[list[float], list[float]]: A tuple of honest and compromised NIS data lists.
    """
    honest_nis = []
    compromised_nis = []
    f_ids = compromised_ids or set()
    states = _get_local_consensus_states(dag)

    for tx, tx_data in extract_nis_transactions(dag):
        sid = tx_data.get("observer")
        state = states.get(tx.hash, {})
        nis = state.get("nis")

        if sid is None or nis is None:
            continue

        if int(sid) in f_ids:
            compromised_nis.append(float(nis))
        else:
            honest_nis.append(float(nis))

    return honest_nis[start_index:], compromised_nis[start_index:]


def calculate_median_percentiles(dof: int = 2) -> None:
    """
    Calculates the chi-squared CDF percentiles for given median values
    and determines their absolute distance from the ideal 50th percentile.

    Args:
    - dof (int): Degrees of Freedom for the distribution calculation.

    Returns:
    - None: Outputs statistical calculations directly to stdout.
    """
    median_values: list[float] = [1.386, 1.703, 1.836, 1.447, 1.330]
    print(f"--- Chi-Squared CDF Percentiles (DOF={dof}) ---")
    print(f"{'Median Value':<15} | {'Percentile (CDF)':<20} | {'Distance from 0.5':<20}")
    print("-" * 60)

    for val in median_values:
        percentile = chi2.cdf(val, df=dof)
        distance_from_ideal = abs(percentile - 0.5)
        print(f"{val:<15.3f} | {percentile:<20.4f} | {distance_from_ideal:<20.4f}")


def check_consensus_outcomes(dag: Any,
                             consensus_threshold: float = 0.5) -> bool:
    """
    Checks if transaction consensus outcomes (confirmed/rejected) are consistent
    with their consensus scores and reports any discrepancies.

    Args:
    - dag (DAG): The DAG containing transaction data.
    - consensus_threshold (float): The consensus threshold used in the simulation.

    Returns:
    - bool: True if all outcomes are consistent, False otherwise.
    """
    inconsistencies = []
    states = _get_local_consensus_states(dag)
    pending_tx = []
    total_tx = 0

    for tx_hash, _ in _get_ledger_items(dag):
        total_tx += 1
        state = states.get(tx_hash, {})

        if not state or "consensus_score" not in state:
            continue

        score = state.get("consensus_score", 0.0)
        is_confirmed = state.get("is_confirmed", False)
        is_rejected = state.get("is_rejected", False)

        # A transaction with a score of exactly 0.0 that is neither confirmed nor rejected
        # is a valid "Pending/Unconfirmed" transaction (e.g., startup non-quorum or
        # tail-end transactions)
        if score == 0.0 and not is_confirmed and not is_rejected:
            pending_tx.append(tx_hash)
            continue

        should_be_confirmed = score >= consensus_threshold

        if should_be_confirmed:
            if not is_confirmed:
                inconsistencies.append(f"TX {tx_hash[:8]}: score {score:.3f} \
                                       >= {consensus_threshold} but was NOT confirmed.")
            if is_rejected:
                inconsistencies.append(f"TX {tx_hash[:8]}: score {score:.3f} \
                                       >= {consensus_threshold} but was REJECTED.")
        else:
            if is_confirmed and "Genesis" not in tx_hash:
                inconsistencies.append(f"TX {tx_hash[:8]}: score {score:.3f} \
                                       < {consensus_threshold} but was CONFIRMED.")
            elif not is_rejected:
                inconsistencies.append(f"TX {tx_hash[:8]}: score {score:.3f} \
                                       < {consensus_threshold} but was NOT rejected.")

    print(
    f"There were {len(pending_tx)} pending/unconfirmed transactions at the end of the simulation. "
    f"This is {(len(pending_tx) / total_tx) * 100:.3f}% of the total transactions."
    )

    if not inconsistencies:
        print("Consensus outcomes are consistent with scores.")
        return True

    print("Found inconsistencies in consensus outcomes:")
    for issue in inconsistencies:
        print(f"- {issue}")
    return False

def calculate_convergence_index(
    rep_history: dict[str, list[float]],
    compromised_ids: set[int],
    threshold: float = 0.5
) -> int:
    """
    Heuristically identifies the convergence index based on when the mean
    reputation of honest satellites starts to rise significantly above neutral.

    Args:
    - rep_history (dict[str, list[float]]): Dictionary of reputation histories.
    - compromised_ids (set[int]): Set of compromised satellite IDs.
    - threshold (float): Reputation threshold to consider "converged".

    Returns:
    - int: The index of the first step where convergence is detected.
    """
    honest_sids = [sid for sid in rep_history.keys() if int(sid) not in compromised_ids]
    if not honest_sids:
        return 0

    honest_histories = [rep_history[sid] for sid in honest_sids]
    max_len = max(len(h) for h in honest_histories)

    padded = [h + [h[-1]] * (max_len - len(h)) for h in honest_histories]
    honest_mean = np.mean(padded, axis=0)

    indices = np.where(honest_mean > threshold)[0]
    return int(indices[0]) if indices.size > 0 else 0


def calculate_nis_convergence_index(
    dag: Any,
    compromised_ids: set[int],
    confidence: float = 0.95,
    window_size: int = 5
) -> int:
    """
    Identifies the convergence index based on when the NIS values of honest
    satellites enter and stay within the expected chi-squared consistency bounds.

    Args:
    - dag (DAG): The DAG object containing transactions with NIS metadata.
    - compromised_ids (set[int]): Set of compromised satellite IDs to exclude.
    - confidence (float): Confidence level for chi-square bounds (default=0.95).
    - window_size (int): Number of consecutive steps NIS must be within bounds.

    Returns:
    - int: The first step where convergence is detected.
    """
    step_nis_data, step_dof_data = _extract_step_data(dag, compromised_ids)

    if not step_nis_data:
        return 0

    sorted_steps = sorted(step_nis_data.keys())
    is_converged = []
    for step in sorted_steps:
        mean_nis = np.mean(step_nis_data[step])
        mean_dof = np.mean(step_dof_data[step])

        chi2_upper = chi2.ppf((1 + confidence) / 2, df=mean_dof)
        is_converged.append(mean_nis <= chi2_upper)

    for i in range(len(is_converged) - window_size + 1):
        if all(is_converged[i : i + window_size]):
            return int(sorted_steps[i])

    return 0


def _extract_step_data(dag: DAG | MockDAG,
                       compromised_ids: set[int]
                       ) -> tuple[dict[int, list[float]], dict[int, list[int]]]:
    """
    Helper to parse the ledger and group honest NIS and DOF data by step.

    Args:
    - dag (DAG): The DAG object containing transactions with NIS metadata.
    - compromised_ids (set[int]): Set of compromised satellite IDs to exclude.

    Returns:
    - tuple[dict[int, list[float]], dict[int, list[int]]]: A tuple
      of NIS and DOF maps indexed by step.
    """
    step_nis_data: dict[int, list[float]] = {}
    step_dof_data: dict[int, list[int]] = {}
    states = _get_local_consensus_states(dag)

    for tx_hash, tx_list in _get_ledger_items(dag):
        for tx in tx_list:
            state = states.get(tx_hash, {})
            if "nis" not in state or "dof" not in state:
                continue

            try:
                tx_data = json.loads(tx.tx_data)
            except (json.JSONDecodeError, TypeError):
                continue

            sid = tx_data.get("observer")
            step = tx_data.get("step")
            if sid is None or step is None or int(sid) in compromised_ids:
                continue

            nis = state.get("nis")
            dof = state.get("dof")
            if nis is None or dof is None:
                continue

            step_nis_data.setdefault(int(step), []).append(float(nis))
            step_dof_data.setdefault(int(step), []).append(int(dof))

    return step_nis_data, step_dof_data


def plot_aggregated_reputation(
    rep_history: dict[str, list[float]],
    compromised_ids: set[int],
    start_at_full_constellation: bool = False,
    convergence_index: Optional[int] = None
) -> None:
    """
    Plots the aggregated median reputation over time for honest vs. compromised satellites.

    Args:
    - rep_history (dict[str, list[float]]): Maps satellite IDs to their reputation histories.
    - compromised_ids (set[int]): Set of satellite IDs that are considered compromised.
    - start_at_full_constellation (bool): If True, chops pre-convergence step indexes from display.
    - convergence_index (Optional[int]): Optional index pointing out filter convergence point.

    Returns:
    - None: Displays a matplotlib plot.
    """
    if not rep_history:
        print("No reputation data to plot.")
        return

    honest_arr, compromised_arr, max_len = _prepare_reputation_matrices(rep_history,
                                                                        compromised_ids)

    start_index = 0
    if start_at_full_constellation:
        start_index = convergence_index if convergence_index is not None else int(0.6 * max_len)
        if start_index >= max_len:
            print("Not enough data to plot with \
                  'start_at_full_constellation'=True. Plotting all data.")
            start_index = 0

    steps = np.arange(max_len)[start_index:]
    if not steps.size:
        print("No data points to plot after filtering.")
        return

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')

    _plot_reputation_spread(steps, honest_arr[:, start_index:] \
                            if honest_arr.size else np.array([]), cmap(0.5), "Honest")
    _plot_reputation_spread(steps, compromised_arr[:, start_index:] \
                            if compromised_arr.size else np.array([]), cmap(0.05), "Faulty")

    plt.axhline(MAX_REPUTATION/2, color="gray", linestyle=":",
                linewidth=2, label="Neutral Reputation(0.5)")
    if convergence_index is not None and not start_at_full_constellation:
        plt.axvline(x=convergence_index, color="black", linestyle="--",
                    linewidth=1, label="Filter Convergence")

    plt.xlabel("Chronological Transaction Index [-]", fontsize=20)
    plt.ylabel("Reputation [-]", fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()


def _prepare_reputation_matrices(rep_history: dict[str, list[float]],
                                 compromised_ids: set[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepares separate matrices for honest and compromised satellite reputations.

    Args:
    - rep_history (dict[str, list[float]]): History map containing chronological values.
    - compromised_ids (set[int]): Evaluated unique identification numbers for standard deviations.

    Returns:
    - Tuple[np.ndarray, np.ndarray, int]: Padded matrices for honest, compromised,
      and length bounds.
    """
    max_len = max(len(h) for h in rep_history.values())
    honest_matrix, compromised_matrix = [], []

    for sid, history in rep_history.items():
        padded_history = history + [history[-1]] * (max_len - len(history))
        if int(sid) in compromised_ids:
            compromised_matrix.append(padded_history)
        else:
            honest_matrix.append(padded_history)

    return np.array(honest_matrix), np.array(compromised_matrix), max_len


def _plot_reputation_spread(steps: np.ndarray,
                            data_matrix: np.ndarray,
                            colour: Any,
                            label_prefix: str) -> None:
    """
    Plots the mean reputation over time with a shaded area representing one standard deviation.

    Args:
    - steps (np.ndarray): 1D array representing step timeline indices.
    - data_matrix (np.ndarray): 2D stacked tracking sequence elements.
    - colour (Any): RGB color space maps passed down for plots.
    - label_prefix (str): Text tag separating plot legends.

    Returns:
    - None: Injects plot components directly to active matplot context.
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
    - truth (np.ndarray): Ground truth configuration tracking parameters.
    - n (int): Size dimensions representing constellation values.

    Returns:
    - None: Saves spatial track visualizations to target folders.
    """
    _, ax = plt.subplots(figsize=(14, 8))

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    _plot_map_background(ax)
    _plot_satellite_tracks(ax, truth, n)

    handles = [
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='black', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='Simulated Orbit')
    ]
    leg = ax.legend(handles=handles, loc='upper right', framealpha=0.7,
                    facecolor='white', fontsize=16)
    leg.set_zorder(10)

    ax.set_xlabel("Longitude [Degrees]", fontsize=20)
    ax.set_ylabel("Latitude [Degrees]", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle=":", alpha=0.4, color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "orbit_map.png"))


def _plot_map_background(ax: Any) -> None:
    """
    Helper to load and display the static Earth map background.

    Args:
    - ax (Any): Active plot context reference handle.

    Returns:
    - None: Injects map textures into plot environment.
    """
    img_path = "images/1024px-Land_ocean_ice_2048.jpg"

    if os.path.exists(img_path):
        ax.imshow(plt.imread(img_path), extent=(-180.0, 180.0, -90.0, 90.0),
                  aspect='auto', alpha=0.2)
    else:
        ax.set_facecolor('lightgray')


def _plot_satellite_tracks(ax: Any, truth: np.ndarray, n: int) -> None:
    """
    Helper to compute and plot lat/lon ground tracks from Cartesian state history.

    Args:
    - ax (Any): Plot context drawing handle canvas.
    - truth (np.ndarray): Coordinate histories mapping 3D values.
    - n (int): Limit sizes evaluating array structures.

    Returns:
    - None: Adds trace geometry loops directly to plots.
    """
    for i in range(n):
        pos_hist = truth[:, i*6:i*6+3]

        r = np.linalg.norm(pos_hist, axis=1)
        lat = np.degrees(np.arcsin(np.clip(pos_hist[:, 2] / r, -1, 1)))
        lon = np.degrees(np.arctan2(pos_hist[:, 1], pos_hist[:, 0]))

        wrap_idx = np.where(np.abs(np.diff(lon)) > 180)[0]

        lon_plot = np.insert(lon, wrap_idx + 1, np.nan)
        lat_plot = np.insert(lat, wrap_idx + 1, np.nan)

        ax.plot(lon_plot, lat_plot, color='black', alpha=0.1, lw=1.2, zorder=5)
        ax.scatter(lon[-1], lat[-1], color='black', s=20, edgecolor='white',
                   linewidth=0.5, zorder=6)





def generate_constellation_df(num_sats: int, seed: int) -> pd.DataFrame:
    """
    Generates valid Keplerian elements for a LEO constellation using vectorised RNG.

    Args:
    - num_sats (int): Total number of targeted satellite entities.
    - seed (int): Initial tracking entropy constraints.

    Returns:
    - pd.DataFrame: A formatted pandas dataframe tracking orbital elements.
    """
    elements = []
    for n in range(num_sats):
        kep_elements = generate_random_keplerian_elements(seed=seed + n)
        a, e, i, raan, argp, ta = kep_elements.a, kep_elements.e, kep_elements.i, \
                                    kep_elements.raan, kep_elements.argp, kep_elements.ta
        elements.append((a, e, i, raan, argp, ta))

    df = pd.DataFrame({
        'Semi-Major Axis\n[km]': [elem[0] for elem in elements],
        'Eccentricity\n[-]': [elem[1] for elem in elements],
        'Inclination\n[deg]': [elem[2] for elem in elements],
        'RAAN\n[deg]': [elem[3] for elem in elements],
        'Arg of Perigee\n[deg]': [elem[4] for elem in elements],
        'True Anomaly\n[deg]': [elem[5] for elem in elements]
    })

    return df


def generate_corner_plot(num_sats_per_run: int = 400,
                         num_runs: int = 40,
                         base_seed: int = 42) -> None:
    """
    Generates a corner plot of the Keplerian elements aggregated across multiple MC runs.

    Args:
    - num_sats_per_run (int): Density metrics sizing the configuration run paths.
    - num_runs (int): Count checking Monte Carlo operations loop limits.
    - base_seed (int): Base element seed initialization boundary parameters.

    Returns:
    - None: Generates and exports corner charts to disk.
    """
    all_dfs = []
    print(f"Aggregating distributions for {num_runs} runs...")

    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        df_run = generate_constellation_df(num_sats=num_sats_per_run, seed=run_seed)
        all_dfs.append(df_run)

    df_sats = pd.concat(all_dfs, ignore_index=True)
    print(f"Total satellites in distribution: {len(df_sats)}")

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)

    cmap = plt.get_cmap('viridis')
    color_main = cmap(0.3)
    color_scatter = cmap(0.1)

    g = sns.PairGrid(df_sats, corner=True, diag_sharey=False, height=2.2)
    g.map_diag(sns.histplot, kde=True, color=color_main, element="step")
    g.map_lower(sns.scatterplot, s=1, alpha=0.1, color=color_scatter)

    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='x', rotation=45)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=5)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=5)

    g.figure.align_labels()
    plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.15, hspace=0.15)

    plt.savefig("images/orbital_elements_corner_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_integrated_walker_delta(truth: np.ndarray,
                                 n: int,
                                 data_dir: str = "images") -> None:
    """
    Plots both the 3D spatial distribution and 2D ground tracks of a
    satellite constellation on a single figure.

    Args:
    - truth (np.ndarray): The history of true stacked state vectors.
    - n (int): The number of satellites.
    - data_dir (str): Directory to save the output image.

    Returns:
    - None: Displays and saves the integrated matplotlib plot.
    """
    # Create a wider figure to accommodate side-by-side subplots comfortably
    fig = plt.figure(figsize=(18, 8))

    # Left Subplot: 3D Spatial Distribution
    ax1 = fig.add_subplot(121, projection='3d')
    _plot_earth_surface(ax1)

    for i in range(n):
        pos_hist = truth[:, i*6:i*6+3]
        ax1.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], color='black', alpha=0.3)
        ax1.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2],
                    color='black', s=10) # type: ignore[misc]

    legend_elements_3d = [
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='black', markersize=8),
        Line2D([0], [0], color='black', lw=1.5, label='Simulated Orbit')
    ]
    ax1.legend(handles=legend_elements_3d, loc='upper right')

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("a)", y=-0.15, fontsize=16)

    _set_axes_equal(ax1)

    # Right subplot: 2D Ground Track
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)

    _plot_map_background(ax2)
    _plot_satellite_tracks(ax2, truth, n)

    legend_elements_2d = [
        Line2D([0], [0], marker='o', color='w', label='Satellite',
               markerfacecolor='black', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='Simulated Orbit')
    ]
    leg = ax2.legend(handles=legend_elements_2d, loc='upper right', framealpha=0.7,
                     facecolor='white', fontsize=14)
    leg.set_zorder(10)

    ax2.set_xlabel("Longitude [Degrees]", fontsize=16)
    ax2.set_ylabel("Latitude [Degrees]", fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True, linestyle=":", alpha=0.4, color='white')
    ax2.set_title("b)", y=-0.18, fontsize=16)

    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, "integrated_walker_delta.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    """
    Main function to parse log and generate plots.

    Args:
    - None.

    Returns:
    - None.
    """
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
        print(f"Error: Log file not found at '{FILENAME}'. Make sure the path is correct.")
        return

    if not data:
        print("No data found in log file matching the pattern.")
        return

    df = pd.DataFrame(data, columns=["nis", "dof", "correctness", "consensus_score", "reputation"])
    plot_nis_vs_consensus(df)


if __name__ == "__main__":
    main()
