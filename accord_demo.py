# pylint: disable=protected-access, too-many-locals, too-many-statements, broad-exception-caught
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

import asyncio
import json
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from scipy.stats import chi2
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG
from src.logger import get_logger
from src.reputation import MAX_REPUTATION, ReputationManager
from src.satellite_node import SatelliteNode
from src.utils import load_json_data

logger = get_logger()

# Consensus demo (JSON-driven)
async def run_consensus_demo() -> tuple[Optional[DAG], Optional[dict]]:
    """
    Run a demo of the consensus mechanism with multiple satellite nodes
    submitting transactions to the DAG.

    Returns:
    - The final DAG object after all transactions have been processed.
    """
    poise = ConsensusMechanism()
    queue: asyncio.Queue = asyncio.Queue()
    dag = DAG(queue=queue, consensus_mech=poise)

    # Load all JSON observations
    observations = load_json_data("sim_output.json")
    if not observations:
        logger.info("No data in JSON file.")
        return None, None

    asyncio.create_task(dag.listen())

    # Create one SatelliteNode per unique observer_id in the JSON
    unique_ids = sorted({obs["observer_id"] for obs in observations})
    satellites: dict[str, SatelliteNode] = {
        sid: SatelliteNode(node_id=sid, queue=queue) for sid in unique_ids
    }
    rep_history: dict[str, list[float]] = {sid: [] for sid in unique_ids}

    # One satellite per observation (or choose grouping logic if needed)
    # Assign each observation to the correct satellite and submit
    for obs in observations:
        sid = obs["observer_id"]
        sat = satellites[sid]
        sat.load_sensor_data(obs)
        logger.info("Satellite %s: submitting transaction.", sid)
        await sat.submit_transaction(recipient_address=123)
        rep_history[sid].append(sat.reputation)

    return dag, rep_history


# Plots
def plot_transaction_dag(dag: DAG) -> None:
    """
    Plot the transaction DAG.

    Args:
    - dag: The DAG object containing the transaction data.

    Returns:
    - None. Displays a plot of the transaction DAG.
    """
    graph: nx.DiGraph = nx.DiGraph()
    tx_timestamps = {}
    tx_status = {}

    for key, tx_list in dag.ledger.items():
        for tx in tx_list:
            graph.add_node(key)
            tx_timestamps[key] = tx.metadata.timestamp
            tx_status[key] = {
                "is_confirmed": tx.metadata.is_confirmed,
                "is_rejected": tx.metadata.is_rejected
            }
            for parent_hash in tx.metadata.parent_hashes:
                graph.add_edge(key, parent_hash)

    sorted_keys = sorted(tx_timestamps, key=lambda k: tx_timestamps[k])
    pos = {k: (i, (hash(k) % 100) / 100.0 - 0.5) for i, k in enumerate(sorted_keys)}

    plt.figure(figsize=(16, 6))

    plt.axvline(x=2,
                color="#000000",
                linestyle="--",
                linewidth=1.5,
                label="Real Data Added",
                zorder=1)

    plt.axvline(x=5,
                color="#000000",
                linestyle="--",
                linewidth=1.5,
                label="BFT Quorum Reached",
                zorder=1)

    for node in graph.nodes():
        outline_color = "black"
        if tx_status[node]["is_confirmed"]:
            outline_color = "green"
        elif tx_status[node]["is_rejected"]:
            outline_color = "red"
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=[node],
            node_color="lightblue",
            node_size=300,
            edgecolors=outline_color,
            linewidths=1
        )

    edge_colors = []
    for _, parent_node in graph.edges():
        if tx_status[parent_node]["is_confirmed"]:
            edge_colors.append("green")
        elif tx_status[parent_node]["is_rejected"]:
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_edges(graph, pos,
                           edge_color=edge_colors, arrowsize=15) # type: ignore [arg-type]

    # Add legend
    node_patch = mpatches.Patch(edgecolor="black", facecolor="lightblue",
                                label="Initial Transaction", linewidth=1)
    confirmed_patch = mpatches.Patch(edgecolor="green", facecolor="lightblue",
                                     label="Confirmed Transaction", linewidth=1)
    rejected_patch = mpatches.Patch(edgecolor="red", facecolor="lightblue",
                                    label="Rejected Transaction", linewidth=1)
    # Use Line2D for edge legend entries to look like lines
    edge_confirmed = Line2D([0], [0], color="green", linewidth=1, label="Strong Edge")
    edge_rejected = Line2D([0], [0], color="red", linewidth=1, label="Weak Edge")
    edge_default = Line2D([0], [0], color="grey", linewidth=1, label="Default Edge")

    # Place legend outside the plot area (top right)
    plt.legend(
        handles=[node_patch, confirmed_patch, rejected_patch, edge_confirmed,
                 edge_rejected, edge_default],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        facecolor='none',
        edgecolor='black',
        labelcolor='black',
        fontsize=18
    )

    plt.title("Transaction DAG", fontsize=12, color='black')
    plt.xlabel("Time Step", color='black', fontsize=12)
    n = len(sorted_keys)
    step = 10
    plt.xticks(
        range(0, n, step),  # positions at every 10 timesteps
        [str(i) for i in range(0, n, step)],  # labels
        color="black",
        fontsize=12
    )

    ax = plt.gca()

    ymin, _ = ax.get_ylim()

    # Add rotated text along the vertical lines
    ax.text(2 - 0.1, ymin + 0.05, "Real Data Added", color="#000000",
        rotation=90, rotation_mode="anchor",
        va="bottom", ha="left", fontsize=10,
        zorder=10,
        bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.7, "pad": 2})

    ax.text(5 - 0.1, ymin + 0.05, "BFT Quorum Reached", color="#000000",
            rotation=90, rotation_mode="anchor",
            va="bottom", ha="left", fontsize=10,
            zorder=10,
            bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.7, "pad": 2})

    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    # Set axis and tick colors to black
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=12)
    ax.xaxis.label.set_color('black')

    # Set transparent background
    ax.set_facecolor('none')
    plt.gcf().patch.set_alpha(0.0)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # type: ignore [arg-type]
    plt.show()


REP_MGR = ReputationManager()

def plot_reputation(rep_history: dict) -> None:
    """
    Plot the reputation history of satellite nodes.
    Args:
    - rep_history: A dictionary where keys are node IDs and
    values are lists of reputation scores over time.

    Returns:
    - None. Displays a plot of reputation over time for each node.
    """
    neutral_level: float = MAX_REPUTATION / 2
    plt.figure(figsize=(8, 5))

    max_len = max((len(h) for h in rep_history.values()), default=0)
    steps = list(range(max_len))

    # Plot reputation histories
    for node_id, history in rep_history.items():
        plt.plot(range(len(history)), history, marker="o", label=f"{node_id} Reputation")

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

        # Byzantine region as offset below
        lag = 5       # number of timesteps delay
        margin = 0.0  # vertical buffer

        # Build lagged version of target curve
        byz_curve = np.zeros_like(target_curve)
        byz_curve[lag:] = target_curve[:-lag] - margin # type: ignore [operator]
        byz_curve[:lag] = target_curve[0] - margin

        # Ensure it's always below the target
        byz_curve = np.clip(byz_curve, 0, target_curve)

        # Shade the band BETWEEN byz_curve and target_curve
        plt.fill_between(steps, 50, byz_curve,
                        color="grey", alpha=0.1, label="Byzantine region")


    # Neutral line
    plt.axhline(neutral_level, color="gray", linestyle=":", label=f"Neutral ({neutral_level})")

    plt.ylim(0, MAX_REPUTATION)
    plt.xlabel("Time step [minutes]")
    plt.ylabel("Reputation Score [-]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_consensus_cdf_dof(dag: DAG) -> None:
    """
    Plot consensus score, CDF, and DOF for each satellite.
    Consensus score on left y-axis, CDF on right y-axis.
    DOF shown as markers.
    """
    # Collect by satellite
    data_by_sat: dict[str, list] = {}
    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "consensus_score"):
                continue
            try:
                tx_data = json.loads(tx.tx_data)
            except Exception:
                continue
            sid = tx_data.get("observer_id")
            if not sid:
                continue

            data_by_sat.setdefault(sid, []).append({
                "consensus": tx.metadata.consensus_score,
                "cdf": getattr(tx.metadata, "cdf", None),
                "dof": getattr(tx.metadata, "dof", None),
                "confirmed": getattr(tx.metadata, "is_confirmed", False),
                "rejected": getattr(tx.metadata, "is_rejected", False),
            })

    # Filter out satellites with no data
    data_by_sat = {sid: vals for sid, vals in data_by_sat.items() if vals}
    if not data_by_sat:
        logger.info("No consensus/CDF/DOF data available to plot.")
        return

    n_sats = len(data_by_sat)
    _, axes = plt.subplots(n_sats, 1, figsize=(10, 5 * n_sats), sharex=True)
    if n_sats == 1:
        axes = [axes]

    for ax, (sid, records) in zip(axes, data_by_sat.items()):
        steps = range(len(records))
        consensus = [r["consensus"] for r in records]
        cdf = [r["cdf"] for r in records]
        dof = [r["dof"] for r in records]

        # Scatter consensus
        colors = ["green" if r["confirmed"] else "red" if r["rejected"] else "gray"
                  for r in records]
        ax.scatter(steps, consensus, c=colors, s=60, label="Consensus Score")
        ax.plot(steps, consensus, linestyle="--", alpha=0.5, color="black")
        ax.axhline(0.4, color="blue", linestyle=":", label="Threshold (0.4)")
        ax.set_ylabel("Consensus Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"Satellite {sid}")
        ax.grid(True, linestyle=":")

        # Add CDF on secondary axis
        ax2 = ax.twinx()
        ax2.plot(steps, cdf, "o-", color="orange", alpha=0.7, label="CDF")
        ax2.set_ylabel("CDF Value", color="orange")
        ax2.tick_params(axis="y", colors="orange")

        # Optional DOF display as text/markers
        for i, d in enumerate(dof):
            ax.text(i, consensus[i] + 0.05, f"DOF={d}", ha="center", fontsize=8, color="gray")

        # Merge legends
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles + handles2, labels + labels2, loc="upper right")

    axes[-1].set_xlabel("Transaction Index")
    plt.tight_layout()
    plt.show()


def plot_nis_consistency(dag: DAG, confidence: float = 0.95) -> None:
    """
    Plot Normalized Innovation Squared (NIS) values for each satellite,
    comparing them to expected chi-squared consistency bounds.

    Args:
    - dag: The final DAG object containing transactions (with NIS + DOF metadata).
    - confidence: Confidence level for chi-square bounds (default=0.95).

    Returns:
    - None. Displays NIS plots with statistical consistency regions.
    """

    # Gather NIS data per satellite
    nis_by_sat: dict[str, dict[str, list]] = {}
    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "nis") or not hasattr(tx.metadata, "dof"):
                continue
            try:
                tx_data = json.loads(tx.tx_data)
            except Exception:
                continue

            sid = tx_data.get("observer_id", "unknown")
            dof = getattr(tx.metadata, "dof", None)
            nis = getattr(tx.metadata, "nis", None)
            if nis is None or dof is None:
                continue

            nis_by_sat.setdefault(sid, {"nis": [], "dof": []})
            nis_by_sat[sid]["nis"].append(nis)
            nis_by_sat[sid]["dof"].append(dof)

    if not nis_by_sat:
        logger.info("No NIS/DOF data found in DAG.")
        return

    n_sats = len(nis_by_sat)
    _, axes = plt.subplots(n_sats, 1, figsize=(10, 5 * n_sats), sharex=True)
    if n_sats == 1:
        axes = [axes]

    for ax, (sid, data) in zip(axes, nis_by_sat.items()):
        nis_vals = np.array(data["nis"])
        dof_vals = np.array(data["dof"])
        mean_dof = np.mean(dof_vals)

        # Compute chi-square confidence bounds
        chi2_lower = chi2.ppf((1 - confidence) / 2, df=mean_dof)
        chi2_upper = chi2.ppf((1 + confidence) / 2, df=mean_dof)
        expected_mean = mean_dof

        steps = np.arange(len(nis_vals))

        # Plot NIS over time
        ax.plot(steps, nis_vals, "o-", color="black", label="NIS")
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

        # Rolling mean NIS (smooth trend)
        if len(nis_vals) > 5:
            window = 5
            rolling_mean = np.convolve(nis_vals, np.ones(window)
                                       / window, mode="valid")
            ax.plot(range(window-1, len(nis_vals)), rolling_mean,
                    color="red", linewidth=2, label="Rolling mean (5)")

        ax.set_ylabel("NIS Value")
        ax.set_title(f"NIS Consistency — Satellite {sid}")
        ax.grid(True, linestyle=":")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Transaction Index")
    plt.tight_layout()
    plt.show()


# Run demo
if __name__ == "__main__":
    final_dag, rep_hist = asyncio.run(run_consensus_demo())
    if final_dag:
        #plot_transaction_dag(final_dag)
        plot_consensus_cdf_dof(final_dag)
        plot_nis_consistency(final_dag)
    if rep_hist:
        plot_reputation(rep_hist)
