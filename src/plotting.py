# pylint: disable=too-many-locals, too-many-statements, protected-access, broad-exception-caught, too-many-branches
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
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from scipy.stats import chi2
from src.reputation import MAX_REPUTATION, ReputationManager

# === Configuration ===
FILENAME = "app.log"  # your log file path
THRESHOLD = 0.5                # consensus threshold
CMAP = "viridis"               # color map for correctness

def plot_consensus_vs_reputation(df):
    """Plots consensus score vs reputation."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["reputation"],
        df["consensus_score"],
        c=df["correctness"],
        cmap=CMAP,
        s=80,
        alpha=0.8,
    )
    plt.axhline(THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold = {THRESHOLD}")
    plt.colorbar(scatter, label="Correctness")

    plt.xlabel("Reputation")
    plt.ylabel("Consensus Score")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_nis_vs_correctness(df):
    """Plots NIS vs correctness."""
    plt.figure(figsize=(10, 7))
    plt.scatter(
        df["nis"],
        df["correctness"],
        cmap=CMAP,
        s=80,
        alpha=0.8,
    )

    plt.xlabel("NIS")
    plt.ylabel("Correctness")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_nis_vs_consensus(df):
    """Plots NIS vs consensus score with a zoomed-in subplot for NIS values 0-10."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Main plot
    scatter = ax.scatter(
        df["nis"],
        df["consensus_score"],
        c=df["correctness"],
        cmap=CMAP,
        s=80,
        alpha=0.8,
    )
    ax.axhline(THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold = {THRESHOLD}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Correctness", fontsize=16)

    ax.set_xlabel("NIS", fontsize=16)
    ax.set_ylabel("Consensus Score", fontsize=16)
    ax.set_xscale('symlog')
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle=":")

    fig.tight_layout()
    plt.show()

def main():
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
    plot_consensus_vs_reputation(df)
    plot_nis_vs_correctness(df)
    plot_nis_vs_consensus(df)


def plot_transaction_dag(dag, max_nodes: int | None = 100, start_index: int = 0) -> None:
    """
    Plot a customizable slice of the transaction DAG.

    Args:
    - dag: The DAG object containing the transaction data.
    - max_nodes: The maximum number of nodes (transactions) to display.
                 If None, all transactions from start_index are shown.
                 Defaults to 100.
    - start_index: The starting index of the sorted transactions to plot.
                   Negative indices count from the end. Defaults to 0.

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

    # --- New slicing logic ---
    if start_index < 0:
        start_index = max(0, len(sorted_keys) + start_index)

    end_index = len(sorted_keys)
    if max_nodes is not None:
        end_index = min(start_index + max_nodes, len(sorted_keys))

    plot_keys = sorted_keys[start_index:end_index]
    subgraph = graph.subgraph(plot_keys)
    # ---

    pos = {k: (i, (hash(k) % 100) / 100.0 - 0.5) for i, k in enumerate(plot_keys)}

    plt.figure(figsize=(16, 6))

    # --- Adjust axvline positions relative to the slice ---
    real_data_line = 2 - start_index
    bft_line = 5 - start_index

    if 0 <= real_data_line < len(plot_keys):
        plt.axvline(x=real_data_line, color="#000000", linestyle="--", linewidth=1.5,
                    label="Real Data Added", zorder=1)

    if 0 <= bft_line < len(plot_keys):
        plt.axvline(x=bft_line, color="#000000", linestyle="--", linewidth=1.5,
                    label="BFT Quorum Reached", zorder=1)

    for node in subgraph.nodes():
        outline_color = "black"
        if tx_status[node]["is_confirmed"]:
            outline_color = "green"
        elif tx_status[node]["is_rejected"]:
            outline_color = "red"
        nx.draw_networkx_nodes(
            subgraph, pos,
            nodelist=[node],
            node_color="lightblue",
            node_size=300,
            edgecolors=outline_color,
            linewidths=1
        )

    edge_colors = []
    # Only draw edges that are fully within the subgraph
    for _, v in subgraph.edges():
        if v in tx_status:
            if tx_status[v]["is_confirmed"]:
                edge_colors.append("green")
            elif tx_status[v]["is_rejected"]:
                edge_colors.append("red")
            else:
                edge_colors.append("gray")
        else:
            edge_colors.append("gray") # Should not happen with subgraph

    nx.draw_networkx_edges(subgraph, pos,
                           edge_color=edge_colors, arrowsize=15) # type: ignore [arg-type]

    # Add legend
    node_patch = mpatches.Patch(edgecolor="black", facecolor="lightblue",
                                label="Initial Transaction", linewidth=1)
    confirmed_patch = mpatches.Patch(edgecolor="green", facecolor="lightblue",
                                     label="Confirmed Transaction", linewidth=1)
    rejected_patch = mpatches.Patch(edgecolor="red", facecolor="lightblue",
                                    label="Rejected Transaction", linewidth=1)
    edge_confirmed = Line2D([0], [0], color="green", linewidth=1, label="Strong Edge")
    edge_rejected = Line2D([0], [0], color="red", linewidth=1, label="Weak Edge")
    edge_default = Line2D([0], [0], color="grey", linewidth=1, label="Default Edge")

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


    plt.xlabel("Transaction Index", color='black', fontsize=12)
    n = len(plot_keys)
    step = max(1, n // 10)

    tick_positions = range(0, n, step)
    tick_labels = [str(i + start_index) for i in tick_positions]

    plt.xticks(
        tick_positions,
        tick_labels,
        color="black",
        fontsize=12
    )

    ax = plt.gca()
    ymin, _ = ax.get_ylim()

    if 0 <= real_data_line < len(plot_keys):
        ax.text(real_data_line - 0.1, ymin + 0.05, "Real Data Added", color="#000000",
                rotation=90, rotation_mode="anchor",
                va="bottom", ha="left", fontsize=10, zorder=10,
                bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.7, "pad": 2})

    if 0 <= bft_line < len(plot_keys):
        ax.text(bft_line - 0.1, ymin + 0.05, "BFT Quorum Reached", color="#000000",
                rotation=90, rotation_mode="anchor",
                va="bottom", ha="left", fontsize=10, zorder=10,
                bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.7, "pad": 2})


    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=12)
    ax.xaxis.label.set_color('black')
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
    # plt.yscale("log") TODO: consider log scale if wide range
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.,
               fontsize=14)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_consensus_correctness_dof(dag) -> None:
    """
    Plot consensus score, Correctness Score, and DOF for each satellite.
    Consensus score on left y-axis, Correctness Score on right y-axis.
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
            sid = tx_data.get("observer") # Corrected: Use "observer" key
            if not sid:
                continue

            data_by_sat.setdefault(str(sid), []).append({ # Ensure sid is string for dict key
                "consensus": tx.metadata.consensus_score,
                "correctness": getattr(tx.metadata, "correctness_score", None),
                "dof": getattr(tx.metadata, "dof", None),
                "confirmed": getattr(tx.metadata, "is_confirmed", False),
                "rejected": getattr(tx.metadata, "is_rejected", False),
            })

    # Filter out satellites with no data
    data_by_sat = {sid: vals for sid, vals in data_by_sat.items() if vals}
    if not data_by_sat:
        print("No consensus/Correctness/DOF data available to plot.")
        return

    n_sats = len(data_by_sat)
    _, axes = plt.subplots(n_sats, 1, figsize=(10, 5 * n_sats), sharex=True)
    if n_sats == 1:
        axes = [axes]

    for ax, (sid, records) in zip(axes, data_by_sat.items()):
        steps = range(len(records))
        consensus = [r["consensus"] for r in records]
        correctness = [r["correctness"] for r in records] # Changed from cdf
        dof = [r["dof"] for r in records]

        # Scatter consensus
        colors = ["green" if r["confirmed"] else "red" if r["rejected"] else "gray"
                  for r in records]
        ax.scatter(steps, consensus, c=colors, s=60, label="Consensus Score")
        ax.plot(steps, consensus, linestyle="--", alpha=0.5, color="black")
        ax.axhline(0.5, color="blue", linestyle=":", label="Threshold (0.5)")
        ax.set_ylabel("Consensus Score")
        ax.set_ylim(0, 1)

        ax.grid(True, linestyle=":")

        # Add Correctness Score on secondary axis
        ax2 = ax.twinx()
        ax2.plot(steps, correctness, "o-", color="orange", alpha=0.7,
                 label="Correctness Score") # Changed from cdf
        ax2.set_ylabel("Correctness Score", color="orange") # Changed from CDF Value
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


def plot_nis_consistency_by_satellite(dag, confidence: float = 0.95) -> None:
    """
    Plots Normalised Innovation Squared (NIS) values for each satellite individually,
    comparing them to expected chi-squared consistency bounds. Each satellite
    is displayed in a separate plot window.

    Args:
    - dag: The final DAG object containing transactions (with NIS + DOF metadata).
    - confidence: Confidence level for chi-square bounds (default=0.95).

    Returns:
    - None. Displays NIS plots with statistical consistency regions for each satellite.
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
        ax.plot(steps, nis_vals, "o", color="black", label=f"NIS (Sat_{sid})")

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

        ax.set_ylabel("NIS Value (symlog scale)", fontsize=16)
        ax.set_yscale("symlog")
        ax.grid(True, linestyle=":")
        ax.legend(loc="upper right", fontsize=16)
        ax.set_xlabel("Transaction Index", fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        plt.tight_layout()

    plt.show()


def plot_nis_boxplot(dag) -> None:
    """
    Generates box plots visualizing the distribution of Normalised Innovation Squared (NIS)
    values for each simulated satellite.

    This function collects NIS data from the DAG for honest and intermittently faulty
    satellites and loads pre-recorded malicious satellite NIS data from
    'sat1_nis_data.json'. The box plots illustrate the spread of NIS values,
    with different satellite types (honest, faulty, malicious) clearly labeled.

    The plot includes horizontal lines indicating:
    - The 95% chi-squared confidence interval (for DOF=2), providing statistical bounds
      for expected NIS values.
    - The expected median of the chi-squared distribution (for DOF=2).

    The y-axis uses a symmetrical log scale to better visualize a wide range of NIS values.

    Args:
    - dag: The DAG object containing transaction data, including NIS metadata
           for honest and intermittently faulty satellites.

    Returns:
    - None. Displays a matplotlib box plot figure.
    """
    # Collect data by satellite
    nis_data_by_sat: dict[str, list[float]] = {}
    for _, tx_list in dag.ledger.items():
        for tx in tx_list:
            if not hasattr(tx.metadata, "nis"):
                continue

            try:
                tx_data = json.loads(tx.tx_data)
            except Exception:
                continue

            sid = tx_data.get("observer")
            if sid is None:
                continue

            nis = getattr(tx.metadata, "nis", None)
            if nis is None:
                continue

            nis_data_by_sat.setdefault(str(sid), []).append(nis)

    # Filter out satellites with no data
    nis_data_by_sat = {sid: vals for sid, vals in nis_data_by_sat.items() if vals}

    # Load data for the malicious satellite from file and slice it
    malicious_nis_data = None
    try:
        with open('sat1_nis_data.json', 'r', encoding='utf-8') as f:
            malicious_nis_data = json.load(f)
            if malicious_nis_data:
                malicious_nis_data = malicious_nis_data[100:]
    except FileNotFoundError:
        print("Warning: sat1_nis_data.json not found. Cannot plot malicious data.")
    except json.JSONDecodeError:
        print("Warning: Could not decode sat1_nis_data.json. Cannot plot malicious data.")

    # Sort satellites by ID, then move sat 1 to the end of the DAG-based data.
    sorted_sids = sorted(nis_data_by_sat.keys(), key=int)
    if '1' in sorted_sids:
        sorted_sids.remove('1')
        sorted_sids.append('1')

    nis_values_for_plot = [nis_data_by_sat[sid] for sid in sorted_sids]
    labels = [f"Honest Satellite (ID: Sat_{sid})" if sid != "1" else \
              "Satellite with Intermittent Fault (ID: Sat_1)" for sid in sorted_sids]

    # Add malicious data if loaded and it has points left
    if malicious_nis_data:
        nis_values_for_plot.append(malicious_nis_data)
        labels.append("Malicious Satellite (ID: Sat_1)")

    if not nis_values_for_plot:
        print("No NIS data available to create a box plot.")
        return

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(nis_values_for_plot,
                     labels=labels) # type: ignore [call-arg]
    for median in bp['medians']:
        median.set_color('blue')

    # Add chi-squared bounds
    dof = 2
    confidence = 0.95
    expected_median = 1.298
    chi2_lower = chi2.ppf((1 - confidence) / 2, df=dof)
    chi2_upper = chi2.ppf((1 + confidence) / 2, df=dof)
    plt.axhline(chi2_lower, color='r', linestyle='--',
                label='95% Chi-squared Confidence Interval (DOF=2)')
    plt.axhline(chi2_upper, color='r', linestyle='--')
    plt.axhline(expected_median, color='black', linestyle=':', label='Expected Median (DOF=2)')

    plt.ylabel("NIS Value", fontsize=14)
    plt.yscale("symlog")
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()

def check_consensus_outcomes(dag, consensus_threshold: float = 0.5) -> bool:
    """
    Checks if transaction consensus outcomes (confirmed/rejected) are consistent
    with their consensus scores and reports any discrepancies.

    This function iterates through all transactions in the DAG that have a consensus
    score and verifies that:
    1. Transactions with a score >= threshold are marked as 'confirmed'.
    2. Transactions with a score < threshold are marked as 'rejected'.

    Args:
    - dag: The DAG containing transaction data.
    - consensus_threshold: The consensus threshold used in the simulation.

    Returns:
    - True if all outcomes are consistent, False otherwise.
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


if __name__ == "__main__":
    main()
