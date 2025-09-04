# pylint: disable=protected-access, too-many-locals
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
from datetime import timedelta
import json
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skyfield.api import EarthSatellite
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG
from src.logger import get_logger
from src.reputation import MAX_REPUTATION, ReputationManager
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata

logger = get_logger(__name__)

def create_noisy_transaction(base_sat: EarthSatellite) -> Transaction:
    """
    Create a single noisy transaction for the given base satellite.
    Optionally override the default random noise.
    """
    noise = {
        "mean_motion": np.random.normal(0, 0.01),       # rev/day
        "eccentricity": np.random.normal(0, 0.00002),   # unitless
        "inclination": np.random.normal(0, 0.05),       # degrees
        "epoch_jitter": np.random.normal(0, 2)          # seconds
    }

    observed_eccentricity = base_sat.model.ecco + noise["eccentricity"]
    observed_epoch = base_sat.epoch.utc_datetime() + timedelta(seconds=noise["epoch_jitter"])
    observed_inclination = (base_sat.model.inclo * 180 / np.pi) + noise["inclination"]
    observed_mean_motion = ((base_sat.model.no_kozai / (2 * np.pi)) * 1440) + noise["mean_motion"]

    return Transaction(
        sender_address=789,
        recipient_address=123,
        sender_private_key="dummy_key",
        metadata=TransactionMetadata(),
        tx_data=json.dumps({
            "OBJECT_NAME": base_sat.name,
            "OBJECT_ID": 25544,
            "EPOCH": observed_epoch.isoformat() + "Z",
            "MEAN_MOTION": observed_mean_motion,
            "ECCENTRICITY": observed_eccentricity,
            "INCLINATION": observed_inclination
        })
    )

async def run_consensus_demo(initial_n_tx: int = 3,
                             n_test_tx: int = 1) -> tuple[Optional[DAG], Optional[list[float]]]:
    """
    This function runs a demonstration of the ACCORD Distributed Ledger.
    The following steps are demonstrated:
    - Loading in TLE data from a JSON file
    - Adding transactions to a ledger. The transactions have simulated 
    noise added to their sensor data to reflect real-wold conditions.
    - Once a 4th transaction is added to the ledger, consensus is 
    executed and nodes and transactions are given reputation and consensus
    scores that reflect the trust in the information and the node
    that provided it.

    Arguments:
    - n_tx: An integer representing the number of transactions to be added
    to the ledger, before a test transaction is then added. For consensus 
    to be possible, n_tx must be at least 3, as consensus cannot be reaced 
    with fewer than 4 observations being added to the ledger.

    Returns:
    - The updated Directed Acyclic Graph (DAG) ledger structure, which can 
    then be parsed into other functions to generate diagrams or analyse data.
    """

    # Initialise PoISE mechanism
    poise = ConsensusMechanism()

    # ----------------------------------------------------------------------------------------
    # Setup the data structures needed for the demo
    queue: asyncio.Queue = asyncio.Queue()
    test_satellite = SatelliteNode(node_id="SAT-001", queue=queue)
    test_dag = DAG(queue=queue, consensus_mech=poise)
    sat_rep_list: list = []

    if test_satellite.tle_data[0] is None:
        logger.info("No satellite data found.")
        return None, None

    base_sat = test_satellite.tle_data[0]

    # Add other transactions into the ledger with some simulated random noise
    # This is bypassing the consensus process to give us foundational data
    # which can then be built upon, purely for testing purposes.
    # TODO - test with bigger variants and non-random noise
    # TODO - can this be simplified??
    for _ in range(initial_n_tx):
        test_dag.add_tx(create_noisy_transaction(base_sat))
    # ----------------------------------------------------------------------------------------

    # Start DAG listener
    asyncio.create_task(test_dag.listen())

    # TODO - get more data to simulate better, rather than adding the same data
    logger.info("Submitting Valid Transactions")
    for _ in range(n_test_tx):
        # Run consensus on a single satellite observation - the test transaction
        await test_satellite.submit_transaction(
             satellite=base_sat,
             recipient_address=123
             )
        sat_rep_list.append(test_satellite.reputation)

    logger.info("Submitting Malicious Transaction")
    test_satellite.is_malicious = True
    await test_satellite.submit_transaction(
             satellite=test_satellite.tle_data[0],
             recipient_address=123
             )
    sat_rep_list.append(test_satellite.reputation)

    logger.info("Submitting Valid Transactions")
    test_satellite.is_malicious = False
    for _ in range(n_test_tx):
        # Run consensus on a single satellite observation - the test transaction
        await test_satellite.submit_transaction(
             satellite=base_sat,
             recipient_address=123
             )
        sat_rep_list.append(test_satellite.reputation)

    # Output results
    # logger.info("Test DAG:")
    # logger.info(test_dag.ledger)
    # logger.info("Satellite Reputations")
    # logger.info(sat_rep_list)
    return test_dag, sat_rep_list


# -----------------------------------------------------------------------------------
def plot_transaction_dag(dag: DAG, draw_labels: bool = False) -> None:
    """
    Plot a graph representing the Directed Acyclic Graph

    Args:
    dag: The Directed Acyclic Graph Distributed Ledger datastructure
    to be plotted
    draw_labels: Whether or not the nodes in the graph will have labels.
    If True, the hashes of each transaction are added as labels to each node.

    Returns:
    None. Shows a plot using MatPlotLib/
    """
    import matplotlib.patches as mpatches

    graph: nx.DiGraph = nx.DiGraph()
    tx_timestamps = {}
    tx_status = {}  # store is_confirmed/is_rejected per tx

    # Add nodes & edges
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

    # Sort nodes by timestamp for X-axis positioning
    sorted_keys = sorted(tx_timestamps, key=lambda k: tx_timestamps[k])
    pos = {}
    for i, key in enumerate(sorted_keys):
        x = i
        y = (hash(key) % 100) / 100.0 - 0.5
        pos[key] = (x, y)

    plt.figure(figsize=(16, 6))

    # Draw nodes with outline color
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
            node_size=1000,
            edgecolors=outline_color,  # outline color
            linewidths=2
        )

    # Draw edges with color matching the target node's status
    edge_colors = []
    for _, parent_node in graph.edges():
        if tx_status[parent_node]["is_confirmed"]:
            edge_colors.append("green")
        elif tx_status[parent_node]["is_rejected"]:
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_edges(graph, pos,
                           edge_color=edge_colors, # type: ignore[arg-type]
                           arrowsize=15)

    # Draw labels (optional)
    if draw_labels:
        nx.draw_networkx_labels(
            graph, pos, font_size=8, font_weight="bold"
        )

    # Add legend
    node_patch = mpatches.Patch(color="lightblue", label="Transaction Node")
    confirmed_patch = mpatches.Patch(edgecolor="green", facecolor="lightblue", label="Confirmed Node", linewidth=2)
    rejected_patch = mpatches.Patch(edgecolor="red", facecolor="lightblue", label="Rejected Node", linewidth=2)
    edge_confirmed = mpatches.Patch(color="green", label="Strong Edge")
    edge_rejected = mpatches.Patch(color="red", label="Weak Edge")
    edge_default = mpatches.Patch(color="gray", label="Default Edge")

    # Place legend outside the plot area (top right)
    plt.legend(
        handles=[node_patch, confirmed_patch, rejected_patch, edge_confirmed, edge_rejected, edge_default],
        loc="upper left",
        bbox_to_anchor=(1, 1)
    )

    plt.title("Transaction DAG", fontsize=14)
    # Add x-axis for time steps
    plt.xlabel("Time Step")
    # Set x-ticks to match time steps
    plt.xticks(range(len(sorted_keys)), [str(i) for i in range(len(sorted_keys))])

    # Only want an x axis in the graph. There is no scale on the Y axis
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# -----------------------------------------------------------------------------------

REP_MGR = ReputationManager()

def plot_reputation(history: list[float]) -> None:
    """
    Plot reputation trajectory over time.

    history: list of reputation values from run_consensus_demo()
    neutral_level: baseline reputation
    max_rep: maximum reputation
    """
    steps = list(range(len(history)))
    neutral_level: float = MAX_REPUTATION / 2

    # Compute Gompertz targets assuming exp_pos increments each step
    # This allows us to plot the Gompertz envelope for reputation
    # tolist() converts a np array of signedInts to list[int]
    exp_pos = (np.arange(len(history))).tolist()
    target_curve = [REP_MGR._gompertz_target(e) for e in exp_pos]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, history, marker='o', label="Actual Reputation")
    plt.plot(steps, target_curve, linestyle="--", color="orange", label="Gompertz Target")
    plt.axhline(neutral_level, color="gray", linestyle=":", label=f"Neutral ({neutral_level})")
    plt.ylim(0, MAX_REPUTATION)
    plt.xlabel("Time step")
    plt.ylabel("Reputation")
    plt.title("Satellite Node Reputation over Time")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":")
    plt.show()

# -----------------------------------------------------------------------------------
# Run demonstration and plot the DAG and reputation of a satellite
final_dag, sat_reps = asyncio.run(run_consensus_demo(n_test_tx=3))
if final_dag:
    plot_transaction_dag(final_dag)
if sat_reps:
    plot_reputation(sat_reps)
