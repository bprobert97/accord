# pylint: disable=protected-access
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
import networkx as nx
import numpy as np
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG
from src.logger import get_logger
from src.reputation import MAX_REPUTATION, ReputationManager
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata
from src.utils import build_tx_data

logger = get_logger(__name__)


# Helpers for loading JSON -> Transactions
def load_sensor_json(file_path: str = "sim_output.json", sender_address: int = 0,
                     recipient_address: int = 123) -> list[Transaction]:
    """
    Load sensor observation data from JSON file and convert to Transaction objects.

    Args:
    - file_path: path to JSON file with sensor observations. Defaults to the
    json file returned by src/satellite_sim.mlx
    - sender_address: address of the sender (default 0)
    - recipient_address: address of the recipient (default 123)

    Returns:
    - List of Transaction objects
    """
    with open(file_path, "r", encoding="utf-8") as f:
        observations = json.load(f)

    transactions = []
    for obs in observations:
        tx = Transaction(
            sender_address=sender_address,
            recipient_address=recipient_address,
            sender_private_key="dummy_key",
            tx_data=json.dumps(obs),
            metadata=TransactionMetadata()
        )
        transactions.append(tx)
    return transactions


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
    # TODO - is this needed? I want each satellite to load its own data
    observations = build_tx_data("sim_output.json")
    if not observations:
        logger.info("No data in JSON file.")
        return None, None

    asyncio.create_task(dag.listen())

    satellites: list[SatelliteNode] = []
    rep_history: dict[str, list[float]] = {}

    # One satellite per observation (or choose grouping logic if needed)
    # TODO - need to fix satelltie IDs as its one per ransaction right now.
    # I need to simulate less satellites.
    # I need the data to match the scenario in here.
    # Also, TODO - update Matlab to use SGP4
    for i, obs in enumerate(observations):
        sat = SatelliteNode(node_id=f"SAT-{i:03d}", queue=queue)
        sat.load_sensor_data(obs)
        satellites.append(sat)
        rep_history[sat.id] = []

    # Submit each transaction via the satellite
    for sat in satellites:
        await sat.submit_transaction(recipient_address=123)
        rep_history[sat.id].append(sat.reputation)

    return dag, rep_history


# Plots
# TODO - need to fix graphs fix malicious nature of things too
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
            node_size=1800,
            edgecolors=outline_color,
            linewidths=2
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
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")
    plt.title("Transaction DAG", fontsize=14)
    plt.axis("off")
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

    for node_id, history in rep_history.items():
        steps = list(range(len(history)))
        exp_pos = (np.arange(len(history))).tolist()
        target_curve = [REP_MGR._gompertz_target(e) for e in exp_pos]
        plt.plot(steps, history, marker='o', label=f"{node_id} Reputation")
        plt.plot(steps, target_curve, linestyle="--", color="orange")

    plt.axhline(neutral_level, color="gray", linestyle=":", label=f"Neutral ({neutral_level})")
    plt.ylim(0, MAX_REPUTATION)
    plt.xlabel("Time step")
    plt.ylabel("Reputation")
    plt.title("Satellite Node Reputation over Time")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()


# Run demo
if __name__ == "__main__":
    final_dag, rep_hist = asyncio.run(run_consensus_demo())
    if final_dag:
        plot_transaction_dag(final_dag)
    if rep_hist:
        plot_reputation(rep_hist)
