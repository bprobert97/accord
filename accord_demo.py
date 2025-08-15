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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from datetime import timedelta
from skyfield.api import EarthSatellite
from typing import Optional
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata


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
                             n_test_tx: int = 1) -> Optional[DAG]:
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
    then be parsed into other funcions to generate diagrams or analyse data.
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
        print("No satellite data found.")
        return None

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
    for _ in range(n_test_tx):
        # Run consensus on a single satellite observation - the test transaction
        await test_satellite.submit_transaction(
             satellite=base_sat,
             recipient_address=123
             )
        sat_rep_list.append(test_satellite.reputation)

    # Output results
    print(test_dag.ledger)
    print(sat_rep_list)
    return test_dag


# -----------------------------------------------------------------------------------
def plot_transaction_dag(dag: DAG) -> None:
    """
    Plot a graph representing the Directed Acyclic Graph

    Args:
    dag: The Directed Acyclic Grap Distributed Ledger datastructure
    to be plotted

    Returns:
    None. Shows a plot using MatPlotLib/
    """
    G: nx.DiGraph = nx.DiGraph()
    tx_timestamps = {}
    tx_status = {}  # store is_confirmed/is_rejected per tx

    # Add nodes & edges
    for key, tx_list in dag.ledger.items():
        for tx in tx_list:
            G.add_node(key)
            tx_timestamps[key] = tx.metadata.timestamp
            tx_status[key] = {
                "is_confirmed": tx.metadata.is_confirmed,
                "is_rejected": tx.metadata.is_rejected
            }
            for parent_hash in tx.metadata.parent_hashes:
                G.add_edge(key, parent_hash)

    # Sort nodes by timestamp for X-axis positioning
    sorted_keys = sorted(tx_timestamps, key=lambda k: tx_timestamps[k])
    pos = {}
    for i, key in enumerate(sorted_keys):
        x = i
        y = (hash(key) % 100) / 100.0 - 0.5
        pos[key] = (x, y)

    plt.figure(figsize=(16, 6))

    # Draw nodes with outline color
    for node in G.nodes():
        outline_color = "black"
        if tx_status[node]["is_confirmed"]:
            outline_color = "green"
        elif tx_status[node]["is_rejected"]:
            outline_color = "red"
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node],
            node_color="lightblue",
            node_size=1800,
            edgecolors=outline_color,  # outline color
            linewidths=2
        )

    # Draw edges with color matching the target node's status
    edge_colors = []
    for _, parent_node in G.edges():
        if tx_status[parent_node]["is_confirmed"]:
            edge_colors.append("green")
        elif tx_status[parent_node]["is_rejected"]:
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    # Ignoring a warning raised by mypy as the sub file for networkx is incorrect
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=15) # type: ignore[arg-type]

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_weight="bold"
    )

    plt.title("Transaction DAG", fontsize=14)
    plt.axis("off")
    plt.show()

# -----------------------------------------------------------------------------------
# Run demonstration and plot the DAG
test_dag = asyncio.run(run_consensus_demo(n_test_tx=5))
if test_dag:
    plot_transaction_dag(test_dag)
