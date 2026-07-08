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

from __future__ import annotations
import asyncio
import copy
import json
import dataclasses
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np
from src.dag import DAG
from src.reputation import ReputationManager, MAX_REPUTATION, \
    ReputationParams
from src.transaction import Transaction, TransactionMetadata, \
    TransactionAddresses
from src.filters.filter_interface import ObservationRecord
from src.logger import get_logger

if TYPE_CHECKING:
    from src.consensus_mech import ConsensusMechanism

logger = get_logger()

@dataclass
class Position:
    """3D spatial coordinates of the satellite."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class SatelliteNode():
    """
    A class representing a node in the network, in this case a LEO satellite.
    This does NOT represent a node in the ledger - these are transactions
    """
    def __init__(self, node_id: int,
                 consensus_mech: ConsensusMechanism) -> None:
        self.id: int = node_id
        self.exp_pos: int = 0
        # Reputation starts at a neutral level
        self.reputation: float = MAX_REPUTATION / 2
        self.performance_ema: float = 0.5  # For tracking recent performance
        rep_params = ReputationParams()
        self.rep_manager = ReputationManager(rep_params)
        self.sensor_data: Optional[ObservationRecord] = None
        self.position = Position()

        # DLT and decentralised communication
        self.local_queue: asyncio.Queue = asyncio.Queue()
        self.dag = DAG(consensus_mech=consensus_mech,
                       queue=self.local_queue)
        # P2P routing table
        self.peers: list[SatelliteNode] = []

    def update_position(self, state_vector: np.ndarray) -> None:
        """
        Updates the satellite's position from a state vector.

        Args:
        - state_vector: A NumPy array containing the satellite's state
                        (at least [px, py, pz, ...]).
        """
        self.position.x, \
            self.position.y, \
                self.position.z = state_vector[0], \
                                    state_vector[1], \
                                        state_vector[2]

    def load_sensor_data(self, observation: ObservationRecord) -> None:
        """
        Attach one observation record (from JSON) to this satellite.

        Args:
        - observation: An ObservationRecord object.

        Returns:
        None. Updates self.sensor_data.
        """
        self.sensor_data = observation

    async def submit_transaction(self, recipient_address: int) -> None:
        """
        Builds a transaction from newly observed sensor data, commits it
        to its own local DAG, and broadcasts it to peers sequentially.

        Args:
        - recipient_address: Cryptographic address of the target satellite node.

        Returns:
        - None.
        """
        if self.sensor_data is None:
            raise ValueError(f"Satellite {self.id} has no sensor data loaded.")

        tx_data = json.dumps(dataclasses.asdict(self.sensor_data))

        # 1. Build pristine local metadata and addresses
        metadata = TransactionMetadata()
        metadata.observer_id = self.id  # Explicitly preserve who recorded the track

        addresses = TransactionAddresses(
            sender_address=hash(self.id),
            recipient_address=recipient_address,
            sender_private_key="PLACEHOLDER_KEY"
        )

        # 1. Use the method to select local tips from this node's view of the graph
        parent1, parent2 = self.dag.get_parents()

        # 2. Immutable instantiation: freeze the data and edges together
        transaction = Transaction(
            addresses=addresses,
            tx_data=tx_data,
            metadata=metadata,
            parent_hashes=(parent1, parent2)  # Securely bound
        )

        # 2. Add our own unvalidated copy to our local ledger
        self.dag.add_tx(transaction)
        logger.info("Satellite %d added its own observation to local DAG.", self.id)

        # 3. Await the broadcast directly so network validations happen
        # inside the current simulation time step
        await self.broadcast_transaction(transaction)

    async def broadcast_transaction(self, transaction: Transaction) -> None:
        """
        Simulates sending a transaction over radio/laser links to visible peers.

        Args:
        - transaction: The core transaction object to broadcast.

        Returns:
        - None.
        """
        for peer in self.peers:
            # We pass the transaction directly to our peers
            await peer.receive_transaction(transaction, sender=self)

    async def receive_transaction(self,
                                  transaction: Transaction,
                                  sender: SatelliteNode) -> None:
        """
        Triggered when a peer sends this satellite a transaction to validate.
        Isolates data states using deep copies to mimic network boundaries.

        Args:
        - transaction: The incoming transaction object sent by a peer.
        - sender: The SatelliteNode instance that originated the broadcast.

        Returns:
        - None.
        """
        # CRITICAL: Deep copy the transaction so this node's validation modifications
        # and DAG parent tip extensions do not corrupt other satellites' states.
        local_tx_copy = copy.deepcopy(transaction)

        # Run POISE consensus locally against our own historical evaluations TODO
        _, _ = self.dag.consensus_mech.proof_of_inter_satellite_evaluation(
            dag=self.dag,
            sat_node=sender,
            transaction=local_tx_copy,
            mean_nis_per_satellite=self.dag.nis_metrics.mean_per_satellite
        )

    async def request_sync_from_peer(self, peer: SatelliteNode) -> None:
        """
        Asks a specific neighbour for any transaction hashes we might have missed
        while offline or out of alignment. Pulls both the transaction payload
        and the peer's consensus state view to catch up database states.

        Args:
        - peer (SatelliteNode): A satellite node within ISL range that we want to synchronise with.

        Returns:
        - None. Receives transactions from peer and synchronises.
        """
        peer_ledger = peer.dag.get_ledger()

        for tx_hash, tx_list in peer_ledger.items():
            if tx_hash not in self.dag.ledger:
                if not tx_list:
                    continue

                # 1. Capture deep copies of the transaction payload and peer state view
                historical_tx = copy.deepcopy(tx_list[0])
                peer_state = copy.deepcopy(peer.dag.local_consensus_states.get(tx_hash, {}))

                # 2. Ingest records smoothly using the public DAG encapsulation interface
                self.dag.import_historical_tx(historical_tx, peer_state)

                logger.info(
                    "Satellite %d synced historical transaction %s from catch-up peer %d.",
                    self.id, tx_hash[:8], peer.id
                )
