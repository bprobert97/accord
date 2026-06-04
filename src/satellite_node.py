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
import json
import dataclasses
from dataclasses import dataclass
import pickle
from typing import Optional, TYPE_CHECKING
import numpy as np
from src.reputation import ReputationManager, MAX_REPUTATION
from src.transaction import Transaction, TransactionMetadata
from src.filter import ObservationRecord
from src.logger import get_logger

if TYPE_CHECKING:
    from .dag import DAG

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
    def __init__(self, node_id: int, queue: asyncio.Queue) -> None:
        self.id: int = node_id
        self.queue = queue
        self.exp_pos: int = 0
        # Reputation starts at a neutral level
        self.reputation: float = MAX_REPUTATION / 2
        self.performance_ema: float = 0.5  # For tracking recent performance
        self.rep_manager = ReputationManager()

        self.sensor_data: Optional[ObservationRecord] = None

        self.position = Position()

        # Local storage for synchronised ledger data
        self.local_ledger: dict[str, list[Transaction]] = {}

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

    async def submit_transaction(self,
                                 recipient_address: int) -> tuple[bool, Optional[float]]:
        """
        Builds a transaction from observed satellite data and submits it to the DAG.

        Args:
        - recipient_address: Cryptographic address of the recipient

        Returns:
        - A tuple containing the consensus result (bool) and the new EMA NIS (float or None).
        """
        if self.sensor_data is None:
            raise ValueError(f"Satellite {self.id} has no sensor data loaded.")

        tx_data = json.dumps(dataclasses.asdict(self.sensor_data))

        # Create metadata and transaction
        metadata = TransactionMetadata()
        transaction = Transaction(sender_address=hash(self.id),
                                  recipient_address=recipient_address,
                                  sender_private_key="PLACEHOLDER_KEY",
                                  tx_data=tx_data,
                                  metadata=metadata)

        future = asyncio.get_running_loop().create_future()
        await self.queue.put((transaction, self, future))
        # Waits until DAG sets the result
        return await future

    def sync_data(self, dag: DAG) -> None:
        """
        Synchronises the local ledger with the global DAG ledger.
        This mimics how a distributed ledger node would update its local state.

        Args:
        - dag: The global DAG object to sync from.
        """
        # In a real DLT, this would involve network communication.
        # Here we just copy the reference or the data.
        self.local_ledger = dag.get_ledger().copy()
        ledger_size = len(pickle.dumps(self.local_ledger))
        logger.info("Satellite %d synced data from DAG. Local ledger now \
                    has %d transactions (%d bytes)." , self.id,
                    len(self.local_ledger), ledger_size)
