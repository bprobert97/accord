# pylint: disable=too-many-instance-attributes
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
import copy
import json
from typing import Optional
from .dag import DAG
from .reputation import ReputationManager, MAX_REPUTATION
from .transaction import Transaction, TransactionMetadata


class SatelliteNode():
    """
    A class representing a node in the network, in this case a LEO satellite.
    This does NOT represent a node in the ledger - these are transactions
    """
    def __init__(self, node_id: str, queue: asyncio.Queue) -> None:
        self.id: str = node_id
        self.queue = queue
        self.exp_pos: int = 0
        # Reputation starts at a neutral level
        self.reputation: float = MAX_REPUTATION / 2
        self.rep_manager = ReputationManager()
        self.local_dag: Optional[DAG] = None

        self.sensor_data: Optional[dict] = None

    def load_sensor_data(self, observation: dict) -> None:
        """
        Attach one observation record (from JSON) to this satellite.

        Args:
        - observation: a dict containing one observation record

        Returns:
        None. Updates self.sensor_data.
        """
        self.sensor_data = observation

    async def submit_transaction(self,
                                 recipient_address: int) -> bool:
        """
        Builds a transaction from observed satellite data and submits it to the DAG.

        Args:
        - recipient_address: Crypographic address of the recipient

        Returns:
        A transaction that is submitted to the ledger
        """
        if self.sensor_data is None:
            raise ValueError(f"Satellite {self.id} has no sensor data loaded.")

        tx_data = json.dumps(self.sensor_data)

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

    def synchronise(self, network_dag: DAG) -> None:
        """
        Synchronise local DAG state with a given network DAG.
        For now, replaces the local DAG with a deep copy.
        """
        self.local_dag = copy.deepcopy(network_dag)
