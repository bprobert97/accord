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
from typing import Optional
from skyfield.api import EarthSatellite
from .dag import DAG
from .transaction import Transaction, TransactionMetadata
from .utils import build_tx_data_str, load_json_data


class SatelliteNode():
    """
    A class representing a node in the network, in this case a LEO satellite. 
    This does NOT represent a node in the ledger - these are transactions
    """
    def __init__(self, node_id: str, queue: asyncio.Queue) -> None:
        self.id: str = node_id
        self.queue = queue

        # Reputation starts at 0, affected by validity and accuracy
        # TODO - need to consider how this affects consensus.
        # If reputation low, does it get allowed? or does it affect
        # consensus score?
        self.reputation: float = 0.0
        self.local_dag: Optional[DAG] = None

        # This is for testing purposes. In reality, data will
        # be loaded from a sensor
        self.tle_data: list[Optional[EarthSatellite]] = load_json_data("od_data.json")

    async def submit_transaction(self,
                                 satellite: EarthSatellite,
                                 recipient_address: int) -> bool:
        """
        Builds a transaction from observed satellite data and submits it to the DAG.

        Args:
        - satellite: The EarthSatellite object containing data relevant for building the transaction
        - recipient_address: TODO may not be needed in transaction

        Returns:
        A transaction that is submitted to the ledger
        """
        # Build TLE/OD data string from the EarthSatellite object
        tx_data_str = build_tx_data_str(satellite)

        # Create metadata and transaction
        metadata = TransactionMetadata()
        transaction = Transaction(sender_address=hash(self.id),
                                  recipient_address=recipient_address,
                                  # TODO Replace with actual key handling
                                  sender_private_key="PLACEHOLDER_KEY",
                                  tx_data=tx_data_str,
                                  metadata=metadata)

        future = asyncio.get_running_loop().create_future()
        print(f"Satellite {self.id}: submitting transaction {transaction.hash}")
        await self.queue.put((transaction, self, future))
        # Waits until DAG sets the result
        return await future

    def synchronise(self, network_dag: DAG) -> None:
        """
        Synchronise local DAG state with a given network DAG.
        For now, replaces the local DAG with a deep copy.
        """
        self.local_dag = copy.deepcopy(network_dag)

# TODO - need to allow node to submit a transaction to the ledger.
# eventually will simulate this properly
# Would likely need something to listen for the signal from the node??
# Then when triggered, run consensus mechanism
