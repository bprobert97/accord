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

# To stop type_checking freaking out at runtime
from __future__ import annotations

import asyncio
import json
import random
from typing import TYPE_CHECKING, OrderedDict
import numpy as np
from .logger import get_logger
from .transaction import Transaction, TransactionMetadata

if TYPE_CHECKING:
    from .consensus_mech import ConsensusMechanism

logger = get_logger()

class DAG():
    """
    A class representing the Directed Acyclic Graph (DAG) Distributed Ledger Technology.
    When a transaction is received, it is added to the DAG. The number of parents
    for each transaction is decided using a tip selection algorithm.
    """

    def __init__(self,
                 consensus_mech: ConsensusMechanism,
                 queue: asyncio.Queue) -> None:
        # Ledger structure is:
        # key: string hash of transaction, value: list[Transaction]
        self.ledger: dict[str, list[Transaction]] = self.create_genesis_tx()
        self.consensus_mech = consensus_mech
        self.queue = queue
        self.mean_nis_per_satellite: dict[int, float] = {}

    async def listen(self) -> None:
        """
        An asynchronous function that continuously listens for transactions
        submitted to the DAG from a satellite node.
        """
        while True:
            transaction, satellite, future = await self.queue.get()
            logger.info("DAG received transaction %s", transaction.hash)
            consensus_result = self.consensus_mech.proof_of_inter_satellite_evaluation(
                dag=self,
                sat_node=satellite,
                transaction=transaction,
                mean_nis_per_satellite=self.calculate_mean_nis()
            )
            future.set_result(consensus_result)

    def create_genesis_tx(self) -> dict[str, list[Transaction]]:
        """
        Creates the two genesis transactions to initialise the DAG and provide parents
        for the first real transaction.
        Have to set consensus_reached and is_confirmed = True here otherwise strong
        parents become impossible.

        Returns:
        - A dictionary of two genesis transactions and their IDs
        """
        genesis_metadata = TransactionMetadata(consensus_reached=True,
                                               is_confirmed=True)

        return {"Genesis Transaction 1": [Transaction(0, 0, "1234",
                                                      "Genesis Transaction 1",
                                                      metadata=genesis_metadata)],
                "Genesis Transaction 2": [Transaction(0, 0, "5678",
                                                      "Genesis Transaction 2",
                                                      metadata=genesis_metadata)]}

    def get_parents(self) -> tuple[str, ...]:
        """
        Randomly select 2 parents for the transaction.
        Weighted towards choosing newer parents in the DAG
        for now, not accounting for node reputation.

        Returns:
        - The hashes of two parent transactions
        """
        keys = list(self.ledger.keys())

        # This error should not happen because of genesis transactions,
        # but just in case
        if len(keys) < 2:
            raise ValueError("Not enough transactions to select parents.")

        # Linear bias in weights, favouring newer transactions
        # which will be later on (higher index) in the DAG as they
        # are ordered by timestamp
        weights = [i + 1 for i in range(len(keys))]

        # Select 2 parents at random, with weighting
        selected_parents = random.choices(keys, weights=weights, k=2)

        # Ensure uniqueness, as choices does not ensure this
        while selected_parents[0] == selected_parents[1]:
            selected_parents[1] = random.choices(keys, weights=weights, k=1)[0]

        return tuple(selected_parents)

    def add_tx(self, transaction: Transaction) -> None:
        """
        Add a transaction to the DAG.

        Args:
        - transaction: the data to be added to the DAG.

        Returns:
        - None. Adds transaction to the DAG.
        """
        parent1, parent2 = self.get_parents()

        # There is guaranteed to be two parents - the genesis transactions in the DAG.
        transaction.metadata.parent_hashes.extend([parent1, parent2])

        # Add transaction to ledger in timestamp order
        self.ledger[transaction.hash] = [transaction]
        self.ledger = OrderedDict(
        sorted(self.ledger.items(), key=lambda item: item[1][0].metadata.timestamp))

    def has_bft_quorum(self) -> bool:
        """
        Check if we have at least 3f + 1 real transactions (f = max faulty nodes tolerated).
        Genesis txs are ignored in this count.

        Returns:
        - A boolean indicating if BFT quorum is reached.
        """
        real_tx_count = max(0, len(self.ledger) - 2)  # exclude genesis
        # If f=1, we need 4 real tx (3*1+1)
        return real_tx_count >= 4

    def calculate_mean_nis(self) -> dict[int, float]:
        """
        Calculate the mean NIS for each satellite from the transactions in the ledger.

        Returns:
        - A dictionary mapping satellite ID to its mean NIS.
        """
        nis_by_sat: dict[int, list[float]] = {}
        for tx_list in self.ledger.values():
            for tx in tx_list:
                if tx.tx_data.startswith("Genesis"):
                    continue
                try:
                    tx_data = json.loads(tx.tx_data)
                    observer_id = tx_data.get("observer")
                    nis = tx_data.get("nis")
                    if observer_id is not None and nis is not None:
                        nis_by_sat.setdefault(observer_id, []).append(nis)
                except (json.JSONDecodeError, TypeError):
                    continue

        mean_nis_per_satellite: dict[int, float] = {
            sat_id: float(np.mean(nis_values)) for sat_id, nis_values in nis_by_sat.items()
        }
        return mean_nis_per_satellite
