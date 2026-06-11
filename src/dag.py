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
import bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from .logger import get_logger
from .transaction import Transaction, TransactionMetadata, \
    TransactionAddresses

if TYPE_CHECKING:
    from .consensus_mech import ConsensusMechanism

logger = get_logger()

@dataclass
class MockDAG():
    """A mock DAG object that only holds a ledger for plotting."""
    ledger: dict
    local_consensus_states: dict = field(default_factory=dict)

@dataclass
class NISMetricsTracker:
    """Tracks Normalised Innovation Squared (NIS) statistics per satellite."""
    mean_per_satellite: dict[int, float] = field(default_factory=dict)
    sums: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)


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

        # Node-local opinion dictionary
        # Key: tx_hash -> Value: dict containing local consensus state
        self.local_consensus_states: dict[str, dict] = {}

        # New: Maintain a separate list for chronological order
        self._chronological_txs: list[tuple[datetime, str]] = []
        for tx_hash, tx_list in self.ledger.items():
            self._chronological_txs.append((tx_list[0].metadata.timestamp, tx_hash))
        self._chronological_txs.sort() # Ensure initial sort of genesis transactions

        self.consensus_mech = consensus_mech
        self.queue = queue

        # Grouped NIS statistics
        self.nis_metrics = NISMetricsTracker()

        # History cache for Persistence of Excitation
        # Key: (observer_id, target_id)
        # Value: {'vector': list[float], 'timestamp': float}
        self.vector_history_cache: dict[tuple[int, int], dict] = {}

    async def listen(self) -> None:
        """
        An asynchronous function that continuously listens for transactions
        submitted to the DAG from a satellite node.
        """
        while True:
            transaction, satellite, future = await self.queue.get()
            logger.info("DAG received transaction %s", transaction.hash)
            consensus_result, mean_ema_nis = \
                self.consensus_mech.proof_of_inter_satellite_evaluation(
                    dag=self,
                    sat_node=satellite,
                    transaction=transaction,
                    mean_nis_per_satellite=self.nis_metrics.mean_per_satellite)
            future.set_result((consensus_result, mean_ema_nis))

            # If the transaction was successfully processed and returned a new_ema_nis,
            # update the DAG's internal running sums/counts and its cached mean_per_satellite.
            if consensus_result and mean_ema_nis is not None:
                try:
                    # Extract observer ID from the transaction data
                    observer_id = transaction.metadata.observer_id
                    if observer_id is not None:
                        # initialise if observer_id is new
                        self.nis_metrics.sums.setdefault(observer_id, 0.0)
                        self.nis_metrics.counts.setdefault(observer_id, 0)

                        # Update running sums and counts
                        self.nis_metrics.sums[observer_id] += mean_ema_nis
                        self.nis_metrics.counts[observer_id] += 1

                        # Update the cached mean_per_satellite for this observer
                        self.nis_metrics.mean_per_satellite[observer_id] = \
                            self.nis_metrics.sums[observer_id] / \
                                self.nis_metrics.counts[observer_id]
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Could not parse transaction data \
                                   for NIS update in DAG.listen().")

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

        genesis_addresses_1 = TransactionAddresses(sender_address=0,
                                                   recipient_address=0,
                                                   sender_private_key="1234")

        genesis_addresses_2 = TransactionAddresses(sender_address=0,
                                                   recipient_address=0,
                                                   sender_private_key="5678")

        return {"Genesis Transaction 1": [Transaction(addresses=genesis_addresses_1,
                                                      tx_data="Genesis Transaction 1",
                                                      metadata=genesis_metadata,
                                                      parent_hashes=())],
                "Genesis Transaction 2": [Transaction(addresses=genesis_addresses_2,
                                                      tx_data="Genesis Transaction 2",
                                                      metadata=genesis_metadata,
                                                      parent_hashes=())]}

    def get_parents(self) -> tuple[str, ...]:
        """
        Randomly select 2 parents for the transaction.
        Weighted towards choosing newer parents in the DAG
        for now, not accounting for node reputation.

        Returns:
        - The hashes of two parent transactions
        """
        # Retrieve keys in chronological order from the dedicated list
        keys = [item[1] for item in self._chronological_txs]

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
        # Read-only write to prevent parent mutations
        self.ledger[transaction.hash] = [transaction]

        # Initialise the local state dictionary entry if it doesn't exist
        if transaction.hash not in self.local_consensus_states:
            self.local_consensus_states[transaction.hash] = {
                "consensus_score": 0.0,
                "is_confirmed": False,
                "is_rejected": False,
                "nis": None,
                "dof": None
            }

        # Insert into the chronological list to maintain order
        new_item = (transaction.metadata.timestamp, transaction.hash)
        bisect.insort_left(self._chronological_txs, new_item)

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

    def get_ledger(self) -> dict[str, list[Transaction]]:
        """
        Returns the current state of the DAG ledger.

        Returns:
        - A dictionary representing the ledger.
        """
        return self.ledger

    def import_historical_tx(self,
                             transaction: Transaction,
                             state: dict) -> None:
        """
        Safely imports a historical transaction and its consensus state from a peer
        during synchronisation, maintaining internal chronological order without
        violating class encapsulation boundaries.

        Args:
        - transaction (Transaction): The historical transaction object to import.
        - state (dict): The consensus state dictionary associated with the transaction.

        Returns:
        - None. Updates internal ledger, chronology, and consensus states in place.
        """
        tx_hash = transaction.hash

        # Enforce ledger dict structure layout matching: dict[str, list[Transaction]]
        self.ledger[tx_hash] = [transaction]

        # Maintain chronological sorting order inside the class owning the attribute
        new_item = (transaction.metadata.timestamp, tx_hash)
        bisect.insort_left(self._chronological_txs, new_item)

        # Ingest the peer's consensus state view
        self.local_consensus_states[tx_hash] = state
