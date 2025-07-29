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

import random
from collections import OrderedDict
from .transaction import Transaction, TransactionMetadata
from .utils import REJECTION_THRESHOLD, CONFIRMATION_STEP, CONFIRMATION_THRESHOLD

class DAG():
    """
    A class representing the Directed Acyclic Graph (DAG) Distributed Ledger Technology.
    When a transaction is received, it is added to the DAG. The number of parents
    for each transaction is decided using a tip selection algorithm.
    """

    def __init__(self) -> None:
        # TODO - need a way to check that the DAG has not been tampered with

        # Ledger structure is:
        # key: string hash of transaction, value: Transaction class
        self.ledger: dict = self.create_genesis_tx()

    def create_genesis_tx(self) -> dict[str, list[Transaction]]:
        """
        Creates the two genesis transactions to initialise the DAG and provide parents
        for the first real transaction.
        Have to set consensus_reached and is_confirmed = True here otherwise strong
        parents become impossible.
        """
        genesis_metadata = TransactionMetadata(consensus_reached=True,
                                               is_confirmed=True)

        return {"Genesis Transaction 1": [Transaction(0, 0, "1234",
                                                      "Genesis Transaction 1",
                                                      metadata=genesis_metadata)],
                "Genesis Transaction 2": [Transaction(0, 0, "5678",
                                                      "Genesis Transaction 2",
                                                      metadata=genesis_metadata)]}

    def check_thresholds(self, transaction: Transaction) -> None:
        """
        Check if the transaction confirmation or rejection thresholds have been crossed. 
        This will affect weighting. is_confirmed = strong weighting, else weak weighting
        """
        if REJECTION_THRESHOLD <= transaction.metadata.confirmation_score <= CONFIRMATION_THRESHOLD:
            transaction.metadata.confirmation_score +=  CONFIRMATION_STEP
        else:
            transaction.metadata.confirmation_score -= CONFIRMATION_STEP

        if transaction.metadata.confirmation_score >= CONFIRMATION_THRESHOLD:
            transaction.metadata.is_confirmed = True
        elif transaction.metadata.confirmation_score <= REJECTION_THRESHOLD:
            transaction.metadata.is_rejected = True

    def get_parents(self) -> tuple[str, str]:
        """
        Randomly select 2 parents for the transaction
        # THRESHOLDS affect tip selection for parents - TODO
        TODO - need to make sure we have an ordered dict of transactions and
        that we pick from latest layer..(line 94 order by timestamp)
        """
        return tuple(random.sample(list(self.ledger.keys()), 2))

    def add_tx(self, transaction: Transaction) -> None:
        """
        Add a transaction to the DAG
        """
        # TODO - tx or blocks?? start with tx for now
        # TODO - fixed number of parents: 2
        parent1, parent2 = self.get_parents()

        # There is guaranteed to be two parent - the genesis transactions in the DAG.
        transaction.metadata.parent_hashes.extend([parent1, parent2])

        # Add transaction to ledger in timestamp order
        self.ledger[transaction.hash] = [transaction]
        self.ledger = OrderedDict(
        sorted(self.ledger.items(), key=lambda item: item[1][0].metadata.timestamp))

        # TODO - need to add consensus mechanism in here, may need to be a function within
        # this class rather than a separate class to avoid circles
        self.check_thresholds(transaction)
