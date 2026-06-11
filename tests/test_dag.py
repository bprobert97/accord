"""
Unit tests for the DAG class.
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock
import pytest
from src.dag import DAG
from src.transaction import Transaction, TransactionMetadata, \
    TransactionAddresses

@pytest.fixture
def mock_consensus_mech():
    """Fixture for a mocked ConsensusMechanism."""
    return MagicMock()

@pytest.fixture
def mock_queue():
    """Fixture for a mocked asyncio.Queue."""
    return AsyncMock(spec=asyncio.Queue)

@pytest.fixture
def dag(mock_consensus_mech, mock_queue):
    """Fixture for a DAG instance with mocked dependencies."""
    return DAG(consensus_mech=mock_consensus_mech, queue=mock_queue)

def test_create_genesis_tx(dag):
    """
    Test that the genesis transactions are created correctly.
    """
    genesis_ledger = dag.create_genesis_tx()
    assert len(genesis_ledger) == 2

    tx1 = genesis_ledger["Genesis Transaction 1"][0]
    tx2 = genesis_ledger["Genesis Transaction 2"][0]

    assert tx1.metadata.is_confirmed is True
    assert tx1.metadata.consensus_reached is True
    assert tx2.metadata.is_confirmed is True

def test_add_tx(dag):
    """
    Test adding a transaction to the DAG.
    """
    initial_len = len(dag.ledger)

    # 1. Dynamically capture the current parents from the DAG view
    parents = dag.get_parents()

    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="key",)

    # Fixed: Supplied parent_hashes to the constructor call
    new_tx = Transaction(addresses=addresses,
                         tx_data="data",
                         metadata=TransactionMetadata(),
                         parent_hashes=parents)

    dag.add_tx(new_tx)

    # Assertions updated to target the new direct immutable properties
    assert len(dag.ledger) == initial_len + 1
    assert new_tx.hash in dag.ledger
    assert len(new_tx.parent_hashes) == 2
    assert new_tx.parent_hashes[0] in parents


def test_get_parents(dag):
    """
    Test the parent selection logic.
    """
    # With only genesis transactions, it should return both of them
    parents = dag.get_parents()
    assert len(parents) == 2
    assert "Genesis Transaction 1" in parents
    assert "Genesis Transaction 2" in parents

    # Add more transactions and check again
    for i in range(5):
        # 2. Get parents dynamically inside the loop to grow a continuous chain
        current_parents = dag.get_parents()

        addresses = TransactionAddresses(sender_address=i,
                                         recipient_address=i+1,
                                         sender_private_key="k")

        # Fixed: Supplied current_parents to the transaction factory loop
        dag.add_tx(Transaction(addresses=addresses,
                               tx_data=f"d{i}",
                               metadata=TransactionMetadata(),
                               parent_hashes=current_parents))

    new_parents = dag.get_parents()
    assert len(new_parents) == 2
    assert new_parents[0] in dag.ledger
    assert new_parents[1] in dag.ledger


def test_has_bft_quorum(dag):
    """
    Test the BFT quorum check.
    """
    # Initially, with 2 genesis tx, we have 0 real tx. No quorum.
    assert not dag.has_bft_quorum()

    # Add 3 real transactions. Not enough for f=1 (needs 4).
    for i in range(3):
        current_parents = dag.get_parents()
        addresses = TransactionAddresses(sender_address=i,
                                         recipient_address=i+1,
                                         sender_private_key="k",)

        # Fixed: Supplied parent_hashes to constructor
        dag.add_tx(Transaction(addresses=addresses,
                               tx_data=f"d{i}",
                               metadata=TransactionMetadata(),
                               parent_hashes=current_parents))
    assert not dag.has_bft_quorum()

    # Add the 4th real transaction. Now we have quorum.
    final_parents = dag.get_parents()
    addresses = TransactionAddresses(sender_address=4,
                                     recipient_address=5,
                                     sender_private_key="k",)

    # Fixed: Supplied final_parents to constructor
    dag.add_tx(Transaction(addresses=addresses,
                           tx_data="d4",
                           metadata=TransactionMetadata(),
                           parent_hashes=final_parents))
    assert dag.has_bft_quorum()
