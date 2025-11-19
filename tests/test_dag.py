# pylint: disable=redefined-outer-name
"""
Unit tests for the DAG class.
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock
import pytest
from src.dag import DAG
from src.transaction import Transaction, TransactionMetadata

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
    new_tx = Transaction(1, 2, "key", "data", TransactionMetadata())

    dag.add_tx(new_tx)

    assert len(dag.ledger) == initial_len + 1
    assert new_tx.hash in dag.ledger
    assert len(new_tx.metadata.parent_hashes) == 2
    assert new_tx.metadata.parent_hashes[0] in ("Genesis Transaction 1", "Genesis Transaction 2")

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
        dag.add_tx(Transaction(i, i+1, "k", f"d{i}", TransactionMetadata()))

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
        dag.add_tx(Transaction(i, i+1, "k", f"d{i}", TransactionMetadata()))
    assert not dag.has_bft_quorum()

    # Add the 4th real transaction. Now we have quorum.
    dag.add_tx(Transaction(4, 5, "k", "d4", TransactionMetadata()))
    assert dag.has_bft_quorum()

def test_calculate_mean_nis(dag):
    """
    Test the calculation of mean NIS per satellite.
    """
    # Add some transactions with NIS data
    tx1_data = '{"observer": 1, "nis": 2.0, "dof": 2, "step": 1, "time": 1, "target": 0}'
    tx2_data = '{"observer": 2, "nis": 4.0, "dof": 2, "step": 1, "time": 1, "target": 0}'
    tx3_data = '{"observer": 1, "nis": 3.0, "dof": 2, "step": 2, "time": 2, "target": 0}'

    dag.add_tx(Transaction(1, 0, "k", tx1_data, TransactionMetadata()))
    dag.add_tx(Transaction(2, 0, "k", tx2_data, TransactionMetadata()))
    dag.add_tx(Transaction(1, 0, "k", tx3_data, TransactionMetadata()))

    mean_nis = dag.calculate_mean_nis()

    assert len(mean_nis) == 2
    assert mean_nis[1] == pytest.approx(2.5) # (2.0 + 3.0) / 2
    assert mean_nis[2] == pytest.approx(4.0)
