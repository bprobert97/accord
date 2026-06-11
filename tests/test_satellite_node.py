"""
Unit tests for the SatelliteNode class.
"""

import pytest
from src.consensus_mech import ConsensusMechanism
from src.dag import Transaction, TransactionAddresses, \
    TransactionMetadata
from src.satellite_node import SatelliteNode
from src.filter import ObservationRecord
from src.reputation import MAX_REPUTATION


def test_satellite_node_init():
    """
    Test the initialization of a SatelliteNode.
    """
    node = SatelliteNode(node_id=5, consensus_mech=ConsensusMechanism())
    assert node.id == 5
    assert node.reputation == MAX_REPUTATION / 2
    assert node.exp_pos == 0
    assert node.performance_ema == 0.5
    assert node.sensor_data is None

def test_load_sensor_data():
    """
    Test that sensor data is loaded correctly.
    """
    node = SatelliteNode(node_id=1, consensus_mech=ConsensusMechanism())
    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2,
                                   r_vector=[1.0, 2.0, 3.0], v_vector=[0.1, 0.2, 0.3])

    assert node.sensor_data is None
    node.load_sensor_data(obs_record)
    assert node.sensor_data is obs_record
    assert node.sensor_data.nis == 2.0


@pytest.mark.asyncio
async def test_submit_transaction_success():
    """
    Test the successful submission and P2P propagation of a transaction.
    """
    # Arrange: Spin up two nodes running independent ledger containers
    mech = ConsensusMechanism()
    node1 = SatelliteNode(node_id=1, consensus_mech=mech)
    node2 = SatelliteNode(node_id=2, consensus_mech=mech)

    # Form our peer routing topology link
    node1.peers = [node2]

    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2,
                                   r_vector=[1.0, 2.0, 3.0], v_vector=[0.1, 0.2, 0.3])
    node1.load_sensor_data(obs_record)

    # Act: Broadcast from node1 over our decentralised network link
    await node1.submit_transaction(recipient_address=node2.id)

    # Assert: Verify transaction cleared boundaries and logged to both ledgers
    # (Note: It will be pending on node2 because BFT quorum requirements aren't met yet,
    # which is perfectly authentic behavior!)
    assert any(tx.metadata.observer_id == 1 for tx_list in \
               node1.dag.ledger.values() for tx in tx_list)
    assert any(tx.metadata.observer_id == 1 for tx_list in \
               node2.dag.ledger.values() for tx in tx_list)


@pytest.mark.asyncio
async def test_sync_data():
    """
    Test that a satellite node can correctly pull historical transaction logs
    and matching database states from a peer during anti-entropy catch-up.
    """
    # Arrange
    mech = ConsensusMechanism()
    node1 = SatelliteNode(node_id=1, consensus_mech=mech)
    node2 = SatelliteNode(node_id=2, consensus_mech=mech)

    # Manually seed a completed transaction log and evaluation opinion inside node2
    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="placeholder",)

    # Fixed: Added parent_hashes=() to pass strict constructor signature matching
    historical_tx = Transaction(addresses=addresses,
                                tx_data="{}",
                                metadata=TransactionMetadata(),
                                parent_hashes=())

    node2.dag.ledger[historical_tx.hash] = [historical_tx]
    node2.dag.local_consensus_states[historical_tx.hash] = {
        "consensus_score": 0.85,
        "is_confirmed": True,
        "is_rejected": False,
        "nis": 1.42,
        "dof": 2
    }

    # Verify node1's ledger is completely blank before synchronization begins
    assert historical_tx.hash not in node1.dag.ledger

    # Act: Fire the catch-up database alignment hook
    await node1.request_sync_from_peer(peer=node2)

    # Assert: Verify transaction records and opinion matrices port cleanly across boundaries
    assert historical_tx.hash in node1.dag.ledger
    assert node1.dag.local_consensus_states[historical_tx.hash]["consensus_score"] == 0.85
    assert node1.dag.local_consensus_states[historical_tx.hash]["is_confirmed"] is True
