"""
Unit tests for the ConsensusMechanism class.
"""
import json
import time
from unittest.mock import MagicMock, patch
import pytest
from src.consensus_mech import ConsensusMechanism
from src.reputation import MAX_REPUTATION
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata, \
    TransactionAddresses
from src.filters.filter_interface import ObservationRecord

@pytest.fixture
def consensus_mech():
    """
    Pytest fixture to provide a ConsensusMechanism instance for tests.
    """
    return ConsensusMechanism()

@pytest.fixture
def mock_sat_node():
    """Fixture for a mocked SatelliteNode."""
    node = MagicMock(spec=SatelliteNode)
    node.reputation = MAX_REPUTATION / 2
    node.exp_pos = 0
    node.performance_ema = 0.5

    # Add the rep_manager attribute to the mock
    node.rep_manager = MagicMock()
    node.rep_manager.apply_positive.return_value = (node.reputation + 10, 1, 0.5)
    node.rep_manager.apply_negative.return_value = (node.reputation - 10, 0, 0.4)
    node.rep_manager.decay.side_effect = lambda rep: rep # No decay in tests
    return node

@pytest.fixture
def mock_dag():
    """Fixture for a mocked DAG."""
    dag = MagicMock()
    dag.has_bft_quorum.return_value = True
    return dag

def test_nis_to_score(consensus_mech):
    """
    Test the nis_to_score function with various inputs.
    """
    dof = 2
    # A "too perfect" NIS should have a low score
    perfect_score = consensus_mech.nis_to_score(nis=0.01, dof=dof)
    assert perfect_score < 0.5

    # An NIS close to the expected value (dof) should have a high score
    good_nis = dof - 0.1
    good_score = consensus_mech.nis_to_score(nis=good_nis, dof=dof)
    assert good_score > 0.75

    # A very high NIS (outlier) should have a score of 0
    outlier_score = consensus_mech.nis_to_score(nis=100.0, dof=dof)
    assert outlier_score == 0.0

    # Test historical improvement
    historical_ema = 10.0 # Bad history
    improving_nis = 3.0 # NIS is high, but closer to dof=2 than 10
    improving_score = consensus_mech.nis_to_score(
        nis=improving_nis, dof=dof, historical_ema_nis=historical_ema
    )

    worsening_nis = 15.0 # NIS is worse than history
    worsening_score = consensus_mech.nis_to_score(
        nis=worsening_nis, dof=dof, historical_ema_nis=historical_ema
    )

    assert improving_score > worsening_score

def test_calculate_dof_score(consensus_mech):
    """
    Test the DOF-based scoring.
    """
    current_r_vector=[1.0, 2.0, 3.0]
    current_v_vector=[0.1, 0.2, 0.3]
    obs = ObservationRecord(step=500,
                            time=time.time(),
                            observer=1,
                            target=2,
                            nis=2.0,
                            dof=1,
                            r_vector=current_r_vector,
                            v_vector=current_v_vector)

    # Test base score - no previous vector or delta_t
    # DOF = 1 initially
    assert consensus_mech.calculate_dof_score(obs_record=obs) == 0

    obs.dof = 2
    assert consensus_mech.calculate_dof_score(obs_record=obs) == 0.2

    obs.dof = 3
    assert consensus_mech.calculate_dof_score(obs_record=obs) == 0.4

    # Test with previous data
    previous_r_vector = [0.5, 1.5, 2.5]
    previous_v_vector = [0.05, 0.15, 0.25]

    prev_data = {
        "r_vector": previous_r_vector,
        "v_vector": previous_v_vector,
        "time": time.time() - 10}

    score_1 = consensus_mech.calculate_dof_score(
        obs_record=obs,
        previous_data=prev_data
    )
    assert score_1 == pytest.approx(0.16, 0.01)

    # Expect unchanged vectors to score worse
    prev_data_unchanged = {
        "r_vector": current_r_vector,
        "v_vector": current_v_vector,
        "time": time.time() - 10}

    score_2 = consensus_mech.calculate_dof_score(
        obs_record=obs,
        previous_data=prev_data_unchanged
    )
    assert score_1 > score_2

def test_calculate_consensus_score(consensus_mech):
    """
    Test the overall consensus score calculation.
    """
    # Scenario 1: Everything is good
    score1 = consensus_mech.calculate_consensus_score(
        correctness=0.9,
        dof_reward=0.8,
        reputation=MAX_REPUTATION * 0.9
    )
    assert score1 > consensus_mech.consensus_threshold
    assert score1 <= 1.0

    # Scenario 2: Correctness is very low
    score2 = consensus_mech.calculate_consensus_score(
        correctness=0.1,
        dof_reward=1.0,
        reputation=MAX_REPUTATION * 0.9
    )
    assert score2 < consensus_mech.consensus_threshold

    # Scenario 3: Reputation is very low
    score3 = consensus_mech.calculate_consensus_score(
        correctness=0.9,
        dof_reward=0.8,
        reputation=0.1 # Very low reputation
    )
    assert score3 < score1

    # Scenario 4: Everything is mediocre
    score4 = consensus_mech.calculate_consensus_score(
        correctness=0.5,
        dof_reward=0.5,
        reputation=MAX_REPUTATION / 2
    )
    # This should be just above the threshold
    assert score4 > consensus_mech.consensus_threshold

def test_poise_empty_transaction(consensus_mech) -> None:
    """
    Test PoISE with an empty transaction, expecting a reputation penalty.
    """
    # 1. Initialize clean explicit mocks
    dag_mock = MagicMock()

    sat_node_mock = MagicMock()
    sat_node_mock.id = 1
    sat_node_mock.reputation = 0.5

    # Pre-configure return tuple to prevent unpacking ValueErrors during early penalty
    sat_node_mock.rep_manager.apply_negative.return_value = (0.4, 0, 0.5)

    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="k",)

    # Fixed: Added parent_hashes=() to pass the updated constructor signature
    empty_tx = Transaction(addresses=addresses,
                           tx_data="",
                           metadata=TransactionMetadata(),
                           parent_hashes=())

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        dag_mock, sat_node_mock, empty_tx, {}
    )

    assert consensus_reached is False
    sat_node_mock.rep_manager.apply_negative.assert_called_once()
    dag_mock.add_tx.assert_not_called()


def test_poise_no_bft_quorum(consensus_mech) -> None:
    """
    Test PoISE when BFT quorum is not met.
    """
    # 1. Mock the DAG to return False for quorum gate checking
    dag_mock = MagicMock()
    dag_mock.has_bft_quorum.return_value = False

    sat_node_mock = MagicMock()
    sat_node_mock.id = 1
    sat_node_mock.reputation = 0.5

    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2,
                                   r_vector=[1.0, 2.0, 3.0], v_vector=[0.1, 0.2, 0.3])
    tx_data = json.dumps(obs_record.__dict__)
    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="k",)

    # Fixed: Added parent_hashes=() to pass the updated constructor signature
    tx = Transaction(addresses=addresses,
                     tx_data=tx_data,
                     metadata=TransactionMetadata(),
                     parent_hashes=())

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        dag_mock, sat_node_mock, tx, {}
    )

    assert consensus_reached is False
    dag_mock.add_tx.assert_called_once_with(tx)

    # Ensure reputation remains unchanged because non-quorum transactions are pending, not malicious
    sat_node_mock.rep_manager.apply_positive.assert_not_called()
    sat_node_mock.rep_manager.apply_negative.assert_not_called()


@patch('src.consensus_mech.chi2')
def test_poise_consensus_reached(mock_chi2, consensus_mech) -> None:
    """
    Test a successful consensus scenario in PoISE using local consensus state storage.

    Args:
    - mock_chi2 (MagicMock): Mocked scipy chi2 distribution object.
    - consensus_mech (ConsensusMechanism): The consensus mechanism engine instance under test.

    Returns:
    - None. Assertions validate local ledger mutations and positive reputation updates.
    """
    # Mock chi2 to ensure NIS is within bounds
    mock_chi2.ppf.side_effect = [0.1, 5.0] # lower, upper bounds

    # Initialise explicit mocks to prevent attribute lookup errors
    dag_mock = MagicMock()
    dag_mock.local_consensus_states = {}
    dag_mock.has_bft_quorum.return_value = True  # Avoid dropping out at quorum validation

    sat_node_mock = MagicMock()
    sat_node_mock.id = 1
    sat_node_mock.reputation = 0.5

    # Configure the mock reputation manager to safely return unpacking values
    sat_node_mock.rep_manager.decay.return_value = 0.5
    sat_node_mock.rep_manager.apply_positive.return_value = (0.6, 0, 0.5)

    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2,
                                   r_vector=[1.0, 2.0, 3.0], v_vector=[0.1, 0.2, 0.3])
    tx_data = json.dumps(obs_record.__dict__)
    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="k",)

    # Fixed: Added parent_hashes=() to pass strict constructor verification
    tx = Transaction(addresses=addresses,
                     tx_data=tx_data,
                     metadata=TransactionMetadata(),
                     parent_hashes=())

    # Make consensus score high to ensure it passes
    consensus_mech.calculate_consensus_score = MagicMock(return_value=0.8)

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        dag_mock, sat_node_mock, tx, {}
    )

    # Assert tracking decisions are written locally to the evaluating node's dictionary
    assert consensus_reached is True
    state = dag_mock.local_consensus_states.get(tx.hash, {})
    assert state.get("is_confirmed") is True
    assert state.get("is_rejected") is False
    sat_node_mock.rep_manager.apply_positive.assert_called_once()
    sat_node_mock.rep_manager.apply_negative.assert_not_called()


@patch('src.consensus_mech.chi2')
def test_poise_consensus_failed(mock_chi2, consensus_mech) -> None:
    """
    Test a failed consensus scenario in PoISE using local consensus state storage.

    Args:
    - mock_chi2 (MagicMock): Mocked scipy chi2 distribution object.
    - consensus_mech (ConsensusMechanism): The consensus mechanism engine instance under test.

    Returns:
    - None. Assertions validate local ledger mutations and negative reputation penalties.
    """
    # Mock chi2 to ensure NIS is outside bounds
    mock_chi2.ppf.side_effect = [0.1, 5.0]

    dag_mock = MagicMock()
    dag_mock.local_consensus_states = {}
    dag_mock.has_bft_quorum.return_value = True

    sat_node_mock = MagicMock()
    sat_node_mock.id = 1
    sat_node_mock.reputation = 0.5

    # Configure the mock reputation manager to safely return unpacking values
    sat_node_mock.rep_manager.decay.return_value = 0.5
    sat_node_mock.rep_manager.apply_negative.return_value = (0.4, 0, 0.5)

    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=10.0, dof=2,
                                   r_vector=[1.0, 2.0, 3.0], v_vector=[0.1, 0.2, 0.3])
    tx_data = json.dumps(obs_record.__dict__)
    addresses = TransactionAddresses(sender_address=1,
                                     recipient_address=2,
                                     sender_private_key="k",)

    # Fixed: Added parent_hashes=() to pass strict constructor verification
    tx = Transaction(addresses=addresses,
                     tx_data=tx_data,
                     metadata=TransactionMetadata(),
                     parent_hashes=())

    # Make consensus score low to ensure it fails
    consensus_mech.calculate_consensus_score = MagicMock(return_value=0.2)

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        dag_mock, sat_node_mock, tx, {}
    )

    assert consensus_reached is False
    state = dag_mock.local_consensus_states.get(tx.hash, {})
    assert state.get("is_confirmed") is False
    assert state.get("is_rejected") is True
    sat_node_mock.rep_manager.apply_positive.assert_not_called()
    sat_node_mock.rep_manager.apply_negative.assert_called_once()
