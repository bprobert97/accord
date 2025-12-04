# pylint: disable=protected-access, duplicate-code, redefined-outer-name
"""
Unit tests for the ConsensusMechanism class.
"""
import json
from unittest.mock import MagicMock, patch
import pytest
from src.consensus_mech import ConsensusMechanism
from src.reputation import MAX_REPUTATION
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata
from src.filter import ObservationRecord

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
    assert consensus_mech.calculate_dof_score(1) == pytest.approx(1/3)
    assert consensus_mech.calculate_dof_score(2) == pytest.approx(2/3)
    assert consensus_mech.calculate_dof_score(3) == 1.0
    # Should cap at 1.0
    assert consensus_mech.calculate_dof_score(10) == 1.0

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
    # This should be around or below the threshold
    assert score4 < consensus_mech.consensus_threshold

def test_poise_empty_transaction(consensus_mech, mock_dag, mock_sat_node):
    """
    Test PoISE with an empty transaction, expecting reputation penalty.
    """
    empty_tx = Transaction(1, 2, "k", "", TransactionMetadata())

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        mock_dag, mock_sat_node, empty_tx, {}
    )

    assert consensus_reached is False
    mock_sat_node.rep_manager.apply_negative.assert_called_once()
    mock_dag.add_tx.assert_not_called()

def test_poise_no_bft_quorum(consensus_mech, mock_dag, mock_sat_node):
    """
    Test PoISE when BFT quorum is not met.
    """
    mock_dag.has_bft_quorum.return_value = False
    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2)
    tx_data = json.dumps(obs_record.__dict__)
    tx = Transaction(1, 2, "k", tx_data, TransactionMetadata())

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        mock_dag, mock_sat_node, tx, {}
    )

    assert consensus_reached is False
    mock_dag.add_tx.assert_called_once_with(tx)
    # No reputation change should happen
    mock_sat_node.rep_manager.apply_positive.assert_not_called()
    mock_sat_node.rep_manager.apply_negative.assert_not_called()

@patch('src.consensus_mech.chi2')
def test_poise_consensus_reached(mock_chi2, consensus_mech, mock_dag, mock_sat_node):
    """
    Test a successful consensus scenario in PoISE.
    """
    # Mock chi2 to ensure NIS is within bounds
    mock_chi2.ppf.side_effect = [0.1, 5.0] # lower, upper bounds

    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=2.0, dof=2)
    tx_data = json.dumps(obs_record.__dict__)
    tx = Transaction(1, 2, "k", tx_data, TransactionMetadata())

    # Make consensus score high to ensure it passes
    consensus_mech.calculate_consensus_score = MagicMock(return_value=0.8)

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        mock_dag, mock_sat_node, tx, {}
    )

    assert consensus_reached is True
    assert tx.metadata.is_confirmed is True
    assert tx.metadata.is_rejected is False
    mock_sat_node.rep_manager.apply_positive.assert_called_once()
    mock_sat_node.rep_manager.apply_negative.assert_not_called()

@patch('src.consensus_mech.chi2')
def test_poise_consensus_failed(mock_chi2, consensus_mech, mock_dag, mock_sat_node):
    """
    Test a failed consensus scenario in PoISE.
    """
    # Mock chi2 to ensure NIS is outside bounds
    mock_chi2.ppf.side_effect = [0.1, 5.0]

    obs_record = ObservationRecord(step=1, time=1, observer=1, target=2, nis=10.0, dof=2)
    tx_data = json.dumps(obs_record.__dict__)
    tx = Transaction(1, 2, "k", tx_data, TransactionMetadata())

    # Make consensus score low to ensure it fails
    consensus_mech.calculate_consensus_score = MagicMock(return_value=0.4)

    consensus_reached, _ = consensus_mech.proof_of_inter_satellite_evaluation(
        mock_dag, mock_sat_node, tx, {}
    )

    assert consensus_reached is False
    assert tx.metadata.is_confirmed is False
    assert tx.metadata.is_rejected is True
    mock_sat_node.rep_manager.apply_positive.assert_not_called()
    mock_sat_node.rep_manager.apply_negative.assert_called_once()
