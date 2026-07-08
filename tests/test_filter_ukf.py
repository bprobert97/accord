"""
Unit and integration tests for the JointUKF class and related functions in src/filters/ukf.py.
"""
import numpy as np
import pytest
from src.filters.ukf import (
    JointUKF,
    fx_joint,
    get_q_matrix,
    robust_cholesky
)
from src.filters.filter_interface import (
    simulate_truth_and_meas,
    STATE_DIM,
    FilterConfig,
    hx_joint
)
from .utilities import validate_observation_records

@pytest.fixture
def filter_config():
    """Provides a default FilterConfig for tests, using a small constellation."""
    # Use a small N and few steps for speed
    return FilterConfig(N=3, steps=2, dt=1.0, seed=42, ISL_range_m=1000e3)

@pytest.fixture
def ukf(filter_config: FilterConfig):
    """Provides a JointUKF instance for tests."""
    # To initialise the UKF, we need an initial truth state.
    initial_truth, _ = simulate_truth_and_meas(config=filter_config)
    return JointUKF(config=filter_config, initial_truth=initial_truth[0])

def test_fx_joint():
    """
    Test the joint state transition function.
    """
    N = 2
    # Fix: Initialize satellites at a valid position (e.g., RE) to avoid division by zero
    x = np.zeros(STATE_DIM * N)
    x[0] = 7000e3  # Sat 1 x-pos
    x[6] = 7000e3  # Sat 2 x-pos
    # Inject velocity
    x[3] = 1000.0
    x[9] = -1000.0

    dt = 1.0
    x_next = fx_joint(x, dt, N)

    assert x_next.shape == (STATE_DIM * N,)
    # Now this will be a valid number, not NaN
    assert x_next[0] > 0

def test_get_q_matrix():
    """
    Test the process noise covariance matrix generation.
    """
    dt = 1.0
    q_acc = 1e-6
    N = 3

    Q = get_q_matrix(dt, q_acc, N)

    assert Q.shape == (STATE_DIM * N, STATE_DIM * N)
    # The matrix must be strictly symmetric
    assert np.allclose(Q, Q.T)
    # The matrix must be positive semi-definite (diagonal elements positive)
    assert np.all(np.diag(Q) > 0)

def test_robust_cholesky():
    """
    Test the robust Cholesky decomposition algorithm with varying matrix health.
    """
    # 1. Standard positive definite matrix
    P_good = np.array([[2.0, 0.5], [0.5, 2.0]])
    L_good = robust_cholesky(P_good)
    assert np.allclose(L_good.T @ L_good, P_good)

    # 2. Singular/highly ill-conditioned matrix (triggers SVD/Jitter fallback)
    P_bad = np.array([[1e-15, 1.0], [1.0, 1e-15]])
    L_bad = robust_cholesky(P_bad)
    # Should safely return a 2x2 matrix without raising LinAlgError
    assert L_bad.shape == (2, 2)

def test_joint_ukf_init(ukf, filter_config):
    """
    Test the initialisation parameters of the JointUKF class.
    """
    N = filter_config.N
    dim_x = STATE_DIM * N
    dim_z = 2 * N * (N - 1)

    assert ukf.config == filter_config
    assert ukf.ukf.x.shape == (dim_x,)
    assert ukf.ukf.P.shape == (dim_x, dim_x)
    assert ukf.ukf.R.shape == (dim_z, dim_z)
    assert np.all(np.diag(ukf.ukf.P) > 0) # Covariance should be positive definite

def test_joint_ukf_predict(ukf):
    """
    Test the predict step of the JointUKF.
    """
    x_prior = ukf.ukf.x.copy()
    P_prior = ukf.ukf.P.copy()

    ukf.predict()

    # State should have moved forward in time
    assert not np.allclose(x_prior, ukf.ukf.x)
    # Covariance should have expanded due to process noise
    assert not np.allclose(P_prior, ukf.ukf.P)
    # The covariance matrix must remain symmetric
    assert np.allclose(ukf.ukf.P, ukf.ukf.P.T)

def test_joint_ukf_update(ukf, filter_config):
    """
    Test the update step of the JointUKF and the correct logging of NIS records.
    """
    x_prior = ukf.ukf.x.copy()

    # 1. Force the filter to trust measurements more by lowering R
    ukf.ukf.R = np.eye(ukf.ukf._dim_z) * 1e-6  # pylint: disable=protected-access

    # 2. Inject a much larger measurement shift
    z_k = hx_joint(x_prior, filter_config.N) + 5000.0

    obs_records = ukf.update(z_k, k=1)
    # State should have corrected based on the simulated measurements
    assert not np.allclose(x_prior, ukf.ukf.x)

    # Check the observation records and their structures
    validate_observation_records(obs_records, filter_config.N, expected_step=1)
