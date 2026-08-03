"""
Unit and integration tests for the JointCKF class in src/filters/ckf.py.
"""
import numpy as np
import pytest
from filterpy.kalman import JulierSigmaPoints
from src.filters.ckf import JointCKF
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
def ckf(filter_config: FilterConfig):
    """Provides a JointCKF instance for tests."""
    # To initialise the CKF, we need an initial truth state.
    initial_truth, _ = simulate_truth_and_meas(config=filter_config)
    return JointCKF(config=filter_config, initial_truth=initial_truth[0])

def test_joint_ckf_init(ckf, filter_config):
    """
    Test the initialisation parameters of the JointCKF class.
    """
    N = filter_config.N
    dim_x = STATE_DIM * N
    dim_z = 2 * N * (N - 1)

    assert ckf.config == filter_config
    assert ckf.ckf.x.shape == (dim_x,)
    assert ckf.ckf.P.shape == (dim_x, dim_x)
    assert ckf.ckf.R.shape == (dim_z, dim_z)
    assert np.all(np.diag(ckf.ckf.P) > 0)  # Covariance should be positive definite

    # Verify the specific sigma point constraints for the Cubature rule
    assert isinstance(ckf.ckf.points_fn, JulierSigmaPoints)
    assert ckf.ckf.points_fn.kappa == 0.0

def test_joint_ckf_predict(ckf):
    """
    Test the predict step of the JointCKF.
    """
    x_prior = ckf.ckf.x.copy()
    P_prior = ckf.ckf.P.copy()

    ckf.predict()

    # State should have moved forward in time
    assert not np.allclose(x_prior, ckf.ckf.x)
    # Covariance should have expanded due to process noise
    assert not np.allclose(P_prior, ckf.ckf.P)
    # The covariance matrix must remain symmetric
    assert np.allclose(ckf.ckf.P, ckf.ckf.P.T)

def test_joint_ckf_update(ckf, filter_config):
    """
    Test the update step of the JointCKF and the correct logging of NIS records.
    """
    # Must predict first to generate and propagate the sigma points!
    ckf.predict()

    # Capture the predicted state to compare against the post-update state
    x_predicted = ckf.ckf.x.copy()

    # Force the filter to trust measurements more by lowering R
    ckf.ckf.R = np.eye(ckf.ckf._dim_z) * 1e-6  # pylint: disable=protected-access

    # Inject a much larger measurement shift relative to the PREDICTED state
    z_k = hx_joint(x_predicted, filter_config.N) + 5000.0

    obs_records = ckf.update(z_k, k=1)

    # State should have corrected based on the simulated measurements
    assert not np.allclose(x_predicted, ckf.ckf.x)

    # Check the observation records and their structures
    validate_observation_records(obs_records, filter_config.N, expected_step=1)
