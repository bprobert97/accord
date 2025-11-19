# pylint: disable=redefined-outer-name, invalid-name
"""
Unit and integration tests for the JointEKF class and related functions in src/filter.py.
"""
import numpy as np
import pytest
from src.filter import (
    FilterConfig,
    JointEKF,
    simulate_truth_and_meas,
    hx_joint,
    H_joint,
    STATE_DIM
)

@pytest.fixture
def filter_config():
    """Provides a default FilterConfig for tests, using a small constellation."""
    # Use a small N and few steps for speed
    return FilterConfig(N=3, steps=2, dt=1.0, seed=42)

@pytest.fixture
def ekf(filter_config):
    """Provides a JointEKF instance for tests."""
    # To initialise the EKF, we need an initial truth state.
    initial_truth, _ = simulate_truth_and_meas(
        N=filter_config.N,
        steps=1,
        dt=filter_config.dt,
        sig_r=filter_config.sig_r,
        sig_rdot=filter_config.sig_rdot
    )
    return JointEKF(config=filter_config, initial_truth=initial_truth[0])

def test_joint_ekf_init(ekf, filter_config):
    """
    Test the initialization of the JointEKF class.
    """
    N = filter_config.N
    dim_x = STATE_DIM * N
    dim_z = 2 * N * (N - 1)

    assert ekf.config == filter_config
    assert ekf.ekf.dim_x == dim_x
    assert ekf.ekf.dim_z == dim_z
    assert ekf.ekf.x.shape == (dim_x,)
    assert ekf.ekf.P.shape == (dim_x, dim_x)
    assert ekf.ekf.R.shape == (dim_z, dim_z)
    assert np.all(np.diag(ekf.ekf.P) > 0) # Covariance should be positive definite

def test_joint_ekf_predict(ekf):
    """
    Test the predict step of the JointEKF.
    """
    x_prior = ekf.ekf.x.copy()
    P_prior = ekf.ekf.P.copy()

    ekf.predict()

    x_posterior = ekf.ekf.x
    P_posterior = ekf.ekf.P

    # State should have changed
    assert not np.allclose(x_prior, x_posterior)
    # Covariance should also have changed
    assert not np.allclose(P_prior, P_posterior)
    # And should remain symmetric
    assert np.allclose(P_posterior, P_posterior.T)

def test_joint_ekf_update(ekf, filter_config):
    """
    Test the update step of the JointEKF.
    """
    x_prior = ekf.ekf.x.copy()

    # Generate a single measurement vector
    z_k = hx_joint(x_prior, filter_config.N) + np.random.randn(ekf.ekf.dim_z) * 0.1

    obs_records = ekf.update(z_k, k=1)

    x_posterior = ekf.ekf.x

    # State should have been updated (moved from the prior)
    assert not np.allclose(x_prior, x_posterior)

    # Check the observation records
    expected_records = filter_config.N * (filter_config.N - 1)
    assert len(obs_records) == expected_records
    for record in obs_records:
        assert record.step == 1
        assert record.nis >= 0
        assert record.dof == 2

def test_hx_and_h_joint_shapes(filter_config):
    """
    Test the shapes of the outputs of hx_joint and H_joint.
    """
    N = filter_config.N
    dim_x = STATE_DIM * N
    dim_z = 2 * N * (N - 1)

    x = np.random.randn(dim_x)

    z = hx_joint(x, N)
    H = H_joint(x, N)

    assert z.shape == (dim_z,)
    assert H.shape == (dim_z, dim_x)
