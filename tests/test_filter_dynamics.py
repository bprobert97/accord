"""
Unit tests for the filter dynamics and measurement model functions.
"""
import numpy as np
from src.filters.ekf import (
    F_jacobian_6,
    van_loan_discretisation,
    H_blocks_target_obs,
)
from src.filters.filter_interface import (
    initialise_state_and_cov,
    rk4_step,
    hx_block,
    two_body_f,
    check_line_of_sight,
    apply_network_faults,
    ObservationRecord,
    STATE_DIM,
    MU_EARTH
)

from src.simulation import RE

def test_two_body_f():
    """
    Test the two-body dynamics function.
    """
    # Satellite in a circular orbit at Re altitude
    x = np.array([RE, 0, 0, 0, np.sqrt(MU_EARTH / RE), 0])
    x_dot = two_body_f(x)

    # Expected acceleration is purely in the -x direction
    expected_a = -MU_EARTH / RE**2

    # Check velocity part of derivative
    assert np.allclose(x_dot[:3], x[3:])
    # Check acceleration part of derivative
    assert np.isclose(x_dot[3], expected_a)
    assert np.isclose(x_dot[4], 0)
    assert np.isclose(x_dot[5], 0)

def test_rk4_step():
    """
    Test the RK4 integration step.
    """
    # Start with satellite at rest far away, it should fall towards Earth
    x = np.array([2 * RE, 0, 0, 0, 0, 0])
    dt = 1.0
    x_next = rk4_step(x, dt)

    # Position should not have changed much in 1s from rest
    assert np.allclose(x_next[:3], x[:3], atol=0.1)
    # Velocity should now be negative in the x-direction (falling)
    assert x_next[3] < 0
    assert np.isclose(x_next[4], 0)
    assert np.isclose(x_next[5], 0)

def test_hx_block():
    """
    Test the measurement model function hx_block.
    """
    # Observer at (1,0,0) looking at target at (2,0,0)
    obs_state = np.array([1, 0, 0, 0, 0, 0])
    target_state = np.array([2, 0, 0, 10, 0, 0]) # Target moving away

    z = hx_block(target_state, obs_state)

    # Expected range is 1
    assert np.isclose(z[0], 1.0)
    # Expected range rate is 10 (target is moving away at 10 m/s)
    assert np.isclose(z[1], 10.0)

    # Test another case: moving towards
    target_state_2 = np.array([2, 0, 0, -5, 0, 0])
    z2 = hx_block(target_state_2, obs_state)
    assert np.isclose(z2[0], 1.0)
    assert np.isclose(z2[1], -5.0)

def test_h_blocks_target_obs():
    """
    Test the measurement Jacobian calculation.
    """
    obs_state = np.array([RE, 0, 0, 0, 0, 0])
    target_state = np.array([RE + 1000, 0, 0, 0, 10, 0])

    Ht, Ho = H_blocks_target_obs(target_state, obs_state)

    # Check shapes
    assert Ht.shape == (2, 6)
    assert Ho.shape == (2, 6)

    # Check some properties. For this simple case (separation along x-axis):
    # Ht[0,0] should be 1 (range change wrt target x-pos)
    # Ho[0,0] should be -1 (range change wrt obs x-pos)
    assert np.isclose(Ht[0, 0], 1.0)
    assert np.isclose(Ho[0, 0], -1.0)
    # Ht[1,4] should be 1 (range-rate change wrt target y-vel)
    # This is because rhat is (1,0,0) and vrel is (0,10,0), so d(rdot)/d(vt_y) is rhat_y which is 0.
    # Let's re-evaluate. rdot = rhat.dot(vrel). H2_t = [d(rdot)/d(pt), d(rdot)/d(vt)].
    # d(rdot)/d(vt) = rhat. So Ht[1, 3:] should be rhat = [1,0,0].
    assert np.isclose(Ht[1, 3], 1.0)
    assert np.isclose(Ht[1, 4], 0.0)

def test_f_jacobian_6():
    """
    Test the dynamics Jacobian matrix calculation.
    This is a simple sanity check. A full validation would require
    numerical differentiation.
    """
    x = np.array([RE, 0, 0, 0, 7600, 0])
    F = F_jacobian_6(x)

    # Check shape
    assert F.shape == (6, 6)

    # Check structure: top-right should be 3x3 identity
    assert np.allclose(F[:3, 3:], np.eye(3))

    # Check structure: top-left should be 3x3 zero
    assert np.all(F[:3, :3] == 0)

    # Check that the bottom-right (d(a)/d(v)) is zero
    assert np.all(F[3:, 3:] == 0)

def test_van_loan_discretisation():
    """
    Test the Van Loan discretisation method.
    """
    dt = 0.1
    F = np.random.randn(6, 6)
    L = np.random.randn(6, 3)
    Qc = np.eye(3)

    Phi, Q = van_loan_discretisation(F, L, Qc, dt)

    # Check shapes
    assert Phi.shape == (6, 6)
    assert Q.shape == (6, 6)

    # Check properties of Q: must be symmetric
    assert np.allclose(Q, Q.T)

def test_initialise_state_and_cov():
    """
    Test the initialisation of state and covariance.
    """
    N = 3
    truth = np.random.randn(1, STATE_DIM * N)
    x0_est, P0 = initialise_state_and_cov(N, truth)

    # Check shapes
    assert x0_est.shape == (STATE_DIM * N,)
    assert P0.shape == (STATE_DIM * N, STATE_DIM * N)

    # Check that P0 is block-diagonal
    for i in range(N):
        for j in range(N):
            if i != j:
                block = P0[STATE_DIM*i:STATE_DIM*(i+1), STATE_DIM*j:STATE_DIM*(j+1)]
                assert np.all(block == 0)

    # Check that diagonal blocks are diagonal matrices
    for i in range(N):
        block = P0[STATE_DIM*i:STATE_DIM*(i+1), STATE_DIM*i:STATE_DIM*(i+1)]
        assert np.count_nonzero(block - np.diag(np.diagonal(block))) == 0

def test_check_line_of_sight():
    """
    Test the line of sight calculation considering Earth blockage.
    """
    # Observer and target close to each other, high altitude
    r_obs = np.array([RE + 1000e3, 0, 0])
    r_tgt = np.array([RE + 1000e3, 1000e3, 0])
    assert check_line_of_sight(r_obs, r_tgt) is True

    # Observer and target on opposite sides of the Earth
    r_obs2 = np.array([RE + 500e3, 0, 0])
    r_tgt2 = np.array([-RE - 500e3, 0, 0])
    assert check_line_of_sight(r_obs2, r_tgt2) is False

def test_apply_network_faults():
    """
    Test the injection of deterministic faults based on satellite ID.
    """
    obs = ObservationRecord(
        step=10, time=100.0, observer=1, target=2,
        nis=1.0, dof=2, r_vector=[1,0,0], v_vector=[0,1,0],
        range_m=1500.0, observer_r_eci=[7000.0, 0.0, 0.0]
    )
    faulty_ids = set()

    # Case 1: sid ends in 1 -> NIS drops to 0.01 (too perfect)
    apply_network_faults(obs, sid=11, n_sats=15, k=250, faulty_ids=faulty_ids)
    assert obs.nis == 0.01
    assert 11 in faulty_ids

    # Case 2: sid ends in 2, n_sats >= 7 -> NIS spikes to 50.0
    obs.nis = 1.0
    apply_network_faults(obs, sid=12, n_sats=10, k=250, faulty_ids=faulty_ids)
    assert obs.nis == 50.0
    assert 12 in faulty_ids

    # Case 3: sid ends in 3, n_sats >= 10, time window 200 <= k < 400
    obs.nis = 3.0
    apply_network_faults(obs, sid=13, n_sats=15, k=300, faulty_ids=faulty_ids)
    assert obs.nis == 30.0 # 3.0 > 2.0, so multiplied by 10
    assert 13 in faulty_ids

    obs.nis = 1.5
    apply_network_faults(obs, sid=23, n_sats=15, k=300, faulty_ids=faulty_ids)
    assert obs.nis == 0.15 # 1.5 < 2.0, so divided by 10
    assert 23 in faulty_ids
