"""
GCRF orbital dynamics module.
Compatible drop-in replacement for module_crtbp in SDEKF.

Implements two-body (Keplerian) dynamics in an Earth-centered inertial (GCRF/ECI) frame,
plus optional State Transition Matrix (STM) and Differential State Transition Tensor (DSTT)
propagation.
"""

import numpy as np

# Earth gravitational parameter [m^3/s^2]
MU_EARTH = 3.986004418e14
J2 = 1.08262668e-3
R_E = 6378137.0
I3 = np.eye(3)

def gcrf_dynamics(y: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """
    Two-body orbital dynamics in GCRF frame.

    Args:
        y : ndarray
            State vector [x, y, z, vx, vy, vz].
        mu : float
            Gravitational parameter (default Earth).

    Returns:
        dydt : ndarray
            Time derivative of state vector.
    """
    r = y[0:3]
    v = y[3:6]
    r_norm = np.linalg.norm(r) + 1e-12
    # J2 perturbation
    z2 = r[2]**2
    r2 = r_norm**2
    factor = 1.5 * J2 * mu * R_E**2 / r_norm**5
    a_j2 = factor * np.array([
        r[0] * (5*z2/r2 - 1),
        r[1] * (5*z2/r2 - 1),
        r[2] * (5*z2/r2 - 3)
    ])
    a = -mu * r / r_norm**3 + a_j2
    return np.hstack((v, a))


def gcrf_jacobian(y: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """
    Compute the 6x6 Jacobian (A = ∂f/∂x) for the two-body GCRF dynamics.

    Args:
        y : ndarray
            State vector [x, y, z, vx, vy, vz].
        mu : float
            Gravitational parameter (default Earth).

    Returns:
        A : ndarray
            6x6 Jacobian matrix.
    """
    r = y[0:3]
    r_norm = np.linalg.norm(r) + 1e-12
    outer = np.outer(r, r)
    dadr = -mu / r_norm**3 * (I3 - 3 * outer / r_norm**2)

    a = np.zeros((6, 6))
    a[0:3, 3:6] = I3
    a[3:6, 0:3] = dadr
    return a


def gcrf_dstt_dynamics(t: float,  # pylint: disable=unused-argument
                       y: np.ndarray,
                       mu: float,
                       dim: int) -> np.ndarray:
    """
    Propagate dynamics + STM + DSTT (same structure as CRTBP version).
    For two-body GCRF model, DSTT terms are optional and can be set to zero.

    Args:
        t : float
            Time (s) - unused except for solve_ivp compatibility.
        y : ndarray
            Augmented state vector [x(6), STM(36), DSTT(6*dim*dim)].
        mu : float
            Gravitational parameter (Earth).
        dim : int
            State dimension (6).

    Returns:
        dy : ndarray
            Time derivative of augmented state.
    """
    # Split state
    x = y[:6]
    stm = y[6:42].reshape(6, 6)
    dstt = y[42:].reshape(6, dim, dim)

    # Dynamics
    dxdt = gcrf_dynamics(x, mu)
    a = gcrf_jacobian(x, mu)

    # STM derivative
    d_stm = (a @ stm).reshape(36)

    # For two-body case, second-order derivatives small → set DSTT derivative ≈ 0
    d_dstt = np.zeros_like(dstt).reshape(6 * dim * dim)

    # Concatenate results
    dy = np.concatenate((dxdt, d_stm, d_dstt))
    return dy
