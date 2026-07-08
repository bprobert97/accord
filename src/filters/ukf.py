"""
The Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework.
Author: Beth Probert
Email: beth.probert@strath.ac.uk

Copyright (C) 2025 Applied Space Technology Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

"""
This file implements the Unscented Kalman Filter (UKF) for the ACCORD framework.
It provides a joint state estimation architecture (JointUKF) that tracks the
orbital states (position and velocity) of a constellation of satellites. The filter
uses a continuous white noise acceleration kinematic model and robust Cholesky
decomposition to handle the numerical scaling challenges inherent in orbital mechanics,
updating the network's joint state based on inter-satellite line-of-sight measurements.
"""
from typing import List
import numpy as np
from scipy import linalg
from numpy.typing import NDArray
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from src.filters.filter_interface import (
    FilterConfig,
    ObservationRecord,
    STATE_DIM,
    MAX_ISL_RANGE,
    rk4_step,
    hx_joint,
    check_line_of_sight,
    initialise_state_and_cov)
from src.logger import get_logger

logger = get_logger()

def fx_joint(x: NDArray[np.float64], dt: float, N: int) -> NDArray[np.float64]:
    """
    State transition function for the Unscented Transform.
    Propagates the joint state vector forward in time.

    Args:
    - x: The joint state vector of all N satellites.
    - dt: Time step.
    - N: Number of satellites.

    Returns:
    - The predicted joint state vector.
    """
    x_next = np.empty_like(x)
    for i in range(N):
        x_next[STATE_DIM*i:STATE_DIM*i+STATE_DIM] = rk4_step(x[STATE_DIM*i:STATE_DIM*i+STATE_DIM], dt)
    return x_next

def get_q_matrix(dt: float, q_acc: float, N: int) -> NDArray[np.float64]:
    """
    Constructs a block-diagonal discrete-time process noise covariance matrix 
    using a continuous white noise acceleration kinematic approximation.

    Args:
    - dt: Time step.
    - q_acc: Continuous-time process noise acceleration magnitude.
    - N: Number of satellites.

    Returns:
    - A block-diagonal Q matrix for the joint system.
    """
    Q_block = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    dt2 = dt * dt
    dt3 = dt2 * dt

    Q_block[0:3, 0:3] = np.eye(3) * (q_acc * dt3 / 3.0)
    Q_block[0:3, 3:6] = np.eye(3) * (q_acc * dt2 / 2.0)
    Q_block[3:6, 0:3] = np.eye(3) * (q_acc * dt2 / 2.0)
    Q_block[3:6, 3:6] = np.eye(3) * (q_acc * dt)

    return np.kron(np.eye(N, dtype=np.float64), Q_block).astype(np.float64)


# ----------------------- UKF ------------------------------

def _log_nis_ukf(y: np.ndarray, S: np.ndarray, x: np.ndarray, 
                 k: int, config: FilterConfig) -> List[ObservationRecord]:
    """
    Calculates and logs the Normalised Innovation Squared (NIS) for each observation
    by extracting the marginal covariance directly from the UKF's joint S matrix.

    Args:
    - y: The innovation (residual) vector.
    - S: The joint system innovation covariance matrix.
    - x: The current joint state vector.
    - k: The current time step index.
    - config: The configuration object containing filter parameters.

    Returns:
    - A list of ObservationRecord objects containing the NIS evaluations.
    """
    obs_records = []
    idx = 0

    for i in range(config.N):
        for j in range(config.N):
            if i == j:
                continue

            rho = x[STATE_DIM*j : STATE_DIM*j+3] - x[STATE_DIM*i : STATE_DIM*i+3]
            r = np.linalg.norm(rho)

            if r <= MAX_ISL_RANGE and check_line_of_sight(
                x[STATE_DIM*i : STATE_DIM*i+3], x[STATE_DIM*j : STATE_DIM*j+3]):

                yij = y[idx:idx+2]
                # Slice the exact 2x2 marginal uncertainty block for this link
                S_ij = S[idx:idx+2, idx:idx+2]

                nis = float(yij.T @ np.linalg.inv(S_ij) @ yij)
                vrel = x[STATE_DIM*j+3 : STATE_DIM*j+6] - x[STATE_DIM*i+3 : STATE_DIM*i+6]

                obs_records.append(ObservationRecord(
                    step=k, observer=i, target=j,
                    nis=nis, dof=2, time=k*config.dt,
                    r_vector=(rho / np.maximum(r, 1e-8)).tolist(),
                    v_vector=(vrel / np.maximum(np.linalg.norm(vrel), 1e-8)).tolist()
                ))

            idx += 2

    return obs_records

def robust_cholesky(P: np.ndarray) -> np.ndarray:
    """
    A robust matrix square root function to replace the default Cholesky decomposition.
    Handles the massive numerical scaling disparities inherent in orbital mechanics.

    Args:
    - P: The positive semi-definite covariance matrix to decompose.

    Returns:
    - The upper-triangular matrix resulting from the decomposition.
    """
    # 1. Enforce strict symmetry
    P = 0.5 * (P + P.T)

    try:
        # 2. Attempt standard scipy upper-triangular Cholesky
        return linalg.cholesky(P, lower=False)
    except linalg.LinAlgError:
        # 3. Fallback 1: Dynamic Jitter
        # Scale the jitter relative to the magnitude of the diagonal elements
        jitter = np.maximum(np.diag(P) * 1e-5, 1e-8)
        try:
            return linalg.cholesky(P + np.diag(jitter), lower=False)
        except linalg.LinAlgError:
            # 4. Fallback 2: Singular Value Decomposition (SVD)
            # Guaranteed to compute a square root for any positive semi-definite matrix
            _, s, Vh = linalg.svd(P)
            return np.diag(np.sqrt(np.maximum(s, 0.0))) @ Vh

class JointUKF:
    """
    A class to manage the state and operations of a joint Unscented Kalman Filter.
    """
    def __init__(self, config: FilterConfig, initial_truth: np.ndarray):
        """
        Initialises the JointUKF.

        Args:
        - config: The configuration for the filter.
        - initial_truth: The initial true state of the satellites.

        Returns:
        - None.
        """
        self.config = config
        dim_x = STATE_DIM * config.N
        M = config.N * (config.N - 1)
        dim_z = 2 * M

        # Generate standard scaling parameters for orbital regimes, passing our custom robust solver
        points = MerweScaledSigmaPoints(n=dim_x, alpha=1e-3, beta=2., kappa=0., sqrt_method=robust_cholesky) # type: ignore

        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=config.dt,
                                         fx=fx_joint, hx=hx_joint, points=points) # type: ignore

        x0_est, P0 = initialise_state_and_cov(config.N, initial_truth[np.newaxis, :])
        self.ukf.x = x0_est
        self.ukf.P = P0
        self.ukf.R = np.diag([config.sig_r**2, config.sig_rdot**2] * M)
        self.ukf.Q = get_q_matrix(config.dt, config.q_acc_target, config.N)

    def predict(self) -> None:
        """
        Performs the prediction step of the UKF by passing joint kwargs to fx.

        Args:
        - None.

        Returns:
        - None.
        """
        self.ukf.predict(N=self.config.N) # type: ignore

    def update(self, z_k: np.ndarray, k: int) -> List[ObservationRecord]:
        """
        Performs the update step of the UKF and returns observation records.

        Args:
        - z_k: The measurement vector for the current step.
        - k: The current step index.

        Returns:
        - A list of ObservationRecord objects for the current step.
        """
        self.ukf.update(z_k, N=self.config.N)
        return _log_nis_ukf(self.ukf.y, self.ukf.S, self.ukf.x, k, self.config)
