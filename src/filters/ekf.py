
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
This file implements the Extended Kalman Filter (EKF) for the ACCORD framework.
It provides a joint state estimation architecture (JointEKF) to track the
orbital states (position and velocity) of a constellation of satellites.
The filter relies on analytical Jacobian matrices for two-body orbital dynamics
and line-of-sight measurements, utilising the Van Loan method for discretisation
and the Joseph form for numerically stable covariance updates.
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from filterpy.kalman import ExtendedKalmanFilter
from src.filters.filter_interface import (
    FilterConfig,
    ObservationRecord,
    ObservationPair,
    STATE_DIM,
    POS_VEL_DIM,
    MAX_ISL_RANGE,
    two_body_f,
    rk4_step,
    hx_joint,
    check_line_of_sight,
    initialise_state_and_cov)
from src.logger import get_logger
from src.simulation import MU_EARTH

logger = get_logger()


def F_jacobian_6(x6: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates the 6x6 Jacobian matrix (F) for the two-body dynamics.

    Args:
    - x6: The 6-element state vector [px, py, pz, vx, vy, vz].

    Returns:
    - The 6x6 Jacobian matrix F.
    """
    r = x6[:POS_VEL_DIM]
    rn = np.linalg.norm(r)
    I3 = np.eye(POS_VEL_DIM)
    dadr = -MU_EARTH * (I3 / rn**3 - 3*np.outer(r, r)/rn**5)
    F = np.zeros((STATE_DIM,STATE_DIM))
    F[:POS_VEL_DIM,POS_VEL_DIM:] = I3
    F[POS_VEL_DIM:,:POS_VEL_DIM] = dadr
    return F

def van_loan_discretisation(F: NDArray[np.float64],
                            L: NDArray[np.float64],
                            Qc: NDArray[np.float64],
                            dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Discretises continuous-time system and noise matrices using the Van Loan method.

    Args:
    - F: Continuous-time state dynamics matrix.
    - L: Noise gain matrix.
    - Qc: Continuous-time process noise covariance matrix.
    - dt: Time step.

    Returns:
    - A tuple containing:
        - Phi: Discrete-time state transition matrix.
        - Q: Discrete-time process noise covariance matrix.
    """
    n = F.shape[0]
    A = L @ Qc @ L.T
    M = np.block([[F, A], [np.zeros((n,n)), -F.T]]) * dt
    EM = expm(M)
    Phi = EM[:n,:n]
    J = EM[:n,n:]
    Q = Phi @ J
    return Phi, 0.5*(Q + Q.T)

def F_midpoint(x: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """
    Calculates the Jacobian matrix F at the midpoint of the integration step.
    This is used for improved accuracy in the discretisation of the process noise.

    Args:
    - x: The current 6-element state vector.
    - dt: The time step.

    Returns:
    - The 6x6 Jacobian matrix F at the midpoint.
    """
    k1 = two_body_f(x)
    x_mid = x + 0.5 * dt * k1
    Fm = F_jacobian_6(x_mid)
    if not np.isfinite(Fm).all():
        Fm = F_jacobian_6(x)
    return Fm

def H_blocks_target_obs(target: NDArray[np.float64],
                        obs: NDArray[np.float64]
                        ) -> tuple[NDArray[np.float64],
                                  NDArray[np.float64]]:
    """
    Calculates the Jacobian matrices for the measurement function with
    respect to target and observer states.

    Args:
    - target: The 6-element state vector of the target satellite.
    - obs: The 6-element state vector of the observing satellite.

    Returns:
    - A tuple containing:
        - Ht: 2x6 Jacobian matrix with respect to the target state.
        - Ho: 2x6 Jacobian matrix with respect to the observer state.
    """
    rho = target[:POS_VEL_DIM] - obs[:POS_VEL_DIM]
    vrel = target[POS_VEL_DIM:] - obs[POS_VEL_DIM:]

    r = np.maximum(np.linalg.norm(rho), 1e-8)
    rhat = rho / r

    # Compute Target Jacobian blocks
    H1_t = np.hstack([rhat, np.zeros(POS_VEL_DIM)])
    # np.eye(POS_VEL_DIM) is identity matrix with POS_VEL_DIM rows
    d_rdot_d_pt = ((np.eye(POS_VEL_DIM) - np.outer(rhat, rhat)) @ vrel) / r
    H2_t = np.hstack([d_rdot_d_pt, rhat])
    Ht = np.vstack([H1_t, H2_t])

    # Exploit relative navigation symmetry: Ho = -Ht
    return Ht, -Ht

def H_joint(x: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    """
    Calculates the stacked Jacobian matrix for the joint measurement model.

    Args:
    - x: The stacked state vector of all N satellites.
    - N: The number of satellites.

    Returns:
    - The stacked Jacobian matrix H for the joint measurement.
    """
    num_measurement_rows = 2 * N * (N - 1)
    dim_x = STATE_DIM * N
    H = np.zeros((num_measurement_rows, dim_x))
    current_row = 0
    for i in range(N):
        xi = x[STATE_DIM*i:STATE_DIM*i+STATE_DIM]
        for j in range(N):
            if i == j:
                continue
            xj = x[STATE_DIM*j:STATE_DIM*j+STATE_DIM]
            Ht, Ho = H_blocks_target_obs(xj, xi)
            H[current_row:current_row+2, STATE_DIM*j:STATE_DIM*j+STATE_DIM] = Ht
            H[current_row:current_row+2, STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Ho
            current_row += 2
    return H

# ----------------------- EKF predict ----------------------
def ekf_predict_joint(ekf: ExtendedKalmanFilter, dt: float, N: int,
                      q_acc_target: float) -> None:
    """
    Performs the prediction step for the joint Extended Kalman Filter.
    Propagates the state and covariance of all satellites forward in time.

    Args:
    - ekf: The EKF object containing the joint state and covariance.
    - dt: The time step for prediction.
    - N: The number of satellites.
    - q_acc_target: The continuous-time process noise acceleration magnitude for targets.

    Returns:
    - None.
    """
    x_prev = ekf.x.copy()
    dim = STATE_DIM*N

    # propagate state
    x = x_prev.copy()
    for i in range(N):
        x[STATE_DIM*i:STATE_DIM*i+STATE_DIM] = rk4_step(x[STATE_DIM*i:STATE_DIM*i+STATE_DIM], dt)
    ekf.x = x

    # propagate covariance (block-diag)
    Phi = np.eye(dim)
    Qd = np.zeros((dim,dim))
    L = np.zeros((STATE_DIM,POS_VEL_DIM))
    L[POS_VEL_DIM:,:] = np.eye(POS_VEL_DIM)

    for i in range(N):
        Fi = F_midpoint(x_prev[STATE_DIM*i:STATE_DIM*i+STATE_DIM], dt)
        Qci = np.eye(POS_VEL_DIM)*q_acc_target
        Phii, Qdi = van_loan_discretisation(Fi, L, Qci, dt)
        Phi[STATE_DIM*i:STATE_DIM*i+STATE_DIM,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Phii
        Qd [STATE_DIM*i:STATE_DIM*i+STATE_DIM,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Qdi

    ekf.P = Phi @ ekf.P @ Phi.T + Qd
    ekf.P = 0.5*(ekf.P + ekf.P.T)


# ----------------------- EKF ------------------------------
def joseph_update(P: NDArray[np.float64],
                  K: NDArray[np.float64],
                  H: NDArray[np.float64],
                  R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Numerically stable Joseph form of covariance update.

    Args:
    - P: The prior covariance matrix.
    - K: The Kalman gain matrix.
    - H: The measurement Jacobian matrix.
    - R: The measurement noise covariance matrix.

    Returns:
    - The posterior covariance matrix, enforced to be symmetric.
    """
    I = np.eye(P.shape[0])
    A = I - K @ H
    Pn = A @ P @ A.T + K @ R @ K.T
    return 0.5 * (Pn + Pn.T)


def _ekf_update(ekf: ExtendedKalmanFilter, z_k: np.ndarray, N: int) -> np.ndarray:
    """
    Performs the update step of the Extended Kalman Filter.

    Args:
    - ekf: The EKF object.
    - z_k: The current measurement vector.
    - N: The number of satellites.

    Returns:
    - The innovation vector (measurement residual).
    """
    H = H_joint(ekf.x, N)
    z_pred = hx_joint(ekf.x, N)
    y = z_k - z_pred

    S = H @ ekf.P @ H.T + ekf.R
    S_inv = np.linalg.inv(S)
    K = ekf.P @ H.T @ S_inv

    ekf.x = ekf.x + K @ y
    ekf.P = joseph_update(ekf.P, K, H, ekf.R)
    return y

def _log_nis(y: np.ndarray, ekf: ExtendedKalmanFilter, k: int,
             config: FilterConfig) -> List[ObservationRecord]:
    """
    Calculates and logs the Normalised Innovation Squared (NIS) for each observation.

    Args:
    - y: The innovation vector (measurement residual).
    - ekf: The EKF object.
    - k: The current time step.
    - config: The filter configuration.

    Returns:
    - A list of ObservationRecord objects for the current step.
    """
    obs_records = []
    idx = 0

    for i in range(config.N): # observer
        for j in range(config.N): # target
            if i == j:
                continue

            obs_pair = ObservationPair(
                i=i,
                j=j,
                yij=y[idx:idx+2],
                diag_R_ij=np.diag([config.sig_r**2,
                                   config.sig_rdot**2])
                )

            record = _process_observation_pair(obs_pair, ekf, k, config)

            if record is not None:
                obs_records.append(record)

            # Advance index by 2 as every pair is range and range rate
            idx += 2

    return obs_records


def _process_observation_pair(obs_pair: ObservationPair,
                              ekf: ExtendedKalmanFilter,
                              k: int,
                              config: FilterConfig
                              ) -> Optional[ObservationRecord]:
    """
    Helper function to calculate NIS and generate a record for a single satellite pair.

    Args:
    - obs_pair: An object containing the integer IDs of the observing and target satellites,
      the innovation vector for the pair, and the diagonal measurement noise covariance matrix.
    - ekf: The ExtendedKalmanFilter instance containing the current state and covariance.
    - k: The current simulation time step.
    - config: The FilterConfig object containing simulation parameters (e.g., dt, sig_r).

    Returns:
    - An ObservationRecord containing the NIS, DOF, and normalised line-of-sight
      vectors if the target is within ISL range. Returns None if the target is out of range.
    """
    xi_idx = slice(STATE_DIM*obs_pair.i, STATE_DIM*obs_pair.i+STATE_DIM)
    xj_idx = slice(STATE_DIM*obs_pair.j, STATE_DIM*obs_pair.j+STATE_DIM)

    # Get position vectors, rho
    # The vectors are generated by the keplerian_to_cartesian function (ECI frame).
    # rho is a relative position vector. No rotation is applied so this is still ECI.
    rho = ekf.x[STATE_DIM*obs_pair.j : STATE_DIM*obs_pair.j+3] - \
        ekf.x[STATE_DIM*obs_pair.i : STATE_DIM*obs_pair.i+3]
    r = np.linalg.norm(rho)

    # Skip records for satellites further than 5000km apart
    if r > MAX_ISL_RANGE:
        return None

    # If the Earth is in the way, the sensor cannot acquire the measurement.
    if not check_line_of_sight(ekf.x[STATE_DIM*obs_pair.i : STATE_DIM*obs_pair.i+3],
                               ekf.x[STATE_DIM*obs_pair.j : STATE_DIM*obs_pair.j+3]):
        return None

    # vrel is a relative velocity vector
    vrel = ekf.x[STATE_DIM*obs_pair.j+3 : STATE_DIM*obs_pair.j+6] \
        - ekf.x[STATE_DIM*obs_pair.i+3 : STATE_DIM*obs_pair.i+6]

    # Recalculate Ht and Ho for the current pair, this is necessary.
    # These are small 2x6 matrices.
    Ht, Ho = H_blocks_target_obs(ekf.x[xj_idx], ekf.x[xi_idx])

    # Calculate H_ij @ ekf.P @ H_ij.T more efficiently
    # This avoids creating the full H_ij (2 x dim_x) matrix repeatedly
    # Extracting relevant 6x6 blocks from ekf.P directly into the formula
    S_ij = (Ho @ ekf.P[xi_idx, xi_idx] @ Ho.T +
            Ho @ ekf.P[xi_idx, xj_idx] @ Ht.T +
            Ht @ ekf.P[xj_idx, xi_idx] @ Ho.T +
            Ht @ ekf.P[xj_idx, xj_idx] @ Ht.T) + obs_pair.diag_R_ij

    # Add unit LOS position and velocity vectors. Avoid division by zero.
    # rhat and vhat are NDArrays, use tolist() inline to make them json serialisable.
    return ObservationRecord(
        step=k, observer=obs_pair.i, target=obs_pair.j,
        nis=float(obs_pair.yij.T @ np.linalg.inv(S_ij) @ obs_pair.yij),
        dof=obs_pair.yij.shape[0], time=k*config.dt,
        r_vector=(rho / np.maximum(r, 1e-8)).tolist(),
        v_vector=(vrel / np.maximum(np.linalg.norm(vrel), 1e-8)).tolist()
    )

class JointEKF:
    """
    A class to manage the state and operations of a joint Extended Kalman Filter.
    """
    def __init__(self, config: FilterConfig, initial_truth: np.ndarray):
        """
        Initialises the JointEKF.

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

        R = np.diag([config.sig_r**2, config.sig_rdot**2] * M)
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)

        x0_est, P0 = initialise_state_and_cov(config.N, initial_truth[np.newaxis, :])
        self.ekf.x, self.ekf.P, self.ekf.R = x0_est, P0, R

    def predict(self) -> None:
        """
        Performs the prediction step of the EKF.

        Args:
        - None.

        Returns:
        - None.
        """
        ekf_predict_joint(self.ekf, self.config.dt, self.config.N,
                          self.config.q_acc_target)

    def update(self, z_k: np.ndarray, k: int) -> List[ObservationRecord]:
        """
        Performs the update step of the EKF and returns observation records.

        Args:
        - z_k: The measurement vector for the current step.
        - k: The current step index.

        Returns:
        - A list of ObservationRecord objects for the current step.
        """
        y = _ekf_update(self.ekf, z_k, self.config.N)
        return _log_nis(y, self.ekf, k, self.config)
