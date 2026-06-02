
# pylint: disable= invalid-name, too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-instance-attributes
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

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2
from scipy.linalg import expm
from filterpy.kalman import ExtendedKalmanFilter  # type: ignore
from src.logger import get_logger
from src.simulation import generate_random_keplerian_elements, \
    keplerian_to_cartesian, generate_walker_delta_constellation, \
    MU_EARTH, RE

logger = get_logger()

# ----------------------- Constants -----------------------
STATE_DIM = 6 # State vector dimension (position and velocity)
POS_VEL_DIM = 3 # Position or velocity dimension
MAX_ISL_RANGE = 5000e3 # Maximum range for ISL observation records (5000km)

# ----------------------- Result Types ---------------------
@dataclass
class ObservationRecord:
    """
    Represents a single observation record, typically used for NIS logging.

    Attributes:
    - step: The simulation step at which the observation was made.
    - time: The time of the observation.
    - observer: The ID of the observing satellite.
    - target: The ID of the target satellite.
    - nis: The Normalised Innovation Squared value for this observation.
    - dof: The degrees of freedom for the NIS calculation.
    - r_vector: A LOS position vector used for looking at persistence of excitation.
      Note: This vector is relative from observer to target but the coordinates
      are expressed in the absolute reference frame of the simulation.
    - v_vector: A LOS velocity vector used for looking at persistence of excitation.
    """
    step: int
    time: float
    observer: int
    target: int
    nis: float
    dof: int
    r_vector: List[float]
    v_vector: List[float]

# ----------------------- Dynamics ------------------------
def two_body_f(x6: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates the state derivative for a two-body orbital system.

    Args:
    - x6: The 6-element state vector [px, py, pz, vx, vy, vz].

    Returns:
    - The 6-element state derivative vector [vx, vy, vz, ax, ay, az].
    """
    r = x6[:POS_VEL_DIM]
    v = x6[POS_VEL_DIM:]
    rn = np.linalg.norm(r)
    a = -MU_EARTH * r / rn**3
    return np.hstack([v, a])

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

def rk4_step(x: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """
    Performs one step of Runge-Kutta 4th order integration for two-body dynamics.

    Args:
    - x: The current 6-element state vector [px, py, pz, vx, vy, vz].
    - dt: The time step for integration.

    Returns:
    - The state vector after one integration step.
    """
    k1 = two_body_f(x)
    k2 = two_body_f(x + 0.5*dt*k1)
    k3 = two_body_f(x + 0.5*dt*k2)
    k4 = two_body_f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def van_loan_discretization(F: NDArray[np.float64],
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

# ----------------------- Truth propagation ----------------
def propagate_truth_kepler(x0_stack: NDArray[np.float64],
                           steps: int, dt: float) -> NDArray[np.float64]:
    """
    Propagates the true state of multiple satellites using Keplerian dynamics.

    Args:
    - x0_stack: Initial stacked state vector for all satellites.
    - steps: Number of time steps to propagate.
    - dt: Time step size.

    Returns:
    - A history of the true stacked state vectors over time.
    """
    x = x0_stack.copy()
    hist = np.zeros((steps, x0_stack.size))
    for k in range(steps):
        for s in range(0, x.size, STATE_DIM):
            x[s:s+STATE_DIM] = rk4_step(x[s:s+STATE_DIM], dt)
        hist[k] = x
    return hist

# ----------------------- Measurement model ----------------
def hx_block(target: NDArray[np.float64], obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates the expected range and range-rate measurement between an observer and a target.
    This is what fixes the DOF to 2.

    Args:
    - target: The 6-element state vector of the target satellite.
    - obs: The 6-element state vector of the observing satellite.

    Returns:
    - A 2-element array [range, range_rate].
    """
    pt, vt = target[:POS_VEL_DIM], target[POS_VEL_DIM:]
    po, vo = obs[:POS_VEL_DIM], obs[POS_VEL_DIM:]
    rho = pt - po
    r = np.linalg.norm(rho)
    r = max(r, 1e-8) # type: ignore
    vrel = vt - vo
    rdot = float(rho.dot(vrel) / r)
    return np.array([r, rdot])

def H_blocks_target_obs(target: NDArray[np.float64],
                        obs: NDArray[np.float64]) -> tuple[NDArray[np.float64],
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
    pt, vt = target[:POS_VEL_DIM], target[POS_VEL_DIM:]
    po, vo = obs[:POS_VEL_DIM], obs[POS_VEL_DIM:]
    rho = pt - po
    r = np.linalg.norm(rho)
    r = max(r, 1e-8) # type: ignore
    rhat = rho / r
    I3 = np.eye(POS_VEL_DIM)
    vrel = vt - vo

    H1_t = np.hstack([rhat, np.zeros(POS_VEL_DIM)])
    d_rdot_d_pt = ((I3 - np.outer(rhat, rhat)) @ vrel) / r
    H2_t = np.hstack([d_rdot_d_pt, rhat])
    Ht = np.vstack([H1_t, H2_t])

    H1_o = np.hstack([-rhat, np.zeros(POS_VEL_DIM)])
    d_rdot_d_po = -((I3 - np.outer(rhat, rhat)) @ vrel) / r
    H2_o = np.hstack([d_rdot_d_po, -rhat])
    Ho = np.vstack([H1_o, H2_o])
    return Ht, Ho

def hx_joint(x: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    """
    Calculates the stacked expected measurements for all inter-satellite links.

    Args:
    - x: The stacked state vector of all N satellites.
    - N: The number of satellites.

    Returns:
    - A stacked array of all expected range and range-rate measurements.
    """
    num_measurements = N * (N - 1)
    z = np.empty(2 * num_measurements)
    idx = 0
    for i in range(N):
        xi = x[STATE_DIM*i:STATE_DIM*i+STATE_DIM]
        for j in range(N):
            if i != j:
                z[idx:idx+2] = hx_block(x[STATE_DIM*j:STATE_DIM*j+STATE_DIM], xi)
                idx += 2
    return z

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
        Phii, Qdi = van_loan_discretization(Fi, L, Qci, dt)
        Phi[STATE_DIM*i:STATE_DIM*i+STATE_DIM,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Phii
        Qd [STATE_DIM*i:STATE_DIM*i+STATE_DIM,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Qdi

    ekf.P = Phi @ ekf.P @ Phi.T + Qd
    ekf.P = 0.5*(ekf.P + ekf.P.T)

# ----------------------- Truth + measurement sim ----------
def simulate_truth_and_meas(N: int, steps: int, dt: float,
                            sig_r: float, sig_rdot: float,
                            seed: int,
                            walker_delta: bool = False) -> tuple[NDArray[np.float64],
                                                NDArray[np.float64]]:
    """
    Simulates the true satellite trajectories and generates noisy
    inter-satellite measurements.

    Args:
    - N: The number of satellites.
    - steps: The number of simulation steps.
    - dt: The time step size.
    - sig_r: Standard deviation of range measurement noise.
    - sig_rdot: Standard deviation of range-rate measurement noise.
    - seed: An integer seed for reproducibility.
    - walker_delta: If True, generates a Walker Delta constellation instead of random orbits.

    Returns:
    - A tuple containing:
        - truth: The history of true stacked state vectors.
        - z_hist: The history of noisy stacked measurements.
    """

    x0 = []
    if walker_delta:
        logger.info("Generating walker_delta satellite constellation with %s satellites", N)

        # Generate elements and convert to Cartesian
        elements = generate_walker_delta_constellation(t=N, p=5, f=1, a=RE+500e3, i=np.radians(53))
        # Sort for deterministic node ordering (e.g. by RAAN then True Anomaly)
        elements.sort(key=lambda x: (x[3], x[5]))

        for el in elements:
            state = keplerian_to_cartesian(*el)
            x0.append(state)
        x0_stack = np.concatenate(x0)
    else:
        logger.info("Generating random satellite constellation with %s satellites", N)

        for n in range(N):
            a, e, i, raan, argp, ta = generate_random_keplerian_elements(seed=seed + n)
            state = keplerian_to_cartesian(a, e, i, raan, argp, ta)
            x0.append(state)
        x0_stack = np.concatenate(x0)

    logger.info("Propagating truth states")
    truth = propagate_truth_kepler(x0_stack, steps, dt)

    logger.info("Generating noisy inter-satellite measurements")
    M = N*(N-1)
    z_hist = np.zeros((steps, 2*M))
    for k in range(steps):
        xk = truth[k]
        z = []
        for i in range(N):
            xi = xk[6*i:6*i+6]
            for j in range(N):
                if i != j:
                    z.append(hx_block(xk[6*j:6*j+6], xi))
        z_true = np.concatenate(z)
        noise = np.zeros(2*M)
        noise[0::2] = np.random.normal(0,sig_r, M)
        noise[1::2] = np.random.normal(0,sig_rdot, M)
        z_hist[k] = z_true + noise
    return truth, z_hist

# ----------------------- EKF ------------------------------
def joseph_update(P: NDArray[np.float64], K: NDArray[np.float64],
                  H: NDArray[np.float64], R: NDArray[np.float64]) -> NDArray[np.float64]:
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

def _initialise_state_and_cov(N: int, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialises the estimated state vector and its covariance matrix.

    Args:
    - N: The number of satellites.
    - truth: The true state history, used to derive an initial estimate.

    Returns:
    - A tuple containing:
        - x0_est: The initial estimated state vector.
        - P0: The initial covariance matrix.
    """
    dim_x = STATE_DIM * N
    x0_est = truth[0].copy()
    for i in range(N):
        x0_est[STATE_DIM*i:STATE_DIM*i+STATE_DIM] += np.array([2e3,0,0,0,-10,0])

    P0 = np.zeros((dim_x,dim_x))
    for i in range(N):
        P0[STATE_DIM*i:STATE_DIM*i+STATE_DIM,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = \
            np.diag([1e8]*POS_VEL_DIM+[1e4]*POS_VEL_DIM)
    return x0_est, P0

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

def _log_nis(y: np.ndarray, ekf: ExtendedKalmanFilter, N: int, k: int,
             dt: float, sig_r: float, sig_rdot: float) -> List[ObservationRecord]:
    """
    Calculates and logs the Normalised Innovation Squared (NIS) for each observation.

    Args:
    - y: The innovation vector (measurement residual).
    - ekf: The EKF object.
    - N: The number of satellites.
    - k: The current simulation step.
    - dt: The time step size.
    - sig_r: Standard deviation of range measurement noise.
    - sig_rdot: Standard deviation of range-rate measurement noise.

    Returns:
    - A list of ObservationRecord objects for the current step.
    """
    obs_records = []
    diag_R_ij = np.diag([sig_r**2, sig_rdot**2])
    idx = 0
    for i in range(N): # observer
        xi_idx_slice = slice(STATE_DIM*i, STATE_DIM*i+STATE_DIM)
        for j in range(N): # target
            if i == j:
                continue

            xj_idx_slice = slice(STATE_DIM*j, STATE_DIM*j+STATE_DIM)

            # Get position vectors, rho
            # The po and pt vectors are generated by the keplerian_to_cartesian function
            # which is in an Earth-Centred Interial (ECI) frame.
            po = ekf.x[STATE_DIM*i : STATE_DIM*i+3]
            pt = ekf.x[STATE_DIM*j : STATE_DIM*j+3]

            # rho is a relative position vector
            # No rotation is applied so this is still in the ECI frame
            rho = pt - po

            # Skip records for satellites further than 5000km apart
            r = np.linalg.norm(rho)
            if r > MAX_ISL_RANGE:
                # Advance index by 2 as every pair is range and range rate
                idx += 2
                continue

            # Get velocity vectors
            vo = ekf.x[STATE_DIM*i+3 : STATE_DIM*i+6]
            vt = ekf.x[STATE_DIM*j+3 : STATE_DIM*j+6]

            # vrel is a relative velocity vector
            vrel = vt - vo

            yij = y[idx:idx+2]

            # Recalculate Ht and Ho for the current pair, this is necessary.
            # These are small 2x6 matrices.
            Ht, Ho = H_blocks_target_obs(ekf.x[xj_idx_slice], ekf.x[xi_idx_slice])

            # Extract relevant blocks from ekf.P
            # These are 6x6 matrices
            P_oo = ekf.P[xi_idx_slice, xi_idx_slice]
            P_ot = ekf.P[xi_idx_slice, xj_idx_slice]
            P_to = ekf.P[xj_idx_slice, xi_idx_slice]
            P_tt = ekf.P[xj_idx_slice, xj_idx_slice]

            # Calculate H_ij @ ekf.P @ H_ij.T more efficiently
            # This avoids creating the full H_ij (2 x dim_x) matrix repeatedly
            innovation_covariance_contribution = (
                Ho @ P_oo @ Ho.T +
                Ho @ P_ot @ Ht.T +
                Ht @ P_to @ Ho.T +
                Ht @ P_tt @ Ht.T
            )

            S_ij = innovation_covariance_contribution + diag_R_ij
            S_ij_inv = np.linalg.inv(S_ij) # Changed from pinv to inv
            nis = float(yij.T @ S_ij_inv @ yij)

            # Add unit LOS vector. Avoid division by zero.
            # rhat is an NDArray so use tolist() later to make it json serialisable.
            rhat = rho / (max(r, 1e-8))  # type: ignore [call-overload]

            # Add unit LOS velocity vector. Avoid division by zero.
            # vhat is an NDArray so use tolist() later to make it json serialisable.
            vhat = vrel / (max(np.linalg.norm(vrel), 1e-8))  # type: ignore [call-overload]

            obs_records.append(
                ObservationRecord(
                    step=k, observer=i, target=j, nis=nis, dof=yij.shape[0],
                    time = k*dt, r_vector=rhat.tolist(), v_vector=vhat.tolist()
            ))
            # Advance index by 2 as every pair is range and range rate
            idx += 2
    return obs_records

@dataclass
class FilterConfig:
    """
    Configuration parameters for the Extended Kalman Filter simulation.

    Attributes:
    - ISL_range_m: The range for inter-satellite link observations in metres.
    - N: Number of satellites in the constellation.
    - steps: Number of simulation steps.
    - dt: Time step size in seconds.
    - sig_r: Standard deviation of range measurement noise in meters.
    - sig_rdot: Standard deviation of range-rate measurement noise in m/s.
    - q_acc_target: Continuous-time process noise acceleration magnitude for target satellites.
    - seed: Random seed for reproducibility.
    """
    ISL_range_m: int
    N: int = 10
    steps: int = 3000
    dt: float = 60.0
    sig_r: float = 10.0
    sig_rdot: float = 0.02
    q_acc_target: float = 1e-6
    seed: int = 42


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
        """
        self.config = config
        dim_x = STATE_DIM * config.N
        M = config.N * (config.N - 1)
        dim_z = 2 * M

        R = np.diag([config.sig_r**2, config.sig_rdot**2] * M)
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)

        x0_est, P0 = _initialise_state_and_cov(config.N, initial_truth[np.newaxis, :])
        self.ekf.x, self.ekf.P, self.ekf.R = x0_est, P0, R

    def predict(self) -> None:
        """
        Performs the prediction step of the EKF.
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
        return _log_nis(y, self.ekf, self.config.N, k, self.config.dt,
                        self.config.sig_r, self.config.sig_rdot)


def apply_network_faults(obs_to_submit: ObservationRecord, sid: int, n_sats: int, k: int, faulty_ids: set) -> None:
    """Injects deterministic faulty NIS values based on satellite ID for testing.

    Args:
    - obs_to_submit: The observation record to potentially modify.
    - sid: The satellite ID.
    - n_sats: The total number of satellites in the simulation.
    - k: The current time step or iteration count in the simulation.
    - faulty_ids: A set to keep track of which satellite IDs have been marked as faulty.

    Returns:
    - None. The function modifies obs_to_submit in place and updates faulty_ids.

    """
    if sid % 10 == 1:
        obs_to_submit.nis = 0.01
        faulty_ids.add(sid)
    elif sid % 10 == 2 and n_sats >= 7:
        obs_to_submit.nis = 50.0
        faulty_ids.add(sid)
    elif sid % 10 == 3 and n_sats >= 10:
        faulty_ids.add(sid)
        if 200 <= k < 400:
            if obs_to_submit.nis > 2.0:
                obs_to_submit.nis = obs_to_submit.nis * 10
            else:
                obs_to_submit.nis = obs_to_submit.nis / 10
