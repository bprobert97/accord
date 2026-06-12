
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
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from filterpy.kalman import ExtendedKalmanFilter
from src.logger import get_logger
from src.simulation import generate_random_keplerian_elements, \
    keplerian_to_cartesian, generate_walker_delta_constellation, \
    MU_EARTH, RE

logger = get_logger()

# ----------------------- Constants -----------------------
STATE_DIM = 6 # State vector dimension (position and velocity)
POS_VEL_DIM = 3 # Position or velocity dimension
MAX_ISL_RANGE = 5000e3 # Maximum range for ISL observation records (5000km)

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
    ISL_range_m: float
    N: int = 10
    steps: int = 3000
    dt: float = 60.0
    sig_r: float = 10.0
    sig_rdot: float = 0.02
    q_acc_target: float = 1e-6
    seed: int = 42

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

@dataclass
class ObservationPair:
    """
    A dataclass to store information about an observation pair for processing.
    """
    i: int
    j: int
    yij: np.ndarray
    diag_R_ij: np.ndarray

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
    r = np.maximum(r, 1e-8)
    vrel = vt - vo
    rdot = float(rho.dot(vrel) / r)
    return np.array([r, rdot])

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
def simulate_truth_and_meas(config: FilterConfig,
                            walker_delta: bool = False
                            ) -> tuple[NDArray[np.float64],
                                       NDArray[np.float64]]:
    """
    Simulates the true satellite trajectories and generates noisy
    inter-satellite measurements.

    Args:
    - config: FilterConfig object with simulation parameters.
    - walker_delta: If True, generates a Walker Delta constellation
      instead of random orbits.

    Returns:
    - A tuple containing:
        - truth: The history of true stacked state vectors.
        - z_hist: The history of noisy stacked measurements.
    """

    # 1. Generate Initial States
    x0_stack = _generate_initial_states(config, walker_delta)

    # 2. Propagate Truth
    logger.info("Propagating truth states")
    truth = propagate_truth_kepler(x0_stack, config.steps, config.dt)

    # 3. Generate Measurements
    logger.info("Generating noisy inter-satellite measurements")
    z_hist = _generate_noisy_measurements(config, truth)

    return truth, z_hist


def _generate_initial_states(config: FilterConfig,
                             walker_delta: bool) -> NDArray[np.float64]:
    """
    Helper to generate the initial Cartesian
    state vectors for the constellation.

    Args:
    - config: FilterConfig object with simulation parameters.
    - walker_delta: If True, generates a Walker Delta constellation
      instead of random orbits.

    Returns:
    - An array of the initial cartesian state vectors.

    """
    x0 = []
    if walker_delta:
        logger.info("Generating walker_delta satellite constellation \
                    with %s satellites", config.N)
        # Generate elements and convert to Cartesian
        elements = generate_walker_delta_constellation(
            t=config.N, p=5, f=1, a=RE+500e3, i=np.radians(53)
        )
        # Sort for deterministic node ordering (e.g. by RAAN then True Anomaly)
        elements.sort(key=lambda x: (x.raan, x.ta))

        for el in elements:
            x0.append(keplerian_to_cartesian(el))
    else:
        logger.info("Generating random satellite constellation with %s satellites",
                    config.N)
        for n in range(config.N):
            kep_elements = generate_random_keplerian_elements(seed=config.seed + n)
            x0.append(keplerian_to_cartesian(kep_elements))

    return np.concatenate(x0)


def _generate_noisy_measurements(config: FilterConfig,
                                 truth: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Helper to compute noisy inter-satellite measurements from the truth states.

    Args:
    - config: FilterConfig object with simulation parameters.
    - truth: The history of true stacked state vectors.

    Returns:
    - z_hist: The history of noisy stacked measurements.
    """
    # M represents the total number of unique directional inter-satellite links.
    # Every satellite observes every other satellite exactly once per step.
    M = config.N * (config.N - 1)

    # Pre-allocate the history array.
    # Size is 2 * M because each observation produces two values: Range and Range-Rate.
    z_hist = np.zeros((config.steps, 2 * M))

    for k in range(config.steps):
        # Extract the stacked state vector for all satellites at the current time step 'k'
        xk = truth[k]
        z = []

        # Iterate through every satellite acting as the observer
        for i in range(config.N):
            # Extract the 6-element state vector (3D position, 3D velocity) for observer 'i'
            xi = xk[6*i:6*i+6]

            # Iterate through every satellite acting as the target
            for j in range(config.N):
                # Satellites do not measure themselves
                if i != j:
                    # Calculate the theoretical, noise-free measurement (hx_block)
                    # using the target's state and the observer's state
                    z.append(hx_block(xk[6*j:6*j+6], xi))

        # Flatten the list of arrays into a single continuous 1D array for this step
        z_true = np.concatenate(z)

        # Prepare an array to hold the generated Gaussian noise
        noise = np.zeros(2 * M)

        # Range measurements are at even indices (0, 2, 4...).
        # Apply standard deviation config.sig_r
        noise[0::2] = np.random.normal(0, config.sig_r, M)

        # Range-rate (velocity) measurements are at odd indices (1, 3, 5...).
        # Apply standard deviation config.sig_rdot
        noise[1::2] = np.random.normal(0, config.sig_rdot, M)

        # Corrupt the true measurements with the generated noise and store in history
        z_hist[k] = z_true + noise

    return z_hist


def check_line_of_sight(r_obs: np.ndarray,
                        r_tgt: np.ndarray) -> bool:
    """
    Determines if a line-of-sight vector between two satellites is clear
    or occluded by the spherical body of the Earth.

    Args:
    - r_obs (np.ndarray): 3D position vector of the observer
                          satellite [x, y, z] in meters.
    - r_tgt (np.ndarray): 3D position vector of the target
                          satellite [x, y, z] in meters.

    Returns:
    - bool: True if line-of-sight is clear, False if occluded by the Earth.
    """
    # Vector pointing from observer to target
    rho = r_tgt - r_obs
    rho_norm = np.linalg.norm(rho)
    if rho_norm == 0:
        return False

    # Normalised line of sight vector
    u = rho / rho_norm

    # Calculate projection parameter of the Earth's centre onto the LOS line segment
    # t represents the fraction along the segment from observer to target
    t = -np.dot(r_obs, u) / rho_norm

    # If the closest point to the Earth's centere lies between the two satellites
    if 0.0 <= t <= 1.0:
        # Compute the minimum distance vector from the Earth's centre to the chord segment
        closest_point = r_obs + t * rho
        min_dist = np.linalg.norm(closest_point)

        # If the minimum distance is less than Earth's radius, the link is blocked
        if min_dist < RE:
            return False

    return True

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
    - Inside obs_pair:
        - i: The integer ID (index) of the observing satellite.
        - j: The integer ID (index) of the target satellite.
        - yij: The innovation vector (measurement residual) for this specific pair.
        - diag_R_ij: The diagonal measurement noise covariance matrix for this pair.
    - ekf: The ExtendedKalmanFilter instance containing the current state and covariance.
    - k: The current simulation time step.
    - config: The FilterConfig object containing simulation parameters (e.g., dt, sig_r).

    Returns:
    - An ObservationRecord containing the NIS, DOF, and normalized line-of-sight
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
        return _log_nis(y, self.ekf, k, self.config)


def apply_network_faults(obs_to_submit: ObservationRecord, sid: int, n_sats: int,
                         k: int, faulty_ids: set) -> None:
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
