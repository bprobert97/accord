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
This file defines the shared interfaces, configurations, and core physical
models used by the orbit determination filters (EKF and UKF) in the ACCORD framework.
It includes Keplerian orbital dynamics propagation, Runge-Kutta integration,
inter-satellite line-of-sight measurement generation, physical occlusion checks
(Earth blockage), and network fault injection for testing consensus resilience.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
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
    Configuration parameters for the orbital filter simulations.

    Attributes:
    - ISL_range_m: The range for inter-satellite link observations in metres.
    - N: Number of satellites in the constellation.
    - steps: Number of simulation steps.
    - dt: Time step size in seconds.
    - sig_r: Standard deviation of range measurement noise in metres.
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
    walker_delta: bool = False


@dataclass
class ObservationRecord:
    """
    Represents a single observation record, typically used for NIS logging.

    Attributes:
    - step: The simulation step at which the observation was made.
    - time: The time of the observation.
    - observer: The ID of the observing satellite.
    - target: The ID of the target satellite.
    - nis: The normalised Innovation Squared value for this observation.
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

    Attributes:
    - i: The integer ID (index) of the observing satellite.
    - j: The integer ID (index) of the target satellite.
    - yij: The innovation vector (measurement residual) for this specific pair.
    - diag_R_ij: The diagonal measurement noise covariance matrix for this pair.
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
    Helper to generate the initial Cartesian state vectors for the constellation.

    Args:
    - config: FilterConfig object with simulation parameters.
    - walker_delta: If True, generates a Walker Delta constellation
      instead of random orbits.

    Returns:
    - An array of the initial Cartesian state vectors.
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
    - The history of noisy stacked measurements.
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
    - r_obs: 3D position vector of the observer satellite [x, y, z] in metres.
    - r_tgt: 3D position vector of the target satellite [x, y, z] in metres.

    Returns:
    - True if line-of-sight is clear, False if occluded by the Earth.
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

    # If the closest point to the Earth's centre lies between the two satellites
    if 0.0 <= t <= 1.0:
        # Compute the minimum distance vector from the Earth's centre to the chord segment
        closest_point = r_obs + t * rho
        min_dist = np.linalg.norm(closest_point)

        # If the minimum distance is less than Earth's radius, the link is blocked
        if min_dist < RE:
            return False

    return True

def initialise_state_and_cov(N: int, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def apply_network_faults(obs_to_submit: ObservationRecord, sid: int, n_sats: int,
                         k: int, faulty_ids: set) -> None:
    """
    Injects deterministic faulty NIS values based on satellite ID for testing.

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
