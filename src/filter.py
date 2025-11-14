
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
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import expm
from filterpy.kalman import ExtendedKalmanFilter  # type: ignore

# ----------------------- Constants -----------------------
MU_EARTH = 3.986004418e14  # m^3/s^2
Re = 6378e3
STATE_DIM = 6 # State vector dimension (position and velocity)
POS_VEL_DIM = 3 # Position or velocity dimension

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
    - nis: The Normalized Innovation Squared value for this observation.
    - dof: The degrees of freedom for the NIS calculation.
    """
    step: int
    time: float
    observer: int
    target: int
    nis: float
    dof: int

@dataclass
class JointResult:
    """
    Stores the results of a joint EKF simulation.

    Attributes:
    - target_ids: A list of identifiers for the target satellites.
    - obs_records: A list of ObservationRecord objects, containing NIS data for each observation.
    - x_hist: History of the estimated stacked state vectors over time.
    - truth: History of the true stacked state vectors over time.
    - z_hist: History of the noisy stacked measurements over time.
    """
    target_ids: List[str]                # ["sat_1", ...]
    obs_records: List[ObservationRecord] # per-observation NIS records
    x_hist: np.ndarray                   # (steps, 6*N)
    truth: np.ndarray                    # (steps, 6*N)
    z_hist: np.ndarray                   # (steps, 2*N*(N-1))

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
    Discretizes continuous-time system and noise matrices using the Van Loan method.

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
    This is used for improved accuracy in the discretization of the process noise.

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
    z = []
    for i in range(N):
        xi = x[STATE_DIM*i:STATE_DIM*i+STATE_DIM]
        for j in range(N):
            if i != j:
                z.append(hx_block(x[STATE_DIM*j:STATE_DIM*j+STATE_DIM], xi))
    return np.concatenate(z)

def H_joint(x: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    """
    Calculates the stacked Jacobian matrix for the joint measurement model.

    Args:
    - x: The stacked state vector of all N satellites.
    - N: The number of satellites.

    Returns:
    - The stacked Jacobian matrix H for the joint measurement.
    """
    rows = []
    for i in range(N):
        xi = x[STATE_DIM*i:STATE_DIM*i+STATE_DIM]
        for j in range(N):
            if i == j:
                continue
            xj = x[STATE_DIM*j:STATE_DIM*j+STATE_DIM]
            Ht, Ho = H_blocks_target_obs(xj, xi)
            R = np.zeros((2, STATE_DIM*N))
            R[:,STATE_DIM*j:STATE_DIM*j+STATE_DIM] = Ht
            R[:,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Ho
            rows.append(R)
    return np.vstack(rows)

# ----------------------- EKF predict ----------------------
def ekf_predict_joint(ekf: ExtendedKalmanFilter, dt: float, N: int,
                      q_acc_target: float, _unused: float) -> None:
    """
    Performs the prediction step for the joint Extended Kalman Filter.
    Propagates the state and covariance of all satellites forward in time.

    Args:
    - ekf: The EKF object containing the joint state and covariance.
    - dt: The time step for prediction.
    - N: The number of satellites.
    - q_acc_target: The continuous-time process noise acceleration magnitude for targets.
    - _unused: An unused parameter, kept for signature compatibility.
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
                            sig_r: float, sig_rdot: float) -> tuple[NDArray[np.float64],
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

    Returns:
    - A tuple containing:
        - truth: The history of true stacked state vectors.
        - z_hist: The history of noisy stacked measurements.
    """
    base = np.array([Re+500e3, 0,0, 0,7600,0])
    x0 = []
    for i in range(N):
        off = np.array([(i-(N-1)/2)*15e3, 0,0, 0,(i-(N-1)/2)*3,0])
        x0.append(base+off)
    x0_stack = np.concatenate(x0)

    truth = propagate_truth_kepler(x0_stack, steps, dt)

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

def _initialize_state_and_cov(N: int, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the estimated state vector and its covariance matrix.

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
    S_inv = np.linalg.pinv(S)
    K = ekf.P @ H.T @ S_inv

    ekf.x = ekf.x + K @ y
    ekf.P = joseph_update(ekf.P, K, H, ekf.R)
    return y

def _log_nis(y: np.ndarray, ekf: ExtendedKalmanFilter, N: int, k: int,
             dt: float, sig_r: float, sig_rdot: float) -> List[ObservationRecord]:
    """
    Calculates and logs the Normalized Innovation Squared (NIS) for each observation.

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
    dim_x = ekf.x.shape[0]
    idx = 0
    for i in range(N): # observer
        for j in range(N): # target
            if i == j:
                continue

            rows = slice(idx, idx+2)
            yij = y[rows]

            H_ij = np.zeros((2, dim_x))
            Ht, Ho = H_blocks_target_obs(ekf.x[STATE_DIM*j:STATE_DIM*j+STATE_DIM],
                                         ekf.x[STATE_DIM*i:STATE_DIM*i+STATE_DIM])
            H_ij[:,STATE_DIM*j:STATE_DIM*j+STATE_DIM] = Ht
            H_ij[:,STATE_DIM*i:STATE_DIM*i+STATE_DIM] = Ho

            S_ij = H_ij @ ekf.P @ H_ij.T + np.diag([sig_r**2, sig_rdot**2])
            S_ij_inv = np.linalg.pinv(S_ij)
            nis = float(yij.T @ S_ij_inv @ yij)

            obs_records.append(
                ObservationRecord(
                    step=k, observer=i, target=j, nis=nis, dof=yij.shape[0], time = k*dt
                )
            )
            idx += 2
    return obs_records

@dataclass
class FilterConfig:
    """
    Configuration parameters for the Extended Kalman Filter simulation.

    Attributes:
    - N: Number of satellites in the constellation.
    - steps: Number of simulation steps.
    - dt: Time step size in seconds.
    - sig_r: Standard deviation of range measurement noise in meters.
    - sig_rdot: Standard deviation of range-rate measurement noise in m/s.
    - q_acc_target: Continuous-time process noise acceleration magnitude for target satellites.
    - q_acc_obs: Continuous-time process noise acceleration magnitude for
      observer satellites (kept for compatibility).
    - seed: Random seed for reproducibility.
    """
    N: int = 10
    steps: int = 3000
    dt: float = 60.0
    sig_r: float = 10.0
    sig_rdot: float = 0.02
    q_acc_target: float = 1e-6
    q_acc_obs: float = 1e-6
    seed: int | None = 42


class JointEKF:
    """
    A class to manage the state and operations of a joint Extended Kalman Filter.
    """
    def __init__(self, config: FilterConfig, initial_truth: np.ndarray):
        """
        Initializes the JointEKF.

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

        x0_est, P0 = _initialize_state_and_cov(config.N, initial_truth[np.newaxis, :])
        self.ekf.x, self.ekf.P, self.ekf.R = x0_est, P0, R

    def predict(self) -> None:
        """
        Performs the prediction step of the EKF.
        """
        ekf_predict_joint(self.ekf, self.config.dt, self.config.N,
                          self.config.q_acc_target, self.config.q_acc_obs)

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

# ----------------------- Plot Helpers --------------------
def extract_mean_nis_per_sat(result: JointResult) -> list[list[float]]:
    """
    Extracts the mean NIS per satellite per time step from the observation records.

    Args:
    - result: The result object containing observation records.

    Returns:
    - A list of lists, where `mean_nis[sat_idx][step]` is the mean NIS for that satellite
      at that step.
    """
    N = len(result.target_ids)
    steps = result.x_hist.shape[0]

    # Initialize storage
    nis_matrix: List[List[List[float]]] = [[[] for _ in range(steps)] for _ in range(N)]

    # Fill list for each observer,step
    for rec in result.obs_records:
        nis_matrix[rec.observer][rec.step].append(rec.nis)

    # Convert lists of values → mean per step
    nis_mean = []
    for i in range(N):
        sat_means = []
        for t in range(steps):
            vals = nis_matrix[i][t]
            if vals:
                sat_means.append(float(np.mean(vals)))
            else:
                sat_means.append(np.nan)  # should not happen
        nis_mean.append(sat_means)

    return nis_mean

def chi2_bounds(dof: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Calculates the lower and upper bounds for a chi-squared distribution.

    Args:
    - dof: Degrees of freedom for the chi-squared distribution.
    - alpha: The confidence level (e.g., 0.95 for 95% confidence).

    Returns:
    - A tuple containing the lower and upper bounds.
    """
    lo = chi2.ppf((1 - alpha) / 2.0, dof)
    hi = chi2.ppf(1 - (1 - alpha) / 2.0, dof)
    return float(lo), float(hi)

def plot_nis(result: JointResult) -> None:
    """
    Plots the mean Normalized Innovation Squared (NIS) for each satellite over time.
    Includes chi-squared bounds for consistency checking.

    Args:
    - result: The result object containing NIS data.
    """
    nis_per_sat = extract_mean_nis_per_sat(result)
    N = len(result.target_ids)
    dof = 2
    lo, hi = chi2_bounds(dof, 0.95)

    plt.figure(figsize=(11, 4))
    for i in range(N):
        plt.plot(nis_per_sat[i], label=f"Sat {i}")

    plt.axhline(lo, ls=':', label=f"χ² 95% lo={lo:.2f}")
    plt.axhline(hi, ls=':', label=f"χ² 95% hi={hi:.2f}")
    plt.axhline(dof, color='r', ls='--', label=f"DoF = {dof}")

    plt.xlabel("Step")
    plt.ylabel("Mean NIS per sat (over N-1 links)")
    plt.title("Per-Satellite Mean NIS (All-to-All Measurements)")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.show()

def plot_nis_consistency(result: JointResult, dof: int = 2, window: int = 50) -> None:
    """
    Plots the NIS consistency check for each satellite, including
    rolling mean and chi-squared bounds.

    Args:
    - result: The result object containing NIS data.
    - dof: Degrees of freedom for the chi-squared distribution.
    - window: Window size for the rolling mean calculation.
    """
    nis_per_sat = extract_mean_nis_per_sat(result)
    N = len(result.target_ids)

    lower = chi2.ppf(0.025, dof)
    upper = chi2.ppf(0.975, dof)

    steps = len(nis_per_sat[0])
    t = np.arange(steps)

    plt.figure(figsize=(13,4))
    for i in range(N):
        x = np.array(nis_per_sat[i])
        roll = np.convolve(x, np.ones(window)/window, mode='same')
        plt.plot(t, x, alpha=0.35, label=f"Sat {i}")
        plt.plot(t, roll, linewidth=2)

        frac = np.mean((x < lower) | (x > upper))
        status = "⚠️" if frac > 0.10 else "✅"
        print(f"{status} Sat {i}: {frac*100:.1f}% outside limits, mean={np.nanmean(x):.2f}")

    plt.axhline(lower, ls='--', color='gray')
    plt.axhline(upper, ls='--', color='gray')
    plt.axhline(dof, ls='-', color='red', label="DoF")

    plt.title(f"NIS Consistency Check (window={window}, DoF={dof})")
    plt.xlabel("Step")
    plt.ylabel("Mean NIS per sat")
    plt.legend(ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.show()

def run_joint_ekf_simulation(config: FilterConfig) -> JointResult:
    """
    Runs a full joint EKF simulation and returns the results.

    Args:
    - config: The configuration for the simulation.

    Returns:
    - A JointResult object containing the simulation results.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    truth, z_hist = simulate_truth_and_meas(
        config.N, config.steps, config.dt, config.sig_r, config.sig_rdot
    )

    ekf = JointEKF(config, truth[0])

    x_hist = np.zeros((config.steps, STATE_DIM * config.N))
    obs_records: List[ObservationRecord] = []

    for k in range(config.steps):
        ekf.predict()
        obs_records_step = ekf.update(z_hist[k], k)
        obs_records.extend(obs_records_step)
        x_hist[k] = ekf.ekf.x

    return JointResult(
        target_ids=[f"sat_{i+1}" for i in range(config.N)],
        obs_records=obs_records,
        x_hist=x_hist,
        truth=truth,
        z_hist=z_hist,
    )
