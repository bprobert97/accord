import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import chi2

"""
Production-grade EKF simulation using FilterPy (single and multi-satellite).

This module simulates fused satellite measurements (range and range-rate) from one or more
stations, runs an Extended Kalman Filter (EKF) for state estimation, computes the Normalized
Innovation Squared (NIS) at each update, and returns the results.

Upgrades for production use:
- Uses FilterPy's Q_discrete_white_noise to build block-diagonal 3D process noise.
- Joseph-form covariance update for numerical stability.
- Optional Saver() per filter for diagnostics.
- Optional adaptive noise hooks (estimate R and Q from residuals and state diffs).
- Condition-number monitoring for P and S matrices.

Key functions:
- hx(): Nonlinear measurement function (range + range-rate for each station)
- H_jacobian(): Analytical Jacobian of hx
- constant_velocity_F(): State transition matrix for constant velocity motion
- make_satellite_ekf(): Factory that builds a 6D CV EKF with 3D block Q
- simulate_truth(): Generates synthetic truth trajectory with process noise
- simulate_and_estimate(): Single-satellite simulation + EKF + NIS
- simulate_constellation_and_estimate(): Multi-satellite version
"""

# -------------------------
# Measurement model helpers
# -------------------------

def hx(x: NDArray[np.float64], stations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Measurement function for multiple ground stations.

    Parameters
    ----------
    x : ndarray (6,)
        State vector [px, py, pz, vx, vy, vz].
    stations : ndarray (n, 3)
        Ground station positions.

    Returns
    -------
    ndarray
        Measurement vector [r1, rdot1, r2, rdot2, ...].
    """
    zs: List[List[float]] = []
    p = x[:3]
    v = x[3:]
    for s in stations:
        dp = p - s
        r = float(np.linalg.norm(dp))
        if r < 1e-8:
            r = 1e-8
        rdot = float(np.dot(dp, v) / r)
        zs.append([r, rdot])
    return np.concatenate(zs)

def H_jacobian(x: NDArray[np.float64], stations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Jacobian of the measurement function hx.

    Parameters
    ----------
    x : ndarray (6,)
        State vector.
    stations : ndarray (n, 3)
        Ground station positions.

    Returns
    -------
    ndarray
        Measurement Jacobian matrix (2n x 6).
    """
    p = x[:3]
    v = x[3:]
    H: List[NDArray[np.float64]] = []
    for s in stations:
        dp = p - s
        r = float(np.linalg.norm(dp))
        if r < 1e-8:
            r = 1e-8
        dr_dp = dp / r
        drdot_dp = (v * r - dp * np.dot(dp, v) / r) / (r * r)
        drdot_dv = dp / r
        H.append(np.hstack([dr_dp, np.zeros(3)]))
        H.append(np.hstack([drdot_dp, drdot_dv]))
    return np.vstack(H)

# -------------------------
# Motion model helpers
# -------------------------

def constant_velocity_F(dt: float) -> NDArray[np.float64]:
    """State transition matrix for constant velocity model.

    Parameters
    ----------
    dt : float
        Time step (seconds).

    Returns
    -------
    ndarray
        6x6 transition matrix.
    """
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F

# -------------------------
# EKF factory and utilities
# -------------------------

def make_satellite_ekf(dt: float, q_var: float, R: NDArray[np.float64], m: int,
                       x0: Optional[NDArray[np.float64]] = None,
                       P0: Optional[NDArray[np.float64]] = None) -> ExtendedKalmanFilter:
    """Create a 6D CV EKF with 3D block process noise using FilterPy helpers.

    Parameters
    ----------
    dt : float
        Time step (seconds).
    q_var : float
        Acceleration noise variance (per axis) for the CV model.
    R : ndarray (m x m)
        Measurement covariance.
    m : int
        Measurement dimension (2*n_stations).
    x0 : ndarray (6,), optional
        Initial state.
    P0 : ndarray (6x6), optional
        Initial covariance.
    """
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=m)
    ekf.F = constant_velocity_F(dt)
    # Build 3D block diagonal Q from 2D CV white-noise model
    Q1 = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)  # 2x2
    ekf.Q = np.kron(np.eye(3), Q1)  # 6x6
    ekf.R = R
    ekf.x = x0 if x0 is not None else np.zeros(6)
    ekf.P = P0 if P0 is not None else np.eye(6)
    return ekf

def joseph_update(P: NDArray[np.float64], K: NDArray[np.float64], H: NDArray[np.float64],
                  R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Joseph-form covariance update for numerical stability.

    P_new = (I - K H) P (I - K H)^T + K R K^T
    """
    I = np.eye(P.shape[0])
    A = I - K @ H
    return A @ P @ A.T + K @ R @ K.T

# -------------------------
# Truth simulation
# -------------------------

def simulate_truth(x0: NDArray[np.float64], steps: int, dt: float, q: float) -> NDArray[np.float64]:
    """Simulate true satellite trajectory with process noise (white accel).

    Parameters
    ----------
    x0 : ndarray (6,)
        Initial true state.
    steps : int
        Number of time steps.
    dt : float
        Time step (seconds).
    q : float
        Acceleration noise variance (per axis) driving the truth.

    Returns
    -------
    ndarray
        True trajectory (steps x 6).
    """
    F = constant_velocity_F(dt)
    G = np.zeros((6, 3))
    G[0:3, :] = 0.5 * dt * dt * np.eye(3)
    G[3:6, :] = dt * np.eye(3)
    x = x0.copy()
    traj: List[NDArray[np.float64]] = []
    for _ in range(steps):
        a = np.random.normal(0.0, np.sqrt(q), 3)
        x = F @ x + G @ a
        traj.append(x.copy())
    return np.array(traj)

# -------------------------
# Adaptive estimation hooks (optional)
# -------------------------

def estimate_measurement_noise(residuals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate R from innovation sequence (simple sample covariance)."""
    return np.cov(residuals.T)

def estimate_process_noise(states: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """Estimate Q from state differences (simple finite-difference model)."""
    diff = np.diff(states, axis=0)
    return np.cov(diff.T) / max(dt, 1e-9)

# -------------------------
# Single-satellite workflow
# -------------------------

def simulate_and_estimate(
    steps: int = 200,
    dt: float = 1.0,
    q_process_truth: float = 1e-4,
    q_process_filter: float = 1e-4,
    sig_r: float = 30.0,
    sig_rdot: float = 0.1,
    seed: Optional[int] = 42,
    use_saver: bool = False,
    adaptive_R_every: Optional[int] = None,
    adaptive_Q_every: Optional[int] = None,
    cond_warn_threshold: float = 1e12,
) -> Tuple[List[float], int, ExtendedKalmanFilter, NDArray[np.float64], NDArray[np.float64]]:
    """Run single-satellite simulation, EKF estimation, and NIS computation.

    Returns NIS list (per step), DoF, EKF object, truth, and measurements.
    Optionally logs via Saver and performs crude adaptive noise updates.
    """
    if seed is not None:
        np.random.seed(seed)

    stations: NDArray[np.float64] = np.array([
        [6378e3, 0.0, 0.0],
        [-2000e3, 5000e3, 3000e3]
    ], dtype=float)
    n_stations = stations.shape[0]
    m = 2 * n_stations

    R = np.diag([sig_r**2, sig_rdot**2] * n_stations)

    x0_truth = np.array([6878e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    truth = simulate_truth(x0_truth, steps, dt, q_process_truth)

    meas = np.zeros((steps, m))
    for k in range(steps):
        h = hx(truth[k], stations)
        noise = np.zeros(m)
        noise[0::2] = np.random.normal(0.0, sig_r, n_stations)
        noise[1::2] = np.random.normal(0.0, sig_rdot, n_stations)
        meas[k] = h + noise

    x_init = x0_truth + np.array([2000.0, 0.0, 0.0, 0.0, -20.0, 0.0])
    P_init = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

    ekf = make_satellite_ekf(dt, q_process_filter, R, m, x_init, P_init)
    saver = Saver() if use_saver else None

    nis_list: List[float] = []
    residual_log = []  # for adaptive R
    state_hist = []    # for adaptive Q

    for k in range(steps):
        ekf.predict()
        H = H_jacobian(ekf.x, stations)
        z = meas[k]
        z_pred = hx(ekf.x, stations)
        y = z - z_pred
        S = H @ ekf.P @ H.T + ekf.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = ekf.P @ H.T @ S_inv
        ekf.x = ekf.x + K @ y
        ekf.P = joseph_update(ekf.P, K, H, ekf.R)

        nis = float(y.T @ S_inv @ y)
        nis_list.append(nis)

        # Logging for adaptive updates
        residual_log.append(y)
        state_hist.append(ekf.x.copy())

        # Optional adaptive R
        if adaptive_R_every and (k + 1) % adaptive_R_every == 0 and len(residual_log) >= 10:
            ekf.R = estimate_measurement_noise(np.array(residual_log))
            residual_log.clear()
        # Optional adaptive Q
        if adaptive_Q_every and (k + 1) % adaptive_Q_every == 0 and len(state_hist) >= 20:
            Q_est = estimate_process_noise(np.array(state_hist), dt)
            # Map 6x6 estimate to 3D CV block structure (simple projection)
            Q1 = Q_est[np.ix_([0,3], [0,3])]  # take x-pos/vel block as proxy
            # ensure symmetry
            Q1 = 0.5 * (Q1 + Q1.T)
            ekf.Q = np.kron(np.eye(3), Q1)
            state_hist.clear()

        # Conditioning checks
        if np.linalg.cond(ekf.P) > cond_warn_threshold:
            print(f"Warning: ill-conditioned P at step {k} (cond={np.linalg.cond(ekf.P):.2e})")
        if np.linalg.cond(S) > cond_warn_threshold:
            print(f"Warning: ill-conditioned S at step {k} (cond={np.linalg.cond(S):.2e})")

        if saver is not None:
            saver.save(ekf)

    if saver is not None:
        saver.to_array()

    return nis_list, m, ekf, truth, meas

# -------------------------
# Multi-satellite workflow
# -------------------------

def simulate_constellation_and_estimate(
    n_sats: int = 5,
    steps: int = 200,
    dt: float = 1.0,
    q_process_truth: float = 1e-4,
    q_process_filter: float = 1e-4,
    sig_r: float = 30.0,
    sig_rdot: float = 0.1,
    seed: Optional[int] = 123,
    use_saver: bool = False,
    adaptive_R_every: Optional[int] = None,
    adaptive_Q_every: Optional[int] = None,
    cond_warn_threshold: float = 1e12,
) -> Tuple[List[List[float]], int, List[ExtendedKalmanFilter], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a constellation of satellites with shared stations and run EKFs.

    Each satellite is tracked by its own EKF. Measurements per update are fused range +
    range-rate from all stations (same stations for all satellites). Returns per-satellite
    NIS series and DoF.
    """
    if seed is not None:
        np.random.seed(seed)

    # Shared ground stations
    stations: NDArray[np.float64] = np.array([
        [6378e3, 0.0, 0.0],
        [-2000e3, 5000e3, 3000e3],
    ], dtype=float)
    n_stations = stations.shape[0]
    m = 2 * n_stations
    R = np.diag([sig_r**2, sig_rdot**2] * n_stations)

    # Distinct initial conditions per satellite
    base_x0: NDArray[np.float64] = np.array([6878e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    offsets_pos = np.linspace(-50e3, 50e3, n_sats)
    offsets_vel = np.linspace(-50.0, 50.0, n_sats)

    truth = np.zeros((n_sats, steps, 6))
    for i in range(n_sats):
        x0_i = base_x0.copy()
        x0_i[0] += offsets_pos[i]
        x0_i[4] += offsets_vel[i]
        truth[i] = simulate_truth(x0_i, steps, dt, q_process_truth)

    # Measurements
    meas = np.zeros((n_sats, steps, m))
    for i in range(n_sats):
        for k in range(steps):
            h = hx(truth[i, k], stations)
            noise = np.zeros(m)
            noise[0::2] = np.random.normal(0.0, sig_r, n_stations)
            noise[1::2] = np.random.normal(0.0, sig_rdot, n_stations)
            meas[i, k] = h + noise

    # EKFs and (optional) Savers
    filters: List[ExtendedKalmanFilter] = []
    savers: List[Saver] = []
    for i in range(n_sats):
        x_init = truth[i, 0] + np.array([2000.0, 0.0, 0.0, 0.0, -20.0, 0.0])
        P_init = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
        ekf = make_satellite_ekf(dt, q_process_filter, R, m, x_init, P_init)
        filters.append(ekf)
        savers.append(Saver() if use_saver else None)

    nis_per_sat: List[List[float]] = [[] for _ in range(n_sats)]
    residual_logs: List[List[NDArray[np.float64]]] = [[] for _ in range(n_sats)]
    state_hists: List[List[NDArray[np.float64]]] = [[] for _ in range(n_sats)]

    for k in range(steps):
        for i, ekf in enumerate(filters):
            ekf.predict()
            H = H_jacobian(ekf.x, stations)
            z = meas[i, k]
            z_pred = hx(ekf.x, stations)
            y = z - z_pred
            S = H @ ekf.P @ H.T + ekf.R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            K = ekf.P @ H.T @ S_inv
            ekf.x = ekf.x + K @ y
            ekf.P = joseph_update(ekf.P, K, H, ekf.R)

            nis = float(y.T @ S_inv @ y)
            nis_per_sat[i].append(nis)

            residual_logs[i].append(y)
            state_hists[i].append(ekf.x.copy())

            # Adaptive updates (per satellite)
            if adaptive_R_every and (k + 1) % adaptive_R_every == 0 and len(residual_logs[i]) >= 10:
                ekf.R = estimate_measurement_noise(np.array(residual_logs[i]))
                residual_logs[i].clear()
            if adaptive_Q_every and (k + 1) % adaptive_Q_every == 0 and len(state_hists[i]) >= 20:
                Q_est = estimate_process_noise(np.array(state_hists[i]), dt)
                Q1 = Q_est[np.ix_([0, 3], [0, 3])]  # proxy from x-pos/vel
                Q1 = 0.5 * (Q1 + Q1.T)
                ekf.Q = np.kron(np.eye(3), Q1)
                state_hists[i].clear()

            # Conditioning checks
            if np.linalg.cond(ekf.P) > cond_warn_threshold:
                print(f"Sat {i}: ill-conditioned P at step {k} (cond={np.linalg.cond(ekf.P):.2e})")
            if np.linalg.cond(S) > cond_warn_threshold:
                print(f"Sat {i}: ill-conditioned S at step {k} (cond={np.linalg.cond(S):.2e})")

            if use_saver and savers[i] is not None:
                savers[i].save(ekf)

    if use_saver:
        for s in savers:
            if s is not None:
                s.to_array()

    # --- Plot NIS ---
    plt.figure()
    for i, nis in enumerate(nis_per_sat):
        plt.plot(nis, label=f"Satellite {i}")
    plt.axhline(dof, color='r', linestyle='--', label=f"DoF = {dof}")
    alpha = 0.95
    lower = chi2.ppf((1 - alpha) / 2, dof)
    upper = chi2.ppf(1 - (1 - alpha) / 2, dof)
    plt.fill_between(range(steps), lower, upper, color='gray', alpha=0.2, label='95% CI')
    plt.xlabel("Step")
    plt.ylabel("NIS")
    plt.title("Constellation NIS Over Time")
    plt.legend()
    plt.show()

    return nis_per_sat, m, filters, truth, meas


if __name__ == "__main__":
    # Single-satellite demo
    nis_list, dof, ekf, truth, meas = simulate_and_estimate(use_saver=False)
    print(f"Single-sat: DoF={dof}, last NIS={nis_list[-1]:.3f}")

    # Constellation demo
    nis_per_sat, dof, filters, truth, meas = simulate_constellation_and_estimate(n_sats=4, use_saver=False)
    print(f"Constellation: DoF per update = {dof}")
    for i, series in enumerate(nis_per_sat):
        print(f"  Sat {i}: last-step NIS = {series[-1]:.3f}")
