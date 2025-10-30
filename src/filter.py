# pylint: disable=too-many-locals, invalid-name, too-many-statements, too-many-branches, too-many-positional-arguments, too-many-arguments
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
from math import sqrt
from typing import Tuple, List, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.stats import chi2
from filterpy.kalman import ExtendedKalmanFilter # type: ignore
from filterpy.common import Q_discrete_white_noise, Saver # type: ignore
import sympy as sp # type: ignore


@dataclass
class ODProcessingResult:
    """
    Output after processing constellation OD.

    Attributes:
    - target_ids: List of satellite IDs
    - nis: NIS history per satellite
    - dof: Degrees of freedom per update
    - filters: EKF instances for each satellite
    - truth: Ground truth state trajectories
    - sim_meas: Simulated measurement data
    """
    target_ids: List[str]
    nis: List[List[float]]
    dof: int
    filters: List[ExtendedKalmanFilter]
    truth: np.ndarray
    sim_meas: np.ndarray


# ---------------- SymPy measurement model ----------------

def _build_sympy_measurement():
    """
    Build symbolic range & range-rate measurement model and Jacobian using SymPy.

    Args:
    - None

    Returns:
    - h_fun: Callable range & range-rate function
    - H_fun: Callable Jacobian function
    """
    px, py, pz, vx, vy, vz = sp.symbols('px py pz vx vy vz', real=True)
    sx, sy, sz = sp.symbols('sx sy sz', real=True)
    dp = sp.Matrix([px - sx, py - sy, pz - sz])
    v = sp.Matrix([vx, vy, vz])
    r = sp.sqrt(dp.dot(dp))
    rdot = dp.dot(v) / r
    h = sp.Matrix([r, rdot])
    x = sp.Matrix([px, py, pz, vx, vy, vz])
    H = sp.simplify(h.jacobian(x))
    h_fun = sp.lambdify((px, py, pz, vx, vy, vz, sx, sy, sz), h, 'numpy')
    H_fun = sp.lambdify((px, py, pz, vx, vy, vz, sx, sy, sz), H, 'numpy')
    return h_fun, H_fun


_H_MEAS, _H_JAC = _build_sympy_measurement()


# ---------------- Utilities ----------------

def hx(x: NDArray[np.float64], stations: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute nonlinear range & range-rate measurement for multiple stations.

    Args:
    - x: State vector [px, py, pz, vx, vy, vz]
    - stations: (N×3) array of station position vectors

    Returns:
    - Measurement vector [r1, rdot1, r2, rdot2, ...]
    """
    p = x[:3]
    v = x[3:]
    zs = []
    for (sx, sy, sz) in stations:
        r, rdot = _H_MEAS(*p, *v, sx, sy, sz).reshape(-1)
        if r < 1e-8:  # Avoid divide-by-zero
            r = 1e-8
            rdot = float(np.dot(p - np.array([sx, sy, sz]), v) / r)
        zs += [float(r), float(rdot)]
    return np.asarray(zs)


def H_jacobian(x: NDArray[np.float64], stations: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute measurement Jacobian stacked for all stations.

    Args:
    - x: Satellite state vector
    - stations: (N×3) station locations

    Returns:
    - Jacobian matrix (2N × 6)
    """
    p = x[:3]
    v = x[3:]
    rows = []
    for (sx, sy, sz) in stations:
        rows.append(np.asarray(_H_JAC(*p, *v, sx, sy, sz), dtype=float))
    return np.vstack(rows)


def hx_rel(x_t: NDArray[np.float64], x_o: NDArray[np.float64]) -> NDArray[np.float64]:
    """Relative range + range-rate: target observed by known observer."""
    pt, vt = x_t[:3], x_t[3:]
    po, vo = x_o[:3], x_o[3:]

    rho = pt - po
    r = np.linalg.norm(rho)
    if r < 1e-8:
        r = 1e-8

    vrel = vt - vo
    rdot = float(rho.dot(vrel) / r)

    return np.array([r, rdot], dtype=float)


def H_jacobian_rel_target(x_t: NDArray[np.float64], x_o: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Jacobian wrt target state only (observer state assumed known).
    H is 2×6, mapping target's [px,py,pz,vx,vy,vz].
    """
    pt, vt = x_t[:3], x_t[3:]
    po, vo = x_o[:3], x_o[3:]

    rho = pt - po
    r = np.linalg.norm(rho)
    if r < 1e-8:
        r = 1e-8

    rhat = rho / r
    vrel = vt - vo
    I = np.eye(3)

    # ∂r/∂p_t , ∂r/∂v_t
    H1 = np.hstack([rhat, np.zeros(3)])

    # ∂ṙ/∂p_t , ∂ṙ/∂v_t
    d_rdot_d_pt = ((I - np.outer(rhat, rhat)) @ vrel) / r
    d_rdot_d_vt = rhat

    H2 = np.hstack([d_rdot_d_pt, d_rdot_d_vt])

    return np.vstack([H1, H2])  # 2x6


def constant_velocity_F(dt: float) -> NDArray[np.float64]:
    """
    Build state transition matrix for 3-D constant velocity model.

    Args:
    - dt: Time step (seconds)

    Returns:
    - 6×6 state transition matrix
    """
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def make_satellite_ekf(dt: float, q_var: float, R: NDArray[np.float64], m: int,
                       x0: Optional[NDArray[np.float64]] = None,
                       P0: Optional[NDArray[np.float64]] = None) -> ExtendedKalmanFilter:
    """
    Build EKF for satellite orbit estimation.

    Args:
    - dt: Time step
    - q_var: Continuous-time process noise variance
    - R: Measurement covariance matrix
    - m: Measurement dimension
    - x0: Optional initial state
    - P0: Optional initial covariance

    Returns:
    - Configured EKF instance
    """
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=m)
    ekf.F = constant_velocity_F(dt)
    Q1 = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)
    ekf.Q = np.kron(np.eye(3), Q1)
    ekf.R = R
    ekf.x = x0 if x0 is not None else np.zeros(6)
    ekf.P = P0 if P0 is not None else np.eye(6)
    return ekf


def joseph_update(P: NDArray[np.float64], K: NDArray[np.float64], H: NDArray[np.float64],
                  R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Perform Joseph-form covariance update (numerically stable).

    Args:
    - P: Prior covariance
    - K: Kalman gain
    - H: Measurement Jacobian
    - R: Measurement noise covariance

    Returns:
    - Updated covariance matrix
    """
    I = np.eye(P.shape[0])
    A = I - K @ H
    Pn = A @ P @ A.T + K @ R @ K.T
    return 0.5 * (Pn + Pn.T)


def simulate_truth(x0: NDArray[np.float64], steps: int, dt: float, q: float) -> NDArray[np.float64]:
    """
    Propagate truth states using a stochastic constant-velocity model.

    Args:
    - x0: Initial state
    - steps: Number of propagation steps
    - dt: Time step (s)
    - q: Continuous-time acceleration noise variance

    Returns:
    - State trajectory array (steps × 6)
    """
    F = constant_velocity_F(dt)
    G = np.zeros((6, 3))
    G[0:3, :] = 0.5 * dt * dt * np.eye(3)
    G[3:6, :] = dt * np.eye(3)
    x = x0.copy()
    traj = []
    for _ in range(steps):
        a = np.random.normal(0.0, np.sqrt(q), 3)
        x = F @ x + G @ a
        traj.append(x.copy())
    return np.asarray(traj)


def chi2_bounds(dof: int, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Compute central chi-square bounds for NIS consistency.

    Args:
    - dof: Degrees of freedom
    - alpha: Confidence level

    Returns:
    - (lower bound, upper bound)
    """
    try:
        lo = chi2.ppf((1 - alpha) / 2.0, dof)
        hi = chi2.ppf(1 - (1 - alpha) / 2.0, dof)
        return float(lo), float(hi)
    except Exception:
        z = 1.959963984540054
        k = float(dof)
        c = 2.0 / (9.0 * k)
        lo = k * (1 - z * sqrt(c))**3
        hi = k * (1 + z * sqrt(c))**3
        return lo, hi


def estimate_measurement_noise(residuals: np.ndarray) -> np.ndarray:
    """
    Estimate measurement covariance from residuals.

    Args:
    - residuals: Innovation matrix (time x dims)

    Returns:
    - Sample covariance matrix
    """
    return np.cov(residuals.T)


def adaptive_R_update(R: NDArray[np.float64], residuals: NDArray[np.float64],
                      beta: float = 0.1) -> NDArray[np.float64]:
    """
    Adaptively update measurement covariance using exponential forgetting.

    Args:
    - R: Current R matrix
    - residuals: Recent innovations
    - beta: Forgetting factor (0-1)

    Returns:
    - Updated R
    """
    R_hat = estimate_measurement_noise(residuals)
    Rn = (1 - beta) * R + beta * R_hat
    return 0.5 * (Rn + Rn.T)


def simulate_constellation_and_estimate(
    n_sats: int = 4,
    steps: int = 200,
    dt: float = 1.0,
    q_process_truth: float = 1e-5,
    q_process_filter: float = 1e-4,
    sig_r: float = 30.0,
    sig_rdot: float = 0.1,
    seed: Optional[int] = 123,
    use_saver: bool = False,
    adaptive_R_every: Optional[int] = 25,
    R_forgetting_beta: float = 0.1,
    cond_warn_threshold: float = 1e12,
) -> ODProcessingResult:
    """
    Simulate satellite constellation, generate measurements, and run EKFs.

    Args:
    - n_sats: Number of satellites
    - steps: Simulation length
    - dt: Time step
    - q_process_truth: True process noise
    - q_process_filter: Filter-assumed process noise
    - sig_r: Range noise stdev
    - sig_rdot: Range-rate noise stdev
    - seed: PRNG seed
    - use_saver: Save EKF history
    - adaptive_R_every: Adapt measurement noise every N steps
    - R_forgetting_beta: Learning rate for adaptive R
    - cond_warn_threshold: Condition number warning limit

    Returns:
    - ODProcessingResult struct
    """

    if seed is not None:
        np.random.seed(seed)

    observer_x0 = np.array([
                6878e3,   # x position (m)
                100e3,    # y position offset from targets (m)
                0.0,      # z
                0.0,      # vx
                7600.0,   # vy (m/s circular LEO)
                0.0       # vz
            ], dtype=float)

    observer_truth = simulate_truth(
        observer_x0,
        steps,
        dt,
        q_process_truth
    )

    # Each measurement has 2 DOF (range and range)
    m = 2
    R = np.diag([sig_r**2, sig_rdot**2])

    base_x0 = np.array([6878e3, 0.0, 0.0, 0.0, 7600.0, 0.0], dtype=float)
    offsets_pos = np.linspace(-50e3, 50e3, n_sats)
    offsets_vel = np.linspace(-50.0, 50.0, n_sats)

    truth = np.zeros((n_sats, steps, 6))
    for j in range(n_sats):
        x0_j = base_x0.copy()
        x0_j[0] += offsets_pos[j]
        x0_j[4] += offsets_vel[j]
        truth[j] = simulate_truth(x0_j, steps, dt, q_process_truth)

    # assume observer_truth is available (size: steps × 6)
    meas = np.zeros((n_sats, steps, m))
    for i in range(n_sats):
        for k in range(steps):
            z_true = hx_rel(truth[i, k], observer_truth[k])
            noise = np.array([
                np.random.normal(0.0, sig_r),
                np.random.normal(0.0, sig_rdot)
            ])
            meas[i, k] = z_true + noise

    filters: List[ExtendedKalmanFilter] = []
    savers: List[Optional[Saver]] = []
    for l in range(n_sats):
        x_init = truth[l, 0] + np.array([2e3, 0.0, 0.0, 0.0, -20.0, 0.0], dtype=float)
        P_init = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
        ekf = make_satellite_ekf(dt, q_process_filter, R.copy(), m, x_init, P_init)
        filters.append(ekf)
        savers.append(Saver(ekf) if use_saver else None)

    nis_per_sat: List[List[float]] = [[] for _ in range(n_sats)]
    residual_logs: List[List[NDArray[np.float64]]] = [[] for _ in range(n_sats)]

    for k in range(steps):
        for g, ekf in enumerate(filters):
            ekf.predict()
            try:
                np.linalg.cholesky(ekf.P)
            except np.linalg.LinAlgError:
                ekf.P += 1e-8 * np.eye(6)

            H = H_jacobian_rel_target(ekf.x, observer_truth[k])
            z = meas[g, k]
            z_pred = hx_rel(ekf.x, observer_truth[k])
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
            nis_per_sat[g].append(nis)
            residual_logs[g].append(y)

            if adaptive_R_every and (k + 1) % adaptive_R_every == 0 \
                and len(residual_logs[g]) >= 10:
                ekf.R = adaptive_R_update(ekf.R, np.asarray(residual_logs[g]),
                                          beta=R_forgetting_beta)
                residual_logs[g].clear()

            if np.linalg.cond(ekf.P) > cond_warn_threshold:
                print(f"Sat {g}: ill-conditioned P at step {k} (cond={np.linalg.cond(ekf.P):.2e})")
            if np.linalg.cond(S) > cond_warn_threshold:
                print(f"Sat {g}: ill-conditioned S at step {k} (cond={np.linalg.cond(S):.2e})")

            if use_saver and savers[g] is not None:
                savers[g].save(ekf) # type: ignore [union-attr]

    if use_saver:
        for s in savers:
            if s is not None:
                s.to_array()
    ids_list = [f"sat_{i+1}" for i in range(len(nis_per_sat))]

    # --- Plot per-satellite NIS with χ² band ---
    lo, hi = chi2_bounds(m, 0.95)
    plt.figure(figsize=(12, 5))
    for sat, nis in enumerate(nis_per_sat): #type: ignore [assignment]
        plt.plot(nis, label=f"Sat {sat}")
    plt.axhline(lo, linestyle=':', label=f"χ² 95% lo={lo:.2f}")
    plt.axhline(hi, linestyle=':', label=f"χ² 95% hi={hi:.2f}")
    plt.axhline(m, color='r', linestyle='--', label=f"DoF = {m}")
    plt.xlabel("Step")
    plt.ylabel("NIS")
    plt.title("Constellation NIS Over Time")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused (required for 3D plotting)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # Target satellite trajectory
    ax.plot(truth[0,:,0], truth[0,:,1], truth[0,:,2], label="Target Sat")

    # Observer satellite trajectory
    ax.plot(observer_truth[:,0], observer_truth[:,1], observer_truth[:,2], label="Observer Sat")

    # Axis labels
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_title("3D Satellite Orbits")
    ax.legend()
    ax.grid(True)

    # Equal aspect ratio for 3D
    max_range = np.array([
        truth[0,:,0].max() - truth[0,:,0].min(),
        truth[0,:,1].max() - truth[0,:,1].min(),
        truth[0,:,2].max() - truth[0,:,2].min()
    ]).max() / 2.0

    mid_x = (truth[0,:,0].max() + truth[0,:,0].min()) * 0.5
    mid_y = (truth[0,:,1].max() + truth[0,:,1].min()) * 0.5
    mid_z = (truth[0,:,2].max() + truth[0,:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()



    return ODProcessingResult(target_ids=ids_list,nis=nis_per_sat, dof=m,
                              filters=filters, truth=truth, sim_meas=meas)

# -------- Demo --------

if __name__ == "__main__":
    # Constellation
    result: ODProcessingResult = simulate_constellation_and_estimate(n_sats=4, use_saver=False)
    print(f"Constellation: DoF per update = {result.dof}")
    print(result.target_ids)
    for i, series in enumerate(result.nis):
        print(f"  Sat {i}: last-step NIS = {series[-1]:.3f}")
