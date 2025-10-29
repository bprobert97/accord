# ekf_constellation_improved.py
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
from dataclasses import dataclass

@dataclass
class ODProcessingResult:
    """
    Output after processing one measurement of a target satellite.

    Attributes:
    - target_id: The identifier of the satellite being tracked.
    - nis: Normalised Innovation Squared. Measures how well the measurement
    agrees with the predicted state. (Small value = fits well, large value =
    measurement is inconsistent).
    - dof: Degrees of freedom of the measurement. i.e.: range only: dof = 1.
    Used with nis to judge consistency.
    - post_cov: A 6x6 posterior covariance matrix that provides a snapshot of
    how uncertain the filter is after the update.
    """
    target_ids: List[int]
    nis: List[float]
    dof: int
    filters: List[ExtendedKalmanFilter]
    truth: np.ndarray
    sim_meas: np.ndarray

# -------- Optional: SymPy Jacobian (compiled with lambdify) --------
import sympy as sp

def _build_sympy_measurement():
    px, py, pz, vx, vy, vz = sp.symbols('px py pz vx vy vz', real=True)
    sx, sy, sz = sp.symbols('sx sy sz', real=True)
    dp = sp.Matrix([px - sx, py - sy, pz - sz])
    v = sp.Matrix([vx, vy, vz])
    r = sp.sqrt(dp.dot(dp))
    rdot = dp.dot(v) / r
    h = sp.Matrix([r, rdot])                 # [range, range-rate]
    x = sp.Matrix([px, py, pz, vx, vy, vz])
    H = sp.simplify(h.jacobian(x))
    # lambdas (vectorized over station by looping in Python)
    h_fun = sp.lambdify((px, py, pz, vx, vy, vz, sx, sy, sz), h, 'numpy')
    H_fun = sp.lambdify((px, py, pz, vx, vy, vz, sx, sy, sz), H, 'numpy')
    return h_fun, H_fun

_H_MEAS, _H_JAC = _build_sympy_measurement()

# -------- Utilities --------

def hx(x: NDArray[np.float64], stations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Nonlinear measurement: concatenated [r_i, rdot_i] for each station."""
    p = x[:3]
    v = x[3:]
    zs = []
    for (sx, sy, sz) in stations:
        r, rdot = _H_MEAS(*p, *v, sx, sy, sz).reshape(-1)
        # guard for extremely small ranges
        if r < 1e-8:
            r = 1e-8
            rdot = float(np.dot(p - np.array([sx, sy, sz]), v) / r)
        zs += [float(r), float(rdot)]
    return np.asarray(zs)

def H_jacobian(x: NDArray[np.float64], stations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytical Jacobian via SymPy lambdas (stacked for all stations)."""
    p = x[:3]
    v = x[3:]
    rows = []
    for (sx, sy, sz) in stations:
        H = np.asarray(_H_JAC(*p, *v, sx, sy, sz), dtype=float)   # 2x6
        # extra tiny guard if needed
        rows.append(H)
    return np.vstack(rows)  # 2n x 6

def constant_velocity_F(dt: float) -> NDArray[np.float64]:
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F

def make_satellite_ekf(dt: float, q_var: float, R: NDArray[np.float64], m: int,
                       x0: Optional[NDArray[np.float64]] = None,
                       P0: Optional[NDArray[np.float64]] = None) -> ExtendedKalmanFilter:
    ekf = ExtendedKalmanFilter(dim_x=6, dim_z=m)
    ekf.F = constant_velocity_F(dt)
    Q1 = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)
    ekf.Q = np.kron(np.eye(3), Q1)  # 6x6 block CV
    ekf.R = R
    ekf.x = x0 if x0 is not None else np.zeros(6)
    ekf.P = P0 if P0 is not None else np.eye(6)
    return ekf

def joseph_update(P: NDArray[np.float64], K: NDArray[np.float64], H: NDArray[np.float64],
                  R: NDArray[np.float64]) -> NDArray[np.float64]:
    I = np.eye(P.shape[0])
    A = I - K @ H
    Pn = A @ P @ A.T + K @ R @ K.T
    # enforce symmetry
    return 0.5 * (Pn + Pn.T)

def simulate_truth(x0: NDArray[np.float64], steps: int, dt: float, q: float) -> NDArray[np.float64]:
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
    """Return two-sided (1-alpha) central interval for Chi^2_dof.
    Uses SciPy if available; otherwise uses Wilson–Hilferty approximation."""
    try:
        from scipy.stats import chi2
        lo = chi2.ppf((1 - alpha) / 2.0, dof)
        hi = chi2.ppf(1 - (1 - alpha) / 2.0, dof)
        return float(lo), float(hi)
    except Exception:
        # Wilson–Hilferty approx: X^(1/3) ~ Normal(mu, sigma)
        from math import sqrt
        z = 1.959963984540054  # ~N^-1(0.975)
        k = float(dof)
        c = 2.0 / (9.0 * k)
        # two-sided: (±z)
        lo = k * (1 - z * sqrt(c))**3
        hi = k * (1 + z * sqrt(c))**3
        return lo, hi

# -------- Adaptive noise helpers --------

def estimate_measurement_noise(residuals: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.cov(residuals.T)

def adaptive_R_update(R: NDArray[np.float64], residuals: NDArray[np.float64], beta: float = 0.1) -> NDArray[np.float64]:
    R_hat = estimate_measurement_noise(residuals)
    # keep positive-definite and blend (exponential forgetting)
    Rn = (1 - beta) * R + beta * R_hat
    return 0.5 * (Rn + Rn.T)

# -------- Multi-satellite (constellation) --------

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
    if seed is not None:
        np.random.seed(seed)

    stations = np.asarray([
        [6378e3, 0.0, 0.0],
        [-2000e3, 5000e3, 3000e3],
    ], dtype=float)

    n_stations = stations.shape[0]
    m = 2 * n_stations
    R = np.diag([sig_r**2, sig_rdot**2] * n_stations)

    base_x0 = np.array([6878e3, 0.0, 0.0, 0.0, 7600.0, 0.0], dtype=float)
    offsets_pos = np.linspace(-50e3, 50e3, n_sats)
    offsets_vel = np.linspace(-50.0, 50.0, n_sats)

    truth = np.zeros((n_sats, steps, 6))
    for i in range(n_sats):
        x0_i = base_x0.copy()
        x0_i[0] += offsets_pos[i]
        x0_i[4] += offsets_vel[i]
        truth[i] = simulate_truth(x0_i, steps, dt, q_process_truth)

    meas = np.zeros((n_sats, steps, m))
    for i in range(n_sats):
        for k in range(steps):
            h = hx(truth[i, k], stations)
            n = np.zeros(m)
            n[0::2] = np.random.normal(0.0, sig_r, n_stations)
            n[1::2] = np.random.normal(0.0, sig_rdot, n_stations)
            meas[i, k] = h + n

    filters: List[ExtendedKalmanFilter] = []
    savers: List[Optional[Saver]] = []
    for i in range(n_sats):
        x_init = truth[i, 0] + np.array([2e3, 0.0, 0.0, 0.0, -20.0, 0.0], dtype=float)
        P_init = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
        ekf = make_satellite_ekf(dt, q_process_filter, R.copy(), m, x_init, P_init)
        filters.append(ekf)
        savers.append(Saver() if use_saver else None)

    nis_per_sat: List[List[float]] = [[] for _ in range(n_sats)]
    residual_logs: List[List[NDArray[np.float64]]] = [[] for _ in range(n_sats)]

    for k in range(steps):
        for i, ekf in enumerate(filters):
            ekf.predict()
            try:
                np.linalg.cholesky(ekf.P)
            except np.linalg.LinAlgError:
                ekf.P += 1e-8 * np.eye(6)

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

            if adaptive_R_every and (k + 1) % adaptive_R_every == 0 and len(residual_logs[i]) >= 10:
                ekf.R = adaptive_R_update(ekf.R, np.asarray(residual_logs[i]), beta=R_forgetting_beta)
                residual_logs[i].clear()

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
    ids_list = [f"sat_{i+1}" for i in range(len(nis_per_sat))]

    # --- Plot per-satellite NIS with χ² band ---
    lo, hi = chi2_bounds(m, 0.95)
    plt.figure(figsize=(12, 5))
    for i, nis in enumerate(nis_per_sat):
        plt.plot(nis, label=f"Sat {i}")
    plt.axhline(lo, linestyle=':', label=f"χ² 95% lo={lo:.2f}")
    plt.axhline(hi, linestyle=':', label=f"χ² 95% hi={hi:.2f}")
    plt.axhline(m, color='r', linestyle='--', label=f"DoF = {m}")
    plt.xlabel("Step"); plt.ylabel("NIS"); plt.title("Constellation NIS Over Time")
    plt.legend(ncol=3); plt.tight_layout(); plt.show()

    return ODProcessingResult(target_ids=ids_list,nis=nis_per_sat, dof=m, filters=filters, truth=truth, sim_meas=meas)

# -------- Demo --------

if __name__ == "__main__":
    # Constellation
    result: ODProcessingResult = simulate_constellation_and_estimate(n_sats=4, use_saver=False)
    print(f"Constellation: DoF per update = {result.dof}")
    print(result.target_ids)
    for i, series in enumerate(result.nis):
        print(f"  Sat {i}: last-step NIS = {series[-1]:.3f}")

