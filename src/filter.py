from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.stats import chi2
from filterpy.kalman import ExtendedKalmanFilter  # type: ignore
import statistics
from scipy.linalg import expm

# ----------------------- Constants -----------------------
MU_EARTH = 3.986004418e14  # m^3/s^2
Re = 6378e3

# ----------------------- Result Types ---------------------
@dataclass
class ObservationRecord:
    step: int
    time: int
    observer: int
    target: int
    nis: float
    dof: int

@dataclass
class JointResult:
    target_ids: List[str]                # ["sat_1", ...]
    obs_records: List[ObservationRecord] # per-observation NIS records
    x_hist: np.ndarray                   # (steps, 6*N)
    truth: np.ndarray                    # (steps, 6*N)
    z_hist: np.ndarray                   # (steps, 2*N*(N-1))

# ----------------------- Dynamics ------------------------
def two_body_f(x6: NDArray[np.float64]) -> NDArray[np.float64]:
    r = x6[:3]; v = x6[3:]
    rn = np.linalg.norm(r)
    a = -MU_EARTH * r / rn**3
    return np.hstack([v, a])

def F_jacobian_6(x6: NDArray[np.float64]) -> NDArray[np.float64]:
    r = x6[:3]; rn = np.linalg.norm(r); I3 = np.eye(3)
    dadr = -MU_EARTH * (I3 / rn**3 - 3*np.outer(r, r)/rn**5)
    F = np.zeros((6,6))
    F[:3,3:] = I3
    F[3:,:3] = dadr
    return F

def rk4_step(x: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    k1 = two_body_f(x)
    k2 = two_body_f(x + 0.5*dt*k1)
    k3 = two_body_f(x + 0.5*dt*k2)
    k4 = two_body_f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def van_loan_discretization(F: NDArray[np.float64],
                            L: NDArray[np.float64],
                            Qc: NDArray[np.float64],
                            dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = F.shape[0]
    A = L @ Qc @ L.T
    M = np.block([[F, A], [np.zeros((n,n)), -F.T]]) * dt
    EM = expm(M)
    Phi = EM[:n,:n]
    J = EM[:n,n:]
    Q = Phi @ J @ Phi.T
    return Phi, 0.5*(Q + Q.T)

def F_midpoint(x: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    k1 = two_body_f(x)
    x_mid = x + 0.5 * dt * k1
    Fm = F_jacobian_6(x_mid)
    if not np.isfinite(Fm).all():
        Fm = F_jacobian_6(x)
    return Fm

# ----------------------- Truth propagation ----------------
def propagate_truth_kepler(x0_stack: NDArray[np.float64], steps: int, dt: float) -> NDArray[np.float64]:
    x = x0_stack.copy()
    hist = np.zeros((steps, x0_stack.size))
    for k in range(steps):
        for s in range(0, x.size, 6):
            x[s:s+6] = rk4_step(x[s:s+6], dt)
        hist[k] = x
    return hist

# ----------------------- Measurement model ----------------
def hx_block(target: NDArray[np.float64], obs: NDArray[np.float64]) -> NDArray[np.float64]:
    pt, vt = target[:3], target[3:]
    po, vo = obs[:3], obs[3:]
    rho = pt - po
    r = np.linalg.norm(rho); r = max(r, 1e-8)
    vrel = vt - vo
    rdot = float(rho.dot(vrel) / r)
    return np.array([r, rdot])

def H_blocks_target_obs(target: NDArray[np.float64], obs: NDArray[np.float64]):
    pt, vt = target[:3], target[3:]
    po, vo = obs[:3], obs[3:]
    rho = pt - po
    r = np.linalg.norm(rho); r = max(r, 1e-8)
    rhat = rho / r
    I3 = np.eye(3)
    vrel = vt - vo

    H1_t = np.hstack([rhat, np.zeros(3)])
    d_rdot_d_pt = ((I3 - np.outer(rhat, rhat)) @ vrel) / r
    H2_t = np.hstack([d_rdot_d_pt, rhat])
    Ht = np.vstack([H1_t, H2_t])

    H1_o = np.hstack([-rhat, np.zeros(3)])
    d_rdot_d_po = -((I3 - np.outer(rhat, rhat)) @ vrel) / r
    H2_o = np.hstack([d_rdot_d_po, -rhat])
    Ho = np.vstack([H1_o, H2_o])
    return Ht, Ho

def hx_joint(x: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    z = []
    for i in range(N):
        xi = x[6*i:6*i+6]
        for j in range(N):
            if i != j:
                z.append(hx_block(x[6*j:6*j+6], xi))
    return np.concatenate(z)

def H_joint(x: NDArray[np.float64], N: int) -> NDArray[np.float64]:
    rows = []
    for i in range(N):
        xi = x[6*i:6*i+6]
        for j in range(N):
            if i == j: continue
            xj = x[6*j:6*j+6]
            Ht, Ho = H_blocks_target_obs(xj, xi)
            R = np.zeros((2, 6*N))
            R[:,6*j:6*j+6] = Ht
            R[:,6*i:6*i+6] = Ho
            rows.append(R)
    return np.vstack(rows)

# ----------------------- EKF predict ----------------------
def ekf_predict_joint(ekf, dt, N, q_acc_target, _unused):
    x_prev = ekf.x.copy()
    dim = 6*N

    # propagate state
    x = x_prev.copy()
    for i in range(N):
        x[6*i:6*i+6] = rk4_step(x[6*i:6*i+6], dt)
    ekf.x = x

    # propagate covariance (block-diag)
    Phi = np.eye(dim)
    Qd = np.zeros((dim,dim))
    L = np.zeros((6,3)); L[3:,:] = np.eye(3)

    for i in range(N):
        Fi = F_midpoint(x_prev[6*i:6*i+6], dt)
        Qci = np.eye(3)*q_acc_target
        Phii, Qdi = van_loan_discretization(Fi, L, Qci, dt)
        Phi[6*i:6*i+6,6*i:6*i+6] = Phii
        Qd [6*i:6*i+6,6*i:6*i+6] = Qdi

    ekf.P = Phi @ ekf.P @ Phi.T + Qd
    ekf.P = 0.5*(ekf.P + ekf.P.T)

# ----------------------- Truth + measurement sim ----------
def simulate_truth_and_meas(N, steps, dt, sig_r, sig_rdot):
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
def run_joint_ekf(
    N=10, steps=3000, dt=60.0,
    sig_r=10.0, sig_rdot=0.02,
    q_acc_target=1e-6, q_acc_obs=1e-6,
    seed=42,
) -> JointResult:

    if seed is not None: np.random.seed(seed)
    truth, z_hist = simulate_truth_and_meas(N, steps, dt, sig_r, sig_rdot)

    dim_x = 6*N
    M = N*(N-1)
    dim_z = 2*M

    R = np.diag([sig_r**2, sig_rdot**2]*M)
    ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)

    x0_est = truth[0].copy()
    for i in range(N):
        x0_est[6*i:6*i+6] += np.array([2e3,0,0,0,-10,0])

    P0 = np.zeros((dim_x,dim_x))
    for i in range(N):
        P0[6*i:6*i+6,6*i:6*i+6] = np.diag([1e8]*3+[1e4]*3)

    ekf.x, ekf.P, ekf.R = x0_est, P0, R

    x_hist = np.zeros((steps,dim_x))
    obs_records: List[ObservationRecord] = []

    for k in range(steps):
        ekf_predict_joint(ekf, dt, N, q_acc_target, q_acc_obs)

        H = H_joint(ekf.x, N)
        z_pred = hx_joint(ekf.x, N)
        y = z_hist[k] - z_pred

        S = H @ ekf.P @ H.T + ekf.R
        S_inv = np.linalg.pinv(S)
        K = ekf.P @ H.T @ S_inv

        ekf.x = ekf.x + K @ y
        I = np.eye(dim_x)
        ekf.P = (I-K@H)@ekf.P@(I-K@H).T + K@ekf.R@K.T
        ekf.P = 0.5*(ekf.P + ekf.P.T)

        # log NIS for each ordered observation
        idx = 0
        for i in range(N): # observer
            for j in range(N): # target
                if i == j: continue

                rows = slice(idx, idx+2)
                yij = y[rows]

                H_ij = np.zeros((2, dim_x))
                Ht, Ho = H_blocks_target_obs(ekf.x[6*j:6*j+6], ekf.x[6*i:6*i+6])
                H_ij[:,6*j:6*j+6] = Ht
                H_ij[:,6*i:6*i+6] = Ho

                S_ij = H_ij @ ekf.P @ H_ij.T + np.diag([sig_r**2, sig_rdot**2])
                S_ij_inv = np.linalg.pinv(S_ij)
                nis = float(yij.T @ S_ij_inv @ yij)

                obs_records.append(
                    ObservationRecord(
                        step=k, observer=i, target=j, nis=nis, dof=yij.shape[0], time = k*dt
                    )
                )
                idx += 2

        x_hist[k] = ekf.x

    print(obs_records)
    return JointResult(
        target_ids=[f"sat_{i+1}" for i in range(N)],
        obs_records=obs_records,
        x_hist=x_hist,
        truth=truth,
        z_hist=z_hist,
    )

# ----------------------- Plot Helpers --------------------
def extract_mean_nis_per_sat(result: JointResult) -> list[list[float]]:
    """
    Converts obs_records (each obs event) into:
    mean_nis[sat][step]
    """
    N = len(result.target_ids)
    steps = result.x_hist.shape[0]

    # Initialize storage
    nis_matrix = [[[] for _ in range(steps)] for _ in range(N)]

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
    lo = chi2.ppf((1 - alpha) / 2.0, dof)
    hi = chi2.ppf(1 - (1 - alpha) / 2.0, dof)
    return float(lo), float(hi)

def plot_nis(result: JointResult):
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

def plot_nis_consistency(result: JointResult, dof: int = 2, window: int = 50):
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
    plt.xlabel("Step"); plt.ylabel("Mean NIS per sat")
    plt.legend(ncol=3); plt.grid(); plt.tight_layout(); plt.show()


# ----------------------- Demo -----------------------------
if __name__ == "__main__":
    result = run_joint_ekf(
        N=10,
        steps=3000,
        dt=60.0,
        sig_r=10.0,
        sig_rdot=0.02,
        q_acc_target=1e-5,
        q_acc_obs=1e-5,   # kept for signature compatibility
        seed=42,
    )

    print("Satellites:", result.target_ids)

    nis_matrix = extract_mean_nis_per_sat(result)
    for i, series in enumerate(nis_matrix):
        print(f"  Sat {i}: last-step mean NIS = {series[-1]:.3f}")

    plot_nis(result)
    plot_nis_consistency(result, window=80)

