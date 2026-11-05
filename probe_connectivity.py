"""
Connectivity probe — run this BEFORE a full harvest.

Sweeps a handful of candidate isl_range_m values on a single short episode and
reports mean/min/max peer count per satellite per step for each. Pick the
smallest candidate that gives a healthy mean peer count (rule of thumb: aim
for mean >= 3-4, so a typical satellite usually has enough neighbours to
reach the BFT quorum of 4 without relying on rare best-case moments) — going
unnecessarily large just makes the physical ISL-range assumption less
realistic without buying you anything once connectivity is already healthy.

This does NOT run the full consensus/harvesting pipeline — it only touches
the orbit simulation and the pure-geometry peer filter, so it's cheap enough
to sweep many candidates quickly before committing to a full 40-episode run.
"""
import numpy as np
from src.filters.filter_interface import FilterConfig, simulate_truth_and_meas

NUM_SATS = 400
DT_S = 60.0
TOTAL_STEPS = 30 + 60 + 300
SEED = 0
WALKER_DELTA = False

CANDIDATE_RANGES_KM = [1000, 2000] #3500, 5000, 7500, 10000, 15000, 20000, 30000, 40000



def probe(isl_range_m: float) -> dict:
    fcfg = FilterConfig(
        ISL_range_m=isl_range_m, N=NUM_SATS, steps=TOTAL_STEPS, dt=DT_S,
        sig_r=10.0, sig_rdot=0.2, q_acc_target=1e-5, seed=SEED, walker_delta=WALKER_DELTA,
    )
    truth, _ = simulate_truth_and_meas(fcfg, fcfg.walker_delta)

    peer_counts = []
    for k in range(TOTAL_STEPS):
        for sid in range(NUM_SATS):
            n_peers = 0
            for other_sid in range(NUM_SATS):
                if other_sid == sid:
                    continue
                d = np.linalg.norm(truth[k, sid * 6:sid * 6 + 3] - truth[k, other_sid * 6:other_sid * 6 + 3])
                if d <= isl_range_m:
                    n_peers += 1
            peer_counts.append(n_peers)

    peer_counts = np.asarray(peer_counts)
    return {
        "mean": float(peer_counts.mean()),
        "min": int(peer_counts.min()),
        "max": int(peer_counts.max()),
        "frac_zero_peers": float((peer_counts == 0).mean()),
        "frac_quorum_capable": float((peer_counts >= 1).mean()),  # needs >=1 peer to broadcast to at all
    }


def main() -> None:
    print(f"{'range (km)':>12} | {'mean peers':>10} | {'min':>4} | {'max':>4} | "
          f"{'% steps w/ 0 peers':>18}")
    print("-" * 70)
    for range_km in CANDIDATE_RANGES_KM:
        stats = probe(range_km * 1000.0)
        print(f"{range_km:>12} | {stats['mean']:>10.2f} | {stats['min']:>4} | {stats['max']:>4} | "
              f"{stats['frac_zero_peers'] * 100:>17.1f}%")

    print("\nPick the smallest range where 'mean peers' is comfortably above ~3-4 and "
          "'% steps w/ 0 peers' is low — that's the smallest isl_range_m that should let "
          "most satellites actually reach BFT quorum (needs >=4 real transactions received, "
          "which requires repeatedly having >=1 peer available to receive from).")


if __name__ == "__main__":
    main()
