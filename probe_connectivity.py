"""Inter-satellite connectivity probe to run before a full dataset harvest.

The probe sweeps a set of candidate inter-satellite-link ranges on a single
short orbit simulation and reports the per-satellite peer count for each. It
touches only the orbit simulation and the pure-geometry peer filter, not the
consensus or harvesting pipeline, so it is cheap enough to evaluate many
candidates before committing to a full multi-episode harvest.

The smallest candidate whose mean peer count is comfortably above three to four
is preferred: a typical satellite then usually has enough neighbours to reach
the Byzantine-fault-tolerant quorum of four without relying on rare best-case
geometry, while an unnecessarily large range weakens the physical link-range
assumption without improving connectivity that is already healthy.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.filters.filter_interface import FilterConfig, simulate_truth_and_meas

NUM_SATS: int = 400
DT_S: float = 60.0
TOTAL_STEPS: int = 30 + 60 + 300
SEED: int = 0
WALKER_DELTA: bool = False
CANDIDATE_RANGES_KM: List[int] = [1000, 2000, 3500, 5000, 7500, 10000, 15000, 20000, 30000, 40000]


def probe(isl_range_m: float) -> Dict[str, float]:
    """Measure per-satellite peer connectivity at one link range.

    Args:
        isl_range_m: Inter-satellite-link range, in metres. Two satellites are
            peers at a given step when the distance between them does not exceed
            this value.

    Returns:
        A dictionary with the mean, minimum, and maximum peer count per
        satellite per step, the fraction of satellite-steps with no peers, and
        the fraction of satellite-steps with at least one peer.
    """
    fcfg = FilterConfig(
        ISL_range_m=isl_range_m, N=NUM_SATS, steps=TOTAL_STEPS, dt=DT_S,
        sig_r=10.0, sig_rdot=0.2, q_acc_target=1e-5, seed=SEED, walker_delta=WALKER_DELTA,
    )
    truth, _ = simulate_truth_and_meas(fcfg, fcfg.walker_delta)

    peer_counts: List[int] = []
    for k in range(TOTAL_STEPS):
        for sid in range(NUM_SATS):
            n_peers = 0
            for other_sid in range(NUM_SATS):
                if other_sid == sid:
                    continue
                distance = np.linalg.norm(
                    truth[k, sid * 6:sid * 6 + 3] - truth[k, other_sid * 6:other_sid * 6 + 3]
                )
                if distance <= isl_range_m:
                    n_peers += 1
            peer_counts.append(n_peers)

    counts = np.asarray(peer_counts)
    return {
        "mean": float(counts.mean()),
        "min": float(counts.min()),
        "max": float(counts.max()),
        "frac_zero_peers": float((counts == 0).mean()),
        "frac_quorum_capable": float((counts >= 1).mean()),
    }


def main() -> None:
    """Sweep the candidate link ranges and print a connectivity summary table.

    Returns:
        None.
    """
    print(
        f"{'range (km)':>12} | {'mean peers':>10} | {'min':>4} | {'max':>4} | "
        f"{'% steps w/ 0 peers':>18}"
    )
    print("-" * 70)
    for range_km in CANDIDATE_RANGES_KM:
        stats = probe(range_km * 1000.0)
        print(
            f"{range_km:>12} | {stats['mean']:>10.2f} | {stats['min']:>4.0f} | "
            f"{stats['max']:>4.0f} | {stats['frac_zero_peers'] * 100:>17.1f}%"
        )

    print(
        "\nPick the smallest range where 'mean peers' is comfortably above three "
        "to four and '% steps w/ 0 peers' is low; that is the smallest link range "
        "that should let most satellites reach the Byzantine-fault-tolerant quorum."
    )


if __name__ == "__main__":
    main()
