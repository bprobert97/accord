"""
Live Evaluation of the Offline CQL Model
"""

import asyncio
import csv
import numpy as np
import torch

from accord_demo import (
    CLUSTER_SIZE, FilterType,
    _cluster_by_orbital_physics, _create_cluster_filters,
    _create_z_map, _process_cluster_step,
)
from src.filters.filter_interface import FilterConfig, simulate_truth_and_meas
from src.satellite_node import SatelliteNode  # use satellite_node_patched.py here
from src.consensus_mech import ConsensusMechanism
from machine_learning.cql_offline import load_actor
from machine_learning.harvest_dag_dataset import HarvestConfig, inject_bias_simple, _lookup_poise_scores

ACTION_DIM = 2


async def evaluate_live_cql(model_path: str = "machine_learning/cql_offline.pt",
                             log_path: str = "live_eval_trajectory.csv",
                             cfg: HarvestConfig = None,
                             seed: int = 10_000):
    # seed offset so this isn't literally replaying a training episode
    cfg = cfg or HarvestConfig()
    engine = ConsensusMechanism()
    actor = load_actor(model_path)
    rng = np.random.default_rng(seed)

    fcfg = FilterConfig(
        ISL_range_m=cfg.isl_range_m, N=cfg.num_sats,
        steps=cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps,
        dt=cfg.dt_s, sig_r=cfg.sig_r_m, sig_rdot=cfg.sig_rdot_mps,
        q_acc_target=cfg.q_acc_target, seed=seed, walker_delta=cfg.walker_delta,
    )
    truth, z_hist = simulate_truth_and_meas(fcfg, fcfg.walker_delta)

    clusters = _cluster_by_orbital_physics(truth[0], cfg.num_sats, CLUSTER_SIZE)
    cluster_filters = _create_cluster_filters(FilterType.EKF, fcfg, truth, clusters, None)
    z_map = _create_z_map(cfg.num_sats)
    x_hist = np.zeros((fcfg.steps, cfg.num_sats * 6))

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))
    cum_bias = np.zeros(ACTION_DIM, dtype=np.float32)

    reputations, scores, drifts = [], [], []
    last_score, last_correctness = 0.5, 0.5

    print("Evaluating CQL Model in live consensus network "
          f"(num_sats={cfg.num_sats}, isl_range_m={cfg.isl_range_m:.0f}, "
          f"real cluster-based observations, isl-filtered peers)...")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "node", "reputation", "consensus_score",
                                                "correctness_score", "score_unavailable", "drift",
                                                "action_0", "action_1"])
        writer.writeheader()

        step_action_cache = {}

        for k in range(fcfg.steps):
            records = []
            for cluster_ids, cluster_filter in zip(clusters, cluster_filters):
                records.extend(
                    _process_cluster_step(cluster_ids, cluster_filter, z_map, (x_hist, z_hist), k)
                )

            for sid, sat in satellites.items():
                sat.update_position(truth[k, sid * 6:(sid + 1) * 6])

            # Same connectivity model as training: filter peers by actual ISL range,
            # not full mesh.
            for sid, sat in satellites.items():
                sat.peers = [other_sat for other_sid, other_sat in satellites.items()
                             if other_sid != sid and
                             np.linalg.norm(truth[k, sid * 6:sid * 6 + 3] - truth[k, other_sid * 6:other_sid * 6 + 3])
                             <= cfg.isl_range_m]

            step_action_cache.clear()  # one action per satellite is valid only within this step

            for rec in records:
                if rec.observer in comp_ids and k > cfg.filter_burn_in_steps + cfg.honest_steps:
                    if rec.observer not in step_action_cache:
                        state = np.concatenate([cum_bias, [float(satellites[rec.observer].reputation)]]).astype(np.float32)
                        state_tensor = torch.as_tensor(state).unsqueeze(0)
                        with torch.no_grad():
                            action = actor(state_tensor).squeeze(0).numpy()
                        cum_bias = np.clip(cum_bias + action * cfg.action_scale_sigma,
                                            -cfg.bias_limit_sigma, cfg.bias_limit_sigma)
                        step_action_cache[rec.observer] = action
                    else:
                        action = step_action_cache[rec.observer]
                    bias_si = cum_bias * np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

                    biased_record = inject_bias_simple(rec, bias_si)
                    observer = satellites[rec.observer]
                    observer.load_sensor_data(biased_record)
                    peers_at_broadcast = list(observer.peers)
                    transaction = await observer.submit_transaction(rec.target)

                    reputation = float(observer.reputation)
                    consensus_score, correctness_score = _lookup_poise_scores(transaction, peers_at_broadcast)
                    is_unavailable = consensus_score is None
                    if is_unavailable:
                        consensus_score, correctness_score = last_score, last_correctness
                    last_score, last_correctness = consensus_score, correctness_score
                    drift = float(np.linalg.norm(cum_bias))

                    reputations.append(reputation)
                    scores.append(consensus_score)
                    drifts.append(drift)

                    writer.writerow({
                        "step": k, "node": rec.observer, "reputation": reputation,
                        "consensus_score": consensus_score, "correctness_score": correctness_score,
                        "score_unavailable": is_unavailable, "drift": drift,
                        "action_0": float(action[0]), "action_1": float(action[1]),
                    })
                elif k > cfg.filter_burn_in_steps:
                    # Honest satellites submit their real sensor data too, exactly like
                    # the harvester — otherwise no ledger ever grows past genesis via
                    # legitimate traffic and BFT quorum starves for reasons that have
                    # nothing to do with the policy being evaluated.
                    satellites[rec.observer].load_sensor_data(rec)
                    await satellites[rec.observer].submit_transaction(rec.target)

    print(f"Live Evaluation Complete. Per-step trajectory written to {log_path}\n")

    if not reputations:
        print("No attack-phase transactions were recorded — nothing to evaluate.")
        return

    reputations = np.asarray(reputations)
    scores = np.asarray(scores)
    drifts = np.asarray(drifts)

    rep_floor, score_floor = cfg.reputation_floor, cfg.score_floor
    frac_rep_ok = float((reputations > rep_floor).mean())
    frac_score_ok = float((scores > score_floor).mean())
    frac_both_ok = float(((reputations > rep_floor) & (scores > score_floor)).mean())

    print(f"Reputation: mean={reputations.mean():.3f}  min={reputations.min():.3f}  "
          f"fraction > {rep_floor}: {frac_rep_ok:.1%}")
    print(f"Consensus score: mean={scores.mean():.3f}  min={scores.min():.3f}  "
          f"fraction > {score_floor}: {frac_score_ok:.1%}")
    print(f"Both constraints simultaneously satisfied: {frac_both_ok:.1%} of steps")
    print(f"Achieved drift: mean={drifts.mean():.3f}  max={drifts.max():.3f}  final={drifts[-1]:.3f}")

    verdict = "PASS" if (frac_rep_ok > 0.9 and frac_score_ok > 0.9) else "FAIL"
    print(f"\nConstraint verdict: {verdict} (threshold: >90% of steps above both floors)")


if __name__ == "__main__":
    asyncio.run(evaluate_live_cql())
