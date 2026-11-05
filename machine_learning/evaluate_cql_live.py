"""
Live Evaluation of the Offline CQL Model
"""

import asyncio
import csv
import numpy as np
import torch
from machine_learning.cql_offline import load_actor
from machine_learning.harvest_dag_dataset import HarvestConfig, inject_bias_simple, _lookup_poise_scores
from src.satellite_node import SatelliteNode  # use satellite_node_patched.py here
from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import ObservationRecord


async def evaluate_live_cql(model_path: str = "machine_learning/cql_offline.pt",
                             log_path: str = "live_eval_trajectory.csv"):
    cfg = HarvestConfig()
    engine = ConsensusMechanism()
    actor = load_actor(model_path)

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))
    cum_bias = np.zeros(2, dtype=np.float32)

    reputations, scores, correctness_scores, drifts = [], [], [], []
    last_score, last_correctness = 0.5, 0.5

    print("Evaluating CQL Model in live consensus network...")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "node", "reputation", "consensus_score",
                                                "correctness_score", "score_unavailable", "drift",
                                                "action_0", "action_1"])
        writer.writeheader()

        for k in range(cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps):
            for sid, sat in satellites.items():
                sat.peers = [s for other, s in satellites.items() if other != sid]

            for sid, sat in satellites.items():
                for peer in sat.peers:
                    if k > cfg.filter_burn_in_steps + cfg.honest_steps and sid in comp_ids:
                        state = np.concatenate([cum_bias, [sat.reputation]]).astype(np.float32)
                        state_tensor = torch.as_tensor(state).unsqueeze(0)

                        with torch.no_grad():
                            action = actor(state_tensor).squeeze(0).numpy()

                        cum_bias = np.clip(cum_bias + action * cfg.action_scale_sigma, -cfg.bias_limit_sigma, cfg.bias_limit_sigma)
                        bias_si = cum_bias * np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

                        mock_record = ObservationRecord(step=k, time=k * cfg.dt_s, observer=sid, target=peer.id,
                                                          nis=2.0, dof=2, r_vector=[1, 0, 0], v_vector=[0, 1, 0],
                                                          range_m=1000.0, observer_r_eci=[0, 0, 0])
                        biased_record = inject_bias_simple(mock_record, bias_si)

                        sat.load_sensor_data(biased_record)
                        peers_at_broadcast = list(sat.peers)
                        transaction = await sat.submit_transaction(peer.id)

                        reputation = float(sat.reputation)
                        consensus_score, correctness_score = _lookup_poise_scores(transaction, peers_at_broadcast)
                        is_unavailable = consensus_score is None
                        if is_unavailable:
                            consensus_score, correctness_score = last_score, last_correctness
                        last_score, last_correctness = consensus_score, correctness_score
                        drift = float(np.linalg.norm(cum_bias))

                        reputations.append(reputation)
                        scores.append(consensus_score)
                        correctness_scores.append(correctness_score)
                        drifts.append(drift)

                        writer.writerow({
                            "step": k, "node": sid, "reputation": reputation,
                            "consensus_score": consensus_score, "correctness_score": correctness_score,
                            "score_unavailable": is_unavailable, "drift": drift,
                            "action_0": float(action[0]), "action_1": float(action[1]),
                        })

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
