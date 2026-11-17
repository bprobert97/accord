"""Live evaluation of a trained offline Conservative Q-Learning attack policy.

The trained policy is deployed in an instance of the Proof of Inter-Satellite
Evaluation (PoISE) consensus simulation that is seeded differently from every
training episode. The evaluation environment matches the harvester: real
cluster-based observations, peers filtered by inter-satellite-link range, and
honest satellites that also submit their observations. Only the action of each
compromised satellite is supplied by the trained policy rather than a scripted
behaviour policy. The per-transaction reputation, consensus score, and injected
drift are written to a trajectory file and summarised against the admission
constraints.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from typing import Dict, Optional

import numpy as np
import torch

from accord_demo import (
    CLUSTER_SIZE, FilterType,
    _cluster_by_orbital_physics, _create_cluster_filters,
    _create_z_map, _process_cluster_step,
)
from machine_learning.cql_offline import load_actor
from machine_learning.harvest_dag_dataset import (
    HarvestConfig, inject_bias_simple, _lookup_poise_scores,
)
from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import FilterConfig, simulate_truth_and_meas
from src.satellite_node import SatelliteNode  # use satellite_node_patched.py here

ACTION_DIM: int = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LiveEval")

# The shared "ACCORD" logger emits one line per PoISE evaluation, which floods
# the console at the scale of this simulation; silence it below the error level.
# This script's own logger, passed into the cluster-filter factory, is untouched.
logging.getLogger("ACCORD").setLevel(logging.ERROR)

# The simulation scaffold is shared verbatim with harvest_dag_dataset so that the
# evaluation environment matches the harvested data exactly.
# pylint: disable=duplicate-code


async def evaluate_live_cql(  # pylint: disable=too-many-locals,too-many-statements
    model_path: str = "machine_learning/cql_offline.pt",
    log_path: str = "live_eval_trajectory.csv",
    cfg: Optional[HarvestConfig] = None,
    seed: int = 10_000,
) -> None:
    """Deploy a trained policy against the live consensus and summarise it.

    Args:
        model_path: Path to the trained actor checkpoint.
        log_path: Destination path for the per-transaction trajectory CSV.
        cfg: Simulation configuration; a default :class:`HarvestConfig` is used
            when omitted.
        seed: Simulation seed, offset from the training seeds so the evaluation
            is not a replay of a training episode.

    Returns:
        None. The trajectory is written to ``log_path`` and a constraint summary
        and pass or fail verdict are printed.
    """
    cfg = cfg or HarvestConfig()
    engine = ConsensusMechanism()
    actor = load_actor(model_path)

    fcfg = FilterConfig(
        ISL_range_m=cfg.isl_range_m, N=cfg.num_sats,
        steps=cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps,
        dt=cfg.dt_s, sig_r=cfg.sig_r_m, sig_rdot=cfg.sig_rdot_mps,
        q_acc_target=cfg.q_acc_target, seed=seed, walker_delta=cfg.walker_delta,
    )
    truth, z_hist = simulate_truth_and_meas(fcfg, fcfg.walker_delta)

    clusters = _cluster_by_orbital_physics(truth[0], cfg.num_sats, CLUSTER_SIZE)
    cluster_filters = _create_cluster_filters(FilterType.EKF, fcfg, truth, clusters, logger)
    z_map = _create_z_map(cfg.num_sats)
    x_hist = np.zeros((fcfg.steps, cfg.num_sats * 6))

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))
    cum_bias = np.zeros(ACTION_DIM, dtype=np.float32)

    reputations: list = []
    scores: list = []
    drifts: list = []
    last_score, last_correctness = 0.5, 0.5
    # Each compromised satellite conditions its action on its own most recent
    # score; a shared scalar would let one satellite's score leak into another's
    # decision when more than one satellite is compromised.
    last_score_by_sat: Dict[int, float] = {}

    print(
        "Evaluating CQL Model in live consensus network "
        f"(num_sats={cfg.num_sats}, isl_range_m={cfg.isl_range_m:.0f}, "
        "real cluster-based observations, isl-filtered peers)..."
    )

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step", "node", "reputation", "consensus_score",
                "correctness_score", "score_unavailable", "drift", "action_0", "action_1",
            ],
        )
        writer.writeheader()

        step_action_cache: Dict[int, np.ndarray] = {}

        # Replay the harvester's simulation structure step by step, substituting the
        # trained policy for the scripted attacker so the two environments match.
        for k in range(fcfg.steps):
            records = []
            for cluster_ids, cluster_filter in zip(clusters, cluster_filters):
                records.extend(
                    _process_cluster_step(cluster_ids, cluster_filter, z_map, (x_hist, z_hist), k)
                )

            for sid, sat in satellites.items():
                sat.update_position(truth[k, sid * 6:(sid + 1) * 6])

            # The connectivity model matches training: peers are filtered by
            # actual link range rather than taken as a full mesh.
            for sid, sat in satellites.items():
                sat.peers = [
                    other_sat for other_sid, other_sat in satellites.items()
                    if other_sid != sid
                    and np.linalg.norm(
                        truth[k, sid * 6:sid * 6 + 3] - truth[k, other_sid * 6:other_sid * 6 + 3]
                    ) <= cfg.isl_range_m
                ]

            step_action_cache.clear()

            for rec in records:
                if rec.observer in comp_ids and k > cfg.filter_burn_in_steps + cfg.honest_steps:
                    if rec.observer not in step_action_cache:
                        # Four-dimensional state matching the harvester:
                        # (cum_bias_x, cum_bias_y, reputation, consensus_score).
                        # The score defaults to 0.5 before this satellite's first
                        # resolved transaction, matching the harvester.
                        prior_score = last_score_by_sat.get(rec.observer, 0.5)
                        state = np.concatenate(
                            [cum_bias, [float(satellites[rec.observer].reputation), prior_score]]
                        ).astype(np.float32)
                        state_tensor = torch.as_tensor(state).unsqueeze(0)
                        with torch.no_grad():
                            action = actor(state_tensor).squeeze(0).numpy()
                        cum_bias = np.clip(
                            cum_bias + action * cfg.action_scale_sigma,
                            -cfg.bias_limit_sigma, cfg.bias_limit_sigma,
                        )
                        step_action_cache[rec.observer] = action
                    else:
                        action = step_action_cache[rec.observer]
                    # Convert the bias from units of the noise standard deviation into the
                    # physical range and range-rate units the observation record expects.
                    bias_si = cum_bias * np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

                    biased_record = inject_bias_simple(rec, bias_si)
                    observer = satellites[rec.observer]
                    observer.load_sensor_data(biased_record)
                    peers_at_broadcast = list(observer.peers)
                    transaction = await observer.submit_transaction(rec.target)

                    reputation = float(observer.reputation)
                    consensus_opt, correctness_opt = _lookup_poise_scores(
                        transaction, peers_at_broadcast
                    )
                    is_unavailable = consensus_opt is None
                    if consensus_opt is None or correctness_opt is None:
                        consensus_score, correctness_score = last_score, last_correctness
                    else:
                        consensus_score, correctness_score = consensus_opt, correctness_opt
                    last_score, last_correctness = consensus_score, correctness_score
                    # Carry this satellite's resolved score forward so that its
                    # next decision conditions on it.
                    last_score_by_sat[rec.observer] = consensus_score
                    drift = float(np.linalg.norm(cum_bias))

                    reputations.append(reputation)
                    scores.append(consensus_score)
                    drifts.append(drift)

                    writer.writerow(
                        {
                            "step": k, "node": rec.observer, "reputation": reputation,
                            "consensus_score": consensus_score,
                            "correctness_score": correctness_score,
                            "score_unavailable": is_unavailable, "drift": drift,
                            "action_0": float(action[0]), "action_1": float(action[1]),
                        }
                    )
                elif k > cfg.filter_burn_in_steps:
                    # Honest satellites also submit their observations, exactly
                    # as in the harvester; otherwise no ledger grows past genesis
                    # through legitimate traffic and the quorum starves for
                    # reasons unrelated to the policy under evaluation.
                    satellites[rec.observer].load_sensor_data(rec)
                    await satellites[rec.observer].submit_transaction(rec.target)

    print(f"Live Evaluation Complete. Per-step trajectory written to {log_path}\n")

    if not reputations:
        print("No attack-phase transactions were recorded; nothing to evaluate.")
        return

    _summarise(np.asarray(reputations), np.asarray(scores), np.asarray(drifts), cfg)


def _summarise(
    reputations: np.ndarray, scores: np.ndarray, drifts: np.ndarray, cfg: HarvestConfig
) -> None:
    """Print the constraint-satisfaction summary and pass or fail verdict.

    Args:
        reputations: Per-transaction attacker reputation.
        scores: Per-transaction consensus score.
        drifts: Per-transaction injected drift magnitude.
        cfg: Simulation configuration, supplying the admission floors.

    Returns:
        None.
    """
    rep_floor, score_floor = cfg.reputation_floor, cfg.score_floor
    frac_rep_ok = float((reputations > rep_floor).mean())
    frac_score_ok = float((scores > score_floor).mean())
    frac_both_ok = float(((reputations > rep_floor) & (scores > score_floor)).mean())

    print(
        f"Reputation: mean={reputations.mean():.3f}  min={reputations.min():.3f}  "
        f"fraction > {rep_floor}: {frac_rep_ok:.1%}"
    )
    print(
        f"Consensus score: mean={scores.mean():.3f}  min={scores.min():.3f}  "
        f"fraction > {score_floor}: {frac_score_ok:.1%}"
    )
    print(f"Both constraints simultaneously satisfied: {frac_both_ok:.1%} of steps")
    print(
        f"Achieved drift: mean={drifts.mean():.3f}  max={drifts.max():.3f}  "
        f"final={drifts[-1]:.3f}"
    )

    # The policy passes only if it stays above both floors on more than ninety per
    # cent of attack transactions; anything less is reported as a failure.
    verdict = "PASS" if (frac_rep_ok > 0.9 and frac_score_ok > 0.9) else "FAIL"
    print(f"\nConstraint verdict: {verdict} (threshold: >90% of steps above both floors)")


if __name__ == "__main__":
    asyncio.run(evaluate_live_cql())
