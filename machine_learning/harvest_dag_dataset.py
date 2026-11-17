"""Asynchronous harvester that scrapes an offline training dataset from the DAG.

The harvester runs the Proof of Inter-Satellite Evaluation (PoISE) consensus
simulation for a number of episodes. In each episode a small number of
compromised satellites inject sensor bias according to a scripted behaviour
policy while the remaining satellites report honestly. Every scored ledger
transaction produced by a compromised satellite is recorded as one reinforcement
-learning transition, together with the reputation and consensus score the
network assigned to it. The resulting archive is the fixed dataset consumed by
the offline Conservative Q-Learning trainer.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from accord_demo import (
    CLUSTER_SIZE, FilterType,
    _cluster_by_orbital_physics, _create_cluster_filters,
    _create_z_map, _process_cluster_step,
)
from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import (
    FilterConfig, ObservationRecord, simulate_truth_and_meas,
)
from src.satellite_node import SatelliteNode  # use satellite_node_patched.py here

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DAGHarvester")

# The simulation scaffold is shared verbatim with evaluate_cql_live so that the
# evaluation environment matches the harvested data exactly.
# pylint: disable=duplicate-code

# The shared "ACCORD" logger emits one line per PoISE evaluation, which floods
# the console at the scale of this simulation; silence it below the error level.
logging.getLogger("ACCORD").setLevel(logging.ERROR)

ACTION_DIM: int = 2
BEHAVIOUR_MODES: Tuple[str, ...] = ("mixed", "persistent", "ramp_hold", "backoff")


@dataclass
class HarvestConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for a dataset harvest.

    Attributes:
        num_episodes: Number of independent episodes to simulate.
        output_path: Destination path for the compressed ``.npz`` dataset.
        seed: Base random seed; episode ``e`` uses ``seed + e``.
        num_sats: Number of satellites in the constellation.
        dt_s: Simulation time step, in seconds.
        sig_r_m: Range measurement-noise standard deviation, in metres.
        sig_rdot_mps: Range-rate measurement-noise standard deviation, in metres
            per second.
        q_acc_target: Target process-noise acceleration for the filter.
        isl_range_m: Inter-satellite-link range, in metres.
        num_compromised: Number of compromised satellites.
        filter_burn_in_steps: Steps before any transactions are recorded.
        honest_steps: Honest-reporting steps following the burn-in.
        attack_steps: Steps during which the compromised satellites attack.
        action_scale_sigma: Scale applied to each action increment, in units of
            the measurement-noise standard deviation.
        bias_limit_sigma: Per-channel clip on the cumulative injected bias, in
            units of the measurement-noise standard deviation.
        score_floor: Consensus-score admission threshold.
        reputation_floor: Reputation admission threshold.
        detection_rule: Name of the detection rule in use.
        mode_probabilities: Sampling probabilities over ``BEHAVIOUR_MODES``.
        include_context_features: Whether to include context features.
        walker_delta: Whether the constellation is a Walker-delta pattern.
        violation_penalty: Reward charged when an admission constraint is
            violated.
        drift_reward_scale: Multiplier applied to the achieved drift to form the
            reward on compliant steps.
        strict_score_check: When True, an excessive unscored fraction raises
            rather than warns.
        max_allowed_unscored_fraction: Unscored-fraction threshold above which
            the dataset is flagged.
    """

    num_episodes: int = 40
    output_path: str = "machine_learning/dag_harvested_dataset.npz"
    seed: int = 0
    num_sats: int = 200
    dt_s: float = 60.0
    sig_r_m: float = 10.0
    sig_rdot_mps: float = 0.2
    q_acc_target: float = 1e-5
    isl_range_m: float = 3000e3
    num_compromised: int = 3
    filter_burn_in_steps: int = 30
    honest_steps: int = 60
    attack_steps: int = 300
    action_scale_sigma: float = 0.25
    bias_limit_sigma: float = 15.0
    score_floor: float = 0.3
    reputation_floor: float = 0.5
    detection_rule: str = "reputation_only"
    mode_probabilities: Tuple[float, ...] = (0.4, 0.2, 0.15, 0.25)
    include_context_features: bool = True
    walker_delta: bool = False
    violation_penalty: float = 5.0
    drift_reward_scale: float = 1.0
    strict_score_check: bool = False
    max_allowed_unscored_fraction: float = 0.3


class BehaviourPolicy:  # pylint: disable=too-few-public-methods
    """Scripted exploration policy for a compromised satellite.

    The policy produces the action increments that drive bias injection during
    the attack phase. Its behaviour depends on the configured mode, which lets a
    harvest span a range of deception strategies rather than a single style.
    """

    def __init__(
        self, mode: str, rng: np.random.Generator, steps: int, score_floor: float
    ) -> None:
        """Initialise the policy.

        Args:
            mode: One of ``BEHAVIOUR_MODES``.
            rng: Random generator used for all stochastic choices.
            steps: Number of attack steps, used to pace the ramp mode.
            score_floor: Consensus-score floor, used by the reactive mode.
        """
        self.mode = mode
        self.rng = rng
        self.steps = steps
        self.retreat_threshold = score_floor + 0.1
        self._direction = self._random_direction()
        self._magnitude = float(self.rng.uniform(0.1, 1.0))
        self._hold_remaining = 0
        self._ramp_target = float(self.rng.uniform(0.3, 1.0))

    def _random_direction(self) -> np.ndarray:
        """Draw a random unit direction in action space.

        Returns:
            A unit-norm action-dimensional direction vector.
        """
        vec = self.rng.normal(size=ACTION_DIM)
        return (vec / max(float(np.linalg.norm(vec)), 1e-8)).astype(np.float32)

    def act(  # pylint: disable=too-many-return-statements
        self, t: int, last_score: float, cum_bias: np.ndarray
    ) -> np.ndarray:
        """Return the action increment for the current attack step.

        Args:
            t: Attack-step index, counted from the start of the attack phase.
            last_score: Most recent consensus score, used by the reactive mode.
            cum_bias: Current cumulative injected bias, used by the reactive
                mode to retreat.

        Returns:
            An action-dimensional increment to the cumulative bias.
        """
        if self.mode == "mixed":
            draw = self.rng.random()
            if draw < 0.5:
                return self.rng.uniform(-1.0, 1.0, ACTION_DIM)
            if draw < 0.8:
                return self.rng.uniform(-0.2, 0.2, ACTION_DIM)
            mask = self.rng.choice([0.0, 1.0], ACTION_DIM)
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM) * mask

        if self.mode == "persistent":
            jitter = self.rng.normal(scale=0.05, size=ACTION_DIM).astype(np.float32)
            return np.clip(self._direction * self._magnitude + jitter, -1.0, 1.0)

        if self.mode == "ramp_hold":
            ramp_frac = min(1.0, (t + 1) / max(1, self.steps // 3))
            return (self._direction * self._ramp_target * ramp_frac).astype(np.float32)

        if self.mode == "backoff":
            if last_score < self.retreat_threshold:
                return (
                    -cum_bias / max(float(np.linalg.norm(cum_bias)), 1e-6) * 0.5
                ).astype(np.float32)
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM)

        return self.rng.uniform(-1.0, 1.0, ACTION_DIM)


def inject_bias_simple(record: ObservationRecord, bias_si: np.ndarray) -> ObservationRecord:
    """Return a copy of an observation record with injected bias applied.

    Args:
        record: The honest observation record.
        bias_si: Bias to apply, in physical units (metres and metres per second).

    Returns:
        A new observation record with an inflated normalised innovation squared
        and a biased range.
    """
    nis = max(0.1, record.nis + float(np.sum(bias_si ** 2) / 100.0))
    return dataclasses.replace(record, nis=nis, range_m=float(record.range_m + bias_si[0]))


def _lookup_poise_scores(
    transaction: object, peers: List[SatelliteNode]
) -> Tuple[Optional[float], Optional[float]]:
    """Read the consensus and correctness scores a transaction received.

    Each peer stores its own evaluation of a transaction in
    ``peer.dag.local_consensus_states[transaction.hash]``. The scores are taken
    as the median across all peers that evaluated the transaction.

    Args:
        transaction: The broadcast transaction to look up.
        peers: Peers present at broadcast time.

    Returns:
        A tuple ``(consensus_score, correctness_score)``, or ``(None, None)`` if
        no peer has scored the transaction yet.
    """
    consensus_scores: List[float] = []
    correctness_scores: List[float] = []
    for peer in peers:
        state = peer.dag.local_consensus_states.get(transaction.hash)  # type: ignore[attr-defined]
        if state is not None and "consensus_score" in state:
            consensus_scores.append(state["consensus_score"])
            correctness_scores.append(state.get("correctness_score", state["consensus_score"]))
    if not consensus_scores:
        return None, None
    return float(np.median(consensus_scores)), float(np.median(correctness_scores))


# pylint: disable-next=too-many-locals,too-many-statements,too-many-branches
async def _simulate_episode_async(
    ep: int, cfg: HarvestConfig, rng: np.random.Generator
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Simulate one episode and return its transitions and diagnostics.

    Args:
        ep: Episode index.
        cfg: Harvest configuration.
        rng: Random generator for episode-level choices.

    Returns:
        A tuple ``(data, diagnostics)``. ``data`` maps dataset field names to
        arrays, or is empty if no transition was recorded. ``diagnostics`` holds
        connectivity and ledger-size summaries for the episode.
    """
    engine = ConsensusMechanism()
    fcfg = FilterConfig(
        ISL_range_m=cfg.isl_range_m, N=cfg.num_sats,
        steps=cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps,
        dt=cfg.dt_s, sig_r=cfg.sig_r_m, sig_rdot=cfg.sig_rdot_mps,
        q_acc_target=cfg.q_acc_target, seed=cfg.seed + ep, walker_delta=cfg.walker_delta,
    )

    truth, z_hist = simulate_truth_and_meas(fcfg, fcfg.walker_delta)

    clusters = _cluster_by_orbital_physics(truth[0], cfg.num_sats, CLUSTER_SIZE)
    cluster_filters = _create_cluster_filters(FilterType.EKF, fcfg, truth, clusters, logger)
    z_map = _create_z_map(cfg.num_sats)
    x_hist = np.zeros((fcfg.steps, cfg.num_sats * 6))

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))

    mode = rng.choice(
        BEHAVIOUR_MODES, p=np.asarray(cfg.mode_probabilities) / np.sum(cfg.mode_probabilities)
    )
    policy = BehaviourPolicy(str(mode), rng, cfg.attack_steps, cfg.score_floor)
    cum_bias = np.zeros(ACTION_DIM, dtype=np.float32)

    keys = (
        "states", "actions", "rewards", "costs", "next_states", "scores",
        "correctness_scores", "reps", "step_idx", "terminals", "score_unavailable",
    )
    rows: Dict[str, List] = {key: [] for key in keys}
    peer_counts: List[int] = []

    last_score = 0.5
    last_correctness = 0.5

    # A satellite can generate several observation records in one step (one per
    # tracked target). One action is drawn per (satellite, step) and the
    # resulting bias is reused across every record that satellite produces that
    # step, so a single step injects a single bias update rather than one per
    # record.
    step_action_cache: Dict[int, np.ndarray] = {}

    for k in range(fcfg.steps):
        records = []
        for cluster_ids, cluster_filter in zip(clusters, cluster_filters):
            records.extend(
                _process_cluster_step(cluster_ids, cluster_filter, z_map, (x_hist, z_hist), k)
            )

        for sid, sat in satellites.items():
            sat.update_position(truth[k, sid * 6:(sid + 1) * 6])

        for sid, sat in satellites.items():
            sat.peers = [
                other_sat for other_sid, other_sat in satellites.items()
                if other_sid != sid
                and np.linalg.norm(
                    truth[k, sid * 6:sid * 6 + 3] - truth[k, other_sid * 6:other_sid * 6 + 3]
                ) <= cfg.isl_range_m
            ]
            peer_counts.append(len(sat.peers))

        step_action_cache.clear()

        for rec in records:
            if rec.observer in comp_ids and k > cfg.filter_burn_in_steps + cfg.honest_steps:
                if rec.observer not in step_action_cache:
                    action = policy.act(
                        k - (cfg.filter_burn_in_steps + cfg.honest_steps), last_score, cum_bias
                    )
                    cum_bias = np.clip(
                        cum_bias + action * cfg.action_scale_sigma,
                        -cfg.bias_limit_sigma, cfg.bias_limit_sigma,
                    )
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
                consensus_opt, correctness_opt = _lookup_poise_scores(
                    transaction, peers_at_broadcast
                )
                unavailable = consensus_opt is None
                if consensus_opt is None or correctness_opt is None:
                    consensus_score, correctness_score = last_score, last_correctness
                else:
                    consensus_score, correctness_score = consensus_opt, correctness_opt
                last_score, last_correctness = consensus_score, correctness_score

                reputation_ok = reputation > cfg.reputation_floor
                score_ok = consensus_score > cfg.score_floor
                constraint_ok = reputation_ok and score_ok

                drift = float(np.linalg.norm(cum_bias))
                cost = (
                    max(0.0, cfg.reputation_floor - reputation)
                    + max(0.0, cfg.score_floor - consensus_score)
                )
                reward = (
                    (cfg.drift_reward_scale * drift) if constraint_ok
                    else -cfg.violation_penalty
                )

                rows["actions"].append(action)
                rows["rewards"].append(reward)
                rows["costs"].append(cost)
                rows["scores"].append(consensus_score)
                rows["correctness_scores"].append(correctness_score)
                rows["reps"].append(reputation)
                rows["step_idx"].append(k)
                # The state includes the consensus score so that the policy can
                # perceive its proximity to the score floor and learn a
                # corrective action.
                rows["states"].append(np.concatenate([cum_bias, [reputation, consensus_score]]))
                # Intermediate constraint violations are not episode boundaries;
                # only the final transition of the episode is marked terminal,
                # below, so the critic still bootstraps the future cost of
                # remaining in violation.
                rows["terminals"].append(False)
                rows["score_unavailable"].append(unavailable)
            elif k > cfg.filter_burn_in_steps:
                satellites[rec.observer].load_sensor_data(rec)
                await satellites[rec.observer].submit_transaction(rec.target)

    n = len(rows["actions"])
    if n > 0:
        rows["terminals"][-1] = True

    final_ledger_sizes = [len(sat.dag.ledger) - 2 for sat in satellites.values()]

    diagnostics = {
        "mean_peers": float(np.mean(peer_counts)) if peer_counts else 0.0,
        "min_peers": float(np.min(peer_counts)) if peer_counts else 0.0,
        "max_peers": float(np.max(peer_counts)) if peer_counts else 0.0,
        "mean_final_ledger_size": float(np.mean(final_ledger_sizes)),
        "min_final_ledger_size": float(np.min(final_ledger_sizes)),
    }

    if n == 0:
        return {}, diagnostics

    states = np.asarray(rows["states"], dtype=np.float32).reshape(n, -1)
    data = {
        "states": states,
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, ACTION_DIM),
        "rewards": np.asarray(rows["rewards"], dtype=np.float32),
        "costs": np.asarray(rows["costs"], dtype=np.float32),
        "next_states": np.vstack([states[1:], states[-1:]]).astype(np.float32),
        "scores": np.asarray(rows["scores"], dtype=np.float32),
        "correctness_scores": np.asarray(rows["correctness_scores"], dtype=np.float32),
        "reps": np.asarray(rows["reps"], dtype=np.float32),
        "step_idx": np.asarray(rows["step_idx"], dtype=np.int32),
        "episode_ids": np.full(n, ep, dtype=np.int32),
        "terminals": np.asarray(rows["terminals"], dtype=bool),
        "score_unavailable": np.asarray(rows["score_unavailable"], dtype=bool),
        "behaviour_mode": np.full(n, str(mode)),
    }
    return data, diagnostics


def run_episode(ep: int, cfg: HarvestConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run one episode synchronously.

    Args:
        ep: Episode index.
        cfg: Harvest configuration.

    Returns:
        The ``(data, diagnostics)`` tuple produced by the episode simulation.
    """
    rng = np.random.default_rng(cfg.seed + ep)
    return asyncio.run(_simulate_episode_async(ep, cfg, rng))


def harvest_ledger_dataset(cfg: HarvestConfig) -> None:  # pylint: disable=too-many-locals
    """Harvest a dataset across all episodes and write it to disk.

    Args:
        cfg: Harvest configuration.

    Returns:
        None. A compressed ``.npz`` archive is written to ``cfg.output_path``,
        and connectivity, violation, and scoring diagnostics are logged.
    """
    logger.info("Generating transparent DAG dataset across %d episodes...", cfg.num_episodes)
    episodes = []
    all_diag = []
    for ep in range(cfg.num_episodes):
        ep_data, diag = run_episode(ep, cfg)
        if ep_data:
            episodes.append(ep_data)
        all_diag.append(diag)
        logger.info("Completed %d/%d episodes", ep + 1, cfg.num_episodes)

    if not episodes:
        logger.error("No valid episodes harvested.")
        return

    merged = {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0]}

    # The violation fraction is read from the reward sign; the terminal flag now
    # marks only true episode ends, not violations.
    frac_violating = float((merged["rewards"] < 0).mean())
    frac_episode_end = float(merged["terminals"].mean())
    frac_unscored = float(merged["score_unavailable"].mean())
    frac_identical = float(np.isclose(merged["scores"], merged["correctness_scores"]).mean())
    modes, counts = np.unique(merged["behaviour_mode"], return_counts=True)

    mean_peers = float(np.mean([d["mean_peers"] for d in all_diag]))
    min_peers = float(np.min([d["min_peers"] for d in all_diag]))
    max_peers = float(np.max([d["max_peers"] for d in all_diag]))
    mean_ledger = float(np.mean([d["mean_final_ledger_size"] for d in all_diag]))
    min_ledger = float(np.min([d["min_final_ledger_size"] for d in all_diag]))

    logger.info(
        "Behaviour-mode mix across episodes: %s", dict(zip(modes.tolist(), counts.tolist()))
    )
    logger.info(
        "Fraction of transitions violating the constraint (rep<=%.2f or score<=%.2f): %.3f",
        cfg.reputation_floor, cfg.score_floor, frac_violating,
    )
    logger.info(
        "Fraction of rows flagged as true episode-end terminals: %.4f "
        "(sanity check: roughly 1/attack_steps per episode, not large)",
        frac_episode_end,
    )
    logger.info(
        "Fraction of rows with no BFT quorum yet (score carried forward): %.3f", frac_unscored
    )
    logger.info("Fraction of rows where consensus_score == correctness_score: %.3f", frac_identical)
    logger.info(
        "[CONNECTIVITY] mean peers/sat/step: %.2f (min %.0f, max %.0f) at isl_range_m=%.0f",
        mean_peers, min_peers, max_peers, cfg.isl_range_m,
    )
    logger.info(
        "[CONNECTIVITY] final per-satellite ledger size (non-genesis), averaged over episodes: "
        "mean %.1f, worst-episode-min %.1f (need >= 4 for quorum)",
        mean_ledger, min_ledger,
    )

    if mean_peers < 2.0:
        logger.warning(
            "Mean peer count per satellite per step is %.2f at isl_range_m=%.0f; this "
            "constellation "
            "is close to disconnected most of the time, which is the likely reason %.1f%% of rows "
            "never reached BFT quorum. Increase isl_range_m or check the orbit geometry.",
            mean_peers, cfg.isl_range_m, frac_unscored * 100,
        )
    elif frac_unscored > cfg.max_allowed_unscored_fraction:
        msg = (
            f"{frac_unscored:.1%} of rows never reached BFT quorum despite mean peer count of "
            f"{mean_peers:.1f}; connectivity looks adequate, so another cause is likely (for "
            f"example transactions failing validation before reaching add_tx)."
        )
        if cfg.strict_score_check:
            raise RuntimeError(msg)
        logger.warning(msg)

    if frac_unscored > 0.3:
        logger.warning(
            "Dataset is not ready for training while %.1f%% of scores are carried-forward "
            "placeholders rather than real PoISE output; resolve connectivity first.",
            frac_unscored * 100,
        )

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    np.savez_compressed(cfg.output_path, **merged)  # type: ignore[arg-type]
    logger.info("Dataset harvesting complete. Saved to %s", cfg.output_path)


if __name__ == "__main__":
    harvest_ledger_dataset(HarvestConfig())
