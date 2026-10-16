"""
Dataset Harvester: Transparent DAG Data Scraping (v3, EKF-driven)

Description:
Simulates a constellation with the framework's own truth propagation and Joint EKF,
submits the resulting observation records to each node's DAG through PoISE, and
records state-action-reward-next_state tuples for Offline Reinforcement Learning.

Why DOF = 2:
The EKF measurement model (`hx_block`) returns [range, range-rate], so every
observation has two degrees of freedom. The compromised nodes therefore act in
measurement space: the action is a drift in the (range, range-rate) bias they add to
the measurements they report. The reported NIS is not a proxy; it is recomputed from
the filter's own innovation and innovation covariance, exactly as the EKF does.

RESEARCH ABSTRACTION NOTE:
In a real-world scenario, an adversary would download a raw, historical DAG from
the network and subsequently reverse-engineer the state-action-reward tuples.
To optimise computational efficiency, this script abstracts that two-step process
by simulating the environment and extracting the transitions concurrently.

Threat-model assumptions (change deliberately, not accidentally):
    * The bias is injected at the record level, after the EKF, as in
      `apply_network_faults`. The joint filter itself ingests clean measurements.
    * A compromised node reports a self-consistent record: NIS, range_m and the
      relative-velocity LOS vector all reflect the biased range / range-rate.
    * The attacker's own position (`observer_r_eci`) is reported truthfully.
"""

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from unittest import mock

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.stats import chi2

from src.consensus_mech import ConsensusMechanism
from src.filters.ekf import JointEKF, H_blocks_target_obs, _ekf_update, _log_nis
from src.filters.filter_interface import (
    STATE_DIM,
    FilterConfig,
    ObservationRecord,
    _generate_noisy_measurements,
    check_line_of_sight,
    propagate_truth_kepler,
    simulate_truth_and_meas,
)
from src.logger import get_logger
from src.satellite_node import SatelliteNode
from src.simulation import (
    LEO_MAX_ALT,
    LEO_MIN_ALT,
    RE,
    generate_random_keplerian_elements,
    keplerian_to_cartesian,
)
from src.transaction import Transaction, TransactionAddresses, TransactionMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DAGHarvester")

OBSERVATION_DOF: int = 2     # Range and range-rate.
ACTION_DIM: int = 2          # Drift of (range bias, range-rate bias).
EXP_POS_CAP: int = 20        # Positive-experience count is normalised by this cap.
EPOCH_START: float = 1_600_000_000.0

BEHAVIOUR_MODES: Tuple[str, ...] = ("mixed", "persistent", "ramp_hold", "backoff")


@dataclass
class HarvestConfig:
    """Configuration for a harvesting run. Biases are expressed in units of measurement sigma."""

    num_episodes: int = 40
    output_path: str = "machine_learning/dag_harvested_dataset.npz"
    seed: int = 0
    num_workers: int = 1                  # Episodes are independent, so they parallelise cleanly.

    # Constellation and filter (mirrors FilterConfig).
    num_sats: int = 10
    dt_s: float = 60.0
    sig_r_m: float = 10.0
    sig_rdot_mps: float = 0.2             # Matches DEFAULT_CONFIG in the demo (NOT FilterConfig's 0.02).
    q_acc_target: float = 1e-5            # Matches DEFAULT_CONFIG in the demo.
    isl_range_m: float = 1000e3           # Records are only submitted between satellites within this true range.
    pool_size: int = 400                  # Random orbits drawn per episode; the best plane-aligned 10 are kept.
    walker_delta: bool = False

    # Episode structure.
    compromised_ids: Optional[Tuple[int, ...]] = None  # None -> the best-connected `num_compromised` nodes.
    num_compromised: int = 3
    filter_burn_in_steps: int = 30        # Filter runs, nothing is submitted (transient).
    honest_steps: int = 60                # Everyone submits honestly to build reputation.
    attack_steps: int = 300               # Steps in which the compromised nodes act.

    # Attacker action model.
    action_scale_sigma: float = 0.25      # Bias change per unit action, in sigma per step.
    bias_limit_sigma: float = 15.0        # Clip on the cumulative bias.
    initial_bias_every: int = 4           # Every n-th episode starts with a bias (0 disables).
    initial_bias_sigma: float = 4.0

    # Reward and episode semantics.
    reputation_floor: float = 0.5         # Neutral reputation (MAX_REPUTATION / 2).
    # "reputation_only": detected when mean reputation < floor (the framework's lasting consequence).
    # "score_or_reputation": the original rule; also detected when mean score < score_floor.
    detection_rule: str = "reputation_only"
    # Score below which the attacker counts as detected. None -> the engine's consensus_threshold (0.3).
    # NOTE: honest nodes average ~0.2 under PoISE, so check the logged honest baseline before relying on 0.3.
    score_floor: Optional[float] = None
    mode_probabilities: Tuple[float, ...] = (0.4, 0.2, 0.15, 0.25)  # Order of BEHAVIOUR_MODES.
    include_context_features: bool = True
    reward_style: str = "level"           # "level" (original) or "progress".
    progress_scale: float = 50.0          # Only used when reward_style == "progress".
    terminate_on_detection: bool = False  # End the episode (true terminal) when detected.
    use_simulated_clock: bool = True      # Patch time.time so reputation decay follows sim time.

    # Filter health gate: episodes whose honest NIS is not roughly chi-squared(2) are discarded.
    min_inband_fraction: float = 0.75     # Fraction of honest NIS inside the 95% band (expect ~0.9+).
    nis_median_range: Tuple[float, float] = (0.5, 4.0)  # Acceptable range for the honest NIS median.


class SimClock:
    """Callable simulated time source, standing in for `time.time` when patched."""

    def __init__(self, start: float) -> None:
        """Initialise the clock at the given epoch time in seconds."""
        self.now = start

    def __call__(self) -> float:
        """Return the current simulated time."""
        return self.now


@contextlib.contextmanager
def patched_time(clock: SimClock, enabled: bool) -> Iterator[None]:
    """Optionally replace `time.time` with the simulated clock for the enclosed block."""
    if not enabled:
        yield
        return
    with mock.patch("time.time", clock):
        yield


@contextlib.contextmanager
def quiet_framework() -> Iterator[None]:
    """Silence the framework's INFO logging and its debug `print` calls during harvesting."""
    framework_logger = get_logger()
    previous_level = framework_logger.level
    framework_logger.setLevel(logging.WARNING)
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            yield
    finally:
        framework_logger.setLevel(previous_level)


class BehaviourPolicy:
    """Temporally correlated exploratory policy used to generate diverse trajectories."""

    def __init__(
        self,
        mode: str,
        rng: np.random.Generator,
        steps: int,
        score_floor: float,
        score_margin: float = 0.1,
    ) -> None:
        """
        Initialise the behaviour policy.

        Args:
            mode: One of BEHAVIOUR_MODES.
            rng: Episode-specific random number generator.
            steps: Number of attack steps, used to time the ramp phase.
            score_floor: The consensus threshold below which a transaction is rejected.
            score_margin: Margin above the floor at which the back-off mode retreats.
        """
        if mode not in BEHAVIOUR_MODES:
            raise ValueError(f"Unknown behaviour mode: {mode}")
        self.mode = mode
        self.rng = rng
        self.steps = steps
        self.retreat_threshold = score_floor + score_margin
        self._direction = self._random_direction()
        self._magnitude = float(self.rng.uniform(0.1, 1.0))
        self._hold_remaining = 0

    def _random_direction(self) -> np.ndarray:
        """Sample a uniformly random unit vector in action space."""
        vec = self.rng.normal(size=ACTION_DIM)
        return (vec / max(float(np.linalg.norm(vec)), 1e-8)).astype(np.float32)

    def _refresh_hold(self) -> None:
        """Pick a new drift direction and magnitude once the current hold expires."""
        if self._hold_remaining <= 0:
            self._direction = self._random_direction()
            self._magnitude = float(self.rng.uniform(0.1, 1.0))
            self._hold_remaining = int(self.rng.integers(20, 80))
        self._hold_remaining -= 1

    def _mixed(self) -> np.ndarray:
        """Per-step mixture: 50% large uniform, 30% small uniform, 20% sparse."""
        draw = self.rng.random()
        if draw < 0.5:
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM)
        if draw < 0.8:
            return self.rng.uniform(-0.2, 0.2, ACTION_DIM)
        mask = self.rng.choice([0.0, 1.0], ACTION_DIM)
        return self.rng.uniform(-1.0, 1.0, ACTION_DIM) * mask

    def act(self, t: int, last_score: float, cum_bias: np.ndarray) -> np.ndarray:
        """
        Produce a raw action in [-1, 1]^2.

        Args:
            t: Attack step index (0-based).
            last_score: Mean consensus score observed on the previous step.
            cum_bias: Current cumulative bias in units of sigma.

        Returns:
            The raw (unscaled) action as a float32 array.
        """
        if self.mode == "mixed":
            action = self._mixed()
        elif self.mode == "persistent":
            self._refresh_hold()
            action = self._direction * self._magnitude + 0.05 * self.rng.normal(size=ACTION_DIM)
        elif self.mode == "ramp_hold":
            # Increase the drift rate for the first third of the episode, then hold the bias.
            ramp_end = max(1, self.steps // 3)
            magnitude = min((t + 1) / ramp_end, 1.0) if t < ramp_end else 0.0
            action = self._direction * magnitude + 0.02 * self.rng.normal(size=ACTION_DIM)
        else:  # "backoff"
            # Drift persistently, but retreat towards zero bias when the previous score nears
            # the rejection threshold. Yields informative data around the boundary.
            self._refresh_hold()
            if last_score < self.retreat_threshold:
                norm = float(np.linalg.norm(cum_bias))
                action = -0.5 * (cum_bias / norm if norm > 1e-8 else self._direction)
            else:
                action = self._direction * self._magnitude
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def pair_innovation(
    ekf: ExtendedKalmanFilter, y: np.ndarray, i: int, j: int, cfg: FilterConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the 2-vector innovation and 2x2 innovation covariance for observer i, target j.

    This replicates `_process_observation_pair`, including its use of the filter's current
    covariance, so an unbiased innovation reproduces the honest record's NIS exactly.
    """
    n = cfg.N
    xi = slice(STATE_DIM * i, STATE_DIM * i + STATE_DIM)
    xj = slice(STATE_DIM * j, STATE_DIM * j + STATE_DIM)
    ht, ho = H_blocks_target_obs(ekf.x[xj], ekf.x[xi])
    s_ij = (ho @ ekf.P[xi, xi] @ ho.T + ho @ ekf.P[xi, xj] @ ht.T
            + ht @ ekf.P[xj, xi] @ ho.T + ht @ ekf.P[xj, xj] @ ht.T
            + np.diag([cfg.sig_r ** 2, cfg.sig_rdot ** 2]))
    idx = 2 * (i * (n - 1) + (j if j < i else j - 1))
    return y[idx:idx + 2], s_ij


def inject_bias(
    record: ObservationRecord, ekf: ExtendedKalmanFilter, y: np.ndarray, cfg: FilterConfig, bias_si: np.ndarray
) -> ObservationRecord:
    """
    Return a self-consistent record in which range and range-rate are biased.

    Args:
        record: The honest record produced by the EKF.
        ekf: The filter object (state and covariance after the update).
        y: The stacked innovation vector for this step.
        cfg: The filter configuration.
        bias_si: Bias as [metres, metres per second].

    Returns:
        A copy of the record with NIS, range and velocity LOS vector updated.
    """
    i, j = record.observer, record.target
    y_ij, s_ij = pair_innovation(ekf, y, i, j, cfg)
    y_biased = y_ij + bias_si
    nis = float(y_biased @ np.linalg.inv(s_ij) @ y_biased)

    rho = ekf.x[STATE_DIM * j: STATE_DIM * j + 3] - ekf.x[STATE_DIM * i: STATE_DIM * i + 3]
    rhat = rho / max(float(np.linalg.norm(rho)), 1e-8)
    vrel = ekf.x[STATE_DIM * j + 3: STATE_DIM * j + 6] - ekf.x[STATE_DIM * i + 3: STATE_DIM * i + 6]
    vrel_biased = vrel + bias_si[1] * rhat      # Shifts the range-rate by the bias along the LOS.
    v_vector = vrel_biased / max(float(np.linalg.norm(vrel_biased)), 1e-8)
    return dataclasses.replace(record, nis=nis, range_m=float(record.range_m + bias_si[0]),
                               v_vector=v_vector.tolist())


def physically_valid(record: ObservationRecord) -> bool:
    """Replicate the altitude test in `_is_physically_valid` (observer and inferred target)."""
    r_obs = np.asarray(record.observer_r_eci, dtype=float)
    target = r_obs + record.range_m * np.asarray(record.r_vector, dtype=float)
    return all(LEO_MIN_ALT <= float(np.linalg.norm(p)) - RE <= LEO_MAX_ALT for p in (r_obs, target))


def build_transaction(node: SatelliteNode, record: ObservationRecord) -> Transaction:
    """Wrap an observation record into a transaction, as a satellite node would."""
    metadata = TransactionMetadata()
    metadata.observer_id = node.id
    addresses = TransactionAddresses(
        sender_address=hash(node.id), recipient_address=record.target, sender_private_key="KEY"
    )
    parent1, parent2 = node.dag.get_parents()
    return Transaction(
        addresses=addresses,
        tx_data=json.dumps(dataclasses.asdict(record)),
        metadata=metadata,
        parent_hashes=(parent1, parent2),
    )


def record_nis_history(node: SatelliteNode, accepted: bool, new_ema: Optional[float]) -> None:
    """
    Mirror the NIS bookkeeping in `DAG.listen`.

    Only accepted transactions update the history, and the stored value is the running
    mean of the returned EMAs (not the latest EMA itself).
    """
    if not (accepted and new_ema is not None):
        return
    metrics = node.dag.nis_metrics
    metrics.sums[node.id] = metrics.sums.get(node.id, 0.0) + new_ema
    metrics.counts[node.id] = metrics.counts.get(node.id, 0) + 1
    metrics.mean_per_satellite[node.id] = metrics.sums[node.id] / metrics.counts[node.id]


def submit_record(
    engine: ConsensusMechanism, node: SatelliteNode, record: ObservationRecord
) -> Tuple[float, bool]:
    """Submit a record through PoISE on the node's own DAG; return (consensus score, accepted)."""
    tx = build_transaction(node, record)
    accepted, new_ema = engine.proof_of_inter_satellite_evaluation(
        node.dag, node, tx, node.dag.nis_metrics.mean_per_satellite
    )
    record_nis_history(node, accepted, new_ema)
    score = float(node.dag.local_consensus_states.get(tx.hash, {}).get("consensus_score", 0.0))
    return score, accepted


def build_state(
    cum_bias_sigma: np.ndarray,
    nodes: List[SatelliteNode],
    last_score: float,
    honest_nis: float,
    link_fraction: float,
    cfg: HarvestConfig,
) -> np.ndarray:
    """
    Assemble the state vector from quantities the compromised nodes can observe.

    Base features: cumulative bias in sigma (2) and mean reputation (1).
    Context features: previous consensus score, normalised positive-experience count,
    performance EMA, historical NIS, the current step's un-biased NIS and the fraction
    of the compromised nodes' links that are visible.
    """
    count = float(len(nodes))
    parts: List[np.ndarray] = [
        np.asarray(cum_bias_sigma, dtype=np.float32),
        np.asarray([sum(n.reputation for n in nodes) / count], dtype=np.float32),
    ]
    if cfg.include_context_features:
        exp_pos = sum(min(n.exp_pos, EXP_POS_CAP) for n in nodes) / count / EXP_POS_CAP
        perf_ema = sum(n.performance_ema for n in nodes) / count
        hist_nis = sum(
            n.dag.nis_metrics.mean_per_satellite.get(n.id, float(OBSERVATION_DOF)) for n in nodes
        ) / count
        parts.append(np.asarray(
            [last_score, exp_pos, perf_ema, hist_nis, honest_nis, link_fraction], dtype=np.float32
        ))
    return np.concatenate(parts).astype(np.float32)


def compute_reward(
    avg_score: float, avg_rep: float, drift: float, prev_drift: float, score_floor: float, cfg: HarvestConfig
) -> Tuple[float, bool]:
    """
    Evaluate the stealth-constrained reward.

    Returns:
        A tuple of (reward, detected) where `detected` flags a boundary violation.
    """
    if cfg.detection_rule == "reputation_only":
        detected = avg_rep < cfg.reputation_floor
        deficit = cfg.reputation_floor - avg_rep
    else:
        detected = avg_score < score_floor or avg_rep < cfg.reputation_floor
        deficit = max(score_floor - avg_score, cfg.reputation_floor - avg_rep)
    if detected:
        return -10.0 * deficit, True
    if cfg.reward_style == "progress":
        # Rewards *changes* in bias so credit is attributable to the action taken.
        progress = (math.log1p(drift) - math.log1p(prev_drift)) * cfg.progress_scale
        return progress + avg_score * 2.0 + avg_rep * 2.0, False
    return math.log1p(drift) * 5.0 + avg_score * 2.0 + avg_rep * 2.0, False


def select_aligned_cluster(pool: List[np.ndarray], size: int, rng: np.random.Generator) -> List[int]:
    """
    Pick `size` satellites whose orbital planes best align with a random seed satellite.

    Mirrors the demo's `_cluster_by_orbital_physics`: the angular momentum h = r x v is
    normalised, and the satellites with the largest dot product with the seed are kept,
    so that cluster members stay within line of sight of one another.
    """
    h = np.asarray([np.cross(s[:3], s[3:]) / np.linalg.norm(np.cross(s[:3], s[3:])) for s in pool])
    seed_idx = int(rng.integers(len(pool)))
    alignment = h @ h[seed_idx]
    alignment[seed_idx] = np.inf
    return [int(i) for i in np.argsort(-alignment)[:size]]


def build_episode_truth(cfg: HarvestConfig, fcfg: FilterConfig, base_seed: int,
                        rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate truth and noisy measurements for one episode's cluster.

    Walker-delta runs use the framework's own generator. Otherwise a pool of random orbits is
    drawn with `generate_random_keplerian_elements`, and the best plane-aligned cluster is kept,
    as in the demo's clustered filter.
    """
    np.random.seed(fcfg.seed)  # The framework's measurement noise uses NumPy's global generator.
    if cfg.walker_delta:
        return simulate_truth_and_meas(fcfg, True)
    pool = [keplerian_to_cartesian(generate_random_keplerian_elements(seed=base_seed + n))
            for n in range(cfg.pool_size)]
    ids = select_aligned_cluster(pool, cfg.num_sats, rng)
    truth = propagate_truth_kepler(np.concatenate([pool[i] for i in ids]), fcfg.steps, fcfg.dt)
    return truth, _generate_noisy_measurements(fcfg, truth)


def select_compromised(cfg: HarvestConfig, truth: np.ndarray, start: int, stop: int) -> List[int]:
    """
    Choose the compromised nodes: those given in the config, else the best-connected ones.

    Connectivity is the number of (step, target) pairs inside the ISL range with a clear
    line of sight during the attack window, so the attacker always has links to act on.
    """
    if cfg.compromised_ids is not None:
        return list(cfg.compromised_ids)
    n = cfg.num_sats
    links = np.zeros(n)
    for k in range(start, stop):
        pos = truth[k].reshape(n, STATE_DIM)[:, :3]
        for i in range(n):
            for j in range(n):
                if i != j and np.linalg.norm(pos[j] - pos[i]) <= cfg.isl_range_m and check_line_of_sight(pos[i], pos[j]):
                    links[i] += 1
    return [int(i) for i in np.argsort(-links)[:cfg.num_compromised]]


def filter_is_healthy(nis_values: List[float], cfg: HarvestConfig) -> Tuple[bool, str]:
    """
    Check that the EKF's honest NIS is statistically consistent with chi-squared(2).

    A diverged filter produces enormous (or negative) NIS for every node, which would silently
    poison reputations and rewards, so such episodes are discarded.

    Returns:
        A tuple of (healthy, message); the message explains a failure and is empty otherwise.
    """
    values = np.asarray(nis_values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or values.min() < 0.0:
        return False, "non-finite or negative honest NIS (filter diverged)"
    lo, hi = chi2.ppf(0.025, OBSERVATION_DOF), chi2.ppf(0.975, OBSERVATION_DOF)
    inband = float(np.mean((values >= lo) & (values <= hi)))
    median = float(np.median(values))
    healthy = inband >= cfg.min_inband_fraction and cfg.nis_median_range[0] <= median <= cfg.nis_median_range[1]
    if not healthy:
        return False, f"honest NIS median {median:.3g}, in-band fraction {inband:.2f} (filter unhealthy)"
    return True, ""


def _simulate_episode(
    ep: int, cfg: HarvestConfig, rng: np.random.Generator, clock: SimClock
) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """Run one episode with the clock installed; return (transitions or None, warning messages)."""
    notes: List[str] = []
    engine = ConsensusMechanism()
    score_floor = cfg.score_floor if cfg.score_floor is not None else engine.consensus_threshold
    attack_start = cfg.filter_burn_in_steps + cfg.honest_steps
    total_steps = attack_start + cfg.attack_steps + 1   # +1 supplies the final next-state.

    fcfg = FilterConfig(
        ISL_range_m=cfg.isl_range_m, N=cfg.num_sats, steps=total_steps, dt=cfg.dt_s,
        sig_r=cfg.sig_r_m, sig_rdot=cfg.sig_rdot_mps, q_acc_target=cfg.q_acc_target,
        seed=cfg.seed + ep * 1000, walker_delta=cfg.walker_delta,
    )
    truth, z_hist = build_episode_truth(cfg, fcfg, cfg.seed + ep * 1000, rng)
    jekf = JointEKF(fcfg, truth[0])

    nodes = [SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)]
    comp_ids = select_compromised(cfg, truth, attack_start, total_steps)
    comp_set = set(comp_ids)
    comp_nodes = [nodes[i] for i in comp_ids]
    sigma = np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

    mode = BEHAVIOUR_MODES[int(rng.choice(
        len(BEHAVIOUR_MODES), p=np.asarray(cfg.mode_probabilities) / sum(cfg.mode_probabilities)
    ))]
    policy = BehaviourPolicy(mode, rng, cfg.attack_steps, score_floor)

    cum_bias = np.zeros(ACTION_DIM, dtype=np.float32)
    if cfg.initial_bias_every > 0 and ep % cfg.initial_bias_every == 0:
        cum_bias = rng.uniform(-cfg.initial_bias_sigma, cfg.initial_bias_sigma, ACTION_DIM).astype(np.float32)

    last_score, last_honest_nis, drift = 1.0, float(OBSERVATION_DOF), float(np.linalg.norm(cum_bias))
    comp_links = len(comp_nodes) * (cfg.num_sats - 1)
    honest_invalid = 0
    all_honest_nis: List[float] = []
    base_scores: List[float] = []
    baseline_rep: Optional[float] = None
    base_accepted: List[bool] = []
    pending: Optional[Dict] = None
    stop_after_finalise = False

    keys = ("states", "actions", "rewards", "next_states", "terminals", "timeouts", "scores", "reps",
            "accepted_frac", "phys_valid", "honest_nis", "attacked_nis", "n_tx", "step_idx")
    rows: Dict[str, List] = {k: [] for k in keys}

    for k in range(total_steps):
        jekf.predict()  # As in the demo's _process_cluster_step: predict on every step, including k = 0.
        y = _ekf_update(jekf.ekf, z_hist[k], cfg.num_sats)
        records = _log_nis(y, jekf.ekf, k, fcfg)
        if k < cfg.filter_burn_in_steps:
            continue
        all_honest_nis.extend(r.nis for r in records)   # Filter health uses every record the EKF produces.
        pos_k = truth[k].reshape(cfg.num_sats, STATE_DIM)[:, :3]
        records = [r for r in records if np.linalg.norm(pos_k[r.target] - pos_k[r.observer]) <= cfg.isl_range_m]
        clock.now = EPOCH_START + k * cfg.dt_s

        # ---- Honest phase: everybody submits unmodified records (builds reputation and EMAs).
        if k < attack_start:
            for rec in records:
                honest_invalid += int(not physically_valid(rec))
                score, accepted = submit_record(engine, nodes[rec.observer], rec)
                if rec.observer in comp_set:
                    last_score = score
                    if k >= cfg.filter_burn_in_steps + 3:  # Skip the pre-quorum transactions (score 0).
                        base_scores.append(score)
                        base_accepted.append(accepted)
            continue

        # ---- Attack phase.
        t = k - attack_start
        if baseline_rep is None:
            baseline_rep = float(np.mean([n.reputation for n in comp_nodes]))
        comp_recs = [r for r in records if r.observer in comp_set]
        other_recs = [r for r in records if r.observer not in comp_set]
        if comp_recs:
            last_honest_nis = float(np.mean([r.nis for r in comp_recs]))
        state = build_state(cum_bias, comp_nodes, last_score, last_honest_nis, len(comp_recs) / comp_links, cfg)

        # Finalise the previous transition now that its next-state is known.
        if pending is not None:
            pending["next_state"] = state
            pending["timeout"] = (t == cfg.attack_steps) and not pending["terminals"]
            for key in keys:
                if key in pending:
                    rows[key].append(pending[key])
            rows["next_states"].append(state)
            rows["timeouts"].append(pending["timeout"])
            pending = None
            if stop_after_finalise:
                break
        if t == cfg.attack_steps:
            break

        for rec in other_recs:
            submit_record(engine, nodes[rec.observer], rec)
        if not comp_recs:
            continue  # No visible links this step, so the attacker cannot act.

        # Act: drift the cumulative (range, range-rate) bias and report biased, self-consistent records.
        action = policy.act(t, last_score, cum_bias)
        prev_drift = drift
        cum_bias = np.clip(cum_bias + action * cfg.action_scale_sigma,
                           -cfg.bias_limit_sigma, cfg.bias_limit_sigma).astype(np.float32)
        bias_si = cum_bias.astype(float) * sigma
        drift = float(np.linalg.norm(cum_bias))

        scores: List[float] = []
        accepted_flags: List[bool] = []
        attacked_nis: List[float] = []
        valid_flags: List[bool] = []
        for rec in comp_recs:
            biased = inject_bias(rec, jekf.ekf, y, fcfg, bias_si)
            valid_flags.append(physically_valid(biased))
            attacked_nis.append(biased.nis)
            score, accepted = submit_record(engine, nodes[rec.observer], biased)
            scores.append(score)
            accepted_flags.append(accepted)

        avg_score = sum(scores) / len(scores)
        avg_rep = sum(n.reputation for n in comp_nodes) / len(comp_nodes)
        reward, detected = compute_reward(avg_score, avg_rep, drift, prev_drift, score_floor, cfg)
        terminal = detected and cfg.terminate_on_detection

        pending = {
            "states": state, "actions": action, "rewards": reward, "terminals": terminal,
            "scores": avg_score, "reps": avg_rep, "accepted_frac": sum(accepted_flags) / len(accepted_flags),
            "phys_valid": all(valid_flags), "honest_nis": last_honest_nis,
            "attacked_nis": float(np.mean(attacked_nis)), "n_tx": len(comp_recs), "step_idx": t,
        }
        last_score = avg_score
        stop_after_finalise = terminal

    if honest_invalid:
        notes.append(f"{honest_invalid} honest records failed the altitude check (orbits outside LEO bounds?)")

    n = len(rows["rewards"])
    if n == 0:
        return None, notes + ["no transitions were recorded"]
    healthy, message = filter_is_healthy(all_honest_nis, cfg)
    if not healthy:
        return None, notes + [f"episode discarded: {message}"]
    out: Dict[str, np.ndarray] = {
        "states": np.asarray(rows["states"], dtype=np.float32).reshape(n, -1),
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, ACTION_DIM),
        "rewards": np.asarray(rows["rewards"], dtype=np.float32),
        "next_states": np.asarray(rows["next_states"], dtype=np.float32).reshape(n, -1),
        "terminals": np.asarray(rows["terminals"], dtype=bool),
        "timeouts": np.asarray(rows["timeouts"], dtype=bool),
        "scores": np.asarray(rows["scores"], dtype=np.float32),
        "reps": np.asarray(rows["reps"], dtype=np.float32),
        "accepted_frac": np.asarray(rows["accepted_frac"], dtype=np.float32),
        "phys_valid": np.asarray(rows["phys_valid"], dtype=bool),
        "honest_nis": np.asarray(rows["honest_nis"], dtype=np.float32),
        "attacked_nis": np.asarray(rows["attacked_nis"], dtype=np.float32),
        "n_tx": np.asarray(rows["n_tx"], dtype=np.int32),
        "step_idx": np.asarray(rows["step_idx"], dtype=np.int32),
        "behaviour_id": np.full(n, BEHAVIOUR_MODES.index(mode), dtype=np.int32),
        "baseline_score": np.full(n, float(np.mean(base_scores)) if base_scores else np.nan, dtype=np.float32),
        "baseline_rep": np.full(n, baseline_rep if baseline_rep is not None else np.nan, dtype=np.float32),
        "baseline_accept": np.full(n, float(np.mean(base_accepted)) if base_accepted else np.nan, dtype=np.float32),
        "episode_ids": np.full(n, ep, dtype=np.int32),
    }
    return out, notes


def run_episode(ep: int, cfg: HarvestConfig) -> Optional[Dict[str, np.ndarray]]:
    """
    Simulate one episode and return its transitions as arrays (None if the filter was unhealthy).

    The simulated clock is installed *before* the nodes are built, because each
    ReputationManager records `time.time()` when it is constructed.
    """
    rng = np.random.default_rng(cfg.seed + ep)
    random.seed(cfg.seed + ep)  # DAG tip selection uses the global `random` module.
    clock = SimClock(EPOCH_START + (cfg.filter_burn_in_steps - 1) * cfg.dt_s)
    with quiet_framework(), patched_time(clock, cfg.use_simulated_clock):
        result, notes = _simulate_episode(ep, cfg, rng, clock)
    for note in notes:  # Logged outside the patched clock so timestamps are real.
        logger.warning("Episode %d: %s", ep, note)
    return result


def harvest_ledger_dataset(cfg: HarvestConfig) -> None:
    """Run all episodes, concatenate the transitions and export a compressed archive."""
    logger.info("Generating transparent DAG dataset across %d episodes...", cfg.num_episodes)
    eps = list(range(cfg.num_episodes))
    if cfg.num_workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
            episodes = list(pool.map(run_episode, eps, [cfg] * len(eps)))
    else:
        episodes = []
        for ep in eps:
            episodes.append(run_episode(ep, cfg))
            logger.info("Completed %d/%d episodes", ep + 1, cfg.num_episodes)

    kept = [e for e in episodes if e is not None]
    logger.info("Kept %d/%d episodes (%d discarded as unhealthy).", len(kept), len(episodes), len(episodes) - len(kept))
    if not kept:
        raise RuntimeError("Every episode was discarded: the EKF is diverging. Check sig_rdot_mps and q_acc_target against DEFAULT_CONFIG in the demo.")

    merged = {key: np.concatenate([e[key] for e in kept]) for key in kept[0]}
    episode_lengths = np.asarray([len(e["rewards"]) for e in kept], dtype=np.int32)

    # Sanity: a calibrated EKF should give honest NIS with mean near DOF and few values outside the 95% band.
    lo, hi = chi2.ppf(0.025, OBSERVATION_DOF), chi2.ppf(0.975, OBSERVATION_DOF)
    honest = merged["honest_nis"]
    logger.info(
        "Transitions: %d | penalised: %.3f | mean acceptance: %.3f",
        len(merged["rewards"]), float(np.mean(merged["rewards"] < 0.0)), float(np.mean(merged["accepted_frac"])),
    )
    logger.info(
        "Honest NIS mean: %.3f (expect ~%d) | outside 95%% band: %.3f (expect ~0.05) | attacked NIS mean: %.3f",
        float(honest.mean()), OBSERVATION_DOF, float(np.mean((honest < lo) | (honest > hi))),
        float(merged["attacked_nis"].mean()),
    )
    logger.info("Honest baseline for the compromised nodes (pre-attack): mean score %.3f, acceptance %.3f, reputation %.3f",
                float(np.nanmean(merged["baseline_score"])), float(np.nanmean(merged["baseline_accept"])),
                float(np.nanmean(merged["baseline_rep"])))
    logger.info("Detection rule: %s (reputation floor %.2f, score floor %s)", cfg.detection_rule,
                cfg.reputation_floor, cfg.score_floor if cfg.score_floor is not None else "engine threshold")
    for idx, name in enumerate(BEHAVIOUR_MODES):
        mask = merged["behaviour_id"] == idx
        if mask.any():
            logger.info("  %-10s n=%6d | mean reward %8.3f | mean score %.3f | acceptance %.3f",
                        name, int(mask.sum()), float(merged["rewards"][mask].mean()),
                        float(merged["scores"][mask].mean()), float(merged["accepted_frac"][mask].mean()))

    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.output_path,
        **merged,
        episode_lengths=episode_lengths,
        behaviour_modes=np.asarray(BEHAVIOUR_MODES),
        config_json=np.asarray(json.dumps(dataclasses.asdict(cfg))),
    )
    logger.info("Dataset harvesting complete! Saved to %s", cfg.output_path)


if __name__ == "__main__":
    harvest_ledger_dataset(HarvestConfig())
