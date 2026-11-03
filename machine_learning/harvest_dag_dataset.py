"""
Dataset Harvester: Transparent DAG Data Scraping (Asynchronous)
"""
import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.consensus_mech import ConsensusMechanism
from src.filters.ekf import JointEKF, _ekf_update, _log_nis
from src.filters.filter_interface import (
    FilterConfig,
    ObservationRecord,
    simulate_truth_and_meas,
)
from src.satellite_node import SatelliteNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DAGHarvester")

ACTION_DIM: int = 2
BEHAVIOUR_MODES: Tuple[str, ...] = ("mixed", "persistent", "ramp_hold", "backoff")

@dataclass
class HarvestConfig:
    num_episodes: int = 40
    output_path: str = "machine_learning/dag_harvested_dataset.npz"
    seed: int = 0
    num_sats: int = 10
    dt_s: float = 60.0
    sig_r_m: float = 10.0
    sig_rdot_mps: float = 0.2
    q_acc_target: float = 1e-5
    isl_range_m: float = 1000e3
    num_compromised: int = 3
    filter_burn_in_steps: int = 30
    honest_steps: int = 60
    attack_steps: int = 300
    action_scale_sigma: float = 0.25
    bias_limit_sigma: float = 15.0
    score_floor: Optional[float] = 0.3
    reputation_floor: float = 0.5
    detection_rule: str = "reputation_only"
    mode_probabilities: Tuple[float, ...] = (0.4, 0.2, 0.15, 0.25)
    include_context_features: bool = True
    walker_delta: bool = False


class BehaviourPolicy:
    def __init__(self, mode: str, rng: np.random.Generator, steps: int, score_floor: float) -> None:
        self.mode = mode
        self.rng = rng
        self.steps = steps
        self.retreat_threshold = score_floor + 0.1
        self._direction = self._random_direction()
        self._magnitude = float(self.rng.uniform(0.1, 1.0))
        self._hold_remaining = 0

    def _random_direction(self) -> np.ndarray:
        vec = self.rng.normal(size=ACTION_DIM)
        return (vec / max(float(np.linalg.norm(vec)), 1e-8)).astype(np.float32)

    def act(self, t: int, last_score: float, cum_bias: np.ndarray) -> np.ndarray:
        if self.mode == "mixed":
            draw = self.rng.random()
            if draw < 0.5: return self.rng.uniform(-1.0, 1.0, ACTION_DIM)
            if draw < 0.8: return self.rng.uniform(-0.2, 0.2, ACTION_DIM)
            mask = self.rng.choice([0.0, 1.0], ACTION_DIM)
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM) * mask
        else:
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM)


def inject_bias_simple(record: ObservationRecord, bias_si: np.ndarray) -> ObservationRecord:
    nis = max(0.1, record.nis + float(np.sum(bias_si**2) / 100.0))
    return dataclasses.replace(record, nis=nis, range_m=float(record.range_m + bias_si[0]))


async def _simulate_episode_async(ep: int, cfg: HarvestConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    engine = ConsensusMechanism()
    fcfg = FilterConfig(
        ISL_range_m=cfg.isl_range_m, N=cfg.num_sats, steps=cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps,
        dt=cfg.dt_s, sig_r=cfg.sig_r_m, sig_rdot=cfg.sig_rdot_mps, q_acc_target=cfg.q_acc_target, seed=cfg.seed + ep, walker_delta=cfg.walker_delta
    )

    # Run real truth and filter simulation
    truth, z_hist = simulate_truth_and_meas(fcfg)
    jekf = JointEKF(fcfg, truth[0])

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))

    policy = BehaviourPolicy("mixed", rng, cfg.attack_steps, cfg.score_floor or 0.3)
    cum_bias = np.zeros(ACTION_DIM, dtype=np.float32)

    keys = ("states", "actions", "rewards", "next_states", "scores", "correctness_scores", "reps", "step_idx")
    rows: Dict[str, List] = {k: [] for k in keys}

    for k in range(fcfg.steps):
        jekf.predict()
        y = _ekf_update(jekf.ekf, z_hist[k], cfg.num_sats)
        records = _log_nis(y, jekf.ekf, k, fcfg)

        # Assign dynamic peers based on truth position
        for sid, sat in satellites.items():
            sat.update_position(truth[k, sid*6:(sid+1)*6])

        for sid, sat in satellites.items():
            sat.peers = [other_sat for other_sid, other_sat in satellites.items()
                         if other_sid != sid and np.linalg.norm(truth[k, sid*6:sid*6+3] - truth[k, other_sid*6:other_sid*6+3]) <= cfg.isl_range_m]

        for rec in records:
            if rec.observer in comp_ids and k > cfg.filter_burn_in_steps + cfg.honest_steps:
                action = policy.act(k - (cfg.filter_burn_in_steps + cfg.honest_steps), 0.5, cum_bias)
                cum_bias = np.clip(cum_bias + action * cfg.action_scale_sigma, -cfg.bias_limit_sigma, cfg.bias_limit_sigma)
                bias_si = cum_bias * np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

                biased_record = inject_bias_simple(rec, bias_si)
                satellites[rec.observer].load_sensor_data(biased_record)
                await satellites[rec.observer].submit_transaction(rec.target)

                rows["actions"].append(action)
                rows["scores"].append(0.5)
                rows["correctness_scores"].append(0.6)
                rows["reps"].append(satellites[rec.observer].reputation)
                rows["step_idx"].append(k)
                rows["states"].append(np.concatenate([cum_bias, [satellites[rec.observer].reputation]]))
            elif k > cfg.filter_burn_in_steps:
                satellites[rec.observer].load_sensor_data(rec)
                await satellites[rec.observer].submit_transaction(rec.target)

    n = len(rows["actions"])
    if n == 0:
        return {}

    return {
        "states": np.asarray(rows["states"], dtype=np.float32).reshape(n, -1),
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, ACTION_DIM),
        "rewards": np.zeros(n, dtype=np.float32),
        "next_states": np.asarray(rows["states"], dtype=np.float32).reshape(n, -1),
        "scores": np.asarray(rows["scores"], dtype=np.float32),
        "correctness_scores": np.asarray(rows["correctness_scores"], dtype=np.float32),
        "reps": np.asarray(rows["reps"], dtype=np.float32),
        "step_idx": np.asarray(rows["step_idx"], dtype=np.int32)
    }

def run_episode(ep: int, cfg: HarvestConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + ep)
    return asyncio.run(_simulate_episode_async(ep, cfg, rng))

def harvest_ledger_dataset(cfg: HarvestConfig) -> None:
    logger.info("Generating transparent DAG dataset across %d episodes...", cfg.num_episodes)
    episodes = []
    for ep in range(cfg.num_episodes):
        ep_data = run_episode(ep, cfg)
        if ep_data:
            episodes.append(ep_data)
        logger.info("Completed %d/%d episodes", ep + 1, cfg.num_episodes)

    if not episodes:
        logger.error("No valid episodes harvested.")
        return

    merged = {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    np.savez_compressed(cfg.output_path, **merged)
    logger.info("Dataset harvesting complete! Saved to %s", cfg.output_path)

if __name__ == "__main__":
    harvest_ledger_dataset(HarvestConfig())