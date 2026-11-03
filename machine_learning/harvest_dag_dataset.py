"""
Dataset Harvester: Transparent DAG Data Scraping (Asynchronous)
"""

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import (
    STATE_DIM,
    FilterConfig,
    ObservationRecord
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

    def act(self) -> np.ndarray:
        if self.mode == "mixed":
            draw = self.rng.random()
            if draw < 0.5: return self.rng.uniform(-1.0, 1.0, ACTION_DIM)
            if draw < 0.8: return self.rng.uniform(-0.2, 0.2, ACTION_DIM)
            mask = self.rng.choice([0.0, 1.0], ACTION_DIM)
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM) * mask
        else:
            return self.rng.uniform(-1.0, 1.0, ACTION_DIM)

def inject_bias_simple(record: ObservationRecord, bias_si: np.ndarray) -> ObservationRecord:
    # A simplified injection that directly offsets the range and recomputes NIS linearly for harvester efficiency
    nis = max(0.1, record.nis + float(np.sum(bias_si**2) / 100.0))
    return dataclasses.replace(record, nis=nis, range_m=float(record.range_m + bias_si[0]))

async def _simulate_episode_async(ep: int, cfg: HarvestConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    engine = ConsensusMechanism()
    fcfg = FilterConfig(
        ISL_range_m=cfg.isl_range_m, N=cfg.num_sats, steps=cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps,
        dt=cfg.dt_s, sig_r=cfg.sig_r_m, sig_rdot=cfg.sig_rdot_mps, q_acc_target=cfg.q_acc_target, seed=cfg.seed + ep
    )

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))

    policy = BehaviourPolicy("mixed", rng, cfg.attack_steps, cfg.score_floor or 0.3)
    cum_bias = np.zeros(ACTION_DIM, dtype=np.float32)

    keys = ("states", "actions", "rewards", "next_states", "scores", "correctness_scores", "reps", "step_idx")
    rows: Dict[str, List] = {k: [] for k in keys}

    for k in range(fcfg.steps):
        # Update peers
        for sid, sat in satellites.items():
            sat.peers = [s for other, s in satellites.items() if other != sid] # Assuming all in range for demo

        # Async transaction submission
        for sid, sat in satellites.items():
            for peer in sat.peers:
                if k > cfg.filter_burn_in_steps + cfg.honest_steps and sid in comp_ids:
                    # Attacker acts
                    action = policy.act(k, 0.5, cum_bias)
                    cum_bias = np.clip(cum_bias + action * cfg.action_scale_sigma, -cfg.bias_limit_sigma, cfg.bias_limit_sigma)
                    bias_si = cum_bias * np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

                    mock_record = ObservationRecord(step=k, time=k*cfg.dt_s, observer=sid, target=peer.id, nis=2.0, dof=2, r_vector=[1,0,0], v_vector=[0,1,0], range_m=1000.0, observer_r_eci=[0,0,0])
                    biased_record = inject_bias_simple(mock_record, bias_si)
                    sat.load_sensor_data(biased_record)
                    await sat.submit_transaction(peer.id)

                    # Gather network score (mocked as average for speed in generation script)
                    score = 0.5
                    correctness = 0.6
                    rows["actions"].append(action)
                    rows["scores"].append(score)
                    rows["correctness_scores"].append(correctness)
                    rows["reps"].append(sat.reputation)
                    rows["step_idx"].append(k)

                    # Track states (simplified)
                    rows["states"].append(np.concatenate([cum_bias, [sat.reputation]]))
                elif k <= cfg.filter_burn_in_steps + cfg.honest_steps:
                    pass # Honest behaviour

    # Format and return arrays
    n = len(rows["actions"])
    if n == 0:
        return {}

    return {
        "states": np.asarray(rows["states"], dtype=np.float32).reshape(n, -1),
        "actions": np.asarray(rows["actions"], dtype=np.float32).reshape(n, ACTION_DIM),
        "rewards": np.zeros(n, dtype=np.float32),  # Mock reward
        "next_states": np.asarray(rows["states"], dtype=np.float32).reshape(n, -1), # Placeholder
        "scores": np.asarray(rows["scores"], dtype=np.float32),
        "correctness_scores": np.asarray(rows["correctness_scores"], dtype=np.float32),
        "reps": np.asarray(rows["reps"], dtype=np.float32),
        "step_idx": np.asarray(rows["step_idx"], dtype=np.int32)
    }

def run_episode(ep: int, cfg: HarvestConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + ep)
    return asyncio.run(_simulate_episode_async(ep, cfg, rng))
