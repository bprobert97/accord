"""
Dataset Harvester: Transparent DAG Data Scraping

Description:
Simulates multiple orbital periods across a 10-satellite constellation to populate
the Directed Acyclic Graph (DAG) with both nominal and exploratory transactions.

RESEARCH ABSTRACTION NOTE:
In a real-world scenario, an adversary would download a raw, historical DAG from
the network and subsequently reverse-engineer the state-action-reward tuples.
To optimise computational efficiency, this script abstracts that two-step process.
It simultaneously simulates the orbital environment (generating the ledger history)
and acts as the attacker extracting the data, structuring it directly into transition
buffers for Offline Reinforcement Learning and Decision Transformer training.
"""

import dataclasses
import json
import logging
import math
from typing import List

import numpy as np

from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import ObservationRecord
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionAddresses, TransactionMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DAGHarvester")


def build_los_transaction(
    node: SatelliteNode,
    target_id: int,
    step: int,
    sim_clock: float,
    obs_r: List[float],
    obs_v: List[float],
    tgt_r: List[float],
    tgt_v: List[float],
    step_nis: float,
) -> Transaction:
    """Constructs a cryptographically formatted transaction with relative LOS unit vectors."""
    rho = [t - o for t, o in zip(tgt_r, obs_r)]
    v_rel = [t - o for t, o in zip(tgt_v, obs_v)]
    r_norm = max(float(np.linalg.norm(rho)), 1e-8)
    v_norm = max(float(np.linalg.norm(v_rel)), 1e-8)

    obs_record = ObservationRecord(
        step=step,
        observer=node.id,
        target=target_id,
        time=sim_clock,
        r_vector=[c / r_norm for c in rho],
        v_vector=[c / v_norm for c in v_rel],
        nis=step_nis,
        dof=2,
    )
    tx_data_json = json.dumps(dataclasses.asdict(obs_record))
    metadata = TransactionMetadata()
    metadata.observer_id = node.id
    addresses = TransactionAddresses(
        sender_address=hash(node.id),
        recipient_address=target_id,
        sender_private_key="KEY",
    )
    parent1, parent2 = node.dag.get_parents()
    return Transaction(
        addresses=addresses,
        tx_data=tx_data_json,
        metadata=metadata,
        parent_hashes=(parent1, parent2),
    )


def harvest_ledger_dataset(
    num_episodes: int = 40,
    steps_per_episode: int = 360,
    output_path: str = "machine_learning/dag_harvested_dataset.npz",
) -> None:
    """
    Executes multi-orbit simulations to populate a transparent DAG, then extracts
    the complete state-action-reward-next_state trajectory tuples.
    """
    consensus_engine = ConsensusMechanism()
    target_id = 1
    honest_ids = [2, 3, 4, 5, 6, 7, 8]
    compromised_ids = [97, 98, 99]

    # Offline Dataset Storage
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[bool] = []
    episode_lengths: List[int] = []

    logger.info("Generating transparent DAG dataset across %d orbital passes...", num_episodes)

    for ep in range(num_episodes):
        sim_clock = 1600000000.0 + (ep * 5400.0)
        honest_nodes = [SatelliteNode(node_id=nid, consensus_mech=consensus_engine) for nid in honest_ids]
        compromised_nodes = [SatelliteNode(node_id=nid, consensus_mech=consensus_engine) for nid in compromised_ids]
        all_nodes = honest_nodes + compromised_nodes
        network_nis_dict = {node.id: 1.386 for node in all_nodes}

        radius = 7000.0
        orbital_rate = (2.0 * math.pi) / 5400.0
        cum_offset = np.zeros(3, dtype=np.float32)

        # Warmup the DAG to establish baseline PoE
        # This provides the network with initial historical behaviour so the EMA
        # reputation system is populated prior to the attack simulation.
        for node in all_nodes:
            for i in range(4):
                w_step = i - 4
                w_ang = (w_step / steps_per_episode) * (2.0 * math.pi)
                w_r = [radius * math.cos(w_ang), radius * math.sin(w_ang), radius * 0.1 * math.sin(w_ang)]
                w_v = [-radius * orbital_rate * math.sin(w_ang), radius * orbital_rate * math.cos(w_ang), radius * 0.1 * orbital_rate * math.cos(w_ang)]
                sim_clock += 0.01
                tx = build_los_transaction(node, target_id, 0, sim_clock, w_r, w_v, w_r, w_v, 1.386)
                _, new_ema = consensus_engine.proof_of_inter_satellite_evaluation(node.dag, node, tx, network_nis_dict)
                if new_ema:
                    network_nis_dict[node.id] = new_ema

        initial_rep = sum(n.reputation for n in compromised_nodes) / len(compromised_nodes)
        nom_r = [radius, 0.0, 0.0]
        current_state = np.concatenate([nom_r, cum_offset, [initial_rep]], dtype=np.float32)

        ep_step_count = 0

        for step in range(1, steps_per_episode + 1):
            ang = (step / steps_per_episode) * (2.0 * math.pi)
            nom_r = [radius * math.cos(ang), radius * math.sin(ang), radius * 0.1 * math.sin(ang)]
            nom_v = [-radius * orbital_rate * math.sin(ang), radius * orbital_rate * math.cos(ang), radius * 0.1 * orbital_rate * math.cos(ang)]

            # 1. Honest Broadcast
            # Simulate legitimate nodes transmitting baseline telemetry with standard
            # hardware noise. This creates the baseline data that the attacker must hide within.
            for node in honest_nodes:
                p_noise = np.random.normal(0, 0.01, 3)
                v_noise = np.random.normal(0, 0.0001, 3)
                h_r = [nom_r[i] + float(p_noise[i]) for i in range(3)]
                h_v = [nom_v[i] + float(v_noise[i]) for i in range(3)]
                h_nis = max(0.1, (float(np.linalg.norm(p_noise)) / 0.01) ** 2)
                sim_clock += 0.01
                tx = build_los_transaction(node, target_id, step, sim_clock, h_r, h_v, nom_r, nom_v, h_nis)
                _, new_ema = consensus_engine.proof_of_inter_satellite_evaluation(node.dag, node, tx, network_nis_dict)
                if new_ema:
                    network_nis_dict[node.id] = new_ema

            # 2. Exploratory Malicious Actions (Mixed Strategy for Diverse Dataset)
            # Instead of scraping an existing external ledger, we simulate the historical
            # behaviour of compromised nodes trying different data injection strategies.
            # 50% coordinated drift, 30% small stealth probing, 20% random noisy errors.
            rand_mode = np.random.rand()
            if rand_mode < 0.5:
                raw_action = np.random.uniform(-1.0, 1.0, 3).astype(np.float32)
            elif rand_mode < 0.8:
                raw_action = np.random.uniform(-0.2, 0.2, 3).astype(np.float32)
            else:
                raw_action = np.random.uniform(-1.0, 1.0, 3).astype(np.float32) * np.random.choice([0.0, 1.0], 3)

            scaled_action = raw_action * 15.0
            cum_offset = np.clip(cum_offset + scaled_action, -20000.0, 20000.0)
            mal_r = [nom_r[i] + float(cum_offset[i]) for i in range(3)]
            d_nis = (float(np.linalg.norm(scaled_action)) / 15.0) ** 2

            scores = []
            for node in compromised_nodes:
                sim_clock += 0.01
                tx = build_los_transaction(node, target_id, step, sim_clock, mal_r, nom_v, nom_r, nom_v, d_nis)
                _, new_ema = consensus_engine.proof_of_inter_satellite_evaluation(node.dag, node, tx, network_nis_dict)
                if new_ema:
                    network_nis_dict[node.id] = new_ema
                st = node.dag.local_consensus_states.get(tx.hash, {})
                scores.append(st.get("consensus_score", 0.0))

            avg_score = sum(scores) / len(scores)
            avg_rep = sum(n.reputation for n in compromised_nodes) / len(compromised_nodes)
            drift_dist = float(np.linalg.norm(cum_offset))

            # 3. Dynamic Reward Calculation
            # Evaluate the attacker's success against the dual-boundary stealth constraints.
            # In reality, the attacker infers this after analysing the scraped DAG.
            # Here, we calculate it concurrently to bypass secondary parsing scripts.
            if avg_score < 0.3 or avg_rep < 0.5:
                reward = -10.0 * max(0.3 - avg_score, 0.5 - avg_rep)
            else:
                reward = math.log1p(drift_dist) * 5.0 + (avg_score * 2.0) + (avg_rep * 2.0)

            next_state = np.concatenate([mal_r, cum_offset, [avg_rep]], dtype=np.float32)
            is_done = step == steps_per_episode

            # 4. Markov Decision Process (MDP) Tuple Extraction
            # Store the state, action, and reward directly into the buffer, representing
            # the final extracted dataset the adversary uses for offline training.
            all_states.append(current_state)
            all_actions.append(raw_action)
            all_rewards.append(reward)
            all_next_states.append(next_state)
            all_dones.append(is_done)

            current_state = next_state
            ep_step_count += 1

        episode_lengths.append(ep_step_count)

    # 5. Export for Offline AI Architectures
    # Compress the extracted tuples into the required format for PyTorch / Stable Baselines3.
    np.savez_compressed(
        output_path,
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_states=np.array(all_next_states, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool),
        episode_lengths=np.array(episode_lengths, dtype=np.int32),
    )
    logger.info("Dataset harvesting complete! Saved %d total transitions to %s", len(all_states), output_path)


if __name__ == "__main__":
    harvest_ledger_dataset()
