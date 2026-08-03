"""
Model 4: Collusive Sybil Drift (BFT Exploitation)
Description: The agent controls a quorum of three compromised nodes. It submits 
identical coordinated offsets across the network to artificially inflate the geometric 
novelty and bypass the local BFT thresholds via cross-validation.
"""

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import math
import json
import dataclasses
import logging
import matplotlib.pyplot as plt

from src.consensus_mech import ConsensusMechanism
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata, TransactionAddresses
from src.filters.filter_interface import ObservationRecord

class SybilDriftEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20000.0, high=20000.0, shape=(6,), dtype=np.float32)

        self.consensus_engine = ConsensusMechanism()
        self.max_steps = 360
        self.current_step = 0
        self.sim_clock = 1600000000.0
        
        self.compromised_ids = [97, 98, 99]
        self.target_node_id = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.compromised_nodes = [
            SatelliteNode(node_id=nid, consensus_mech=self.consensus_engine) 
            for nid in self.compromised_ids
        ]

        self.network_nis_dict = {nid: 2.366 for nid in self.compromised_ids}
        self.cumulative_r_offset = np.zeros(3, dtype=np.float32)

        self.episode_history = {
            "nominal_x": [], "nominal_y": [], "nominal_z": [],
            "attack_x": [], "attack_y": [], "attack_z": [],
            "rewards": []
        }

        radius = 7000.0
        orbital_rate = (2.0 * math.pi) / 5400.0
        initial_angle = 0.0

        self.nominal_r_vector = [radius * math.cos(initial_angle), radius * math.sin(initial_angle), radius * 0.1 * math.sin(initial_angle)]
        self.nominal_v_vector = [-radius * orbital_rate * math.sin(initial_angle), radius * orbital_rate * math.cos(initial_angle), radius * 0.1 * 2.0 * orbital_rate * math.cos(initial_angle)]

        for node in self.compromised_nodes:
            for i in range(4):
                warmup_step = i - 4
                warmup_angle = (warmup_step / self.max_steps) * (2.0 * math.pi)
                warmup_r_vector = [radius * math.cos(warmup_angle), radius * math.sin(warmup_angle), radius * 0.1 * math.sin(warmup_angle)]
                warmup_v_vector = [-radius * orbital_rate * math.sin(warmup_angle), radius * orbital_rate * math.cos(warmup_angle), radius * 0.1 * orbital_rate * math.cos(warmup_angle)]
                
                nominal_tx = self._build_transaction(node, warmup_r_vector, warmup_v_vector, 2.366)
                self.consensus_engine.proof_of_inter_satellite_evaluation(
                    dag=node.dag, sat_node=node,
                    transaction=nominal_tx, mean_nis_per_satellite=self.network_nis_dict
                )

        self.current_state = np.concatenate([self.nominal_r_vector, self.cumulative_r_offset], dtype=np.float32)
        return self.current_state, {}

    def _build_transaction(self, node: SatelliteNode, r_vector, v_vector, step_nis):
        self.sim_clock += 0.1
        obs_record = ObservationRecord(
            step=self.current_step, observer=node.id, target=self.target_node_id, time=self.sim_clock,
            r_vector=list(r_vector), v_vector=list(v_vector), nis=step_nis, dof=3
        )
        tx_data_json = json.dumps(dataclasses.asdict(obs_record))
        metadata = TransactionMetadata()
        metadata.observer_id = node.id
        addresses = TransactionAddresses(sender_address=hash(node.id), recipient_address=self.target_node_id, sender_private_key="KEY")
        parent1, parent2 = node.dag.get_parents()
        return Transaction(addresses=addresses, tx_data=tx_data_json, metadata=metadata, parent_hashes=(parent1, parent2))

    def step(self, action):
        self.current_step += 1
        angle = (self.current_step / self.max_steps) * (2.0 * math.pi)
        radius = 7000.0
        orbital_rate = (2.0 * math.pi) / 5400.0

        self.nominal_r_vector = [radius * math.cos(angle), radius * math.sin(angle), radius * 0.1 * math.sin(angle)]
        self.nominal_v_vector = [-radius * orbital_rate * math.sin(angle), radius * orbital_rate * math.cos(angle), radius * 0.1 * orbital_rate * math.cos(angle)]

        scaled_action = action * 15.0
        self.cumulative_r_offset += scaled_action

        malicious_r_vector = [self.nominal_r_vector[i] + self.cumulative_r_offset[i] for i in range(3)]
        step_distance = float(np.linalg.norm(scaled_action))
        dynamic_nis = (step_distance / 15.0)**2

        node_scores = []
        for node in self.compromised_nodes:
            mock_transaction = self._build_transaction(node, malicious_r_vector, self.nominal_v_vector, dynamic_nis)
            consensus_reached, _ = self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=node.dag, sat_node=node,
                transaction=mock_transaction, mean_nis_per_satellite=self.network_nis_dict
            )
            consensus_state = node.dag.local_consensus_states.get(mock_transaction.hash, {})
            node_scores.append(consensus_state.get("consensus_score", 0.0))

        avg_score = sum(node_scores) / len(node_scores)
        drift_distance = float(np.linalg.norm(self.cumulative_r_offset))
        
        if avg_score < 0.3:
            reward = -10.0 * (0.3 - avg_score)
        else:
            reward = math.log1p(drift_distance) * 5.0 + (avg_score * 2.0)

        self.episode_history["nominal_x"].append(self.nominal_r_vector[0])
        self.episode_history["nominal_y"].append(self.nominal_r_vector[1])
        self.episode_history["nominal_z"].append(self.nominal_r_vector[2])
        self.episode_history["attack_x"].append(malicious_r_vector[0])
        self.episode_history["attack_y"].append(malicious_r_vector[1])
        self.episode_history["attack_z"].append(malicious_r_vector[2])
        self.episode_history["rewards"].append(reward)

        self.current_state = np.concatenate([malicious_r_vector, self.cumulative_r_offset], dtype=np.float32)
        
        return self.current_state, reward, False, self.current_step >= self.max_steps, {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler("sybil_attack.log", mode="w", encoding="utf-8"), logging.StreamHandler()])
    logging.getLogger("src").setLevel(logging.WARNING)
    logger = logging.getLogger("PoISELogger")

    env = SybilDriftEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

    logger.info("Training Collusive Sybil Drift Model...")
    model.learn(total_timesteps=300000)
    model.save("sybil_drift_injector")

    obs, info = env.reset()
    for i in range(360):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        logger.info(f"Step {i+1} | Action: {np.round(action, 4)} | Reward: {reward:.4f} | Cumulative Drift: {np.linalg.norm(env.cumulative_r_offset):.2f} km")
        if truncated: break

    history = env.episode_history
    np.savez_compressed("sybil_attack_log.npz",
                        nominal_track=np.array([history["nominal_x"], history["nominal_y"], history["nominal_z"]]),
                        attack_track=np.array([history["attack_x"], history["attack_y"], history["attack_z"]]),
                        rewards=np.array(history["rewards"]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(history["nominal_x"], history["nominal_y"], history["nominal_z"], label='True Orbital Track', color='blue', linestyle='--')
    ax.plot(history["attack_x"], history["attack_y"], history["attack_z"], label='Sybil Quorum Track', color='red', marker='o', markersize=3)
    ax.set_title('AI-Driven Collusive Sybil Attack')
    ax.legend()
    plt.tight_layout()
    plt.show()
