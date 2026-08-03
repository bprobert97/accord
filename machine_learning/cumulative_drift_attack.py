import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import math
import json
import dataclasses
import logging
import matplotlib.pyplot as plt

# Import your real ACCORD classes
from src.consensus_mech import ConsensusMechanism
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata, TransactionAddresses
from src.filters.filter_interface import ObservationRecord

class PoISEAttackEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Action Space: Small incremental adjustments to position
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation Space: Current true position + current cumulative drift
        self.observation_space = spaces.Box(
            low=-20000.0, high=20000.0, shape=(6,), dtype=np.float32
        )

        self.consensus_engine = ConsensusMechanism()
        self.max_steps = 100
        self.current_step = 0
        self.sim_clock = 1600000000.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.attacking_node_id = 99
        self.target_node_id = 1

        self.attacking_satellite_node = SatelliteNode(
            node_id=self.attacking_node_id,
            consensus_mech=self.consensus_engine
        )

        self.network_nis_dict = {self.attacking_node_id: 2.366}
        self.cumulative_r_offset = np.zeros(3, dtype=np.float32)

        self.episode_history = {
            "nominal_x": [], "nominal_y": [], "nominal_z": [],
            "attack_x": [], "attack_y": [], "attack_z": [],
            "rewards": []
        }

        self.nominal_r_vector = [7000.0, 0.0, 0.0]
        self.nominal_v_vector = [0.0, 7.5, 0.0]

        # Warm up the DAG instantly
        for i in range(4):
            warmup_angle = (i - 4) * 0.05
            radius = 7000.0

            warmup_r_vector = [
                radius * math.cos(warmup_angle),
                radius * math.sin(warmup_angle),
                radius * 0.5 * math.sin(warmup_angle * 0.5)
            ]
            warmup_v_vector = [
                -radius * 0.05 * math.sin(warmup_angle),
                radius * 0.05 * math.cos(warmup_angle),
                radius * 0.5 * 0.05 * 0.5 * math.cos(warmup_angle * 0.5)
            ]

            nominal_tx = self._build_transaction(warmup_r_vector, warmup_v_vector, 2.366)
            self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=self.attacking_satellite_node.dag,
                sat_node=self.attacking_satellite_node,
                transaction=nominal_tx,
                mean_nis_per_satellite=self.network_nis_dict
            )

        self.current_state = np.concatenate([self.nominal_r_vector, self.cumulative_r_offset], dtype=np.float32)
        return self.current_state, {}

    def _build_transaction(self, r_vector, v_vector, step_nis):
        self.sim_clock += 1.0
        obs_record = ObservationRecord(
            step=self.current_step,
            observer=self.attacking_node_id,
            target=self.target_node_id,
            time=self.sim_clock,
            r_vector=list(r_vector),
            v_vector=list(v_vector),
            nis=step_nis,
            dof=3
        )

        tx_data_json = json.dumps(dataclasses.asdict(obs_record))
        metadata = TransactionMetadata()
        metadata.observer_id = self.attacking_node_id

        addresses = TransactionAddresses(
            sender_address=hash(self.attacking_node_id),
            recipient_address=self.target_node_id,
            sender_private_key="PLACEHOLDER_KEY"
        )

        parent1, parent2 = self.attacking_satellite_node.dag.get_parents()
        return Transaction(
            addresses=addresses,
            tx_data=tx_data_json,
            metadata=metadata,
            parent_hashes=(parent1, parent2)
        )

    def step(self, action):
        self.current_step += 1

        # Base orbital physics
        angle = self.current_step * 0.05
        radius = 7000.0
        self.nominal_r_vector = [
            radius * math.cos(angle),
            radius * math.sin(angle),
            radius * 0.5 * math.sin(angle * 0.5)
        ]
        self.nominal_v_vector = [
            -radius * 0.05 * math.sin(angle),
            radius * 0.05 * math.cos(angle),
            radius * 0.5 * 0.05 * 0.5 * math.cos(angle * 0.5)
        ]

        # Force a baseline movement vector so the agent cannot just output zeros
        scaled_action = action * 5.0  # Allow up to 5.0 km per step
        self.cumulative_r_offset += scaled_action

        malicious_r_vector = [
            self.nominal_r_vector[0] + self.cumulative_r_offset[0],
            self.nominal_r_vector[1] + self.cumulative_r_offset[1],
            self.nominal_r_vector[2] + self.cumulative_r_offset[2]
        ]

        step_distance = float(np.linalg.norm(scaled_action))
        dynamic_nis = (step_distance / 15.0)**2

        mock_transaction = self._build_transaction(
            malicious_r_vector,
            self.nominal_v_vector,
            dynamic_nis
        )

        consensus_reached, _ = self.consensus_engine.proof_of_inter_satellite_evaluation(
            dag=self.attacking_satellite_node.dag,
            sat_node=self.attacking_satellite_node,
            transaction=mock_transaction,
            mean_nis_per_satellite=self.network_nis_dict
        )

        drift_distance = float(np.linalg.norm(self.cumulative_r_offset))

        # Softer penalty + aggressive distance scaling to force outward exploration
        if not consensus_reached:
            reward = -2.0  # Mild nudge back, not a brick wall
        else:
            reward = drift_distance * 10.0  # Massive payoff for cumulative separation

        # Log history
        self.episode_history["nominal_x"].append(self.nominal_r_vector[0])
        self.episode_history["nominal_y"].append(self.nominal_r_vector[1])
        self.episode_history["nominal_z"].append(self.nominal_r_vector[2])
        self.episode_history["attack_x"].append(malicious_r_vector[0])
        self.episode_history["attack_y"].append(malicious_r_vector[1])
        self.episode_history["attack_z"].append(malicious_r_vector[2])
        self.episode_history["rewards"].append(reward)

        self.current_state = np.concatenate([malicious_r_vector, self.cumulative_r_offset], dtype=np.float32)

        terminated = False  # Let the episode run its full course!
        truncated = self.current_step >= self.max_steps

        return self.current_state, reward, terminated, truncated, {}

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)

    env = PoISEAttackEnv()
    print("Training cumulative drift PPO model...")

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    model.learn(total_timesteps=50000)
    print("Training complete!\n")

    print("--- Running Test Episode ---")
    obs, info = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if truncated:
            break

    # Plotting results
    history = env.episode_history
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(history["nominal_x"], history["nominal_y"], history["nominal_z"],
            label='True Orbital Track', color='blue', linestyle='--', linewidth=2)
    ax.plot(history["attack_x"], history["attack_y"], history["attack_z"],
            label='Drift Spoofed Track', color='red', linewidth=2, marker='o', markersize=3)

    ax.set_title('AI-Driven Cumulative Drift Attack')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    ax.legend()
    plt.tight_layout()
    plt.show()
