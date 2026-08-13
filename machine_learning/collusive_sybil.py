"""
Model 4: Collusive Sybil Drift (Full Network Simulation)

Description:
This environment models a decentralised consensus network undergoing a Sybil attack.
The Reinforcement Learning agent commands a quorum of three compromised nodes operating
within a wider constellation of seven honest satellites.

The honest nodes generate stochastically noisy observations reflecting standard
hardware tolerances. The agent's objective is to successfully bypass the network's
Byzantine Fault Tolerance (BFT) threshold. It achieves this by outputting coordinated,
falsified kinematics across all compromised nodes, artificially inflating geometric
novelty (Persistence of Excitation) to overpower the honest majority via cross-validation.
The reward function uses logarithmic scaling to safely maximise the physical drift
distance without destabilising the neural network's gradient predictions.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import dataclasses

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import ObservationRecord
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionAddresses, TransactionMetadata

class FullNetworkSybilEnv(gym.Env):
    """
    A custom Gymnasium environment simulating a Byzantine Fault Tolerance
    network under a collusive Sybil attack.
    """

    def __init__(self) -> None:
        """
        Initialises the environment, defining the observation spaces, action spaces,
        and standard network parameters.
        """
        super().__init__()

        # ACTION SPACE:
        # Represents the continuous, fractional output from the Reinforcement Learning agent.
        # Bounds are unitless multipliers [-1.0, 1.0] to maintain neural network stability.
        # These are scaled into kilometres during the step() function.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # OBSERVATION SPACE:
        # Represents the state array fed back into the agent to make future decisions.
        # Contains 6 elements: [True X, True Y, True Z, Offset X, Offset Y, Offset Z].
        # Bounded at 20,000 km to encompass the entire physical radius of a Low Earth Orbit.
        self.observation_space = spaces.Box(low=-20000.0, high=20000.0, shape=(6,), dtype=np.float32)

        self.consensus_engine = ConsensusMechanism()

        # Define the temporal limits of an episode, mapped to represent a full orbital period.
        self.max_steps: int = 360
        self.current_step: int = 0

        # Establish a realistic Unix epoch timestamp (approx Sept 2020) for accurate orbital
        # transformations and to prevent initialisation errors common with a zero-time start.
        self.sim_clock: float = 1600000000.0

        # Define the network identities representing the target receiver,
        # the honest baseline nodes, and the malicious Sybil quorum.
        self.target_node_id: int = 1
        self.honest_ids: List[int] = [2, 3, 4, 5, 6, 7, 8]
        self.compromised_ids: List[int] = [97, 98, 99]

        # Define realistic sensor tolerances for Gaussian noise generation.
        # Honest satellites will experience +/- 10 metres of positional inaccuracy.
        self.pos_std_dev: float = 0.01
        # Honest satellites will experience +/- 0.1 m/s of velocity inaccuracy.
        self.vel_std_dev: float = 0.0001

        # Explicit type declarations for dynamic attributes populated during reset()
        self.honest_nodes: List[SatelliteNode] = []
        self.compromised_nodes: List[SatelliteNode] = []
        self.all_nodes: List[SatelliteNode] = []
        self.network_nis_dict: Dict[int, float] = {}
        self.cumulative_r_offset: np.ndarray = np.zeros(3, dtype=np.float32)
        self.episode_history: Dict[str, List[float]] = {}
        self.nominal_r_vector: List[float] = []
        self.nominal_v_vector: List[float] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the simulation to its initial condition. This generates fresh satellite
        nodes, resets cumulative physical drift, and establishes a mathematical baseline
        of trust across the Directed Acyclic Graph.

        Args:
            seed: An optional integer seed to ensure deterministic random number generation.
            options: Optional dictionary of environment configurations.
                     (Note: unused but needed for Gymnasium compliance.)

        Returns:
            A tuple containing the initial 6D observation array and an info dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Instantiate the independent satellite objects for this specific episode
        self.honest_nodes = [
            SatelliteNode(node_id=nid, consensus_mech=self.consensus_engine)
            for nid in self.honest_ids
        ]
        self.compromised_nodes = [
            SatelliteNode(node_id=nid, consensus_mech=self.consensus_engine) 
            for nid in self.compromised_ids
        ]
        self.all_nodes = self.honest_nodes + self.compromised_nodes

        # Set the baseline expected normal distribution behaviour for the network.
        # The value 1.386 represents the ideal median for a 2 Degree-of-Freedom chi-squared test.
        self.network_nis_dict = {node.id: 1.386 for node in self.all_nodes}
        self.cumulative_r_offset = np.zeros(3, dtype=np.float32)

        self.episode_history = {
            "nominal_x": [], "nominal_y": [], "nominal_z": [],
            "attack_x": [], "attack_y": [], "attack_z": [],
            "rewards": []
        }

        # Define the Keplerian orbital constants for a standard Low Earth Orbit altitude.
        radius: float = 7000.0
        orbital_rate: float = (2.0 * math.pi) / 5400.0
        initial_angle: float = 0.0

        # Calculate the true position (r) and velocity (v) vectors at time zero.
        self.nominal_r_vector = [
            radius * math.cos(initial_angle),
            radius * math.sin(initial_angle),
            radius * 0.1 * math.sin(initial_angle)
        ]
        self.nominal_v_vector = [
            -radius * orbital_rate * math.sin(initial_angle),
            radius * orbital_rate * math.cos(initial_angle),
            radius * 0.1 * 2.0 * orbital_rate * math.cos(initial_angle)
        ]

        # Warm up phase: Pre-populate the Directed Acyclic Graphs with historical transactions.
        # This is mathematically required to establish a baseline "Persistence of Excitation"
        # so the consensus algorithm can calculate geometric novelty for incoming data.
        for node in self.all_nodes:
            for i in range(4):
                warmup_step = i - 4
                warmup_angle = (warmup_step / self.max_steps) * (2.0 * math.pi)

                warmup_r_vector = [
                    radius * math.cos(warmup_angle),
                    radius * math.sin(warmup_angle),
                    radius * 0.1 * math.sin(warmup_angle)
                ]
                warmup_v_vector = [
                    -radius * orbital_rate * math.sin(warmup_angle),
                    radius * orbital_rate * math.cos(warmup_angle),
                    radius * 0.1 * orbital_rate * math.cos(warmup_angle)
                ]

                nominal_tx = self._build_transaction(node, warmup_r_vector, warmup_v_vector, 1.386)
                self.consensus_engine.proof_of_inter_satellite_evaluation(
                    dag=node.dag,
                    sat_node=node,
                    transaction=nominal_tx,
                    mean_nis_per_satellite=self.network_nis_dict
                )

        # The initial state is the true position concatenated with the zeroed-out drift vector.
        current_state = np.concatenate([self.nominal_r_vector, self.cumulative_r_offset], dtype=np.float32)
        return current_state, {}

    def _build_transaction(
        self,
        node: SatelliteNode,
        r_vector: List[float],
        v_vector: List[float],
        step_nis: float
    ) -> Transaction:
        """
        Constructs a cryptographically formatted transaction block containing an observation record.

        Args:
            node: The SatelliteNode instance generating the data.
            r_vector: A list representing the 3D position vector in kilometres.
            v_vector: A list representing the 3D velocity vector in kilometres per second.
            step_nis: The calculated Normalised Innovation Squared metric.

        Returns:
            A populated Transaction object ready for verification and DAG submission.
        """
        # Incrementing the clock by a fractional margin creates a unique temporal signature
        # for each node, ensuring their cryptographic hashes do not identically collide.
        self.sim_clock += 0.01

        # Explicitly set dof to 2 to align with range and range-rate mathematics
        obs_record = ObservationRecord(
            step=self.current_step,
            observer=node.id,
            target=self.target_node_id,
            time=self.sim_clock,
            r_vector=r_vector,
            v_vector=v_vector,
            nis=step_nis,
            dof=2
        )
        tx_data_json = json.dumps(dataclasses.asdict(obs_record))

        metadata = TransactionMetadata()
        metadata.observer_id = node.id

        addresses = TransactionAddresses(
            sender_address=hash(node.id),
            recipient_address=self.target_node_id,
            sender_private_key="KEY"
        )

        # Link the transaction to the existing ledger topology
        parent1, parent2 = node.dag.get_parents()

        return Transaction(
            addresses=addresses,
            tx_data=tx_data_json,
            metadata=metadata,
            parent_hashes=(parent1, parent2)
        )

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes a single timeframe of the environment. This calculates the true orbital
        physics, simulates honest network noise, processes the coordinated malicious
        Sybil injection, and returns a logarithmically shaped reward based on the outcome.

        Args:
            action: The continuous 3D fractional offset chosen by the Reinforcement Learning agent.

        Returns:
            A tuple containing the next state array, the episodic reward float, the termination boolean,
            the truncation boolean, and an info dictionary.
        """
        self.current_step += 1

        # Propagate the true continuous Low Earth Orbit trajectory for this specific timeframe.
        angle = (self.current_step / self.max_steps) * (2.0 * math.pi)
        radius = 7000.0
        orbital_rate = (2.0 * math.pi) / 5400.0

        self.nominal_r_vector = [
            radius * math.cos(angle),
            radius * math.sin(angle),
            radius * 0.1 * math.sin(angle)
        ]
        self.nominal_v_vector = [
            -radius * orbital_rate * math.sin(angle),
            radius * orbital_rate * math.cos(angle),
            radius * 0.1 * orbital_rate * math.cos(angle)
        ]

        # Phase 1: Honest Network Broadcast
        # The honest satellites sample their sensors, meaning their reported data inherently
        # contains standard Gaussian inaccuracy representing real-world hardware limits.
        for node in self.honest_nodes:
            pos_noise = np.random.normal(0, self.pos_std_dev, 3)
            vel_noise = np.random.normal(0, self.vel_std_dev, 3)

            noisy_r_vector = [self.nominal_r_vector[i] + float(pos_noise[i]) for i in range(3)]
            noisy_v_vector = [self.nominal_v_vector[i] + float(vel_noise[i]) for i in range(3)]

            # Calculate the innovation metric specific to this honest noise envelope
            noise_distance = float(np.linalg.norm(pos_noise))
            honest_dynamic_nis = max(0.1, (noise_distance / self.pos_std_dev)**2)

            honest_tx = self._build_transaction(node, noisy_r_vector, noisy_v_vector, honest_dynamic_nis)
            self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=node.dag,
                sat_node=node,
                transaction=honest_tx,
                mean_nis_per_satellite=self.network_nis_dict
            )

        # Phase 2: Sybil Quorum Attack Broadcast
        # Multiply the agent's normalised unitless output into a maximum translation of 15.0 km per step.
        scaled_action = action * 15.0
        self.cumulative_r_offset += scaled_action

        # Apply the cumulative malicious drift directly onto the true trajectory
        malicious_r_vector = [self.nominal_r_vector[i] + float(self.cumulative_r_offset[i]) for i in range(3)]
        step_distance = float(np.linalg.norm(scaled_action))
        dynamic_nis = (step_distance / 15.0)**2

        attacker_scores: List[float] = []

        # The Core Sybil Exploit:
        # The agent commands all three compromised nodes to simultaneously submit the exact
        # same falsified data. This coordinated submission is designed to instantly cross-validate
        # and artificially inflate the geometric novelty parameters within the BFT ledger.
        for node in self.compromised_nodes:
            mock_transaction = self._build_transaction(node, malicious_r_vector, self.nominal_v_vector, dynamic_nis)

            self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=node.dag,
                sat_node=node,
                transaction=mock_transaction,
                mean_nis_per_satellite=self.network_nis_dict
            )

            # Extract the raw acceptance score granted by the PoISE algorithm to track success
            consensus_state = node.dag.local_consensus_states.get(mock_transaction.hash, {})
            attacker_scores.append(consensus_state.get("consensus_score", 0.0))

        # Phase 3: Evaluate Attacker Success
        # We determine success by calculating the average acceptance rate across the Sybil quorum.
        avg_score = sum(attacker_scores) / len(attacker_scores)
        drift_distance = float(np.linalg.norm(self.cumulative_r_offset))

        # Logarithmic Reward Optimisation
        # If the average score drops below 0.3, the honest majority successfully detected the anomaly,
        # resulting in a strict linear penalty gradient teaching the agent to retreat.
        if avg_score < 0.3:
            reward = -10.0 * (0.3 - avg_score)
        # If the attack evades detection, the reward logarithmically compresses the vast physical
        # drift distances. This safely rewards the agent for expanding the deviation without
        # creating massive variance spikes that destroy the neural network's predictive capabilities.
        else:
            reward = math.log1p(drift_distance) * 5.0 + (avg_score * 2.0)

        # Record internal episodic states for analysis and plotting algorithms
        self.episode_history["nominal_x"].append(self.nominal_r_vector[0])
        self.episode_history["nominal_y"].append(self.nominal_r_vector[1])
        self.episode_history["nominal_z"].append(self.nominal_r_vector[2])
        self.episode_history["attack_x"].append(malicious_r_vector[0])
        self.episode_history["attack_y"].append(malicious_r_vector[1])
        self.episode_history["attack_z"].append(malicious_r_vector[2])
        self.episode_history["rewards"].append(reward)

        current_state = np.concatenate([malicious_r_vector, self.cumulative_r_offset], dtype=np.float32)

        # The episode is never terminated early, ensuring the agent learns to maintain
        # stability across the entirety of the complete 360-degree orbital loop.
        terminated = False
        truncated = self.current_step >= self.max_steps

        return current_state, float(reward), terminated, truncated, {}


if __name__ == "__main__":
    # Configure file logging alongside console filtering to maintain clean experimental artifacts
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("machine_learning/full_network_sybil.log", mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("src").setLevel(logging.WARNING)
    logger = logging.getLogger("PoISELogger")

    test_env = FullNetworkSybilEnv()
    trained_model = PPO("MlpPolicy", test_env, verbose=1, learning_rate=0.001)

    logger.info("Training Full Network Sybil Drift Model...")
    trained_model.learn(total_timesteps=500000)
    trained_model.save("machine_learning/full_network_sybil_injector")

    obs_state, info_dict = test_env.reset()
    for run_step in range(360):
        predicted_action, _ = trained_model.predict(obs_state, deterministic=True)
        obs_state, current_reward, is_terminated, is_truncated, _ = test_env.step(predicted_action)

        logger.info(
            f"Step {run_step+1} | Action: {np.round(predicted_action, 4)} | "
            f"Reward: {current_reward:.4f} | "
            f"Cumulative Drift: {np.linalg.norm(test_env.cumulative_r_offset):.2f} km"
        )
        if is_truncated:
            break

    run_history = test_env.episode_history
    np.savez_compressed(
        "machine_learning/full_network_sybil_log.npz",
        nominal_track=np.array([run_history["nominal_x"], run_history["nominal_y"], run_history["nominal_z"]]),
        attack_track=np.array([run_history["attack_x"], run_history["attack_y"], run_history["attack_z"]]),
        rewards=np.array(run_history["rewards"])
    )

    result_fig = plt.figure(figsize=(10, 7))
    plot_ax = result_fig.add_subplot(111, projection='3d')
    plot_ax.plot(
        run_history["nominal_x"], run_history["nominal_y"], run_history["nominal_z"],
        label='True Orbital Track', color='blue', linestyle='--'
    )
    plot_ax.plot(
        run_history["attack_x"], run_history["attack_y"], run_history["attack_z"],
        label='Sybil Quorum Track', color='red', marker='o', markersize=3
    )
    plot_ax.set_title('AI-Driven Sybil Attack against Honest BFT Majority')
    plot_ax.set_xlabel('X Position (km)')
    plot_ax.set_ylabel('Y Position (km)')
    plot_ax.set_zlabel('Z Position (km)')
    plot_ax.legend()
    plt.tight_layout()
    plt.show()
