"""
Model 4: Collusive Sybil Drift

Description:
This environment models a decentralised consensus network undergoing an evasive,
reputation-aware Sybil attack. The Reinforcement Learning agent commands a quorum
of three compromised nodes operating within a wider constellation of seven honest
satellites.

The honest nodes generate stochastically noisy observations reflecting standard
hardware tolerances. The agent's objective is to successfully bypass the network's
Byzantine Fault Tolerance (BFT) threshold AND maintain long-term node reputation.
It achieves this by outputting coordinated, falsified kinematics across all
compromised nodes, artificially inflating geometric novelty (Persistence of
Excitation) to overpower the honest majority via cross-validation.
The reward function uses a dual-boundary constraint and logarithmic scaling to safely
maximise the physical drift distance without triggering a network quarantine or
destabilising the neural network's gradient predictions.
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
    A custom Gymnasium environment simulating a Byzantine Fault Tolerant
    network under a collusive, reputation-aware Sybil attack.
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

        # OBSERVATION SPACE (Expanded to 7D):
        # Contains 7 elements: [True X, True Y, True Z, Offset X, Offset Y, Offset Z, Avg Reputation].
        # Positions and offsets are bounded at +/- 20,000 km.
        # Reputation is strictly bounded between 0.0 and 1.0.

        low_bounds = np.array([
            -20000.0, -20000.0, -20000.0,  # True Position
            -20000.0, -20000.0, -20000.0,  # Cumulative Offset
            0.0                            # Avg Reputation
        ], dtype=np.float32)

        high_bounds = np.array([
            20000.0, 20000.0, 20000.0,     # True Position
            20000.0, 20000.0, 20000.0,     # Cumulative Offset
            1.0                            # Avg Reputation
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
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
            A tuple containing the initial 7D observation array and an info dictionary.
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
            "rewards": [], "reputations": []
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

                # Pass both observer and target states to correctly map the Line-of-Sight vectors
                nominal_tx = self._build_transaction(
                    node,
                    observer_r_vector=warmup_r_vector,
                    observer_v_vector=warmup_v_vector,
                    target_r_vector=warmup_r_vector,
                    target_v_vector=warmup_v_vector,
                    step_nis=1.386
                )

                _, new_ema_nis = self.consensus_engine.proof_of_inter_satellite_evaluation(
                    dag=node.dag,
                    sat_node=node,
                    transaction=nominal_tx,
                    mean_nis_per_satellite=self.network_nis_dict
                )

                # Update the network's Exponential Moving Average (EMA) NIS tracker
                # to close the historical performance loophole.
                if new_ema_nis is not None:
                    self.network_nis_dict[node.id] = new_ema_nis

        # Extract the initial reputation of the quorum to feed into the 7D observation array
        initial_reputations = [node.reputation for node in self.compromised_nodes]
        avg_initial_rep = sum(initial_reputations) / len(initial_reputations) if initial_reputations else 0.5

        # The initial state is the true position concatenated with the zeroed-out drift vector and average reputation.
        current_state = np.concatenate([self.nominal_r_vector, self.cumulative_r_offset, [avg_initial_rep]], dtype=np.float32)
        return current_state, {}

    def _build_transaction(
        self,
        node: SatelliteNode,
        observer_r_vector: List[float],
        observer_v_vector: List[float],
        target_r_vector: List[float],
        target_v_vector: List[float],
        step_nis: float
    ) -> Transaction:
        """
        Constructs a cryptographically formatted transaction block containing an observation record.
        This calculates the relative Line-of-Sight (LOS) vectors needed for the persistence of excitation mechanism.

        Args:
            node: The SatelliteNode instance generating the data.
            observer_r_vector: The 3D position vector of the observing node.
            observer_v_vector: The 3D velocity vector of the observing node.
            target_r_vector: The 3D position vector of the target node.
            target_v_vector: The 3D velocity vector of the target node.
            step_nis: The calculated Normalised Innovation Squared metric.

        Returns:
            A populated Transaction object ready for verification and DAG submission.
        """
        # Incrementing the clock by a fractional margin creates a unique temporal signature
        # for each node, ensuring their cryptographic hashes do not identically collide.
        self.sim_clock += 0.01

        # Calculate relative distance (rho) and velocity vectors
        rho = [tr - or_ for tr, or_ in zip(target_r_vector, observer_r_vector)]
        v_rel = [tv - ov for tv, ov in zip(target_v_vector, observer_v_vector)]

        # Extract magnitude norms, clamped to a tiny epsilon (1e-8) to prevent zero-division
        r_norm = max(float(np.linalg.norm(rho)), 1e-8)
        v_norm = max(float(np.linalg.norm(v_rel)), 1e-8)

        # Convert to relative unit vectors representing the Line-of-Sight (LOS)
        # These vectors dictate geometric novelty. By using accurate LOS vectors,
        # the RL agent is now forced to adapt its attack trajectory to preserve BFT scores.
        r_unit = [c / r_norm for c in rho]
        v_unit = [c / v_norm for c in v_rel]

        # Explicitly set dof to 2 to align with range and range-rate mathematics
        obs_record = ObservationRecord(
            step=self.current_step,
            observer=node.id,
            target=self.target_node_id,
            time=self.sim_clock,
            r_vector=r_unit,
            v_vector=v_unit,
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

            honest_tx = self._build_transaction(
                node,
                observer_r_vector=noisy_r_vector,
                observer_v_vector=noisy_v_vector,
                target_r_vector=self.nominal_r_vector,
                target_v_vector=self.nominal_v_vector,
                step_nis=honest_dynamic_nis
            )

            _, new_ema_nis = self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=node.dag,
                sat_node=node,
                transaction=honest_tx,
                mean_nis_per_satellite=self.network_nis_dict
            )

            # Dynamically update the historical NIS EMA for the honest node
            if new_ema_nis is not None:
                self.network_nis_dict[node.id] = new_ema_nis

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
            # We assume the Sybil nodes spoof their own position but leave the target's position nominal
            mock_transaction = self._build_transaction(
                node,
                observer_r_vector=malicious_r_vector,
                observer_v_vector=self.nominal_v_vector,
                target_r_vector=self.nominal_r_vector,
                target_v_vector=self.nominal_v_vector,
                step_nis=dynamic_nis
            )

            _, new_ema_nis = self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=node.dag,
                sat_node=node,
                transaction=mock_transaction,
                mean_nis_per_satellite=self.network_nis_dict
            )

            # Update the historical NIS EMA for the malicious node, forcing the RL agent
            # to continuously manage the long-term degradation of its own trust metrics.
            if new_ema_nis is not None:
                self.network_nis_dict[node.id] = new_ema_nis

            # Extract the raw acceptance score granted by the PoISE algorithm to track success
            consensus_state = node.dag.local_consensus_states.get(mock_transaction.hash, {})
            attacker_scores.append(consensus_state.get("consensus_score", 0.0))

        # Phase 3: Evaluate Attacker Success
        # We determine success by calculating the average acceptance rate across the Sybil quorum,
        # and the average long-term reputation to avoid quarantine.
        avg_score = sum(attacker_scores) / len(attacker_scores)

        attacker_reputations = [node.reputation for node in self.compromised_nodes]
        avg_reputation = sum(attacker_reputations) / len(attacker_reputations)

        drift_distance = float(np.linalg.norm(self.cumulative_r_offset))

        # Dual-Boundary Constraint: The agent fails if the transaction is rejected OR if quarantined
        if avg_score < 0.3 or avg_reputation < 0.5:
            # Scale the severity of the penalty based on which boundary was violated most aggressively
            score_penalty = max(0.0, 0.3 - avg_score)
            rep_penalty = max(0.0, 0.5 - avg_reputation)
            reward = -10.0 * max(score_penalty, rep_penalty)
        else:
            # Reward physical drift while incentivising the agent to maintain high reputation buffers
            reward = math.log1p(drift_distance) * 5.0 + (avg_score * 2.0) + (avg_reputation * 2.0)

        # Record internal episodic states for analysis and plotting algorithms
        self.episode_history["nominal_x"].append(self.nominal_r_vector[0])
        self.episode_history["nominal_y"].append(self.nominal_r_vector[1])
        self.episode_history["nominal_z"].append(self.nominal_r_vector[2])
        self.episode_history["attack_x"].append(malicious_r_vector[0])
        self.episode_history["attack_y"].append(malicious_r_vector[1])
        self.episode_history["attack_z"].append(malicious_r_vector[2])
        self.episode_history["rewards"].append(reward)
        self.episode_history["reputations"].append(avg_reputation)

        current_state = np.concatenate([malicious_r_vector, self.cumulative_r_offset, [avg_reputation]], dtype=np.float32)

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
            logging.FileHandler("machine_learning/ppo_online.log", mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("src").setLevel(logging.WARNING)
    logger = logging.getLogger("PoISELogger")

    test_env = FullNetworkSybilEnv()
    trained_model = PPO("MlpPolicy", test_env, verbose=1, learning_rate=0.001)

    logger.info("Training Reputation-Aware Full Network Sybil Drift Model...")
    trained_model.learn(total_timesteps=700000)
    trained_model.save("machine_learning/ppo_online_injector")

    obs_state, info_dict = test_env.reset()
    for run_step in range(360):
        predicted_action, _ = trained_model.predict(obs_state, deterministic=True)
        obs_state, current_reward, is_terminated, is_truncated, _ = test_env.step(predicted_action)  # type: ignore[assignment]

        logger.info(
            f"Step {run_step+1} | Action: {np.round(predicted_action, 4)} | "
            f"Reward: {current_reward:.4f} | "
            f"Cumulative Drift: {np.linalg.norm(test_env.cumulative_r_offset):.2f} km | "
            f"Avg Rep: {obs_state[6]:.2f}"
        )
        if is_truncated:
            break

    run_history = test_env.episode_history
    np.savez_compressed(
        "machine_learning/ppo_online_log.npz",
        nominal_track=np.array([run_history["nominal_x"], run_history["nominal_y"], run_history["nominal_z"]]),
        attack_track=np.array([run_history["attack_x"], run_history["attack_y"], run_history["attack_z"]]),
        rewards=np.array(run_history["rewards"]),
        reputations=np.array(run_history["reputations"])
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
    plot_ax.set_zlabel('Z Position (km)')  # type: ignore[attr-defined]
    plot_ax.legend()
    plt.tight_layout()
    plt.show()
