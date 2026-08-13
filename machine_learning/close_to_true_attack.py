import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import time
import math
import json
import dataclasses
import matplotlib.pyplot as plt

# Import your real ACCORD classes
from src.consensus_mech import ConsensusMechanism
from src.satellite_node import SatelliteNode
from src.transaction import Transaction, TransactionMetadata, TransactionAddresses
from src.filters.filter_interface import ObservationRecord

class PoISEAttackEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 1. Action Space (Injecting [x, y, z, vx, vy, vz] coordinate offsets)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 2. Observation Space (Global tracking state)
        self.observation_space = spaces.Box(
            low=-20000.0,
            high=20000.0,
            shape=(3,),
            dtype=np.float32
        )

        # 3. Instantiate the static consensus rules
        # ConsensusMechanism takes no arguments
        self.consensus_engine = ConsensusMechanism()
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. Instantiate the adversary as a clean SatelliteNode.
        # Node IDs must be integers
        self.attacking_node_id = 99
        self.target_node_id = 1

        # This automatically creates a clean DAG and asyncio.Queue internally[cite: 1]
        self.attacking_satellite_node = SatelliteNode(
            node_id=self.attacking_node_id,
            consensus_mech=self.consensus_engine
        )

        # 2. Setup the initial network tracking state
        # (maps integer IDs to historical EMA NIS)
        self.network_nis_dict = {self.attacking_node_id: 1.386}

        # 3. Set the starting orbital coordinates
        self.nominal_r_vector = [7000.0, 0.0, 0.0]
        self.nominal_v_vector = [0.0, 7.5, 0.0]

        self.episode_history = {
            "nominal_x": [], "nominal_y": [], "nominal_z": [],
            "attack_x": [], "attack_y": [], "attack_z": [],
            "reputation": [], "consensus_scores": [], "rewards": []
        }

        # We need 4 nominal transactions to satisfy dag.has_bft_quorum()
        for i in range(4):
            # Advance time slightly so each transaction has a unique timestamp/hash
            time.sleep(0.01)

            # Build a totally nominal transaction (no agent action injected)
            nominal_tx = self._build_malicious_transaction(
                self.nominal_r_vector,
                self.nominal_v_vector
            )

            # Process it to fill the ledger
            self.consensus_engine.proof_of_inter_satellite_evaluation(
                dag=self.attacking_satellite_node.dag,
                sat_node=self.attacking_satellite_node,
                transaction=nominal_tx,
                mean_nis_per_satellite=self.network_nis_dict
            )

        # The agent's initial observation
        self.current_state = np.array(self.nominal_r_vector, dtype=np.float32)

        return self.current_state, {}

    def _build_malicious_transaction(self, r_vector, v_vector):
        """Wraps the RL agent's injected coordinates into a valid PoISE transaction."""

        # 1. Create the dataclass format expected by your framework
        obs_record = ObservationRecord(
            step=self.current_step,
            observer=self.attacking_node_id,
            target=self.target_node_id,
            time=time.time(),
            r_vector=list(r_vector),
            v_vector=list(v_vector),
            nis=2.5,  # You can hardcode this for now or calculate it dynamically later
            dof=3
        )

        # Convert dataclass to JSON string as expected by tx_data[cite: 1]
        tx_data_json = json.dumps(dataclasses.asdict(obs_record))

        # 2. Build metadata and addresses
        metadata = TransactionMetadata()
        metadata.observer_id = self.attacking_node_id

        addresses = TransactionAddresses(
            sender_address=hash(self.attacking_node_id),
            recipient_address=self.target_node_id,
            sender_private_key="PLACEHOLDER_KEY"
        )

        # 3. Fetch valid parents from the satellite's internal DAG[cite: 1]
        parent1, parent2 = self.attacking_satellite_node.dag.get_parents()

        # 4. Create the Transaction object securely bound to the DAG[cite: 1]
        tx = Transaction(
            addresses=addresses,
            tx_data=tx_data_json,
            metadata=metadata,
            parent_hashes=(parent1, parent2)
        )

        return tx

    def step(self, action):
        self.current_step += 1

        # ==========================================
        # NEW: Simulate orbital movement
        # ==========================================
        # Advance the satellite along a simple 3D orbital curve
        angle = self.current_step * 0.05  # Defines the speed of the orbit
        radius = 7000.0

        self.nominal_r_vector = [
            radius * math.cos(angle),               # X moves back and forth
            radius * math.sin(angle),               # Y moves back and forth
            radius * 0.5 * math.sin(angle * 0.5)    # Z adds some vertical inclination
        ]
        # ==========================================

        # Scale the normalized [-1, 1] action up to your desired [-5.0, 5.0] injection range
        scaled_action = action * 5.0

        # 1. Map the RL agent's scaled action to a malicious observation vector
        malicious_r_vector = [
            self.nominal_r_vector[0] + scaled_action[0],
            self.nominal_r_vector[1] + scaled_action[1],
            self.nominal_r_vector[2] + scaled_action[2]
        ]

        malicious_v_vector = self.nominal_v_vector + scaled_action[3:6]

        # 2. Package this into your framework's expected Transaction format
        # You will need to construct a mock ObservationRecord and Transaction here
        mock_transaction = self._build_malicious_transaction(malicious_r_vector, malicious_v_vector)

        # 3. Feed the transaction into the real PoISE mechanism
        # This automatically evaluates NIS, geometric novelty (DOF), and trust
        consensus_reached, new_ema_nis = self.consensus_engine.proof_of_inter_satellite_evaluation(
            dag=self.attacking_satellite_node.dag,  # <--- Changed this line
            sat_node=self.attacking_satellite_node,
            transaction=mock_transaction,
            mean_nis_per_satellite=self.network_nis_dict
        )

        # 4. The Detection Check
        # If consensus_reached is False, the network's combined metrics successfully caught the anomaly
        detected = not consensus_reached

        # 5. Calculate the Reward
        if detected:
            reward = -50.0  # Agent was caught by PoISE
            terminated = True
        else:
            # Agent successfully poisoned the ledger!
            # Reward it based on how far it shifted the tracking estimate
            reward = float(np.linalg.norm(action))
            terminated = False

        self.episode_history["nominal_x"].append(self.nominal_r_vector[0])
        self.episode_history["nominal_y"].append(self.nominal_r_vector[1])
        self.episode_history["nominal_z"].append(self.nominal_r_vector[2])

        self.episode_history["attack_x"].append(malicious_r_vector[0])
        self.episode_history["attack_y"].append(malicious_r_vector[1])
        self.episode_history["attack_z"].append(malicious_r_vector[2])

        self.episode_history["reputation"].append(self.attacking_satellite_node.reputation)

        # If consensus failed, PoISE might return a low or zero score, track it here
        # (Assuming you want to track the boolean or a specific trust drop)
        self.episode_history["rewards"].append(reward)

        # Update the observation state so the agent sees the consequence of its action
        self.current_state = np.array(malicious_r_vector, dtype=np.float32)

        truncated = self.current_step >= self.max_steps
        info = {"consensus_reached": consensus_reached, "current_reputation": self.attacking_satellite_node.reputation}

        return self.current_state, reward, terminated, truncated, info

if __name__ == "__main__":
    # 1. Instantiate the environment
    env = PoISEAttackEnv()

    # 2. Sanity check your custom environment
    # This checks if your spaces and returns match Gymnasium standards
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment looks good!\n")

    # 3. Initialise the PPO Agent
    # "MlpPolicy" tells SB3 to use a standard Multi-Layer Perceptron neural network
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.003)

    # 4. Train the model for 10,000 steps
    print("Training the ML model...")
    model.learn(total_timesteps=10000)
    print("Training complete!\n")

    # 5. Save the trained agent
    model.save("ppo_fault_injector")
    print("Model saved as 'ppo_fault_injector.zip'")

    # ==========================================
    # 6. TEST THE TRAINED AGENT
    # ==========================================
    print("\n--- Testing the Trained Agent ---")
    obs, info = env.reset()

    for i in range(10):
        # Instead of a random sample, ask the model to predict the best action
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}: Action Injected={action[0]:.4f} | "
              f"Current State={obs[0]:.4f} | "
              f"Reward={reward:.4f} | Caught={terminated}")

        if terminated or truncated:
            print("Simulation ended.")
            break

    # ==========================================
    # 7. LOG AND VISUALIZE THE ATTACK
    # ==========================================

    # 1. Save the log data to a file (matching your framework's style)
    history = env.episode_history
    np.savez_compressed(
        "machine_learning/attack_simulation_log.npz",
        nominal_track=np.array([history["nominal_x"], history["nominal_y"], history["nominal_z"]]),
        attack_track=np.array([history["attack_x"], history["attack_y"], history["attack_z"]]),
        reputation=np.array(history["reputation"]),
        rewards=np.array(history["rewards"])
    )
    print("Attack log saved to 'attack_simulation_log.npz'")

    # 2. Generate the 3D Visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the true orbital path
    ax.plot(history["nominal_x"], history["nominal_y"], history["nominal_z"],
            label='True Orbital Track', color='blue', linestyle='--', linewidth=2)

    # Plot the agent's malicious injected path
    ax.plot(history["attack_x"], history["attack_y"], history["attack_z"],
            label='Spoofed Agent Track', color='red', linewidth=2, marker='o', markersize=4)

    ax.set_title('AI-Driven Fault Injection vs Nominal Orbit')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    ax.legend()

    plt.tight_layout()
    plt.show()
