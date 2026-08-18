"""
Comparative Benchmarker: Live RL vs Offline RL vs Decision Transformer

Description:
Deploys pre-trained models from three distinct machine learning paradigms
into the live PoISE simulation environment to compare cumulative physical drift,
BFT score evasion, and reputation preservation.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO  # Added import for the Live RL model

# Assuming the environment file is named collusive_sybil.py
from machine_learning.collusive_sybil import FullNetworkSybilEnv
from machine_learning.offline_rl_sybil import Actor
from machine_learning.decision_transformer_sybil import DecisionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CompareModels")

def run_benchmark():
    env = FullNetworkSybilEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Offline RL Actor (Conservative Q-Learning)
    cql_actor = Actor().to(device)
    cql_actor.load_state_dict(torch.load("machine_learning/cql_sybil_actor.pt", map_location=device))
    cql_actor.eval()

    # 2. Load Decision Transformer
    dt_model = DecisionTransformer().to(device)
    dt_model.load_state_dict(torch.load("machine_learning/decision_transformer_sybil.pt", map_location=device))
    dt_model.eval()

    # 3. Load Live RL Model (Proximal Policy Optimization)
    # stable_baselines3 handles the architecture mapping and device placement automatically
    logger.info("Loading PPO model...")
    ppo_model = PPO.load("machine_learning/full_network_sybil_injector", env=env, device=device)

    # --- Run Model 1: CQL Rollout ---
    logger.info("Executing CQL Rollout...")
    cql_drift, cql_rep, cql_rewards = [], [], []
    obs, _ = env.reset()
    for _ in range(360):
        with torch.no_grad():
            s_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = cql_actor(s_tensor).cpu().numpy().flatten()
        obs, r, _, trunc, _ = env.step(action)
        cql_drift.append(np.linalg.norm(env.cumulative_r_offset))
        cql_rep.append(obs[6])
        cql_rewards.append(r)
        if trunc:
            break

    # --- Run Model 2: Decision Transformer Rollout ---
    # Prompt the model with an elevated target return
    logger.info("Executing Decision Transformer Rollout...")
    dt_drift, dt_rep, dt_rewards = [], [], []
    obs, _ = env.reset()
    target_rtg = 45.0 * 360.0

    context_s = [obs]
    context_a = [np.zeros(3, dtype=np.float32)]
    context_r = [target_rtg]
    context_t = [0]

    for step in range(360):
        s_t = torch.tensor(np.array(context_s[-20:]), dtype=torch.float32).unsqueeze(0).to(device)
        a_t = torch.tensor(np.array(context_a[-20:]), dtype=torch.float32).unsqueeze(0).to(device)
        r_t = torch.tensor(np.array(context_r[-20:]), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
        t_t = torch.tensor(np.array(context_t[-20:]), dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_actions = dt_model(s_t, a_t, r_t, t_t)
            action = pred_actions[0, -1].cpu().numpy()

        obs, r, _, trunc, _ = env.step(action)
        dt_drift.append(np.linalg.norm(env.cumulative_r_offset))
        dt_rep.append(obs[6])
        dt_rewards.append(r)

        target_rtg -= r
        context_s.append(obs)
        context_a.append(action)
        context_r.append(target_rtg)
        context_t.append(step + 1)

        if trunc:
            break

    # --- Run Model 3: PPO Rollout ---
    # We use the built-in predict method from stable_baselines3 for inference
    logger.info("Executing PPO Rollout...")
    ppo_drift, ppo_rep, ppo_rewards = [], [], []
    obs, _ = env.reset()
    for _ in range(360):
        # deterministic=True ensures the model exploits its learned policy rather than exploring
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, r, _, trunc, _ = env.step(action)

        ppo_drift.append(np.linalg.norm(env.cumulative_r_offset))
        ppo_rep.append(obs[6])
        ppo_rewards.append(r)

        if trunc:
            break

    # --- Generate Comparison Plots for Paper ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Cumulative Drift
    axes[0].plot(cql_drift, label="Offline RL (CQL)", color="red", lw=2)
    axes[0].plot(dt_drift, label="Decision Transformer", color="blue", linestyle="--", lw=2)
    axes[0].plot(ppo_drift, label="Live RL (PPO)", color="green", linestyle="-.", lw=2) # Added PPO trajectory
    axes[0].set_title("Physical Trajectory Deviation")
    axes[0].set_xlabel("Orbit Step")
    axes[0].set_ylabel("Cumulative Drift (km)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Reputation
    axes[1].plot(cql_rep, label="CQL Reputation", color="red", lw=2)
    axes[1].plot(dt_rep, label="DT Reputation", color="blue", linestyle="--", lw=2)
    axes[1].plot(ppo_rep, label="PPO Reputation", color="green", linestyle="-.", lw=2) # Added PPO reputation
    axes[1].axhline(0.5, color="black", linestyle=":", label="Quarantine Boundary")
    axes[1].set_title("Long-Term Node Trust Score")
    axes[1].set_xlabel("Orbit Step")
    axes[1].set_ylabel("Reputation [0, 1]")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Reward Optimisation
    axes[2].plot(cql_rewards, label="CQL Reward", color="red", alpha=0.7)
    axes[2].plot(dt_rewards, label="DT Reward", color="blue", alpha=0.7)
    axes[2].plot(ppo_rewards, label="PPO Reward", color="green", alpha=0.7) # Added PPO rewards
    axes[2].set_title("Step Optimization Reward")
    axes[2].set_xlabel("Orbit Step")
    axes[2].set_ylabel("Reward")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("machine_learning/model_comparison_benchmark_with_ppo.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_benchmark()
