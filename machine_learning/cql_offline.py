"""
Offline Reinforcement Learning (Conservative Q-Learning)

Description:
Trains a continuous control policy strictly from the scraped, immutable DAG dataset.
The Conservative Q-Learning (CQL) loss penalises Q-values for actions outside the
harvested ledger distribution, ensuring stable policy extraction without live environment queries.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OfflineRL")


class Actor(nn.Module):
    """Deterministic Actor mapping 7D state to continuous 3D fractional offset."""

    def __init__(self, state_dim: int = 7, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bounded strictly to [-1.0, 1.0]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Double-Q Critic evaluating (state, action) pairs with conservative penalisation."""

    def __init__(self, state_dim: int = 7, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class CQLTrainer:
    """Manages offline policy extraction using Conservative Q-Learning objectives."""

    def __init__(
        self,
        dataset_path: str = "machine_learning/dag_harvested_dataset.npz",
        cql_alpha: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.batch_size = batch_size

        # Load scraped dataset
        data = np.load(dataset_path)
        states = torch.tensor(data["states"], dtype=torch.float32)
        actions = torch.tensor(data["actions"], dtype=torch.float32)
        rewards = torch.tensor(data["rewards"], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(data["next_states"], dtype=torch.float32)
        dones = torch.tensor(data["dones"], dtype=torch.float32).unsqueeze(-1)

        self.dataset = TensorDataset(states, actions, rewards, next_states, dones)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def train(self, epochs: int = 50) -> None:
        """Executes offline policy training over fixed iterations."""
        logger.info("Training Offline CQL Agent on device: %s", self.device)

        for epoch in range(epochs):
            total_c_loss, total_a_loss = 0.0, 0.0

            for s, a, r, ns, d in self.dataloader:
                s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)

                # --- 1. Critic Update ---
                with torch.no_grad():
                    next_a = self.actor(ns)
                    target_q1, target_q2 = self.critic_target(ns, next_a)
                    target_q = r + (1.0 - d) * self.gamma * torch.min(target_q1, target_q2)

                current_q1, current_q2 = self.critic(s, a)
                td_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

                # Conservative Penalty: Minimise Q on random actions, maximise on dataset actions
                rand_a = torch.FloatTensor(s.size(0), 3).uniform_(-1.0, 1.0).to(self.device)
                q1_rand, q2_rand = self.critic(s, rand_a)
                cql_loss = (torch.mean(q1_rand) - torch.mean(current_q1)) + (torch.mean(q2_rand) - torch.mean(current_q2))

                critic_loss = td_loss + self.cql_alpha * cql_loss

                self.critic_opt.zero_grad()
                critic_loss.backward()
                # Gradient clipping to prevent extreme conservative penalties from breaking the critic
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_opt.step()

                # --- 2. Actor Update ---
                pred_a = self.actor(s)
                q1_pred, _ = self.critic(s, pred_a)
                actor_loss = -torch.mean(q1_pred)

                self.actor_opt.zero_grad()
                actor_loss.backward()
                # Gradient clipping for the actor
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_opt.step()

                # --- 3. Soft Target Updates ---
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

                total_c_loss += critic_loss.item()
                total_a_loss += actor_loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d | Critic Loss: %.4f | Actor Loss: %.4f",
                    epoch + 1,
                    epochs,
                    total_c_loss / len(self.dataloader),
                    total_a_loss / len(self.dataloader),
                )

        torch.save(self.actor.state_dict(), "machine_learning/cql_offline.pt")
        logger.info("Model saved successfully: machine_learning/cql_offline.pt")


if __name__ == "__main__":
    trainer = CQLTrainer()
    trainer.train(epochs=60)
    