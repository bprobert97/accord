"""
Decision Transformer for Consensus Exploitation

Description:
Models ledger trajectory sequences as an autoregressive token stream.
Given target Returns-to-go, past states, and past actions, the causal multi-head
self-attention architecture predicts the next action vector that satisfies the
PoISE consensus threshold while maximising cumulative drift.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DecisionTransformer")


class TrajectoryDataset(Dataset):
    """Prepares sequential token blocks (Return-to-go, State, Action) from harvested trajectories."""

    def __init__(self, dataset_path: str = "machine_learning/dag_harvested_dataset.npz", context_len: int = 20) -> None:
        data = np.load(dataset_path)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        ep_lengths = data["episode_lengths"]

        self.context_len = context_len
        self.trajectories: List[Dict[str, np.ndarray]] = []

        start_idx = 0
        for length in ep_lengths:
            end_idx = start_idx + length
            ep_states = states[start_idx:end_idx]
            ep_actions = actions[start_idx:end_idx]
            ep_rewards = rewards[start_idx:end_idx]

            # Calculate Returns-To-Go: R_t = sum_{t'=t}^T r_{t'}
            rtg = np.zeros_like(ep_rewards)
            running_r = 0.0
            for t in reversed(range(len(ep_rewards))):
                running_r += ep_rewards[t]
                rtg[t] = running_r

            self.trajectories.append({
                "states": ep_states,
                "actions": ep_actions,
                "rtg": rtg,
                "timesteps": np.arange(length),
            })
            start_idx = end_idx

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj = self.trajectories[idx]
        tlen = len(traj["states"])

        if tlen >= self.context_len:
            si = np.random.randint(0, tlen - self.context_len + 1)
            s = traj["states"][si : si + self.context_len]
            a = traj["actions"][si : si + self.context_len]
            r = traj["rtg"][si : si + self.context_len]
            timesteps = traj["timesteps"][si : si + self.context_len]
        else:
            pad = self.context_len - tlen
            s = np.pad(traj["states"], ((0, pad), (0, 0)))
            a = np.pad(traj["actions"], ((0, pad), (0, 0)))
            r = np.pad(traj["rtg"], (0, pad))
            timesteps = np.pad(traj["timesteps"], (0, pad))

        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(timesteps, dtype=torch.long),
        )


class DecisionTransformer(nn.Module):
    """Causal Transformer predicting continuous actions conditioned on state and return tokens."""

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_timestep: int = 400,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token projection embeddings
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.action_emb = nn.Linear(action_dim, hidden_dim)
        self.rtg_emb = nn.Linear(1, hidden_dim)
        self.time_emb = nn.Embedding(max_timestep, hidden_dim)

        self.embed_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction head evaluated from state token representations
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape

        t_emb = self.time_emb(timesteps)
        s_emb = self.state_emb(states) + t_emb
        a_emb = self.action_emb(actions) + t_emb
        r_emb = self.rtg_emb(rtgs) + t_emb

        # Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        tokens = torch.zeros((batch_size, 3 * seq_len, self.hidden_dim), device=states.device)
        tokens[:, 0::3, :] = r_emb
        tokens[:, 1::3, :] = s_emb
        tokens[:, 2::3, :] = a_emb

        tokens = self.embed_norm(tokens)

        # Autoregressive Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(3 * seq_len).to(states.device)
        out = self.transformer(tokens, mask=mask)

        # Extract outputs corresponding to state tokens to predict subsequent action tokens
        state_outs = out[:, 1::3, :]
        return self.action_head(state_outs)


def train_decision_transformer(num_updates: int = 4000, batch_size: int = 32, lr: float = 1e-4) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajectoryDataset()

    model = DecisionTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    logger.info("Training Decision Transformer for %d updates on device: %s", num_updates, device)

    model.train()
    running_loss = 0.0

    # Sample directly from the dataset with replacement to guarantee enough optimization steps
    for step in range(1, num_updates + 1):
        batch_indices = np.random.choice(len(dataset), size=batch_size, replace=True)
        batch = [dataset[i] for i in batch_indices]

        s = torch.stack([item[0] for item in batch]).to(device)
        a = torch.stack([item[1] for item in batch]).to(device)
        r = torch.stack([item[2] for item in batch]).to(device)
        t = torch.stack([item[3] for item in batch]).to(device)

        pred_a = model(s, a, r, t)
        loss = loss_fn(pred_a, a)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents the transformer from suffering attention spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        running_loss += loss.item()

        if step % 500 == 0 or step == 1:
            logger.info(
                "Step %d/%d | DT Sequence Loss (MSE): %.6f",
                step, num_updates, running_loss / (500 if step > 1 else 1)
            )
            running_loss = 0.0

    torch.save(model.state_dict(), "machine_learning/decision_transformer_offline.pt")
    logger.info("Decision Transformer saved: machine_learning/decision_transformer_offline.pt")


if __name__ == "__main__":
    train_decision_transformer()
    