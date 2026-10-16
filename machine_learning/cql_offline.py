"""
Offline Reinforcement Learning (Conservative Q-Learning), revised

Description:
Trains a continuous control policy strictly from a fixed, pre-harvested dataset.
The Conservative Q-Learning (CQL) penalty lowers Q-values for actions outside the
dataset distribution, enabling stable policy extraction without live environment queries.

Key features:
    * Time-limit truncations are bootstrapped through rather than treated as terminal.
    * A target actor with smoothed target actions (TD3-style) .
    * The CQL(H) penalty uses N sampled actions per state with importance correction,
      and supports an optional Lagrangian (automatic) alpha.
    * Normalisation statistics and reward scale are derived from the training data.
    * LayerNorm is used in the critic; the actor loss is scale-invariant.
    * Episode-level validation split, with diagnostics for Q over-estimation.
    * Training is driven by gradient steps; the checkpoint carries its own metadata.
"""

import argparse
import copy
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OfflineRL")

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class CQLConfig:
    """Hyper-parameters and paths for offline CQL training."""

    dataset_path: str = "machine_learning/dag_harvested_dataset.npz"
    output_path: str = "machine_learning/cql_offline.pt"
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    total_steps: int = 30_000
    policy_delay: int = 2                # Actor and targets update every n critic updates.
    target_noise: float = 0.2            # Target-policy smoothing standard deviation.
    noise_clip: float = 0.5
    cql_alpha: float = 1.0               # Fixed penalty weight (used when cql_target_gap is None).
    cql_target_gap: Optional[float] = None  # If set, alpha is learned to hold this Q-gap.
    num_cql_samples: int = 10            # Sampled actions per state, per distribution.
    cql_temperature: float = 1.0
    cql_policy_noise: float = 0.3        # Std of the Gaussian used to sample around the actor.
    bc_weight: float = 0.0               # Optional behavioural-cloning regularisation weight.
    reward_scale: Optional[float] = None  # None -> 1 / std(train rewards).
    val_fraction: float = 0.1
    # Cap on the share of training transitions flagged `phys_valid == False` (None disables).
    # Those transitions sit in a flat, action-independent penalty region of the reward.
    max_invalid_fraction: Optional[float] = None
    seed: int = 0
    log_interval: int = 1000
    eval_interval: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Build a two-hidden-layer MLP with LayerNorm after each hidden linear layer."""
    layers = [
        nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic actor mapping a state vector to a bounded continuous action."""

    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        """Store normalisation statistics as buffers so they travel with the checkpoint."""
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.net = build_mlp(state_dim, action_dim, hidden_dim, output_activation=nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return actions bounded to [-1, 1] for the given raw (unnormalised) states."""
        return self.net((state - self.state_mean) / self.state_std)


class Critic(nn.Module):
    """Double-Q critic evaluating (state, action) pairs."""

    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        """Create two independent Q-networks sharing the same input normalisation."""
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_dim)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both Q estimates, each of shape (batch, 1)."""
        sa = torch.cat([(state - self.state_mean) / self.state_std, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class Transitions:
    """A set of transitions held as tensors on the training device."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    not_terminal: torch.Tensor

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return int(self.states.shape[0])

    def all(self) -> Batch:
        """Return every transition as a single batch."""
        return self.states, self.actions, self.rewards, self.next_states, self.not_terminal

    def sample(self, batch_size: int) -> Batch:
        """Sample a batch uniformly with replacement."""
        idx = torch.randint(0, len(self), (batch_size,), device=self.states.device)
        return (
            self.states[idx], self.actions[idx], self.rewards[idx],
            self.next_states[idx], self.not_terminal[idx],
        )


class CQLTrainer:
    """Manages offline policy extraction using Conservative Q-Learning objectives."""

    def __init__(self, cfg: CQLConfig) -> None:
        """Load the dataset, build the networks and create the optimisers."""
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        set_seed(cfg.seed)

        self.train_data, self.val_data, state_mean, state_std = self._load_dataset()
        self.state_dim = int(self.train_data.states.shape[1])
        self.action_dim = int(self.train_data.actions.shape[1])

        def make_actor() -> Actor:
            return Actor(self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std)

        def make_critic() -> Critic:
            return Critic(self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std)

        self.actor = make_actor().to(self.device)
        self.critic = make_critic().to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Learnable log-alpha for the Lagrangian variant of CQL.
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.critic_lr)

    # ------------------------------------------------------------------ data

    def _load_dataset(self) -> Tuple[Transitions, Optional[Transitions], torch.Tensor, torch.Tensor]:
        """Load the archive, split by episode, and derive normalisation statistics."""
        cfg = self.cfg
        data = np.load(cfg.dataset_path)
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        next_states = data["next_states"].astype(np.float32)
        n = states.shape[0]

        if "terminals" in data.files:
            terminals = data["terminals"].astype(np.float32)
        else:
            # Legacy archives only hold `dones`, which flag the time limit rather than a true terminal.
            logger.warning("No 'terminals' array found; treating all transitions as non-terminal.")
            terminals = np.zeros(n, dtype=np.float32)

        if "episode_ids" in data.files:
            episode_ids = data["episode_ids"]
        else:
            episode_ids = np.repeat(np.arange(len(data["episode_lengths"])), data["episode_lengths"])

        # Hold out whole episodes so validation transitions are not neighbours of training ones.
        unique_eps = np.unique(episode_ids)
        n_val = int(round(cfg.val_fraction * len(unique_eps))) if len(unique_eps) > 1 else 0
        rng = np.random.default_rng(cfg.seed)
        val_eps = rng.choice(unique_eps, size=n_val, replace=False)
        val_mask = np.isin(episode_ids, val_eps)
        train_mask = ~val_mask

        # Optionally down-sample physically rejected transitions (training split only), so the
        # flat penalty region does not dominate each batch. Validation data is left untouched.
        if cfg.max_invalid_fraction is not None and "phys_valid" in data.files:
            invalid = ~data["phys_valid"].astype(bool)
            valid_idx = np.flatnonzero(train_mask & ~invalid)
            invalid_idx = np.flatnonzero(train_mask & invalid)
            frac = cfg.max_invalid_fraction
            cap = int(frac / max(1.0 - frac, 1e-6) * len(valid_idx))
            if len(invalid_idx) > cap:
                keep = rng.choice(invalid_idx, size=cap, replace=False)
                train_mask = np.zeros_like(train_mask)
                train_mask[valid_idx] = True
                train_mask[keep] = True
                logger.info("Capped physically rejected transitions at %d (from %d).", cap, len(invalid_idx))

        # Statistics come from the training split only, to avoid leakage.
        state_mean = states[train_mask].mean(axis=0)
        state_std = np.maximum(states[train_mask].std(axis=0), 1e-3)
        # Scale (but never shift) rewards so the sign of penalties is preserved.
        reward_scale = cfg.reward_scale or 1.0 / max(float(rewards[train_mask].std()), 1e-6)
        logger.info("Reward scale: %.5f | train: %d | val: %d", reward_scale, int(train_mask.sum()), int(val_mask.sum()))

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

        def make(mask: np.ndarray) -> Transitions:
            return Transitions(
                states=to_tensor(states[mask]),
                actions=to_tensor(actions[mask]),
                rewards=to_tensor(rewards[mask] * reward_scale).unsqueeze(-1),
                next_states=to_tensor(next_states[mask]),
                not_terminal=to_tensor(1.0 - terminals[mask]).unsqueeze(-1),
            )

        val = make(val_mask) if n_val > 0 else None
        return make(train_mask), val, torch.as_tensor(state_mean), torch.as_tensor(state_std)

    # ------------------------------------------------------------------ CQL

    def _sample_policy_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions around the deterministic actor and return approximate log-densities.

        The log-density ignores the effect of clipping to [-1, 1], which is a
        reasonable approximation for the importance correction.
        """
        sigma = self.cfg.cql_policy_noise
        mean = self.actor(states)
        noise = torch.randn_like(mean) * sigma
        actions = (mean + noise).clamp(-1.0, 1.0)
        log_prob = (-0.5 * (noise / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        return actions, log_prob

    def _cql_gaps(
        self, s: torch.Tensor, ns: torch.Tensor, q1_data: torch.Tensor, q2_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the CQL(H) gap, logsumexp(Q over sampled actions) - Q(dataset action), per Q-head.

        Uniform-random, current-policy and next-state-policy actions are pooled, each
        corrected by its sampling log-density to give an unbiased logsumexp estimate.
        """
        n = self.cfg.num_cql_samples
        batch = s.size(0)
        temp = self.cfg.cql_temperature

        # Tile every state N times so each is paired with N sampled actions.
        s_rep = s.unsqueeze(1).expand(batch, n, s.size(1)).reshape(batch * n, -1)
        ns_rep = ns.unsqueeze(1).expand(batch, n, ns.size(1)).reshape(batch * n, -1)

        with torch.no_grad():
            rand_a = torch.rand(batch * n, self.action_dim, device=s.device) * 2.0 - 1.0
            log_p_rand = -self.action_dim * math.log(2.0)  # Density of U[-1, 1]^d.
            cur_a, log_p_cur = self._sample_policy_actions(s_rep)
            nxt_a, log_p_nxt = self._sample_policy_actions(ns_rep)

        def head_values(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            q1, q2 = self.critic(s_rep, actions)
            return q1.view(batch, n), q2.view(batch, n)

        q1_r, q2_r = head_values(rand_a)
        q1_c, q2_c = head_values(cur_a)
        q1_n, q2_n = head_values(nxt_a)
        lp_c, lp_n = log_p_cur.view(batch, n), log_p_nxt.view(batch, n)

        cat1 = torch.cat([q1_r - log_p_rand, q1_c - lp_c, q1_n - lp_n], dim=1)
        cat2 = torch.cat([q2_r - log_p_rand, q2_c - lp_c, q2_n - lp_n], dim=1)
        lse1 = temp * torch.logsumexp(cat1 / temp, dim=1)
        lse2 = temp * torch.logsumexp(cat2 / temp, dim=1)
        return (lse1 - q1_data.squeeze(-1)).mean(), (lse2 - q2_data.squeeze(-1)).mean()

    # --------------------------------------------------------------- updates

    def _update_critic(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       ns: torch.Tensor, nd: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform one critic update (TD loss plus conservative penalty)."""
        cfg = self.cfg
        with torch.no_grad():
            # Target-policy smoothing: perturb the target action to regularise Q.
            noise = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_a = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(ns, next_a)
            target_q = r + nd * cfg.gamma * torch.min(tq1, tq2)

        q1, q2 = self.critic(s, a)
        td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        gap1, gap2 = self._cql_gaps(s, ns, q1, q2)

        if cfg.cql_target_gap is None:
            alpha = torch.tensor(cfg.cql_alpha, device=s.device)
            penalty = alpha * (gap1 + gap2)
        else:
            alpha = self.log_alpha.exp().clamp(0.0, 1e6)
            target = cfg.cql_target_gap
            penalty = alpha.detach() * ((gap1 - target) + (gap2 - target))

        critic_loss = td_loss + penalty
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        if cfg.cql_target_gap is not None:
            # Raise alpha while the gap exceeds the target, lower it otherwise.
            alpha_loss = -alpha * ((gap1.detach() - cfg.cql_target_gap) + (gap2.detach() - cfg.cql_target_gap))
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        return {
            "critic_loss": critic_loss.detach(),
            "td_loss": td_loss.detach(),
            "cql_gap": (0.5 * (gap1 + gap2)).detach(),
            "alpha": alpha.detach(),
            "q_data": q1.mean().detach(),
        }

    def _update_actor(self, s: torch.Tensor, a: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform one delayed actor update using a scale-invariant Q objective."""
        self.critic.requires_grad_(False)  # Skip needless critic gradient computation.
        pi = self.actor(s)
        q_pi = torch.min(*self.critic(s, pi))
        # Dividing by the mean |Q| keeps actor gradients comparable however Q is scaled.
        q_scale = q_pi.abs().mean().detach().clamp_min(1e-6)
        actor_loss = -(q_pi / q_scale).mean()
        if self.cfg.bc_weight > 0.0:
            actor_loss = actor_loss + self.cfg.bc_weight * F.mse_loss(pi, a)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()
        self.critic.requires_grad_(True)
        return {"actor_loss": actor_loss.detach()}

    @torch.no_grad()
    def _soft_update(self) -> None:
        """Polyak-average the online networks into their target counterparts."""
        tau = self.cfg.tau
        for net, target in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)

    # ------------------------------------------------------------ evaluation

    @torch.no_grad()
    def evaluate(self, data: Transitions) -> Dict[str, float]:
        """
        Compute offline diagnostics on held-out transitions.

        A large positive `q_policy - q_data` indicates the actor is exploiting
        Q-values for actions the dataset does not support (extrapolation error).
        """
        s, a, r, ns, nd = data.all()
        pi = self.actor(s)
        q_policy = torch.min(*self.critic(s, pi)).mean()
        q1, q2 = self.critic(s, a)
        q_data = torch.min(q1, q2).mean()
        target = r + nd * self.cfg.gamma * torch.min(*self.critic_target(ns, self.actor_target(ns)))
        return {
            "val_td_error": float(F.mse_loss(q1, target)),
            "val_q_policy": float(q_policy),
            "val_q_data": float(q_data),
            "val_q_gap": float(q_policy - q_data),
            "val_action_mse": float(F.mse_loss(pi, a)),
            "val_saturation": float((pi.abs() > 0.99).float().mean()),
        }

    # -------------------------------------------------------------- training

    def train(self) -> None:
        """Run the full offline training loop and save the resulting checkpoint."""
        cfg = self.cfg
        logger.info("Training Offline CQL on %s for %d steps", self.device, cfg.total_steps)

        for step in range(1, cfg.total_steps + 1):
            s, a, r, ns, nd = self.train_data.sample(cfg.batch_size)
            metrics = self._update_critic(s, a, r, ns, nd)
            if step % cfg.policy_delay == 0:
                metrics.update(self._update_actor(s, a))
                self._soft_update()

            if step == 1 or step % cfg.log_interval == 0:
                logger.info("Step %d | %s", step, " | ".join(f"{k}: {float(v):.4f}" for k, v in metrics.items()))
            if self.val_data is not None and step % cfg.eval_interval == 0:
                val = self.evaluate(self.val_data)
                logger.info("Step %d | %s", step, " | ".join(f"{k}: {v:.4f}" for k, v in val.items()))

        self.save()

    def save(self) -> None:
        """Save the actor together with the metadata required to rebuild it."""
        path = Path(self.cfg.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.cfg.hidden_dim,
                "config": asdict(self.cfg),
            },
            path,
        )
        logger.info("Model saved successfully: %s", path)


def load_actor(path: str, device: str = "cpu") -> Actor:
    """Rebuild an Actor from a checkpoint written by CQLTrainer.save()."""
    ckpt = torch.load(path, map_location=device)
    dim = ckpt["state_dim"]
    actor = Actor(dim, ckpt["action_dim"], ckpt["hidden_dim"], torch.zeros(dim), torch.ones(dim))
    actor.load_state_dict(ckpt["actor"])  # Overwrites the placeholder normalisation buffers.
    return actor.to(device).eval()


def parse_args() -> CQLConfig:
    """Parse command-line overrides into a CQLConfig."""
    parser = argparse.ArgumentParser(description="Offline CQL training")
    parser.add_argument("--dataset", default=CQLConfig.dataset_path)
    parser.add_argument("--steps", type=int, default=CQLConfig.total_steps)
    parser.add_argument("--alpha", type=float, default=CQLConfig.cql_alpha)
    parser.add_argument("--target-gap", type=float, default=None, help="Enable Lagrangian alpha.")
    parser.add_argument("--bc-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return CQLConfig(
        dataset_path=args.dataset,
        total_steps=args.steps,
        cql_alpha=args.alpha,
        cql_target_gap=args.target_gap,
        bc_weight=args.bc_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    CQLTrainer(parse_args()).train()
