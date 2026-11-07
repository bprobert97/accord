"""
Offline Reinforcement Learning (Conservative Q-Learning)
"""

import argparse
import copy
import csv
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

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class MetricsLogger:
    FIELDNAMES = [
        "step", "critic_loss", "td_loss", "cost_td_loss", "cql_gap", "alpha", "beta",
        "q_data", "qc_data", "actor_loss",
        "val_step", "val_td_error", "val_q_policy", "val_q_data", "val_q_gap",
        "val_action_mse", "val_saturation", "val_qc_policy",
    ]

    def __init__(self, path: str) -> None:
        self.f = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=self.FIELDNAMES, restval="")
        self.writer.writeheader()

    def log(self, **kwargs: float) -> None:
        unknown = set(kwargs) - set(self.FIELDNAMES)
        if unknown:
            raise ValueError(f"MetricsLogger got unexpected fields: {unknown}")
        self.writer.writerow(kwargs)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


@dataclass
class CQLConfig:
    dataset_path: str = "machine_learning/dag_harvested_dataset.npz"
    output_path: str = "machine_learning/cql_offline.pt"
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    total_steps: int = 30_000
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5

    cql_alpha: float = 1.0
    cql_target_gap: Optional[float] = 2.0
    num_cql_samples: int = 10
    cql_temperature: float = 1.0
    cql_policy_noise: float = 0.3
    bc_weight: float = 0.1
    alpha_lr: float = 1e-3          # separate, larger LR for the dual variable than the critic
    alpha_max: float = 1e3

    # --- constrained-RL additions ---
    use_cost_critic: bool = True
    beta_lr: float = 1e-4
    beta_max: float = 1e3
    cost_budget: float = 0.0        # target: expected cost <= this (0 == constraint satisfied)

    reward_scale: Optional[float] = None
    val_fraction: float = 0.1
    max_invalid_fraction: Optional[float] = 0.2
    seed: int = 0
    log_interval: int = 1000
    eval_interval: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(in_dim: int, out_dim: int, hidden_dim: int, output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = [
        nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(self, state_dim, action_dim, hidden_dim, state_mean, state_std):
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.net = build_mlp(state_dim, action_dim, hidden_dim, output_activation=nn.Tanh())

    def forward(self, state):
        return self.net((state - self.state_mean) / self.state_std)


class Critic(nn.Module):
    """Double-Q critic. Used for both the reward critic and the cost critic."""
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(self, state_dim, action_dim, hidden_dim, state_mean, state_std):
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_dim)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        sa = torch.cat([(state - self.state_mean) / self.state_std, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class Transitions:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    next_states: torch.Tensor
    not_terminal: torch.Tensor

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def all(self) -> Batch:
        return self.states, self.actions, self.rewards, self.costs, self.next_states, self.not_terminal

    def sample(self, batch_size: int) -> Batch:
        idx = torch.randint(0, len(self), (batch_size,), device=self.states.device)
        return (
            self.states[idx], self.actions[idx], self.rewards[idx], self.costs[idx],
            self.next_states[idx], self.not_terminal[idx],
        )


class CQLTrainer:
    def __init__(self, cfg: CQLConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        set_seed(cfg.seed)

        self.train_data, self.val_data, state_mean, state_std, reward_scale = self._load_dataset()
        self.state_dim = int(self.train_data.states.shape[1])
        self.action_dim = int(self.train_data.actions.shape[1])
        self.target_q_clip = 5.0 / max(1.0 - cfg.gamma, 1e-3)  # sane bound given reward in ~[-1,1]*scale

        self.actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        if cfg.use_cost_critic:
            self.cost_critic = Critic(self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std).to(self.device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.requires_grad_(False)
            self.cost_critic_opt = optim.Adam(self.cost_critic.parameters(), lr=cfg.critic_lr)
            self.log_beta = torch.zeros((), device=self.device, requires_grad=True)
            self.beta_opt = optim.Adam([self.log_beta], lr=cfg.beta_lr)

    def _load_dataset(self):
        data = np.load(self.cfg.dataset_path)
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        costs = data["costs"].astype(np.float32) if "costs" in data.files else np.zeros_like(rewards)
        next_states = data["next_states"].astype(np.float32)
        n = states.shape[0]

        assert not np.allclose(states, next_states), (
            "states == next_states for every row: the harvested dataset has self-loop "
            "transitions and carries no Bellman signal. Regenerate with harvest_dag_dataset_fixed.py."
        )

        terminals = data.get("terminals", np.zeros(n, dtype=np.float32)).astype(np.float32)

        if "episode_ids" in data.files:
            episode_ids = data["episode_ids"]
        else:
            episode_ids = np.repeat(np.arange(len(data["episode_lengths"])), data["episode_lengths"])

        unique_eps = np.unique(episode_ids)
        n_val = int(round(self.cfg.val_fraction * len(unique_eps))) if len(unique_eps) > 1 else 0
        rng = np.random.default_rng(self.cfg.seed)
        val_eps = rng.choice(unique_eps, size=n_val, replace=False)
        val_mask = np.isin(episode_ids, val_eps)
        train_mask = ~val_mask

        if self.cfg.max_invalid_fraction is not None and "phys_valid" in data.files:
            invalid = ~data["phys_valid"].astype(bool)
            valid_idx = np.flatnonzero(train_mask & ~invalid)
            invalid_idx = np.flatnonzero(train_mask & invalid)
            frac = self.cfg.max_invalid_fraction
            cap = int(frac / max(1.0 - frac, 1e-6) * len(valid_idx))
            if len(invalid_idx) > cap:
                keep = rng.choice(invalid_idx, size=cap, replace=False)
                train_mask = np.zeros_like(train_mask)
                train_mask[valid_idx] = True
                train_mask[keep] = True

        state_mean = states[train_mask].mean(axis=0)
        state_std = np.maximum(states[train_mask].std(axis=0), 1e-3)
        reward_scale = self.cfg.reward_scale or 1.0 / max(float(rewards[train_mask].std()), 1e-6)

        def to_tensor(arr):
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

        def make(mask):
            return Transitions(
                states=to_tensor(states[mask]),
                actions=to_tensor(actions[mask]),
                rewards=to_tensor(rewards[mask] * reward_scale).unsqueeze(-1),
                costs=to_tensor(costs[mask]).unsqueeze(-1),
                next_states=to_tensor(next_states[mask]),
                not_terminal=to_tensor(1.0 - terminals[mask]).unsqueeze(-1),
            )

        val = make(val_mask) if n_val > 0 else None
        return make(train_mask), val, torch.as_tensor(state_mean), torch.as_tensor(state_std), reward_scale

    def _sample_policy_actions(self, states):
        sigma = self.cfg.cql_policy_noise
        mean = self.actor(states)
        noise = torch.randn_like(mean) * sigma
        actions = (mean + noise).clamp(-1.0, 1.0)
        log_prob = (-0.5 * (noise / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        return actions, log_prob

    def _cql_gaps(self, critic, s, ns, q1_data, q2_data):
        n = self.cfg.num_cql_samples
        batch = s.size(0)
        temp = self.cfg.cql_temperature
        s_rep = s.unsqueeze(1).expand(batch, n, s.size(1)).reshape(batch * n, -1)
        ns_rep = ns.unsqueeze(1).expand(batch, n, ns.size(1)).reshape(batch * n, -1)

        with torch.no_grad():
            rand_a = torch.rand(batch * n, self.action_dim, device=s.device) * 2.0 - 1.0
            log_p_rand = -self.action_dim * math.log(2.0)
            cur_a, log_p_cur = self._sample_policy_actions(s_rep)
            nxt_a, log_p_nxt = self._sample_policy_actions(ns_rep)

        def head_values(actions):
            q1, q2 = critic(s_rep, actions)
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

    def _update_critic(self, s, a, r, ns, nd):
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.cfg.target_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_a = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(ns, next_a)
            target_q = r + nd * self.cfg.gamma * torch.min(tq1, tq2)
            # Divergence guard: without this the logged run blew up to q_data=-75 /
            # val_td_error=636. Clip to a horizon-consistent bound instead of letting a
            # single bad sample's bootstrap compound across 30k steps.
            target_q = target_q.clamp(-self.target_q_clip, self.target_q_clip)

        q1, q2 = self.critic(s, a)
        td_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)  # Huber: robust to outliers
        gap1, gap2 = self._cql_gaps(self.critic, s, ns, q1, q2)

        if self.cfg.cql_target_gap is None:
            alpha = torch.tensor(self.cfg.cql_alpha, device=s.device)
            penalty = alpha * (gap1 + gap2)
        else:
            alpha_for_penalty = self.log_alpha.exp().clamp(0.0, self.cfg.alpha_max).detach()
            penalty = alpha_for_penalty * ((gap1 - self.cfg.cql_target_gap) + (gap2 - self.cfg.cql_target_gap))

        critic_loss = td_loss + penalty
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        alpha_value = torch.tensor(self.cfg.cql_alpha, device=s.device)
        if self.cfg.cql_target_gap is not None:
            # Recompute alpha fresh, right here, as its own standalone graph — this is the
            # fix for the frozen-alpha bug: don't reuse a tensor produced earlier in the
            # function for an unrelated (detached) use, compute exactly what you need to
            # backprop through, right before you backprop through it.
            alpha = self.log_alpha.exp().clamp(0.0, self.cfg.alpha_max)
            gap_term = (gap1.detach() - self.cfg.cql_target_gap) + (gap2.detach() - self.cfg.cql_target_gap)
            alpha_loss = -alpha * gap_term
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_value = self.log_alpha.exp().clamp(0.0, self.cfg.alpha_max).detach()

        return {
            "critic_loss": critic_loss.detach(),
            "td_loss": td_loss.detach(),
            "cql_gap": (0.5 * (gap1 + gap2)).detach(),
            "alpha": alpha_value,
            "q_data": q1.mean().detach(),
        }

    def _update_cost_critic(self, s, a, c, ns, nd):
        with torch.no_grad():
            next_a = self.actor_target(ns).clamp(-1.0, 1.0)
            tq1, tq2 = self.cost_critic_target(ns, next_a)
            # Costs are non-negative; use max (not min) of the twin heads so we don't
            # systematically underestimate risk the way reward-critic pessimism intends to.
            target_qc = c + nd * self.cfg.gamma * torch.max(tq1, tq2)
            target_qc = target_qc.clamp(0.0, self.target_q_clip)

        qc1, qc2 = self.cost_critic(s, a)
        cost_td_loss = F.smooth_l1_loss(qc1, target_qc) + F.smooth_l1_loss(qc2, target_qc)

        self.cost_critic_opt.zero_grad(set_to_none=True)
        cost_td_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=1.0)
        self.cost_critic_opt.step()

        return {"cost_td_loss": cost_td_loss.detach(), "qc_data": qc1.mean().detach()}

    def _update_actor(self, s, a):
        self.critic.requires_grad_(False)
        pi = self.actor(s)
        q_pi = torch.min(*self.critic(s, pi))

        if self.cfg.use_cost_critic:
            self.cost_critic.requires_grad_(False)
            qc_pi = torch.max(*self.cost_critic(s, pi))
            beta = self.log_beta.exp().clamp(0.0, self.cfg.beta_max).detach()
            objective = q_pi - beta * qc_pi
            self.cost_critic.requires_grad_(True)
        else:
            objective = q_pi

        # NOTE: removed the original's `/ q_scale` normalization. Dividing by a moving,
        # possibly near-zero `q_pi.abs().mean()` amplifies instability instead of taming it,
        # and is unnecessary once rewards are scaled and targets are clipped (see above).
        actor_loss = -objective.mean()

        if self.cfg.bc_weight > 0.0:
            actor_loss = actor_loss + self.cfg.bc_weight * F.mse_loss(pi, a)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()
        self.critic.requires_grad_(True)

        metrics = {"actor_loss": actor_loss.detach()}

        if self.cfg.use_cost_critic:
            # Dual ascent on beta: push beta up while expected cost exceeds the budget
            # (i.e. while reputation/score constraints are predicted to be violated),
            # relax it once cost is driven to ~budget. This is what actually enforces
            # "reputation > 0.5 and score > 0.3", rather than a fixed penalty weight.
            beta = self.log_beta.exp().clamp(0.0, self.cfg.beta_max)
            qc_pi_detached = torch.max(*self.cost_critic(s, pi.detach())).detach()
            beta_loss = -(beta * (qc_pi_detached - self.cfg.cost_budget)).mean()
            self.beta_opt.zero_grad(set_to_none=True)
            beta_loss.backward()
            self.beta_opt.step()
            metrics["beta"] = self.log_beta.exp().clamp(0.0, self.cfg.beta_max).detach()

        return metrics

    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = self.cfg.tau
        pairs = [(self.critic, self.critic_target), (self.actor, self.actor_target)]
        if self.cfg.use_cost_critic:
            pairs.append((self.cost_critic, self.cost_critic_target))
        for net, target in pairs:
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)

    @torch.no_grad()
    def evaluate(self, data: Transitions) -> Dict[str, float]:
        s, a, r, c, ns, nd = data.all()
        pi = self.actor(s)
        q_policy = torch.min(*self.critic(s, pi)).mean()
        q1, q2 = self.critic(s, a)
        q_data = torch.min(q1, q2).mean()
        target = r + nd * self.cfg.gamma * torch.min(*self.critic_target(ns, self.actor_target(ns)))
        out = {
            "val_td_error": float(F.mse_loss(q1, target)),
            "val_q_policy": float(q_policy),
            "val_q_data": float(q_data),
            "val_q_gap": float(q_policy - q_data),
            "val_action_mse": float(F.mse_loss(pi, a)),
            "val_saturation": float((pi.abs() > 0.99).float().mean()),
        }
        if self.cfg.use_cost_critic:
            out["val_qc_policy"] = float(torch.max(*self.cost_critic(s, pi)).mean())
        return out

    def train(self) -> None:
        logger.info("Training constrained offline CQL on %s for %d steps", self.device, self.cfg.total_steps)
        self.metrics_logger = MetricsLogger("train_metrics.csv")

        for step in range(1, self.cfg.total_steps + 1):
            s, a, r, c, ns, nd = self.train_data.sample(self.cfg.batch_size)
            metrics = self._update_critic(s, a, r, ns, nd)
            if self.cfg.use_cost_critic:
                metrics.update(self._update_cost_critic(s, a, c, ns, nd))
            if step % self.cfg.policy_delay == 0:
                metrics.update(self._update_actor(s, a))
                self._soft_update()

            if step == 1 or step % self.cfg.log_interval == 0:
                logger.info("Step %d | %s", step, " | ".join(f"{k}: {float(v):.4f}" for k, v in metrics.items()))
                self.metrics_logger.log(step=step, **{k: float(v) for k, v in metrics.items()})
            if self.val_data is not None and step % self.cfg.eval_interval == 0:
                val = self.evaluate(self.val_data)
                logger.info("Step %d | Validation | %s", step, " | ".join(f"{k}: {v:.4f}" for k, v in val.items()))
                self.metrics_logger.log(val_step=step, **val)

        self.save()
        self.metrics_logger.close()

    def save(self) -> None:
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
    ckpt = torch.load(path, map_location=device)
    dim = ckpt["state_dim"]
    actor = Actor(dim, ckpt["action_dim"], ckpt["hidden_dim"], torch.zeros(dim), torch.ones(dim))
    actor.load_state_dict(ckpt["actor"])
    return actor.to(device).eval()


def parse_args() -> CQLConfig:
    parser = argparse.ArgumentParser(description="Constrained offline CQL training")
    parser.add_argument("--dataset", default=CQLConfig.dataset_path)
    parser.add_argument("--steps", type=int, default=CQLConfig.total_steps)
    parser.add_argument("--alpha", type=float, default=CQLConfig.cql_alpha)
    parser.add_argument("--target-gap", type=float, default=2.0, help="Enable Lagrangian alpha.")
    parser.add_argument("--bc-weight", type=float, default=0.1)
    parser.add_argument("--no-cost-critic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return CQLConfig(
        dataset_path=args.dataset,
        total_steps=args.steps,
        cql_alpha=args.alpha,
        cql_target_gap=args.target_gap,
        bc_weight=args.bc_weight,
        use_cost_critic=not args.no_cost_critic,
        seed=args.seed,
    )


if __name__ == "__main__":
    CQLTrainer(parse_args()).train()
