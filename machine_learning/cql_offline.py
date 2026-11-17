"""Constrained offline reinforcement learning with Conservative Q-Learning.

The trainer fits an attack policy to the fixed dataset harvested from the DAG.
It combines Conservative Q-Learning, which penalises value estimates for actions
absent from the dataset, with a constrained formulation that keeps the attacker
admitted to the network. A second (cost) critic estimates the admission cost and
is coupled to the policy through a Lagrangian multiplier, so the actor maximises
reward while the predicted cost is driven toward its budget. Training minibatches
are stratified by injected-drift magnitude to counter the dataset's imbalance.
"""

from __future__ import annotations

# Single-letter tensor names (s, a, r, ns, nd, c) follow the reinforcement
# learning convention and are used throughout; the module is long because
# every public object carries a full docstring.
# pylint: disable=invalid-name,too-many-lines

import argparse
import copy
import csv
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OfflineRL")

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class MetricsLogger:
    """Append-only CSV writer for training and validation metrics."""

    FIELDNAMES = [
        "step", "critic_loss", "td_loss", "cost_td_loss", "cql_gap", "cost_cql_gap",
        "alpha", "beta", "q_data", "qc_data", "actor_loss",
        "val_step", "val_td_error", "val_q_policy", "val_q_data", "val_q_gap",
        "val_action_mse", "val_saturation", "val_qc_policy",
    ]

    def __init__(self, path: str) -> None:
        """Open the metrics file and write its header.

        Args:
            path: Destination path for the metrics CSV.
        """
        # pylint: disable-next=consider-using-with
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=self.FIELDNAMES, restval="")
        self.writer.writeheader()

    def log(self, **kwargs: float) -> None:
        """Write one metrics row.

        Args:
            **kwargs: Field-value pairs; every key must be a known field name.

        Returns:
            None.

        Raises:
            ValueError: If any key is not in :attr:`FIELDNAMES`.
        """
        unknown = set(kwargs) - set(self.FIELDNAMES)
        if unknown:
            raise ValueError(f"MetricsLogger got unexpected fields: {unknown}")
        self.writer.writerow(kwargs)
        self.f.flush()

    def close(self) -> None:
        """Close the metrics file.

        Returns:
            None.
        """
        self.f.close()


@dataclass
class CQLConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters and run configuration for the constrained CQL trainer.

    Attributes:
        dataset_path: Path to the harvested ``.npz`` dataset.
        output_path: Destination path for the trained actor checkpoint.
        hidden_dim: Width of the hidden layers.
        gamma: Discount factor.
        tau: Soft-update coefficient for the target networks.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.
        batch_size: Minibatch size.
        total_steps: Number of gradient steps.
        policy_delay: Actor and target update period, in critic steps.
        target_noise: Standard deviation of target-action smoothing noise.
        noise_clip: Clip applied to the target-action noise.
        cql_alpha: Conservative penalty weight when the Lagrangian form is off.
        cql_target_gap: Target conservative gap; ``None`` disables the Lagrangian
            adjustment of the penalty weight.
        num_cql_samples: Number of sampled actions used in the conservative term.
        cql_temperature: Temperature of the conservative log-sum-exp.
        cql_policy_noise: Standard deviation of the policy-action samples.
        bc_weight: Behaviour-cloning regularisation weight.
        alpha_lr: Learning rate of the conservative dual variable.
        alpha_min: Lower bound on the conservative penalty weight.
        alpha_max: Upper bound on the conservative penalty weight.
        use_cost_critic: Whether to train the cost critic and enforce the
            admission constraint.
        beta_lr: Learning rate of the constraint dual variable.
        beta_max: Upper bound on the constraint dual variable.
        cost_budget: Target expected cost; zero means the constraint is met.
        cost_cql_alpha: Weight of the cost critic's own conservative penalty;
            zero disables it.
        stratify_by_drift: Whether to stratify minibatches by drift magnitude.
        num_drift_bins: Number of drift-magnitude bins used for stratification.
        num_capped_top_bins: Number of highest-drift bins sampled at their
            natural frequency rather than equalised.
        reward_scale: Explicit reward scale; computed from the data when ``None``.
        val_fraction: Fraction of episodes held out for validation.
        max_invalid_fraction: Cap on the fraction of physically invalid
            transitions retained in training, when such a flag is present.
        seed: Random seed.
        log_interval: Logging period, in steps.
        eval_interval: Validation period, in steps.
        device: Compute device string.
    """

    dataset_path: str = "machine_learning/dag_harvested_dataset.npz"
    output_path: str = "machine_learning/cql_offline.pt"
    hidden_dim: int = 256
    gamma: float = 0.95
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
    alpha_lr: float = 1e-3
    alpha_min: float = 0.1
    alpha_max: float = 1e3

    use_cost_critic: bool = True
    beta_lr: float = 1e-4
    beta_max: float = 1e3
    cost_budget: float = 0.0
    cost_cql_alpha: float = 0.0

    stratify_by_drift: bool = True
    num_drift_bins: int = 8
    num_capped_top_bins: int = 1

    reward_scale: Optional[float] = None
    val_fraction: float = 0.1
    max_invalid_fraction: Optional[float] = 0.2
    seed: int = 0
    log_interval: int = 1000
    eval_interval: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch random number generators.

    Args:
        seed: Seed value applied to both libraries and, when available, CUDA.

    Returns:
        None.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(
    in_dim: int, out_dim: int, hidden_dim: int, output_activation: Optional[nn.Module] = None
) -> nn.Sequential:
    """Build a two-hidden-layer multilayer perceptron with layer normalisation.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        hidden_dim: Hidden-layer width.
        output_activation: Optional activation appended after the final layer.

    Returns:
        The assembled sequential network.
    """
    layers: list = [
        nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy mapping a normalised state to a bounded action."""

    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int,
        state_mean: torch.Tensor, state_std: torch.Tensor,
    ) -> None:
        """Initialise the actor.

        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden-layer width.
            state_mean: Per-feature mean used to normalise the input.
            state_std: Per-feature standard deviation used to normalise the input.
        """
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.net = build_mlp(state_dim, action_dim, hidden_dim, output_activation=nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map a state to an action.

        Args:
            state: Batch of states.

        Returns:
            Batch of actions in the range minus one to one.
        """
        return self.net((state - self.state_mean) / self.state_std)


class Critic(nn.Module):
    """Twin action-value critic, used for both the reward and cost critics."""

    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int,
        state_mean: torch.Tensor, state_std: torch.Tensor,
    ) -> None:
        """Initialise the critic.

        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden-layer width.
            state_mean: Per-feature mean used to normalise the state.
            state_std: Per-feature standard deviation used to normalise the state.
        """
        super().__init__()
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_dim)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both value heads.

        Args:
            state: Batch of states.
            action: Batch of actions.

        Returns:
            A tuple of the two value-head outputs.
        """
        sa = torch.cat([(state - self.state_mean) / self.state_std, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class Transitions:
    """A set of transitions with optional drift-stratified sampling.

    Attributes:
        states: Batch of states.
        actions: Batch of actions.
        rewards: Batch of scaled rewards.
        costs: Batch of admission costs.
        next_states: Batch of next states.
        not_terminal: One minus the terminal flag for each transition.
        drift_bin: Per-transition drift-magnitude bin index, or ``None`` to
            disable stratified sampling.
        num_drift_bins: Number of drift bins.
        num_capped_top_bins: Number of highest-drift bins sampled at their
            natural frequency rather than equalised.
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    next_states: torch.Tensor
    not_terminal: torch.Tensor
    drift_bin: Optional[torch.Tensor] = None
    num_drift_bins: int = 0
    num_capped_top_bins: int = 0
    _bin_row_cache: Dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __len__(self) -> int:
        """Return the number of transitions."""
        return int(self.states.shape[0])

    def all(self) -> Batch:
        """Return every transition as a single batch.

        Returns:
            The full batch tuple.
        """
        return (
            self.states, self.actions, self.rewards, self.costs,
            self.next_states, self.not_terminal,
        )

    def _gather(self, idx: torch.Tensor) -> Batch:
        """Gather the transitions at the given indices.

        Args:
            idx: Row indices to select.

        Returns:
            The selected batch tuple.
        """
        return (
            self.states[idx], self.actions[idx], self.rewards[idx], self.costs[idx],
            self.next_states[idx], self.not_terminal[idx],
        )

    def sample(self, batch_size: int) -> Batch:
        """Sample a minibatch, optionally stratified by drift magnitude.

        When stratification is enabled, the highest drift bins are sampled at
        their natural share of the dataset while the remaining bins are sampled
        in equal proportion. This raises the representation of the sparsely
        populated intermediate-drift transitions without inflating the
        saturated-drift transitions.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            The sampled batch tuple.
        """
        if self.drift_bin is None or self.num_drift_bins <= 1:
            idx = torch.randint(0, len(self), (batch_size,), device=self.states.device)
            return self._gather(idx)

        # Separate the highest-drift bins (sampled at their natural rate) from the rest
        # (equalised). Boosting the saturated-drift bins once caused the policy to drift
        # straight to the clip limit, so those bins are deliberately left un-boosted.
        capped = set(range(self.num_drift_bins - self.num_capped_top_bins, self.num_drift_bins))
        boosted_bins = [b for b in range(self.num_drift_bins) if b not in capped]

        # Reserve a share of the batch for the capped bins equal to their real share of
        # the dataset, so saturated-drift transitions are neither inflated nor erased.
        capped_total_rows = sum(len(self._bin_row_cache.get(b, [])) for b in capped)
        capped_budget = round(batch_size * capped_total_rows / max(1, len(self))) if capped else 0
        idx_parts = []
        if capped_budget > 0 and capped_total_rows > 0:
            for b in capped:
                bin_rows = self._bin_row_cache.get(b)
                if bin_rows is None or len(bin_rows) == 0:
                    continue
                share = max(1, round(capped_budget * len(bin_rows) / capped_total_rows))
                idx_parts.append(
                    bin_rows[torch.randint(0, len(bin_rows), (share,), device=self.states.device)]
                )

        # Fill the remainder of the batch with an equal number drawn from each of the
        # other bins, which lifts the sparsely populated mid-drift bins up to parity.
        remaining = batch_size - sum(len(part) for part in idx_parts)
        if boosted_bins and remaining > 0:
            per_bin = max(1, remaining // len(boosted_bins))
            for b in boosted_bins:
                bin_rows = self._bin_row_cache.get(b)
                if bin_rows is None or len(bin_rows) == 0:
                    continue
                idx_parts.append(
                    bin_rows[torch.randint(0, len(bin_rows), (per_bin,), device=self.states.device)]
                )

        idx = (
            torch.cat(idx_parts)[:batch_size] if idx_parts
            else torch.randint(0, len(self), (batch_size,), device=self.states.device)
        )
        # Rounding can leave the batch a few rows short; top it up with a uniform draw.
        if len(idx) < batch_size:
            extra = torch.randint(0, len(self), (batch_size - len(idx),), device=self.states.device)
            idx = torch.cat([idx, extra])
        return self._gather(idx)

    def __post_init__(self) -> None:
        """Cache the row indices belonging to each drift bin."""
        if self.drift_bin is not None:
            for b in range(self.num_drift_bins):
                rows = torch.nonzero(self.drift_bin == b, as_tuple=True)[0]
                if len(rows) > 0:
                    self._bin_row_cache[b] = rows


def bin_by_drift_magnitude(states: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Discretise the cumulative-drift magnitude of each state into bins.

    The drift magnitude is the Euclidean norm of the first two state dimensions
    (the two components of the cumulative injected bias). Equal-width bins span
    zero to the observed maximum.

    Args:
        states: Batch of states.
        num_bins: Number of bins.

    Returns:
        A per-state integer bin index.
    """
    # The first two state dimensions are the components of the cumulative injected
    # bias, so their norm is the injected-drift magnitude the bins are built on.
    mag = np.linalg.norm(states[:, :2], axis=1)
    max_mag = max(float(mag.max()), 1e-6)
    edges = np.linspace(0.0, max_mag, num_bins + 1)
    bin_idx = np.clip(np.digitize(mag, edges[1:-1]), 0, num_bins - 1)
    return bin_idx.astype(np.int64)


class CQLTrainer:  # pylint: disable=too-many-instance-attributes
    """Constrained offline Conservative Q-Learning trainer."""

    def __init__(self, cfg: CQLConfig) -> None:
        """Build the networks, optimisers, and dual variables.

        Args:
            cfg: Training configuration.
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.metrics_logger: Optional[MetricsLogger] = None
        set_seed(cfg.seed)

        self.train_data, self.val_data, state_mean, state_std, _ = self._load_dataset()
        self.state_dim = int(self.train_data.states.shape[1])
        self.action_dim = int(self.train_data.actions.shape[1])
        # Horizon-consistent bound on bootstrapped targets, given the scaled
        # reward magnitude.
        self.target_q_clip = 5.0 / max(1.0 - cfg.gamma, 1e-3)

        self.actor = Actor(
            self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std
        ).to(self.device)
        self.critic = Critic(
            self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        if cfg.use_cost_critic:
            self.cost_critic = Critic(
                self.state_dim, self.action_dim, cfg.hidden_dim, state_mean, state_std
            ).to(self.device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_target.requires_grad_(False)
            self.cost_critic_opt = optim.Adam(self.cost_critic.parameters(), lr=cfg.critic_lr)
            self.log_beta = torch.zeros((), device=self.device, requires_grad=True)
            self.beta_opt = optim.Adam([self.log_beta], lr=cfg.beta_lr)

    def _load_dataset(  # pylint: disable=too-many-locals
        self,
    ) -> Tuple[Transitions, Optional[Transitions], torch.Tensor, torch.Tensor, float]:
        """Load, split, normalise, and package the dataset.

        Returns:
            A tuple ``(train, val, state_mean, state_std, reward_scale)``, where
            ``val`` is ``None`` when there are too few episodes to hold any out.

        Raises:
            AssertionError: If every state equals its own next state, which
                indicates self-loop transitions carrying no learning signal.
        """
        data = np.load(self.cfg.dataset_path)
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        costs = (
            data["costs"].astype(np.float32) if "costs" in data.files
            else np.zeros_like(rewards)
        )
        next_states = data["next_states"].astype(np.float32)
        n = states.shape[0]

        assert not np.allclose(states, next_states), (
            "states == next_states for every row: the harvested dataset has self-loop "
            "transitions and carries no Bellman signal."
        )

        terminals = data.get("terminals", np.zeros(n, dtype=np.float32)).astype(np.float32)

        if "episode_ids" in data.files:
            episode_ids = data["episode_ids"]
        else:
            episode_ids = np.repeat(
                np.arange(len(data["episode_lengths"])), data["episode_lengths"]
            )

        # Hold out whole episodes rather than individual transitions, so no validation
        # transition has its own next-state sitting in the training split.
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

        # Normalisation statistics come from the training split alone, to avoid leaking
        # the validation distribution into the networks' input scaling.
        state_mean = states[train_mask].mean(axis=0)
        state_std = np.maximum(states[train_mask].std(axis=0), 1e-3)
        # Rescale rewards to roughly unit variance so the value targets stay in a
        # predictable range whatever the raw drift magnitudes happen to be.
        reward_scale = self.cfg.reward_scale or 1.0 / max(float(rewards[train_mask].std()), 1e-6)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

        def make(mask: np.ndarray, stratify: bool) -> Transitions:
            drift_bin = None
            num_bins = 0
            num_capped = 0
            if stratify and self.cfg.stratify_by_drift and mask.sum() > 0:
                num_bins = self.cfg.num_drift_bins
                num_capped = self.cfg.num_capped_top_bins
                drift_bin = to_tensor(bin_by_drift_magnitude(states[mask], num_bins)).long()
            return Transitions(
                states=to_tensor(states[mask]),
                actions=to_tensor(actions[mask]),
                rewards=to_tensor(rewards[mask] * reward_scale).unsqueeze(-1),
                costs=to_tensor(costs[mask]).unsqueeze(-1),
                next_states=to_tensor(next_states[mask]),
                not_terminal=to_tensor(1.0 - terminals[mask]).unsqueeze(-1),
                drift_bin=drift_bin,
                num_drift_bins=num_bins,
                num_capped_top_bins=num_capped,
            )

        # Only the training split is stratified; validation reflects the real
        # distribution so its metrics stay a faithful check.
        val = make(val_mask, stratify=False) if n_val > 0 else None
        train = make(train_mask, stratify=True)
        if self.cfg.stratify_by_drift and train.drift_bin is not None:
            bin_counts = [
                int((train.drift_bin == b).sum()) for b in range(self.cfg.num_drift_bins)
            ]
            capped_from = self.cfg.num_drift_bins - self.cfg.num_capped_top_bins
            logger.info(
                "Drift-magnitude bin counts: %s | bins >= %d sampled at natural frequency, "
                "bins < %d equalised", bin_counts, capped_from, capped_from,
            )
        return train, val, torch.as_tensor(state_mean), torch.as_tensor(state_std), reward_scale

    def _sample_policy_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy policy actions and their log probabilities.

        Args:
            states: Batch of states.

        Returns:
            A tuple of the sampled actions and their Gaussian log probabilities.
        """
        sigma = self.cfg.cql_policy_noise
        mean = self.actor(states)
        noise = torch.randn_like(mean) * sigma
        actions = (mean + noise).clamp(-1.0, 1.0)
        log_prob = (
            -0.5 * (noise / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return actions, log_prob

    def _cql_gaps(  # pylint: disable=too-many-locals
        self, critic: Critic, s: torch.Tensor, ns: torch.Tensor,
        q1_data: torch.Tensor, q2_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the conservative gap for each value head.

        The gap is the log-sum-exp of values over random and policy-sampled
        actions minus the value on the dataset action. Raising it pushes value
        down on actions the dataset does not support.

        Args:
            critic: Critic to evaluate.
            s: Batch of states.
            ns: Batch of next states.
            q1_data: First value head on the dataset actions.
            q2_data: Second value head on the dataset actions.

        Returns:
            The conservative gap for each of the two value heads.
        """
        n = self.cfg.num_cql_samples
        batch = s.size(0)
        temp = self.cfg.cql_temperature
        # Repeat each state n times so n candidate actions can be scored per state in a
        # single forward pass.
        s_rep = s.unsqueeze(1).expand(batch, n, s.size(1)).reshape(batch * n, -1)
        ns_rep = ns.unsqueeze(1).expand(batch, n, ns.size(1)).reshape(batch * n, -1)

        with torch.no_grad():
            # Three action sources feed the conservative term: uniform random actions and
            # the policy's actions at the current and next states. Each carries its own
            # log-probability so the later log-sum-exp can correct for how it was drawn.
            rand_a = torch.rand(batch * n, self.action_dim, device=s.device) * 2.0 - 1.0
            log_p_rand = -self.action_dim * math.log(2.0)
            cur_a, log_p_cur = self._sample_policy_actions(s_rep)
            nxt_a, log_p_nxt = self._sample_policy_actions(ns_rep)

        def head_values(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            q1, q2 = critic(s_rep, actions)
            return q1.view(batch, n), q2.view(batch, n)

        q1_r, q2_r = head_values(rand_a)
        q1_c, q2_c = head_values(cur_a)
        q1_n, q2_n = head_values(nxt_a)
        lp_c, lp_n = log_p_cur.view(batch, n), log_p_nxt.view(batch, n)

        # log-sum-exp over the candidate actions approximates the soft-maximum value;
        # subtracting the dataset action's value gives the gap the penalty minimises,
        # which pushes value down on actions the dataset does not support.
        cat1 = torch.cat([q1_r - log_p_rand, q1_c - lp_c, q1_n - lp_n], dim=1)
        cat2 = torch.cat([q2_r - log_p_rand, q2_c - lp_c, q2_n - lp_n], dim=1)
        lse1 = temp * torch.logsumexp(cat1 / temp, dim=1)
        lse2 = temp * torch.logsumexp(cat2 / temp, dim=1)
        return (lse1 - q1_data.squeeze(-1)).mean(), (lse2 - q2_data.squeeze(-1)).mean()

    def _cost_cql_penalty(  # pylint: disable=too-many-locals
        self, s: torch.Tensor, qc1_data: torch.Tensor, qc2_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the cost critic's conservative penalty for each head.

        The penalty pushes the cost estimate up on actions the dataset does not
        support, the opposite bias to the reward critic, so an unfamiliar action
        is never implicitly assumed safer than the data supports. The out-of-
        distribution predictions are clamped to the same bound as the cost
        target so the penalty cannot grow without limit.

        Args:
            s: Batch of states.
            qc1_data: First cost head on the dataset actions.
            qc2_data: Second cost head on the dataset actions.

        Returns:
            The conservative gap for each of the two cost heads.
        """
        n = self.cfg.num_cql_samples
        batch = s.size(0)
        s_rep = s.unsqueeze(1).expand(batch, n, s.size(1)).reshape(batch * n, -1)

        with torch.no_grad():
            rand_a = torch.rand(batch * n, self.action_dim, device=s.device) * 2.0 - 1.0
            cur_a, _ = self._sample_policy_actions(s_rep)

        def head_values(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            q1, q2 = self.cost_critic(s_rep, actions)
            # Clamp to the same bound as the cost target so minimising the gap below
            # cannot inflate the out-of-distribution cost estimate without limit.
            q1 = q1.clamp(0.0, self.target_q_clip)
            q2 = q2.clamp(0.0, self.target_q_clip)
            return q1.view(batch, n), q2.view(batch, n)

        q1_r, q2_r = head_values(rand_a)
        q1_c, q2_c = head_values(cur_a)
        ood1 = torch.cat([q1_r, q1_c], dim=1).mean(dim=1)
        ood2 = torch.cat([q2_r, q2_c], dim=1).mean(dim=1)

        # A positive gap means the unfamiliar-action cost sits below the data cost;
        # minimising it raises the cost the critic assigns to unfamiliar actions.
        gap1 = (qc1_data.detach().squeeze(-1) - ood1).mean()
        gap2 = (qc2_data.detach().squeeze(-1) - ood2).mean()
        return gap1, gap2

    def _update_critic(  # pylint: disable=too-many-locals
        self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
        ns: torch.Tensor, nd: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Update the reward critic and its conservative dual variable.

        Args:
            s: Batch of states.
            a: Batch of dataset actions.
            r: Batch of scaled rewards.
            ns: Batch of next states.
            nd: One minus the terminal flag.

        Returns:
            A dictionary of reward-critic metrics.
        """
        with torch.no_grad():
            # Target-policy smoothing: perturb the target action so the critic cannot
            # latch onto a sharp, exploitable peak in the value surface.
            noise = (torch.randn_like(a) * self.cfg.target_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            # Clipped double-Q target: the smaller of the twin target heads curbs the
            # value overestimation that bootstrapping otherwise accumulates.
            next_a = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(ns, next_a)
            target_q = r + nd * self.cfg.gamma * torch.min(tq1, tq2)
            # Clamp to a horizon-consistent bound so a single outlying bootstrap
            # cannot compound across steps.
            target_q = target_q.clamp(-self.target_q_clip, self.target_q_clip)

        q1, q2 = self.critic(s, a)
        td_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)
        gap1, gap2 = self._cql_gaps(self.critic, s, ns, q1, q2)

        # With a target gap configured, the penalty weight alpha is tuned automatically
        # by dual ascent below; otherwise it stays a fixed constant.
        if self.cfg.cql_target_gap is None:
            alpha = torch.tensor(self.cfg.cql_alpha, device=s.device)
            penalty = alpha * (gap1 + gap2)
        else:
            alpha_for_penalty = self.log_alpha.exp().clamp(
                self.cfg.alpha_min, self.cfg.alpha_max
            ).detach()
            penalty = alpha_for_penalty * (
                (gap1 - self.cfg.cql_target_gap) + (gap2 - self.cfg.cql_target_gap)
            )

        critic_loss = td_loss + penalty
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        alpha_value = torch.tensor(self.cfg.cql_alpha, device=s.device)
        if self.cfg.cql_target_gap is not None:
            # Recompute the penalty weight on its own graph before its gradient
            # step. The lower and upper bounds are enforced each time the weight
            # is read, since its value is never used unclamped.
            alpha = self.log_alpha.exp().clamp(self.cfg.alpha_min, self.cfg.alpha_max)
            gap_term = (
                (gap1.detach() - self.cfg.cql_target_gap)
                + (gap2.detach() - self.cfg.cql_target_gap)
            )
            alpha_loss = -alpha * gap_term
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_value = (
                self.log_alpha.exp().clamp(self.cfg.alpha_min, self.cfg.alpha_max).detach()
            )

        return {
            "critic_loss": critic_loss.detach(),
            "td_loss": td_loss.detach(),
            "cql_gap": (0.5 * (gap1 + gap2)).detach(),
            "alpha": alpha_value,
            "q_data": q1.mean().detach(),
        }

    def _update_cost_critic(  # pylint: disable=too-many-locals
        self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
        ns: torch.Tensor, nd: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Update the cost critic and its optional conservative penalty.

        Args:
            s: Batch of states.
            a: Batch of dataset actions.
            c: Batch of admission costs.
            ns: Batch of next states.
            nd: One minus the terminal flag.

        Returns:
            A dictionary of cost-critic metrics.
        """
        with torch.no_grad():
            next_a = self.actor_target(ns).clamp(-1.0, 1.0)
            tq1, tq2 = self.cost_critic_target(ns, next_a)
            # Costs are non-negative, so the maximum of the twin heads is used to
            # avoid systematically underestimating risk.
            target_qc = c + nd * self.cfg.gamma * torch.max(tq1, tq2)
            target_qc = target_qc.clamp(0.0, self.target_q_clip)

        qc1, qc2 = self.cost_critic(s, a)
        cost_td_loss = F.smooth_l1_loss(qc1, target_qc) + F.smooth_l1_loss(qc2, target_qc)

        # The cost critic's own conservative penalty is optional and off by default; it
        # adds pessimism on unfamiliar actions only when explicitly enabled.
        cost_cql_gap: torch.Tensor = torch.tensor(0.0, device=s.device)
        if self.cfg.cost_cql_alpha > 0.0:
            gap1, gap2 = self._cost_cql_penalty(s, qc1, qc2)
            cost_cql_gap = 0.5 * (gap1 + gap2)
            cost_penalty = self.cfg.cost_cql_alpha * (gap1 + gap2)
            cost_critic_loss = cost_td_loss + cost_penalty
        else:
            cost_critic_loss = cost_td_loss

        self.cost_critic_opt.zero_grad(set_to_none=True)
        cost_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=1.0)
        self.cost_critic_opt.step()

        return {
            "cost_td_loss": cost_td_loss.detach(),
            "cost_cql_gap": cost_cql_gap.detach(),
            "qc_data": qc1.mean().detach(),
        }

    def _update_actor(self, s: torch.Tensor, a: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update the actor and, when constrained, the constraint dual variable.

        The actor maximises the reward value minus the constraint multiplier
        times the cost value. The multiplier is then raised while the predicted
        cost exceeds its budget, which is what enforces the admission floors.

        Args:
            s: Batch of states.
            a: Batch of dataset actions.

        Returns:
            A dictionary of actor metrics.
        """
        # Freeze the critics while the actor updates, so their parameters are untouched
        # by the actor's backward pass and only the policy moves.
        self.critic.requires_grad_(False)
        pi = self.actor(s)
        q_pi = torch.min(*self.critic(s, pi))

        if self.cfg.use_cost_critic:
            self.cost_critic.requires_grad_(False)
            qc_pi = torch.max(*self.cost_critic(s, pi))
            beta = self.log_beta.exp().clamp(0.0, self.cfg.beta_max).detach()
            # The actor maximises reward value while the multiplier beta discourages
            # actions the cost critic predicts will break the admission constraint.
            objective = q_pi - beta * qc_pi
            self.cost_critic.requires_grad_(True)
        else:
            objective = q_pi

        actor_loss = -objective.mean()

        # Behaviour cloning anchors the policy to actions seen in the data, keeping it
        # inside the region the critics can score reliably.
        if self.cfg.bc_weight > 0.0:
            actor_loss = actor_loss + self.cfg.bc_weight * F.mse_loss(pi, a)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()
        self.critic.requires_grad_(True)

        metrics = {"actor_loss": actor_loss.detach()}

        if self.cfg.use_cost_critic:
            # Dual ascent on beta: raise it while the predicted cost exceeds its budget,
            # so the admission constraint is enforced rather than traded off at a fixed
            # weight the way a plain penalty would be.
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
        """Soft-update the target networks toward the online networks.

        Returns:
            None.
        """
        # Polyak averaging: nudge each target a small step tau towards its online
        # network, which gives the bootstrapped targets a slowly moving, stable anchor.
        tau = self.cfg.tau
        pairs = [(self.critic, self.critic_target), (self.actor, self.actor_target)]
        if self.cfg.use_cost_critic:
            pairs.append((self.cost_critic, self.cost_critic_target))
        for net, target in pairs:
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)

    @torch.no_grad()
    def evaluate(self, data: Transitions) -> Dict[str, float]:
        """Compute validation metrics on a held-out split.

        Args:
            data: Validation transitions.

        Returns:
            A dictionary of scalar validation metrics.
        """
        s, a, r, c, ns, nd = data.all()  # pylint: disable=unused-variable
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
        """Run the full training loop and save the trained actor.

        Returns:
            None.
        """
        logger.info(
            "Training constrained offline CQL on %s for %d steps "
            "(alpha_min=%.3f, bc_weight=%.3f, cost_cql_alpha=%.3f)",
            self.device, self.cfg.total_steps, self.cfg.alpha_min,
            self.cfg.bc_weight, self.cfg.cost_cql_alpha,
        )
        self.metrics_logger = MetricsLogger("train_metrics.csv")

        for step in range(1, self.cfg.total_steps + 1):
            s, a, r, c, ns, nd = self.train_data.sample(self.cfg.batch_size)
            metrics = self._update_critic(s, a, r, ns, nd)
            if self.cfg.use_cost_critic:
                metrics.update(self._update_cost_critic(s, a, c, ns, nd))
            # Delayed updates: the actor and target networks update less often than the
            # critics, letting the value estimates settle before the policy chases them.
            if step % self.cfg.policy_delay == 0:
                metrics.update(self._update_actor(s, a))
                self._soft_update()

            if step == 1 or step % self.cfg.log_interval == 0:
                logger.info(
                    "Step %d | %s", step,
                    " | ".join(f"{k}: {float(v):.4f}" for k, v in metrics.items()),
                )
                self.metrics_logger.log(step=step, **{k: float(v) for k, v in metrics.items()})
            if self.val_data is not None and step % self.cfg.eval_interval == 0:
                val = self.evaluate(self.val_data)
                logger.info(
                    "Step %d | Validation | %s", step,
                    " | ".join(f"{k}: {v:.4f}" for k, v in val.items()),
                )
                self.metrics_logger.log(val_step=step, **val)

        self.save()
        self.metrics_logger.close()

    def save(self) -> None:
        """Save the trained actor and configuration to disk.

        Returns:
            None.
        """
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
    """Load a trained actor from a checkpoint.

    Args:
        path: Path to the checkpoint written by :meth:`CQLTrainer.save`.
        device: Device to place the actor on.

    Returns:
        The loaded actor in evaluation mode.
    """
    ckpt = torch.load(path, map_location=device)
    dim = ckpt["state_dim"]
    actor = Actor(dim, ckpt["action_dim"], ckpt["hidden_dim"], torch.zeros(dim), torch.ones(dim))
    actor.load_state_dict(ckpt["actor"])
    return actor.to(device).eval()


def parse_args() -> CQLConfig:
    """Parse command-line arguments into a configuration.

    Returns:
        The training configuration assembled from the command line.
    """
    parser = argparse.ArgumentParser(description="Constrained offline CQL training")
    parser.add_argument("--dataset", default=CQLConfig.dataset_path)
    parser.add_argument("--steps", type=int, default=CQLConfig.total_steps)
    parser.add_argument("--alpha", type=float, default=CQLConfig.cql_alpha)
    parser.add_argument(
        "--target-gap", type=float, default=2.0, help="Enable the Lagrangian penalty weight."
    )
    parser.add_argument(
        "--alpha-min", type=float, default=CQLConfig.alpha_min,
        help="Lower bound on the conservative penalty weight.",
    )
    parser.add_argument("--bc-weight", type=float, default=CQLConfig.bc_weight)
    parser.add_argument(
        "--cost-cql-alpha", type=float, default=CQLConfig.cost_cql_alpha,
        help="Weight of the cost critic's own conservative penalty; zero disables it.",
    )
    parser.add_argument("--no-cost-critic", action="store_true")
    parser.add_argument(
        "--no-stratify-by-drift", action="store_true",
        help="Disable drift-magnitude-stratified sampling and sample rows uniformly.",
    )
    parser.add_argument("--num-drift-bins", type=int, default=CQLConfig.num_drift_bins)
    parser.add_argument(
        "--num-capped-top-bins", type=int, default=CQLConfig.num_capped_top_bins,
        help="Number of highest-drift bins sampled at natural frequency rather than equalised.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return CQLConfig(
        dataset_path=args.dataset,
        total_steps=args.steps,
        cql_alpha=args.alpha,
        cql_target_gap=args.target_gap,
        alpha_min=args.alpha_min,
        bc_weight=args.bc_weight,
        cost_cql_alpha=args.cost_cql_alpha,
        use_cost_critic=not args.no_cost_critic,
        stratify_by_drift=not args.no_stratify_by_drift,
        num_drift_bins=args.num_drift_bins,
        num_capped_top_bins=args.num_capped_top_bins,
        seed=args.seed,
    )


if __name__ == "__main__":
    CQLTrainer(parse_args()).train()
