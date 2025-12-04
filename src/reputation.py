# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-instance-attributes
"""
The Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework.
Author: Beth Probert
Email: beth.probert@strath.ac.uk

Copyright (C) 2025 Applied Space Technology Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import time
import numpy as np

# Global Variables for consensus
MAX_REPUTATION: float = 1

class ReputationManager:
    """
    A class for calculating and updating a satellite node's reputation
    """
    def __init__(self,
                 max_rep: float = MAX_REPUTATION,
                 offset: float = 0.693,
                 growth_rate: float = 0.6,
                 decay_rate: float = 0.002,
                 alpha: float = 0.12,
                 performance_ema_alpha: float = 0.05,
                 min_drop_factor: float = 0.65,
                 max_drop_factor: float = 0.95) -> None:
        """
        max_rep: max possible reputation
        offset, growth_rate: Gompertz curve parameters
        decay_rate: exponential decay per second (or tick)
        alpha: % of distance toward Gompertz target per positive event
        performance_ema_alpha: Smoothing factor for the performance EMA.
        min_drop_factor: Multiplier for the worst-case negative event.
        max_drop_factor: Multiplier for the mildest negative event.
        """
        self.max_rep = max_rep
        self.offset = offset
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.performance_ema_alpha = performance_ema_alpha
        self.min_drop_factor = min_drop_factor
        self.max_drop_factor = max_drop_factor
        self._last_update = time.time()

    def decay(self, current_rep: float) -> float:
        """
        Calculate the time decay for a node's reputation calculation

        Args:
        - current_rep: The node's reputation before a time decay is applied.

        Returns:
        - The node's reputation after a time decay is applied. The lower
          bound from this decay is neutral reputation (max reputation / 2)
          so as to not overly penalise inactive nodes, especially in async
          situations where nodes may be out of contact for a longer period.
        """
        neutral_rep: float = MAX_REPUTATION / 2
        now: float = time.time()
        delta_t: float = now - self._last_update
        self._last_update = now

        # Exponential decay towards neutral reputation
        decayed_rep = \
            neutral_rep + ((current_rep - neutral_rep) * np.exp(-self.decay_rate * delta_t))

        return min(max(decayed_rep, 0.0), MAX_REPUTATION)

    def _gompertz_target(self, exp_pos: int) -> float:
        """
        Calculate Gompertz function impact on reputation.

        Args:
        - exp_pos: The number of positive experiences the node has had.

        Returns:
        - The gompertz function impact on a node's reputation, used
          as an upper bound for reputation.
        """
        return self.max_rep * np.exp(-self.offset * np.exp(-self.growth_rate * exp_pos))

    def apply_positive(self, current_rep: float, exp_pos: int,
                       current_performance_ema: float) -> tuple[float, int, float]:
        """
        Apply reputation effect for a positive node interaction.

        Args:
        - current_rep: The node's reputation before a time decay is applied.
        - exp_pos: The number of positive experiences the node has had.
        - current_performance_ema: The node's current performance EMA score.

        Returns:
        - The updated reputation, updated number of positive experiences,
          and the new performance EMA.
        """
        current_rep = self.decay(current_rep)
        target = self._gompertz_target(exp_pos)
        new_rep = current_rep + self.alpha * (target - current_rep)

        # Update performance EMA towards 1 for a positive outcome
        new_performance_ema = (self.performance_ema_alpha * 1.0) + \
                              (1 - self.performance_ema_alpha) * current_performance_ema

        return float(min(self.max_rep, new_rep)), exp_pos + 1, new_performance_ema

    def apply_negative(self, current_rep: float, exp_pos: int,
                       current_performance_ema: float) -> tuple[float, int, float]:
        """
        Apply reputation effect for a negative node interaction.

        The penalty is scaled based on the node's recent performance (EMA).

        Args:
        - current_rep: The node's reputation before a time decay is applied.
        - exp_pos: The number of positive experiences the node has had.
        - current_performance_ema: The node's current performance EMA score.

        Returns:
        - The updated reputation, unchanged number of positive experiences,
          and the new performance EMA.
        """
        current_rep = self.decay(current_rep)

        # Update performance EMA towards 0 for a negative outcome
        new_performance_ema = (1 - self.performance_ema_alpha) * current_performance_ema

        # Calculate a dynamic drop factor based on the stable performance EMA.
        # A high performance EMA results in a milder penalty
        # (drop factor closer to max_drop_factor).
        bonus_range = self.max_drop_factor - self.min_drop_factor
        dynamic_drop_factor = self.min_drop_factor + bonus_range * current_performance_ema

        new_rep = current_rep * dynamic_drop_factor

        return float(max(0.0, new_rep)), exp_pos, new_performance_ema
