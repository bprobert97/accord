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
MAX_REPUTATION: float = 100.0

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
                 min_drop_factor: float = 0.65,
                 max_drop_factor: float = 0.95) -> None:
        """
        max_rep: max possible reputation
        B, C: Gompertz curve parameters
        decay_rate: exponential decay per second (or tick)
        alpha: % of distance toward Gompertz target per positive event
        min_drop_factor: Multiplier for the worst-case negative event (low rep, bad data).
        max_drop_factor: Multiplier for the mildest negative event (high rep, ok data).
        """
        self.max_rep = max_rep
        self.offset = offset
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.alpha = alpha
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

    def apply_positive(self, current_rep: float, exp_pos: int) -> tuple[float, int]:
        """
        Apply reputation effect for a positive node interaction.

        Args:
        - current_rep: The node's reputation before a time decay is applied.
        - exp_pos: The number of positive experiences the node has had.

        Returns:
        - The updated reputation and updated number of positive experiences,
          increased by one.
        """
        current_rep = self.decay(current_rep)
        target = self._gompertz_target(exp_pos)
        new_rep = current_rep + self.alpha * (target - current_rep)
        return float(min(self.max_rep, new_rep)), exp_pos + 1

    def apply_negative(self, current_rep: float, exp_pos: int,
                       correctness_score: float = 0.0) -> tuple[float, int]:
        """
        Apply reputation effect for a negative node interaction.

        The penalty is scaled based on the node's current reputation and the
        correctness of the data submitted. A higher reputation and a higher
        correctness score lead to a larger reputation multiplier (a smaller penalty).

        Args:
        - current_rep: The node's reputation before a time decay is applied.
        - exp_pos: The number of positive experiences the node has had.
        - correctness_score: The correctness score [0,1] of the submission.
          Defaults to 0 for cases like invalid data where score is not applicable.

        Returns:
        - The updated reputation and updated number of positive experiences,
          that does not change.
        """
        current_rep = self.decay(current_rep)

        # Calculate a dynamic reputation multiplier (drop factor).
        # A higher factor means a smaller penalty.
        # The factor is scaled between min_drop_factor and max_drop_factor.
        bonus_range = self.max_drop_factor - self.min_drop_factor

        # Merit is a weighted average of reputation and correctness (0 to 1)
        rep_merit = current_rep / self.max_rep
        # Using 50/50 weighting for reputation and correctness
        combined_merit = 0.8 * rep_merit + 0.2 * correctness_score

        dynamic_drop_factor = self.min_drop_factor + bonus_range * combined_merit

        new_rep = current_rep * dynamic_drop_factor
        return float(max(0.0, new_rep)), exp_pos
