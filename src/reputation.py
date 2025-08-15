# pylint: disable=too-many-arguments, too-many-positional-arguments
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
                 drop_factor: float = 0.35) -> None:
        """
        max_rep: max possible reputation
        B, C: Gompertz curve parameters
        decay_rate: exponential decay per second (or tick)
        alpha: % of distance toward Gompertz target per positive event
        drop_factor: multiplicative penalty for negative events
        """
        self.max_rep = max_rep
        self.offset = offset
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.drop_factor = drop_factor
        self._last_update = time.time()

    def decay(self, current_rep: float) -> float:
        """
        Calculate the time decay for a node's reputation calculation
        """
        now: float = time.time()
        delta_t: float = now - self._last_update
        self._last_update = now
        return current_rep * np.exp(-self.decay_rate * delta_t)

    def _gompertz_target(self, exp_pos: int) -> float:
        """
        Calculate Gompertz function impact on reputation
        """
        return self.max_rep * np.exp(-self.offset * np.exp(-self.growth_rate * exp_pos))

    def apply_positive(self, current_rep: float, exp_pos: int) -> tuple[float, int]:
        """
        Apply reputation effect for a positive node interaction
        """
        current_rep = self.decay(current_rep)
        target = self._gompertz_target(exp_pos)
        new_rep = current_rep + self.alpha * (target - current_rep)
        return float(min(self.max_rep, new_rep)), exp_pos + 1

    def apply_negative(self, current_rep: float, exp_pos: int) -> tuple[float, int]:
        """
        Apply reputation effect for a negative node interaction
        """
        current_rep = self.decay(current_rep)
        new_rep = current_rep * self.drop_factor
        return float(max(0.0, new_rep)), exp_pos
