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

from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    """
    Dataclass representing the target state estimate.

    Attributes:
    - state_estimate: 6x1 ,matrix [r (position); v (velocity)]. The current estimate of the
    satellite’s state vector in the Earth-Centered Inertial (ECI) frame.
    - covariance: 6x6 covariance matrix, representing uncertainty in the state estimate.
    - last_update_seconds: The time of he last update in seconds (monotonic or UNIX)
    """
    state_estimate: np.ndarray
    covariance: np.ndarray
    last_update_seconds: float

@dataclass
class ODProcessingResult:
    """
    Output after processing one measurement of a target.
    Attributes:
    - target_id: The identifier of the satellite being tracked.
    - nis: Normalised Innovation Squared. Measures how well the measurement
    agrees with the predicted state. (Small value = fits well, large value =
    measurement is inconsistent)
    - dof: Degrees of freedom of the measurement. i.e.: range only: dof = 1.
    Used with nis to judge consistency.
    post_cov: A 6x6 posterior covariance matrix that provides a snapshot of
    how uncertain the filter is after the update.
    """
    target_id: str
    nis: float
    dof: int
    post_cov: np.ndarray

# class SDEKFFilter: # TODO - ref Mals paper. Check this actually is doing what I want
#     """
#     TODO
#     """
