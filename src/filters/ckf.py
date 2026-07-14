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

"""
This file implements the Cubature Kalman Filter (CKF) for the ACCORD framework.
It provides a joint state estimation architecture that tracks the orbital states
(position and velocity) of a constellation of satellites. By applying the
spherical-radial cubature rule (kappa=0.0) to the standard Unscented Transform,
the filter uses a continuous white noise acceleration kinematic model and robust
Cholesky decomposition to handle the numerical scaling challenges inherent in
orbital mechanics, updating the network's joint state based on inter-satellite
line-of-sight measurements.
"""
from typing import List
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
from src.filters.filter_interface import (
    FilterConfig,
    ObservationRecord,
    STATE_DIM,
    hx_joint,
    initialise_state_and_cov)
from src.filters.ukf import get_q_matrix, robust_cholesky, _log_nis_ukf, fx_joint
from src.logger import get_logger

logger = get_logger()

class JointCKF:
    """
    A class to manage the state and operations of a joint Cubature Kalman Filter.
    """
    def __init__(self, config: FilterConfig, initial_truth: np.ndarray):
        """
        Initialises the JointCKF.

        Args:
        - config: The configuration for the filter.
        - initial_truth: The initial true state of the satellites.

        Returns:
        - None.
        """
        self.config = config
        dim_x = STATE_DIM * config.N
        M = config.N * (config.N - 1)
        dim_z = 2 * M

        # Setting kappa=0 exactly replicates the Cubature Kalman Filter (CKF)
        # The centre point weight becomes 0, generating the spherical-radial cubature points
        points = JulierSigmaPoints(n=dim_x, kappa=0.0, sqrt_method=robust_cholesky) # type: ignore

        self.ckf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=config.dt,
                                         fx=fx_joint, hx=hx_joint, points=points) # type: ignore

        x0_est, P0 = initialise_state_and_cov(config.N, initial_truth[np.newaxis, :])
        self.ckf.x = x0_est
        self.ckf.P = P0
        self.ckf.R = np.diag([config.sig_r**2, config.sig_rdot**2] * M)
        self.ckf.Q = get_q_matrix(config.dt, config.q_acc_target, config.N)

    def predict(self) -> None:
        """
        Performs the prediction step of the CKF by passing joint kwargs to fx.

        Args:
        - None.

        Returns:
        - None.
        """
        self.ckf.predict(N=self.config.N) # type: ignore

    def update(self, z_k: np.ndarray, k: int) -> List[ObservationRecord]:
        """
        Performs the update step of the CKF and returns observation records.

        Args:
        - z_k: The measurement vector for the current step.
        - k: The current step index.

        Returns:
        - A list of ObservationRecord objects for the current step.
        """
        self.ckf.update(z_k, N=self.config.N)
        return _log_nis_ukf(self.ckf.y, self.ckf.S, self.ckf.x, k, self.config)

# TODO- needs unit testing, check non-gaussian noise assumptions, check if this is any better than UKF time-wise