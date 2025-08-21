# pylint: disable=too-few-public-methods
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
from typing import Dict, Tuple
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
    measurement is inconsistent).
    - dof: Degrees of freedom of the measurement. i.e.: range only: dof = 1.
    Used with nis to judge consistency.
    - post_cov: A 6x6 posterior covariance matrix that provides a snapshot of
    how uncertain the filter is after the update.
    """
    target_id: str
    nis: float
    dof: int
    post_cov: np.ndarray

class SDEKF: # TODO - ref Mals paper. Check this actually is doing what I wantt
    """
    Simple square-root EKF-like estimator (improves numerical stability)
    - State: [r, v] (6x1) in ECI
    - Dynamics: nearly-constant velocity (NCV) for robustness (note: this is not
    high-fidelity orbital dynamics, but sufficient for innovation-based
    consensus scoring)
    - Measurements: range (1d), azimuth/elevation (2D), right ascension/declination (2D)
    """

    # TODO tune init values
    def __init__(self,
                 pos_noise: float = 10.0,
                 vel_noise: float = 1e-2,
                 meas_floor: float = 1.0) -> None:
        """
        Initialise the SDEKF.

        Arguments:
        - pos_noise: Process noise for position [m^2 / s].
        - vel_noise: Process noise for velocity [(m/s)^2 / s].
        - meas_floor: Minimum variance floor to avoid singularities.
        """

        self.targets: Dict[str, State] = {}
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.meas_floor = meas_floor

    def process_measurement(self, measurements: Dict) -> ODProcessingResult:
        """
        Process one cooperative measurement and update the per-target filter.

        Arguments:
        - measurements: Measurement dictionary containing target_id, timestamp,
        and observables

        Returns:
        - Innovation statistics and posterior covariance.
        """

        target_id = str(measurements["target_id"])
        timestamp = self._timestamp_to_seconds(measurements["timestamp"])

        # Initialise if needed (only if we can get a position)
        if target_id not in self.targets and self._can_initialise(measurements):
            self.targets[target_id] = self._initialise_state(measurements, timestamp)

        # If still not initialised, keep a very weak prior and use a neutral NIS
        if target_id not in self.targets:
            x0 = np.zeros((6,1))
            p0 = np.diag([1e10, 1e10, 1e10, 1e6, 16, 1e6])
            self.targets[target_id] = State(state_estimate=x0,
                                            covariance=p0,
                                            last_update_seconds=timestamp)
            return ODProcessingResult(target_id, nis=1.0, dof=1, post_cov=p0)

        # Predict to time t
        state = self.targets[target_id]
        delta_t = max(0.0, timestamp - state.last_update_seconds)
        x_pred, p_pred = self._predict(state.state_estimate, state.covariance, delta_t)
        state.last_update_seconds = timestamp

        # Build measurement and Jacobian
        h, z, z_hat, r = self._build_measurement_model(measurements, x_pred)

    # --------------------------------------------------------------------------------------
    # Internal methods
    @staticmethod
    def _timestamp_to_seconds(ts) -> float:
        """
        TODO - update and type hint
        """
        if isinstance(ts, (int, float)):
            return float(ts)
        # If string (ISO), we can fallback to 0; consensus uses relative stats anyway.
        # Upstream can pass UNIX seconds to get proper timing.
        return 0.0

    def _can_initialise(self, measurement_dict: Dict) -> bool:
        """
        This function is a guard that checks whether there is enough information to start
        the filter — i.e., to initialise the state and covariance.

        Arguments:
        - measurement_dict: The measurement dictionary for one observation.

        Returns:
        - True is the filter can initialise, False otherwise.
        """
        # Need an observer state and either range and angles, or at least range + LOS unit
        has_obs: bool = "observer_state_eci" in measurement_dict and \
                  "r_m" in measurement_dict["observer_state_eci"]

        has_range: bool = "range_m" in measurement_dict

        has_any_angle: bool = "az_el_rad" in measurement_dict or \
            "ra_dec_rad" in measurement_dict or "los_eci" in measurement_dict

        return bool(has_obs and (has_range and has_any_angle))

    @staticmethod
    def _norm(v: np.ndarray) -> float:
        """Return Euclidean norm of vector v."""
        return float(np.linalg.norm(v))

    def _initialise_state(self, measurement_dict: Dict, timestamp: float) -> State:
        """
        Creates the initial state estimate (x0) and its covariance (p0).

        Arguments:
        - measurement_dict: The measurement dictionary for one observation.
        - timestamp: The measurement timestamp in seconds

        Returns: A state dataclass.
        """
        # Exract observer position and convert to a 3x1 column vector
        obs = measurement_dict["observer_state_eci"]
        r_obs: np.ndarray = np.array(obs["r_m"], dtype=float).reshape(3, 1)

        # Build line-of-sight unit vector u
        # If a LOS vecto in ECI coords is provided, normalise it
        if "los_eci" in measurement_dict:
            u = np.array(measurement_dict["los_eci"], dtype=float).reshape(3, 1)
            u = u / max(1e-12, self._norm(u))
        # If only azimuth/elevation is provided, compute an approx LOS vector
        elif "az_el_rad" in measurement_dict:
            az, el = [float(x) for x in measurement_dict["az_el_rad"]]
            # Local-ENU LOS in observer frame would require a frame map; use a simple placeholder:
            # assume azimuth measured from x-y plane and
            # elevation from horizon; form ECI-like LOS (approx).
            u = np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ], dtype=float).reshape(3, 1)
        # If only RA/Dec (astronomical coordinates) are given, convert to a unit vector
        else:
            ra, dec = [float(x) for x in measurement_dict["ra_dec_rad"]]
            u = np.array([
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ], dtype=float).reshape(3, 1)

        # Compute target position. rho = range from observer to target
        rho = float(measurement_dict.get("range_m", 0.0))
        r_tgt = r_obs + rho * u

        # Build initial state vector
        x0 = np.zeros((6, 1))
        x0[0:3, :] = r_tgt
        # Start with unknown velocity
        # High uncertainty for initial position (1e6 m²) and moderate for velocity (1e4 (m/s)²).
        p0 = np.diag([1e6, 1e6, 1e6, 1e4, 1e4, 1e4]).astype(float)
        return State(state_estimate=x0, covariance=p0, last_update_seconds=timestamp)

    def _update_noise_covariance(self, dt: float) -> np.ndarray:
        """
        Process noise covariance for NCV model.

        Arguments:
        - dt: Time step (s).

        Returns:
        - Covariance matrix
        """
        x_noise = self.pos_noise
        v_noise = self.vel_noise
        updated_x_noise = x_noise * dt * np.eye(3)
        updated_v_noise = v_noise * dt * np.eye(3)

        return np.block([[updated_x_noise, np.zeros((3, 3))], [np.zeros((3, 3)), updated_v_noise]])

    def _predict(self, x: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propogate state and covariance forward by dt using NCV model.
        # TODO - NCV vs sgp4??

        Arguments:
        - x: Current state vector.
        - p: Current covariance.
        - dt: Time step (s).

        Returns:
        - Tuple of predicted state and covariance.
        """

        if dt <= 0.0:
            return x, p

        # stm is the state transition matrix
        stm = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3,3)), np.eye(3)]])
        noise_covariance = self._update_noise_covariance(dt)

        # Make predictions (@ is matrix multiplication)
        x_pred = stm @ x
        p_pred = stm @ p @ stm.T + noise_covariance

        # Ensure covariance remains symmetric
        p_pred = 0.5 * (p_pred + p_pred.T)
        return x_pred, p_pred

    def _build_measurement_model(self, measurement_dict: Dict,
                                 x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a measurement model for a given observation.

        Arguments:
        - measurement_dict: Measurement dictionary.
        - x: predicted state vector.

        Returns:
        A tuple of: # TODo - rename here and above where called
        - h: Measurement Jacobian # TODO - what?
        - z: Actual measurement vector.
        - z_hat: Predicted measurement vector.
        - r: Measurement covariance.
        """

        # TODO - finish

