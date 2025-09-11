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
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.integrate import solve_ivp
from .module_crtbp import crtbp_dstt_dynamics
from .module_stt import dstt_pred_mu_p

@dataclass
class State:
    """
    Dataclass representing the target state estimate.

    Attributes:
    - state_estimate: 6x1 ,matrix [r (position); v (velocity)]. The current estimate of the
    satellite’s state vector in the Earth-Centered Inertial (ECI) frame.
    - covariance: 6x6 covariance matrix, representing uncertainty in the state estimate.
    - last_update_seconds: The time of the last update in seconds (monotonic or UNIX).
    """
    state_estimate: np.ndarray
    covariance: np.ndarray
    last_update_seconds: float

@dataclass
class ODProcessingResult:
    """
    Output after processing one measurement of a target satellite.

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

class SDEKF: # TODO - ref Mals paper. Check this actually is doing what I want.
    # TODO - put this in part of the actual consensus mechanism using the innovation score
    """
    Simple square-root EKF-like estimator (improves numerical stability).
    Designed for processing one measurement at a time from multiple observers.
    - State: [r, v] (6x1) in ECI.
    - Dynamics: CRTBP (Circular Restricted Three-Body Problem) with
    STM (State Transition Matrix) and DSTT (Dynamic State Transition Tensor).
    - Measurements: range (1d), azimuth/elevation (2D), right ascension/declination (2D)
    """

    # TODO tune init values
    def __init__(self,
                 meas_floor: float = 1.0) -> None:
        """
        Initialise the SDEKF.

        Arguments:
        - meas_floor: Minimum variance floor to avoid singularities.
        """

        self.targets: Dict[str, State] = {}
        self.meas_floor = meas_floor
        self.dimensions: int = 6
        self.mu: float = 0.0122
        self.q: np.ndarray = np.zeros([self.dimensions, self.dimensions])

    def process_measurement(self, measurements: Dict) -> ODProcessingResult:

        # TODO - need to find uncertainty refs for noise measurements - if narrow gaussian, its okay
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

        # Initialise if needed
        if target_id not in self.targets and self._can_initialise(measurements):
            self.targets[target_id] = self._initialise_state(measurements, timestamp)

        # If still not initialised, use weak prior
        if target_id not in self.targets:
            return self._weak_prior_result(target_id, timestamp)

        # Predict step
        state = self.targets[target_id]
        x_pred, p_pred = self._predict_to_time(state, timestamp)

        # Update step
        x_upd, p_post, nis, dof = self._update_with_measurement(x_pred, p_pred, measurements)

        # Store updated state
        self.targets[target_id] = State(state_estimate=x_upd,
                                        covariance=p_post,
                                        last_update_seconds=timestamp)

        return ODProcessingResult(target_id=target_id, nis=nis, dof=dof, post_cov=p_post)

    # --------------------------------------------------------------------------------------
    # Internal methods

    def _weak_prior_result(self, target_id: str, timestamp: float) -> ODProcessingResult:
        """
        Return a weak prior state and neutral NIS for uninitialised targets.

        Args:
        - target_id: The identifier of the target satellite.
        - timestamp: The current timestamp in seconds.

        Returns:
        - An ODProcessingResult with a weak prior state and neutral NIS.

        Side effect:
        - Mutates self.targets by adding a new entry for the target with a weak prior state.
        """
        # x0 is a zero state, p0 is large uncertainty
        # TODO - tune these values
        x0 = np.zeros((6, 1))
        p0 = np.diag([1e10, 1e10, 1e10, 1e6, 1e6, 1e6])

        self.targets[target_id] = State(state_estimate=x0,
                                        covariance=p0,
                                        last_update_seconds=timestamp)

        # Expected (neutral) value of NIS is equal to the degrees of freedom
        # Simplest case: range only, dof = 1
        return ODProcessingResult(target_id=target_id, nis=1.0, dof=1, post_cov=p0)


    def _predict_to_time(self, state: State, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate state forward to the given timestamp.

        Args:
        - state: The current state of the target satellite.
        - timestamp: The target timestamp in seconds.

        Returns:
        - A tuple of the predicted state and covariance at the target timestamp.
        """
        # Ensure non-negative time difference
        delta_t = max(0.0, timestamp - state.last_update_seconds)

        # Perform the prediction step
        x_pred, p_pred = self._predict(state.state_estimate, state.covariance, delta_t)
        state.last_update_seconds = timestamp
        return x_pred, p_pred


    def _update_with_measurement( # pylint: disable=too-many-locals
        self, x_pred: np.ndarray, p_pred: np.ndarray, measurements: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Perform the update step for the kalman filter.

        Args:
        - x_pred: The predicted state vector.
        - p_pred: The predicted covariance matrix.
        - measurements: The measurement dictionary containing the observation data.

        Returns:
        - A tuple of the updated state vector, updated covariance matrix,
        """
        # Build measurement model
        h, z, z_hat, r = self._build_measurement_model(measurements, x_pred)

        # Kalman gain and update
        # y = innovation
        # s = innovation covariance
        y = z - z_hat
        s = h @ p_pred @ h.T + r
        s = 0.5 * (s + s.T)

        # Use pseudo-inverse in case s is singular
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s)

        # Kalman gain
        k = p_pred @ h.T @ s_inv

        # Posterior update
        x_upd = x_pred + k @ y
        p_post = (np.eye(6) - k @ h) @ p_pred @ (np.eye(6) - k @ h).T + k @ r @ k.T

        # Symmetrising the covariance matrix helps maintain numerical stability
        # and ensures the matrix remains positive semi-definite, which can be
        # affected by floating-point errors.
        p_post = 0.5 * (p_post + p_post.T)

        nis = float(y.T @ s_inv @ y)
        dof = int(z.shape[0])

        return x_upd, p_post, nis, dof

    @staticmethod
    def _timestamp_to_seconds(ts: Union[int, float, str]) -> float:
        """
        Convert a timestamp to a seconds value as a float.

        Args:
        - ts: The timestamp, either as a float/int (UNIX seconds)
        or string (ISO 8601).

        Returns:
        - The timestamp in seconds as a float.
        If input is a string, returns 0.0
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

        can_initialise = has_obs and has_range and has_any_angle
        return bool(can_initialise)

    @staticmethod
    def _norm(v: np.ndarray) -> float:
        """
        Return Euclidean norm of vector v.
        The Euclidean norm of a vector is a measure of
        its length or magnitude

        Args:
        - v: Input vector.

        Returns:
        - Euclidean norm of v.
        """
        return float(np.linalg.norm(v))

    def _initialise_state(self, measurement_dict: Dict, timestamp: float) -> State:
        """
        Creates the initial state estimate (x0) and its covariance (p0).

        Arguments:
        - measurement_dict: The measurement dictionary for one observation.
        - timestamp: The measurement timestamp in seconds

        Returns: A state dataclass.
        """
        # Extract observer position and convert to a 3x1 column vector so
        # that NIS is a scalar at the end of the matrix calculations.
        obs = measurement_dict["observer_state_eci"]
        r_obs: np.ndarray = np.array(obs["r_m"], dtype=float).reshape(3, 1)

        # Build line-of-sight unit vector u
        # If a LOS vector in ECI coords is provided, normalise it
        if "los_eci" in measurement_dict:
            u = np.array(measurement_dict["los_eci"], dtype=float).reshape(3, 1)
            u = u / max(1e-12, self._norm(u))

        # If only azimuth/elevation is provided, compute an approx LOS vector
        elif "az_el_rad" in measurement_dict:
            az, el = [float(x) for x in measurement_dict["az_el_rad"]]
            # Local-ENU LOS in observer frame would require a frame map; use a simple placeholder:
            # assume azimuth measured from x-y plane and
            # elevation from horizon; form ECI-like LOS (approx).
            # Convert spherical to Cartesian
            # (This is a simplification; proper conversion would need observer's lat/lon)
            u = np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ], dtype=float).reshape(3, 1)

        # If only RA/Dec (astronomical coordinates) are given, convert to a Cartesian unit vector
        else:
            ra, dec = [float(x) for x in measurement_dict["ra_dec_rad"]]
            u = np.array([
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ], dtype=float).reshape(3, 1)

        # Compute target position. rho = measured range from observer to target
        rho = float(measurement_dict.get("range_m", 0.0))
        r_tgt = r_obs + rho * u

        # Build initial state vector
        x0 = np.zeros((6, 1))

        # Set the first three rows to the estimated target position
        x0[0:3, :] = r_tgt

        # Start with unknown velocity
        # TODO - find paper The initial covariance values (1e6 m² for position,
        # 1e4 (m/s)² for velocity) are chosen to reflect high initial uncertainty,
        # similar to values used in practical orbit determination literature
        # (see e.g. Vallado, "Fundamentals of Astrodynamics and Applications", 4th Ed.).
        p0 = np.diag([1e6, 1e6, 1e6, 1e4, 1e4, 1e4]).astype(float)
        return State(state_estimate=x0, covariance=p0, last_update_seconds=timestamp)


    def _predict(self, x: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate state and covariance forward by dt using orbital dynamics
        and STM/DSTT.

        Args:
        - x: Current state vector (6x1).
        - p: Current covariance matrix (6x6).
        - dt: Time step (s).

        Returns:
        - Tuple of predicted state (6x1) and covariance (6x6).
        """
        # No prediction needed for zero time step
        if dt <= 0.0:
            return x, p

        # Augmented state vector
        # x (6), STM (6x6), DSTT (6 x dim x dim)
        r_matrix = np.eye(self.dimensions)

        x_aug = np.concatenate([
            x.flatten(),                              # initial state
            np.eye(6).reshape(-1),                    # STM initialised to I6
            np.zeros((6 * (self.dimensions ** 2)))    # DSTT initialised to 0
        ])

        # Propagate dynamics + STM + DSTT
        sol = solve_ivp( # type: ignore[call-overload]
            crtbp_dstt_dynamics,
            [0, dt],                                    # integrate over [0, dt]
            x_aug,                                      # initial condition
            args=(self.mu, r_matrix, self.dimensions),
            method="RK45", max_step=np.inf, rtol=1e-12, atol=1e-12
        )

        final = sol.y[:, -1] # State at final time (last column)

        # Extract results
        x_pred = final[0:6].reshape(6, 1)                     # propagated state
        stm = final[6:42].reshape(6, 6)                       # 6x6 state transition matrix
        dstt = final[42:].reshape(6,                          # 6 x dim x dim tensor
                                  self.dimensions,
                                  self.dimensions)

        # Propagate covariance
        mf, pf = dstt_pred_mu_p(p, stm, dstt, r_matrix, self.dimensions)
        p_pred = pf + self.q  # add process noise
        x_pred = x_pred + mf.reshape(6, 1)

        return x_pred, p_pred

    def _build_measurement_model(self, measurement_dict: Dict,
                                 x: np.ndarray) -> Tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:
        """
        Build a measurement model for a given observation.

        Arguments:
        - measurement_dict: Measurement dictionary.
        - x: predicted state vector.

        Returns:
        A tuple of:
        - h: Measurement Jacobian
        - z: Actual measurement vector.
        - z_hat: Predicted measurement vector.
        - r: Measurement covariance.
        """
        r_target = x[0:3, :]

        # Default observer at origin if not provided
        # (OK for innovation scoring)

        if "observer_state_eci" in measurement_dict and \
           "r_m" in measurement_dict["observer_state_eci"]:
            obs = measurement_dict["observer_state_eci"]
            r_obs = np.array(obs["r_m"], dtype=float).reshape(3, 1)
        else:
            r_obs = np.zeros((3, 1))

        delta_r = r_target - r_obs
        r = self._ensure_covariance(measurement_dict.get("R_meas"))

        if "range_m" in measurement_dict and "az_el_rad" not in measurement_dict \
            and "ra_dec_rad" not in measurement_dict:
            return self._build_range_model(delta_r, measurement_dict, r)
        if "az_el_rad" in measurement_dict:
            return self._build_azel_model(delta_r, measurement_dict, r)
        return self._build_radec_model(delta_r, measurement_dict, r)


    def _build_range_model(self, delta_r: np.ndarray,
                           measurement_dict: Dict, r: Optional[np.ndarray]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle pure range measurement.

        Args:
        - delta_r: Relative position vector (target - observer).
        - measurement_dict: Measurement dictionary.
        - r: Optional measurement covariance matrix.

        Returns a tuple of:
        - h: Measurement Jacobian.
        - z: Actual measurement vector.
        - z_hat: Predicted measurement vector.
        - r: Measurement covariance.
        """
        # rho_hat is the predicted range
        rho_hat = np.linalg.norm(delta_r)
        z = np.array([[float(measurement_dict["range_m"])]], dtype=float)
        z_hat = np.array([[rho_hat]], dtype=float)

        # u is the line-of-sight unit vector from observer to target
        if rho_hat < 1e-6:
            u = np.zeros((3, 1))
        else:
            u = delta_r / rho_hat

        # Jacobian wrt position
        h = np.hstack([u.T, np.zeros((1, 3))])

        if r is None:
            r = np.array([[max(self.meas_floor, 25.0)]], dtype=float)  # (5 m)^2 as floor

        return h, z, z_hat, r


    def _build_azel_model(self, delta_r: np.ndarray, # pylint: disable=too-many-locals
                          measurement_dict: Dict, r: Optional[np.ndarray]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle azimuth/elevation measurement.

        Args:
        - delta_r: Relative position vector (target - observer).
        - measurement_dict: Measurement dictionary.
        - r: Optional measurement covariance matrix.

        Returns a tuple of:
        - h: Measurement Jacobian.
        - z: Actual measurement vector.
        - z_hat: Predicted measurement vector.
        - r: Measurement covariance.
        """
        # Extract azimuth and elevation from measurement dictionary
        az, el = [float(x) for x in measurement_dict["az_el_rad"]]
        # Actual measurement vector
        z = np.array([[az], [el]], dtype=float)

        # Predicted measurement vector
        x_, y_, z_ = delta_r.flatten()
        # Compute ranges
        r_xy = np.hypot(x_, y_)
        r_tot = np.linalg.norm(delta_r)

        # Predicted azimuth and elevation
        az_hat = np.arctan2(y_, x_)
        el_hat = np.arctan2(z_, r_xy)

        # Predicted measurement vector
        z_hat = np.array([[az_hat], [el_hat]], dtype=float)

        if r is None:
            r = np.diag([max(self.meas_floor, (1.7e-3)**2),
                        max(self.meas_floor, (1.7e-3)**2)]).astype(float)

        # Jacobian wrt position
        # eps is a small value added for numerical stability to prevent
        # division by zero in Jacobian calculations.
        eps = 1e-9
        d_az_dx = -y_ / (r_xy**2 + eps)
        d_az_dy = x_ / (r_xy**2 + eps)
        d_az_dz = 0.0
        d_el_dx = -(x_ * z_) / ((r_tot**2) * (r_xy + eps)) if r_tot > eps else 0.0
        d_el_dy = -(y_ * z_) / ((r_tot**2) * (r_xy + eps)) if r_tot > eps else 0.0
        d_el_dz = r_xy / (r_tot**2 + eps)

        h_pos = np.array([[d_az_dx, d_az_dy, d_az_dz],
                        [d_el_dx, d_el_dy, d_el_dz]], dtype=float)
        h = np.hstack([h_pos, np.zeros((2, 3))])

        return h, z, z_hat, r


    def _build_radec_model(self, delta_r: np.ndarray, # pylint: disable=too-many-locals
                           measurement_dict: Dict, r: Optional[np.ndarray]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle right ascension/declination measurement.


        Args:
        - delta_r: Relative position vector (target - observer).
        - measurement_dict: Measurement dictionary.
        - r: Optional measurement covariance matrix.

        Returns a tuple of:
        - h: Measurement Jacobian.
        - z: Actual measurement vector.
        - z_hat: Predicted measurement vector.
        - r: Measurement covariance.
        """

        # Extract right ascension and declination from measurement dictionary
        ra, dec = [float(x) for x in measurement_dict.get("ra_dec_rad", [0.0, 0.0])]

        # Actual measurement vector
        z = np.array([[ra], [dec]], dtype=float)

        # Predicted measurement vector
        x_, y_, z_ = delta_r.flatten()
        rho = float(np.linalg.norm(delta_r))

        # Predicted right ascension and declination
        ra_hat = np.arctan2(y_, x_)
        dec_hat = np.arcsin(z_ / max(rho, 1e-9))
        z_hat = np.array([[ra_hat], [dec_hat]], dtype=float)

        # Jacobian wrt position
        # eps is a small value added for numerical stability to prevent
        # division by zero in Jacobian calculations.
        eps = 1e-9
        d_ra_dx = -y_ / (x_**2 + y_**2 + eps)
        d_ra_dy = x_ / (x_**2 + y_**2 + eps)
        d_ra_dz = 0.0
        d_dec_dx = -x_ * z_ / ((rho**2) * np.sqrt(1.0 - (z_ / max(eps, rho))**2) + eps) \
            if rho > eps else 0.0
        d_dec_dy = -y_ * z_ / ((rho**2) * np.sqrt(1.0 - (z_ / max(eps, rho))**2) + eps) \
            if rho > eps else 0.0
        d_dec_dz = np.sqrt(1.0 - (z_ / max(eps, rho))**2) / (rho + eps)

        h_pos = np.array([[d_ra_dx, d_ra_dy, d_ra_dz],
                        [d_dec_dx, d_dec_dy, d_dec_dz]], dtype=float)
        h = np.hstack([h_pos, np.zeros((2, 3))])

        if r is None:
            r = np.diag([max(self.meas_floor, (1.7e-3)**2),
                        max(self.meas_floor, (1.7e-3)**2)]).astype(float)

        return h, z, z_hat, r

    def _ensure_covariance(self, r_meas: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Ensure measurement covariance is valid, applying a floor if needed.

        Arguments:
        - r_meas: Input measurement covariance.

        Returns:
        - Validated measurement covariance.
        """
        if r_meas is None:
            return None

        r_meas = np.array(r_meas, dtype=float)

        # Symmetrisation step
        r_meas = 0.5 * (r_meas + r_meas.T)

        # Regularise tiny/negative eigenvalues
        # w = eigenvalues, v = eigenvectors
        w, v = np.linalg.eigh(r_meas)

        # Prevent there being eigenvalues smaller than the measurement floor
        w = np.clip(w, self.meas_floor, None)
        return v @ np.diag(w) @ v.T
