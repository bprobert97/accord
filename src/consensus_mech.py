# pylint: disable=too-many-return-statements too-many-branches
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

import json
import math
from typing import Optional
import numpy as np
from scipy.stats import chi2
from .dag import DAG
from.filter import ObservationRecord
from .logger import get_logger
from .reputation import MAX_REPUTATION
from .satellite_node import SatelliteNode
from .transaction import Transaction

R_EARTH = 6371e3  # metres

logger = get_logger()

class ConsensusMechanism():
    """
    The Proof of Inter-Satellite Evaluation (PoISE) consensus mechanism.
    """
    def __init__(self) -> None:
        self.consensus_threshold: float = 0.6
        # Define a simple mapping: normalize by a maximum useful DOF
        # Theoretically, this could be up to 6 (full 3D position+velocity), but
        # in practice, most measurements will have fewer DOF - maximum of 3.
        self.max_dof: int = 3

    def data_is_valid(self, obs: dict) -> bool:
        """
        Validate observation data for physical and logical consistency.
        Works with a CRTBP-based model.

        Args:
        - obs: Observation data dictionary.

        Returns:
        - True if data is valid, False otherwise.
        """

        sat_pos = None

        # Check observer position
        r_obs = np.array(obs.get("observer_state_eci", {}).get("r_m", []), dtype=float)
        if r_obs.shape != (3,) or not np.isfinite(r_obs).all():
            logger.info("Observer position invalid or missing.")
            return False

        # Ensure at least one measurement exists
        has_measurement = any(key in obs for key in ["range_m",
                                                     "az_el_rad",
                                                     "ra_dec_rad",
                                                     "los_eci"])
        if not has_measurement:
            logger.info("No valid measurement data present.")
            return False

        # Check range
        if "range_m" in obs:
            rng = obs["range_m"]
            if not (isinstance(rng, (int, float)) and np.isfinite(rng)):
                logger.info("Range measurement invalid.")
                return False
            if rng < 1e5 or rng > 5e7:  # 100 km to 50,000 km
                logger.info("Range measurement out of bounds: %.1f m", rng)
                return False

        # Check LOS vector
        if "los_eci" in obs:
            los = np.array(obs["los_eci"], dtype=float)
            if los.shape != (3,) or not np.isfinite(los).all():
                logger.info("LOS vector invalid.")
                return False
            norm = np.linalg.norm(los)
            if not np.isclose(norm, 1.0, atol=1e-3):
                logger.info("LOS vector not unit length: norm=%.6f", norm)
                return False
            if "range_m" in obs:  # can form satellite position
                sat_pos = r_obs + rng * los

        # Check az/el
        if "az_el_rad" in obs:
            az, el = obs["az_el_rad"]
            if not (0 <= az < 2*np.pi and -np.pi/2 <= el <= np.pi/2):
                logger.info("Azimuth/Elevation out of bounds: az=%.3f, el=%.3f", az, el)
                return False

        # Check RA/Dec
        if "ra_dec_rad" in obs:
            ra, dec = obs["ra_dec_rad"]
            if not (0 <= ra < 2*np.pi and -np.pi/2 <= dec <= np.pi/2):
                logger.info("RA/Dec out of bounds: ra=%.3f, dec=%.3f", ra, dec)
                return False

        # Check covariance matrix
        if "R_meas" in obs:
            r = np.array(obs["R_meas"], dtype=float)
            # Account for issues converting from MATLAB to json to Python
            if r.ndim == 0:  # scalar
                r = np.array([[float(r)]])
            elif r.ndim == 1:  # vector
                r = np.diag(r)

            if r.ndim != 2 or r.shape[0] != r.shape[1]:
                logger.info("Covariance matrix R_meas invalid shape.")
                return False
            # Must be symmetric and positive semidefinite
            if not np.allclose(r, r.T, atol=1e-8):
                logger.info("Covariance matrix R_meas not symmetric.")
                return False
            eigvals = np.linalg.eigvalsh(r)
            if np.any(eigvals < -1e-12):
                logger.info("Covariance matrix R_meas not positive semidefinite.")
                return False

        # LEO altitude bound check (200 to 2000km)
        # only possible with range + LOS
        if sat_pos is not None:
            h = np.linalg.norm(sat_pos) - R_EARTH
            if not 200e3 <= h <= 2000e3:
                logger.info("Inferred satellite altitude out of LEO bounds: %.1f m", h)
                return False

        return True

    def nis_to_score(self, nis: float, dof: int, historical_nis: Optional[float] = None) -> float:
        """
        Convert NIS into a normalised [0,1] correctness score.
        The score rewards NIS values that are close to the expected mean (DOF),
        promoting long-term statistical consistency. Values that are too high or
        too low are penalised.

        Args:
        - nis: Normalised Innovation Squared value (>=0).
        - dof: Degrees of freedom of the measurement.
        - historical_nis: Optional historical mean NIS for the satellite.

        Returns:
        - Correctness score in [0,1].
        """
        # Ensure valid inputs
        nis = max(0.0, float(nis))
        dof = max(1, int(dof))

        # Smooth the current NIS with the historical mean if available
        if historical_nis is not None:
            nis = (nis + historical_nis) / 2.0
            logger.info("BCP123 Updated NIS %s", nis)

        # The score is based on the chi-squared distribution. The expected value
        # (mean) of the NIS is equal to the degrees of freedom (DOF), and the
        # variance is 2 * DOF.
        # This scoring function is a Gaussian centered on the mean (dof), which
        # rewards NIS values close to the expected value and penalises values
        # that are too high or too low ("too good to be true").
        variance = 2 * dof
        score = math.exp(-((nis - dof) ** 2) / (2 * variance))

        return float(score)

    def get_correctness_score(self, dag: DAG, obs_record: ObservationRecord, mean_nis_per_satellite: dict[int, float]) -> float:
        """
        Calculate correctness score based on NIS and past observations of the same target.
        Args:
        - dag: The current DAG containing past transactions.
        - obs_record: The observation record for the current measurement.
        - mean_nis_per_satellite: A dictionary mapping satellite ID to its mean NIS.

        Returns:
        - Correctness score in [0,1]. 0 = low agreement, 1 = high agreement.
        """

        target_id = obs_record.target
        nis = obs_record.nis
        dof = obs_record.dof
        matches = []

        # If we've never seen this target_id before -> neutral score
        for tx_list in dag.ledger.values():
            for tx in tx_list:
                if tx.tx_data.startswith("Genesis"):
                    continue
                try:
                    past_data = json.loads(tx.tx_data)
                    if past_data.get("target") == target_id:
                        matches.append(past_data)
                except (json.JSONDecodeError, TypeError):
                    continue

        # If this satellite has not been seen before, give it a neutral score
        # Also, if the filter has been initialised for the first time, return a
        # neutral score
        if not matches or math.isnan(nis):
            # Neutral value is the expected NIS for the given dof
            # Expected[NIS] = dof, so neutral score = nis_to_score(dof, dof)
            return self.nis_to_score(float(dof), dof)

        # Get historical mean NIS for the observer satellite
        observer_id = obs_record.observer
        historical_nis = mean_nis_per_satellite.get(observer_id)

        return self.nis_to_score(nis, dof, historical_nis)


    def calculate_dof_score(self, dof: int) -> float:
        """
        Estimate a relative accuracy/reward score based on measurement DOF.
        Higher DOF -> higher score (since it reduces OD computational effort).
        Returns a value in [0,1].

        Args:
        - dof: Degrees of freedom of the measurement.

        Returns:
        - DOF score in [0,1]. 0 = low accuracy, 1 = high accuracy.
        """

        # Normalise DOF
        # dof = 1 returns 0.33, dof = 2 returns 0.66, dof of 3 returns 1.0.
        return min(1.0, dof / self.max_dof)

    def calculate_consensus_score(self, correctness: float,
                                  dof_reward: float, reputation: float,
                                  gamma: float = 0.35,
                                  alpha: float = 0.8) -> float:
        """
        Calculate overall consensus score from correctness, DOF reward, and node reputation.
        Weights can be adjusted to tune the influence of each factor.

        Args:
        - correctness: Correctness score in [0,1].
        - dof_reward: DOF-based reward score in [0,1].
        - reputation: Node reputation in [0, 1].

        Returns:
        - Consensus score in [0,1]. Higher is better.
        """
        # Normalise reputation
        rep_norm = min(max(reputation / MAX_REPUTATION, 0.0), 1.0)

        # Normalise each variable relative to its threshold point
        # Min acceptable correctness for consensus = 0.5
        # Min DOF score = 0.33
        # Min reputation = 0
        # TODO - look at more tomorrow. Updates:
        # Need to check that 0.5 correctness is right? May need to switch to PDF from CDF?
        # Because 0.4 correctness might actually be okay? shee chi2 plots

        # Relative to baselines (can go negative for correctness)
        c_rel = max(min((correctness - 0.5) / 0.5, 1.0), -1.0)      # [-1,1]
        d_rel = max(min((dof_reward - (1/3)) / (2/3), 1.0), 0.0)      # [0,1] with baseline at 0
        r_rel = rep_norm                                            # [0,1] with baseline at 0

        # Nonlinear emphasis on correctness (continuous across 0.5)
        # gamma < 1 makes correctness more influential above 0.5 and penalizes below 0.5
        c_scale = ( (abs(c_rel) ** gamma) * (1 if c_rel >= 0 else -1) + 1 ) / 2

        # Cooperative DOF–reputation term (no weights, monotonic, bounded)
        dr_term = (1 - (1 - d_rel) * (1 - r_rel)) ** alpha

        # Combine and calibrate to the threshold anchor
        combined = c_scale * dr_term

        consensus = self.consensus_threshold + (combined - 0.5) * \
        2 * (1 - self.consensus_threshold)

        logger.info("[FOR PLOT] correctness: %.6f, reputation: %.6f, dof_norm: %.6f, \
                    consensus score: %.6f", correctness, reputation, dof_reward, consensus)

        return min(max(consensus, 0.0), 1.0)

        # c = max(0.0, min(1.0, correctness))
        # r = max(0.0, min(1.0, rep_norm))
        # t = self.consensus_threshold

        # # Coefficients solving the four anchor equations
        # b  = t
        # c2 = 3*t - 1
        # d  = 2 - 4*t

        # s = b*c + c2*r + d*c*r  # bilinear surface S(c, r)

        # return max(0.0, min(1.0, s))  # clamp

    def proof_of_inter_satellite_evaluation(self, dag: DAG,
                                            sat_node: SatelliteNode,
                                            transaction: Transaction,
                                            mean_nis_per_satellite: dict[int, float]) -> bool:
        """
        Returns a bool of it consensus has been reached
        NOTE: Assume one witnessed satellite per transaction
        """

        # 2) If the list is empty, there is no data that can be valid
        if not transaction.tx_data:
            # Reduce node reputation for providing no or invalid data
            sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_negative(
                sat_node.reputation, sat_node.exp_pos
                )
            return False

        # 1) Convert tx_data to dict
        transaction_data: dict = json.loads(transaction.tx_data)
        obs_record = ObservationRecord(**transaction_data)

        # 2a) If we have data, submit a transaction
        dag.add_tx(transaction)

        # 3) Check we have enough data to be bft (3f + 1)
        # If not, consensus cannot be reached.
        # Length of DAG is 2 by default with genesis transactions. These are dummy data,
        # so we must have at least 4 real transactions on top to be BFT (2 + 4 =  total)
        if not dag.has_bft_quorum():
            logger.info("Not enough transactions for BFT quorum.")
            logger.info("Satellite reputation unchanged at %.2f", sat_node.reputation)
            return False

        # If we have valid data, try to reach consensus on it
        # The data validation part is removed as the observation record is assumed to be valid

        # 4) Check if satellite has been witnessed before
        #4a if yes, does this data agree with other data/ is it correct?
        correctness_score = self.get_correctness_score(dag, obs_record, mean_nis_per_satellite)

        # 5) Reward measurements with higher DOF (more accurate, reduced comp. intensity)
        dof_score = self.calculate_dof_score(obs_record.dof)

        # 6) Calculate consensus score
        consensus_score = self.calculate_consensus_score(correctness_score,
                                                         dof_score,
                                                         sat_node.reputation)

        # Store scores in metadata for later analysis
        transaction.metadata.consensus_score = consensus_score
        transaction.metadata.nis = obs_record.nis
        transaction.metadata.dof = obs_record.dof

        logger.info("NIS=%.3f, DOF=%d, correctness=%.3f, consensus_score=%.3f, \
                    reputation=%.3f",
        obs_record.nis, obs_record.dof,
        correctness_score, consensus_score, sat_node.reputation)

        sat_node.reputation = sat_node.rep_manager.decay(sat_node.reputation)
        logger.info("Satellite reputation decayed to %.3f.",
                    sat_node.reputation)

        # 7) if consensus reached - strong node (maybe affects node reputation?),
        # else weak node (like IOTA)
        if consensus_score >= self.consensus_threshold:
            transaction.metadata.consensus_reached = True
            sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_positive(
                sat_node.reputation, sat_node.exp_pos
            )
            transaction.metadata.is_confirmed = True
            logger.info("Satellite reputation increased to %.2f", sat_node.reputation)
            logger.info("Successful consensus score: %.2f", consensus_score)
            return True

        logger.info("Consensus threshold of %.2f does not met threshold.",
                    consensus_score)

        # If data is invalid, or consensus score is below threshold
        # the transaction is rejected and the node's reputation is penalised.
        transaction.metadata.consensus_reached = False
        sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_negative(
            sat_node.reputation, sat_node.exp_pos, correctness_score
        )
        transaction.metadata.is_rejected = True
        logger.info("Satellite reputation decreased to %.2f",
                    sat_node.reputation)
        # If data is invalid, or consensus sco
        return False
