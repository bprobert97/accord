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
from typing import List, Optional
import numpy as np
from scipy.stats import chi2
from .dag import DAG
from .filter import ObservationRecord
from .logger import get_logger
from .satellite_node import SatelliteNode
from .transaction import Transaction

logger = get_logger()

class ConsensusMechanism():
    """
    The Proof of Inter-Satellite Evaluation (PoISE) consensus mechanism.
    """
    def __init__(self) -> None:
        self.consensus_threshold: float = 0.5
        self.ema_alpha: float = 0.1  # Smoothing factor for EMA
        # Define a simple mapping: normalise by a maximum useful DOF
        # Theoretically, this could be up to 6 (full 3D position+velocity), but
        # in practice, most measurements will have fewer DOF - maximum of 3.
        self.max_dof: int = 3

    def nis_to_score(self, nis: float, dof: int,
                     historical_ema_nis: Optional[float] = None) -> float:
        """
        Convert NIS into a normalised [0,1] correctness score.
        The base score is calculated based on the two-sided probability of the NIS
        value occurring in a chi-squared distribution (penalising "too perfect"
        and "too high" values). This score is then modulated by the satellite's
        historical performance (the EMA of its NIS values).

        Args:
        - nis: Normalised Innovation Squared value (>=0).
        - dof: Degrees of freedom of the measurement.
        - historical_ema_nis: The historical Exponential Moving Average of NIS for the satellite.

        Returns:
        - Correctness score in [0,1].
        """
        nis = max(0.0, float(nis))
        dof = max(1, int(dof))

        def _calculate_twosided_chi2_score(current_nis: float, current_dof: int) -> float:
            """
            Calculates a score based on how far the NIS is from the median of the
            chi-squared distribution. It penalises values that are too high (in the
            right tail) and too low (in the left tail, i.e., "too perfect").
            """
            cdf_val = chi2.cdf(current_nis, current_dof)
            # The score is 1.0 at the median (cdf=0.5) and drops towards 0.0 at the extremes.
            # Median is about 1.386 for dof=2.
            return 1.0 - abs(cdf_val - 0.5) * 2

        if historical_ema_nis is None:
            # First observation. Score is based on how close the first NIS is to dof.
            return _calculate_twosided_chi2_score(nis, dof)

        # Hard penalty for very large NIS values, indicating an outlier
        if nis > 5 * dof:
            return 0

        # Calculate new EMA
        new_ema_nis = (nis * self.ema_alpha) + (historical_ema_nis * (1 - self.ema_alpha))

        # Score based on whether the new NIS brings the EMA closer to the expected value (dof)
        dist_before = abs(historical_ema_nis - dof)
        dist_after = abs(new_ema_nis - dof)

        improvement = dist_before - dist_after

        # The base score is now a two-sided check on the instantaneous NIS value.
        base_score = _calculate_twosided_chi2_score(nis, dof)

        # The improvement factor modulates the score based on historical performance.
        # A positive improvement (moving closer to dof) increases the score.
        # A negative improvement (moving away) decreases it.
        improvement_factor = np.tanh(improvement)

        # Combine instantaneous score with historical improvement.
        # A good current NIS can receive a high score even with a poor history,
        # especially if it shows improvement.
        final_score = base_score * ((1 + improvement_factor) * 0.5)
        return max(0.0, min(1.0, final_score))

    def get_correctness_score(self, obs_record: ObservationRecord,
                              mean_nis_per_satellite: dict[int, float]) -> tuple[float, float]:
        """
        Calculate correctness score based on NIS and historical performance.
        This function now calculates and returns the new EMA of the NIS.

        Args:
        - obs_record: The observation record for the current measurement.
        - mean_nis_per_satellite: A dictionary mapping satellite ID to its historical EMA NIS.

        Returns:
        - A tuple containing:
            - Correctness score in [0,1]. 0 = low agreement, 1 = high agreement.
            - The new EMA NIS value for the observing satellite.
        """
        nis = obs_record.nis
        dof = obs_record.dof

        historical_ema_nis = mean_nis_per_satellite.get(obs_record.observer)

        score = self.nis_to_score(nis, dof, historical_ema_nis)

        if historical_ema_nis is None:
            new_ema_nis = nis
        else:
            new_ema_nis = (nis * self.ema_alpha) + (historical_ema_nis * (1 - self.ema_alpha))

        return score, new_ema_nis

    def calc_normalised_dot(self, curr: List[float], prev: List[float]) -> float:
        """
        Calculates the normalised dot product between two vectors.

        Args:
        - curr: The current vector.
        - prev: The previous vector.

        Returns:
        - A float representing the normalised dot product, which indicates
          the change in direction of the vector.
        """
        dot_prod = sum(c * p for c, p in zip(curr, prev))
        v1_sq_norm = sum(c * c for c in curr)
        v2_sq_norm = sum(p * p for p in prev)

        if v1_sq_norm == 0.0 or v2_sq_norm == 0.0:
            return 1.0 # Fully redundant data

        return abs(dot_prod) / math.sqrt(v1_sq_norm * v2_sq_norm)

    def calculate_dof_score(self,
                            obs_record: ObservationRecord,
                            previous_data: Optional[dict] = None,
                            decay_rate: float = 0.05,
                            velocity_weight: float = 0.5) -> float:
        """
        Estimate a relative accuracy/reward score based on measurement DOF.
        Higher DOF -> higher score (since it reduces OD computational effort).
        Returns a value in [0,1].

        Args:
        - obs_record: The observation record containing the DOF information.
        - previous_data: A dictionary containing the previous r_vector, v_vector, and
                         timestamp for the same observer-target pair.
        - decay_rate: Rate at which the DOF score decays if the measurement
          direction changes significantly.
        - velocity_weight: Weighting factor for the velocity vector in the blended
          persistence of excitation calculation. Should be in [0,1].

        Returns:
        - DOF score 0 = low DOF/ highly redundant, 1+ = high DOF/novelty.
        - Note: assumed to be bounded in [0,1] where max k is assumed to be 3.
        """

        # Calculate base score of (k-1)/2
        # dof = 1 returns 0, dof = 2 returns 0.5 and dof = 3 returns 1.
        base_score = (obs_record.dof - 1) / 2

        if previous_data is None:
            return base_score

        previous_r_vector = previous_data.get('r_vector')
        previous_v_vector = previous_data.get('v_vector')
        delta_t = obs_record.time - previous_data['time'] if 'time' in previous_data else None

        if previous_r_vector is None or previous_v_vector is None or delta_t is None:
            return base_score

        # Calculate persistence of excitation term
        r_dot = self.calc_normalised_dot(obs_record.r_vector, previous_r_vector)
        v_dot = self.calc_normalised_dot(obs_record.v_vector, previous_v_vector)

        # If velocity_weight is 0.5, a change in either position or velocity
        # lowers the blended_dot, triggering a higher reward.
        blended_dot = ((1.0 - velocity_weight) * r_dot) + (velocity_weight * v_dot)
        time_decay = math.exp(-decay_rate * delta_t)
        pe_multiplier = 1.0 - (time_decay * blended_dot)

        return base_score * pe_multiplier

    def calculate_consensus_score(self, correctness: float,
                                  dof_reward: float, reputation: float,
                                  alpha: float = 0.8) -> float:
        """
        Calculate overall consensus score from correctness, DOF reward, and node reputation.
        Weights can be adjusted to tune the influence of each factor.

        Args:
        - correctness: Correctness score in [0,1].
        - dof_reward: DOF-based reward score in [0,1].
        - reputation: Node reputation in the range [0, 1].
        - alpha: Non-linear scaling factor for the combined DOF and reputation term.

        Returns:
        - Consensus score in [0,1]. Higher is better.
        """
        # Min acceptable correctness for consensus = 0.5
        # Min DOF score = 0
        # Min reputation = 0

        # Cooperative DOF–reputation term (no weights, monotonic, bounded)
        dr_term = (1 - (1 - dof_reward) * (1 - reputation)) ** alpha

        # Combine terms
        consensus = correctness * dr_term

        logger.info("[FOR PLOT] correctness: %.6f, reputation: %.6f, dof_norm: %.6f, \
                    consensus score: %.6f", correctness, reputation, dof_reward, consensus)

        return min(max(consensus, 0.0), 1.0)

    def proof_of_inter_satellite_evaluation(self, dag: DAG,
                                            sat_node: SatelliteNode,
                                            transaction: Transaction,
                                            mean_nis_per_satellite: dict[int, float],
                                            ) -> tuple[bool, Optional[float]]:
        """
        Returns a bool of if consensus has been reached, and the new EMA NIS for
        the satellite.
        NOTE: Assume one witnessed satellite per transaction

        Args:
        - dag: The DAG instance to access current transactions and history.
        - sat_node: The SatelliteNode that submitted the transaction.
        - transaction: The Transaction object containing the observation data.
        - mean_nis_per_satellite: A dictionary mapping satellite ID to its historical
        EMA NIS.

        Returns:
        - A tuple containing:
            - A boolean indicating whether consensus was reached for this transaction.
            - The new EMA NIS value for the observing satellite
              (or None if consensus not reached).
        """
        new_ema_nis: Optional[float] = None

        if not self._is_transaction_valid(transaction, sat_node):
            return False, new_ema_nis

        transaction_data: dict = json.loads(transaction.tx_data)
        obs_record = ObservationRecord(**transaction_data)
        transaction.metadata.observer_id = obs_record.observer

        if not self._is_dof_valid(obs_record, transaction, sat_node):
            return False, new_ema_nis

        dag.add_tx(transaction)
        if not dag.has_bft_quorum():
            logger.info("Not enough transactions for BFT quorum. Satellite \
                        reputation unchanged at %.2f", sat_node.reputation)
            return False, new_ema_nis

        consensus_score, correctness_score, new_ema_nis = self._calculate_poise_scores(
            dag, obs_record, sat_node, mean_nis_per_satellite
        )

        self._update_transaction_metadata(transaction, obs_record, consensus_score,
                                          correctness_score)
        self._update_satellite_reputation(sat_node, obs_record)

        if consensus_score >= self.consensus_threshold:
            transaction.metadata.consensus_reached = True
            transaction.metadata.is_confirmed = True
            logger.info("Successful consensus score: %.2f", consensus_score)
            return True, new_ema_nis

        logger.info("Consensus score of %.2f does not meet threshold.", consensus_score)
        transaction.metadata.consensus_reached = False
        transaction.metadata.is_rejected = True
        return False, new_ema_nis

    def _is_transaction_valid(self, transaction: Transaction, sat_node: SatelliteNode) -> bool:
        """
        Checks if the transaction contains valid data. If not, applies
        a penalty to the satellite's reputation.

        Args:
        - transaction: The Transaction object to validate.
        - sat_node: The SatelliteNode that submitted the transaction, used for applying penalties.

        Returns:
        - A boolean indicating whether the transaction is valid.
        """
        if not transaction.tx_data:
            sat_node.reputation, sat_node.exp_pos, sat_node.performance_ema = \
                sat_node.rep_manager.apply_negative(sat_node.reputation,
                                                    sat_node.exp_pos,
                                                    sat_node.performance_ema)
            return False
        return True

    def _is_dof_valid(self, obs_record: ObservationRecord,
                      transaction: Transaction, sat_node: SatelliteNode) -> bool:
        """
        Checks if the degrees of freedom (DOF) of the observation are within expected bounds.
        If the DOF is invalid, applies a penalty to the satellite's reputation.

        Args:
        - obs_record: The ObservationRecord extracted from the transaction data.
        - transaction: The Transaction object being evaluated, used for updating metadata.
        - sat_node: The SatelliteNode that submitted the transaction, used for applying penalties.

        Returns:
        - A boolean indicating whether the DOF is valid.
        """
        if obs_record.dof > 6 or obs_record.dof < 1:
            logger.info("Invalid DOF of %d in transaction. Penalising reputation.", obs_record.dof)
            sat_node.reputation, sat_node.exp_pos, sat_node.performance_ema = \
                sat_node.rep_manager.apply_negative(sat_node.reputation,
                                                    sat_node.exp_pos,
                                                    sat_node.performance_ema)
            transaction.metadata.consensus_reached = False
            transaction.metadata.is_rejected = True
            return False
        return True

    def _calculate_poise_scores(self,
                                dag: DAG,
                                obs_record: ObservationRecord,
                                sat_node: SatelliteNode,
                                mean_nis_per_satellite: dict):
        """
        Calculates the correctness score, DOF score, and overall consensus score for a
        given observation record.
        Also updates the DAG's vector history cache for DOF scoring.

        Args:
        - dag: The DAG instance to access current transactions and history.
        - obs_record: The ObservationRecord extracted from the transaction data.
        - sat_node: The SatelliteNode that submitted the transaction, used for accessing reputation.
        - mean_nis_per_satellite: A dictionary mapping satellite ID to its historical EMA NIS.

        Returns:
        - A tuple containing:
            - The overall consensus score for the transaction.
            - The correctness score based on NIS and historical performance.
            - The new EMA NIS value for the observing satellite.
        """
        correctness_score, new_ema_nis = self.get_correctness_score(obs_record,
        mean_nis_per_satellite)

        cache_key = (obs_record.observer, obs_record.target)
        prev_data = dag.vector_history_cache.get(cache_key, {})

        dof_score = self.calculate_dof_score(
            obs_record,
            prev_data
        )

        if obs_record.r_vector is not None and obs_record.v_vector is not None:
            dag.vector_history_cache[cache_key] = {
                'r_vector': obs_record.r_vector,
                'v_vector': obs_record.v_vector,
                'time': obs_record.time
            }

        consensus_score = self.calculate_consensus_score(correctness_score, dof_score,
                                                         sat_node.reputation)
        return consensus_score, correctness_score, new_ema_nis

    def _update_transaction_metadata(self, transaction: Transaction, obs_record: ObservationRecord,
                                     consensus_score: float, correctness_score: float) -> None:
        """
        Updates the transaction metadata with the calculated consensus score, correctness score,
        and NIS value.
        Args:
        - transaction: The Transaction object to update.
        - obs_record: The ObservationRecord containing the NIS and DOF values.
        - consensus_score: The overall consensus score calculated for this transaction.
        - correctness_score: The correctness score calculated based on NIS and historical
        performance.

        Returns:
        None. Updates the transaction's metadata in place.
        """
        transaction.metadata.consensus_score = consensus_score
        transaction.metadata.correctness_score = correctness_score
        transaction.metadata.nis = obs_record.nis
        transaction.metadata.dof = obs_record.dof

    def _update_satellite_reputation(self, sat_node: SatelliteNode,
                                     obs_record: ObservationRecord) -> None:
        """
        Updates the satellite's reputation based on the NIS value of the observation
        and its historical performance.
        The reputation is decayed towards neutral on every observation,
        then adjusted up or down based on
        whether the NIS is within the expected bounds of the chi-squared distribution.

        Args:
        - sat_node: The SatelliteNode whose reputation is to be updated.
        - obs_record: The ObservationRecord containing the NIS and DOF values.

        Returns:
        None. Updates the satellite's reputation in place.
        """
        lower_bound = chi2.ppf(0.025, obs_record.dof)
        upper_bound = chi2.ppf(0.975, obs_record.dof)

        sat_node.reputation = sat_node.rep_manager.decay(sat_node.reputation)
        logger.info("Satellite reputation decayed to %.3f.", sat_node.reputation)

        if lower_bound <= obs_record.nis <= upper_bound:
            sat_node.reputation, sat_node.exp_pos, sat_node.performance_ema = \
                sat_node.rep_manager.apply_positive(sat_node.reputation,
                                                    sat_node.exp_pos,
                                                    sat_node.performance_ema)
            logger.info("NIS within bounds. Reputation slowly increased to %.2f",
                        sat_node.reputation)
        else:
            sat_node.reputation, sat_node.exp_pos, sat_node.performance_ema = \
                sat_node.rep_manager.apply_negative(sat_node.reputation,
                                                    sat_node.exp_pos,
                                                    sat_node.performance_ema)
            logger.info("NIS outside bounds. Reputation decreased to %.2f",
                        sat_node.reputation)
