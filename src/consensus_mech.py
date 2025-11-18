# pylint: disable=too-many-return-statements too-many-branches too-many-arguments too-many-positional-arguments
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
from .dag import DAG
from.filter import ObservationRecord
from .logger import get_logger
from .reputation import MAX_REPUTATION
from .satellite_node import SatelliteNode
from .transaction import Transaction

logger = get_logger()

class ConsensusMechanism():
    """
    The Proof of Inter-Satellite Evaluation (PoISE) consensus mechanism.
    """
    def __init__(self) -> None:
        self.consensus_threshold: float = 0.6
        self.ema_alpha: float = 0.1  # Smoothing factor for EMA
        # Define a simple mapping: normalise by a maximum useful DOF
        # Theoretically, this could be up to 6 (full 3D position+velocity), but
        # in practice, most measurements will have fewer DOF - maximum of 3.
        self.max_dof: int = 3

    def nis_to_score(self, nis: float, dof: int,
                     historical_ema_nis: Optional[float] = None) -> float:
        """
        Convert NIS into a normalised [0,1] correctness score, considering historical performance.
        The score is high if the new NIS brings the historical EMA
        closer to the expected value (dof).
        The score is low if the new NIS pulls the EMA further away, or if the NIS is an outlier.

        Args:
        - nis: Normalised Innovation Squared value (>=0).
        - dof: Degrees of freedom of the measurement.
        - historical_ema_nis: The historical Exponential Moving Average of NIS for the satellite.

        Returns:
        - Correctness score in [0,1].
        """
        nis = max(0.0, float(nis))
        dof = max(1, int(dof))
        variance = 2 * dof

        def _calculate_gaussian_score(current_nis: float, current_dof: int) -> float:
            """Calculates a simple score based on a Gaussian function centered on dof."""
            return math.exp(-((current_nis - current_dof) ** 2) / (2 * variance))

        if historical_ema_nis is None:
            # First observation. Score is based on how close the first NIS is to dof.
            return _calculate_gaussian_score(nis, dof)

        # Hard penalty for very large NIS values, indicating an outlier
        # TODO - might be too heavily influencing reputation on NIS at the moment.Need
        # to remember to look at reputation and consensus differently
        if nis > 5 * dof:
            return 0

        # Calculate new EMA
        new_ema_nis = (nis * self.ema_alpha) + (historical_ema_nis * (1 - self.ema_alpha))
        logger.info("BCP999 old %s, new %s", historical_ema_nis, new_ema_nis)

        # Score based on whether the new NIS brings the EMA closer to the expected value (dof)
        dist_before = abs(historical_ema_nis - dof)
        dist_after = abs(new_ema_nis - dof)

        improvement = dist_before - dist_after

        # The base score is determined by how close the current NIS is to the ideal value (dof).
        # This provides an instantaneous measure of the measurement's quality.
        base_score = _calculate_gaussian_score(nis, dof)
        logger.info("BCP002 base score: %.6f", base_score)

        # The improvement factor modulates the score based on historical performance.
        # A positive improvement (moving closer to dof) increases the score.
        # A negative improvement (moving away) decreases it.
        improvement_factor = np.tanh(improvement)
        logger.info("BCP001 improvement factor: %.6f", improvement_factor)

        # Combine instantaneous score with historical improvement.
        # A good current NIS can receive a high score even with a poor history,
        # especially if it shows improvement.
        final_score = base_score * (1 + improvement_factor * 0.5)
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

    def proof_of_inter_satellite_evaluation(self, dag: DAG,
                                            sat_node: SatelliteNode,
                                            transaction: Transaction,
                                            mean_nis_per_satellite: dict[int, float]
                                            ) -> tuple[bool, Optional[float]]:
        """
        Returns a bool of it consensus has been reached, and the new EMA NIS for the satellite.
        NOTE: Assume one witnessed satellite per transaction
        """
        new_ema_nis: Optional[float] = None
        # 1) If the list is empty, there is no data that can be valid
        if not transaction.tx_data:
            # Reduce node reputation for providing no or invalid data
            sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_negative(
                sat_node.reputation, sat_node.exp_pos
                )
            return False, new_ema_nis

        # 2) Convert tx_data to dict
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
            return False, new_ema_nis

        # 4) Check if satellite has been witnessed before
        #4a if yes, does this data agree with other data/ is it correct?
        correctness_score, new_ema_nis = self.get_correctness_score(obs_record,
                                                                    mean_nis_per_satellite)

        # 5) Reward measurements with higher DOF (more accurate, reduced comp. intensity)
        dof_score = self.calculate_dof_score(obs_record.dof)

        # 6) Calculate consensus score
        consensus_score = self.calculate_consensus_score(correctness_score,
                                                         dof_score,
                                                         sat_node.reputation)

        # Store scores in metadata for later analysis
        transaction.metadata.consensus_score = consensus_score
        transaction.metadata.correctness_score = correctness_score
        transaction.metadata.nis = obs_record.nis
        transaction.metadata.dof = obs_record.dof

        logger.info("NIS=%.3f, DOF=%d, correctness=%.3f, consensus_score=%.3f, \
                    reputation=%.3f",
        obs_record.nis, obs_record.dof,
        correctness_score, consensus_score, sat_node.reputation)

        sat_node.reputation = sat_node.rep_manager.decay(sat_node.reputation)
        logger.info("Satellite reputation decayed to %.3f.",
                    sat_node.reputation)

        # 7) Check if consensus reached
        if consensus_score >= self.consensus_threshold:
            transaction.metadata.consensus_reached = True
            sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_positive(
                sat_node.reputation, sat_node.exp_pos
            )
            transaction.metadata.is_confirmed = True
            logger.info("Satellite reputation increased to %.2f", sat_node.reputation)
            logger.info("Successful consensus score: %.2f", consensus_score)
            return True, new_ema_nis

        logger.info("Consensus threshold of %.2f does not met threshold.",
                    consensus_score)

        # If data is invalid, or consensus score is below threshold
        # the transaction is rejected and the node's reputation is penalised.
        transaction.metadata.consensus_reached = False
        sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_negative(
            sat_node.reputation, sat_node.exp_pos
        )
        transaction.metadata.is_rejected = True
        logger.info("Satellite reputation decreased to %.2f",
                    sat_node.reputation)

        return False, new_ema_nis
