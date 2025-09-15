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
import numpy as np
from scipy.stats import chi2
from .dag import DAG
from .od_filter import ODProcessingResult
from .satellite_node import SatelliteNode
from .transaction import Transaction
from .utils import build_earth_satellite_list_from_str

R_EARTH = 6371e3  # metres

class ConsensusMechanism():
    """
    The Proof of Inter-Satellite Evaluation (PoISE) consensus mechanism.

    TODO - need to check: if physically possible, how many times its been seen (maybe new param?)
    """
    # TODO - proof of Location XYO paper, and bidirectional heuristics (need to check for invalid,
    # or valid but incorrect/inaccurate)(4.3, proof of proximity)
    # TODO - format for data? class for verification/consensus? Need some calcs
    # TODO TODAY - SLAM DRONE PAPER, and consensus on position, doppler shift?? what data so I
    # have to choose from?
    # DAGmap - map consensus could be something to build upon? does it have a ground truth?
    # PowerGraph - consensus for trust level, calculates validity of transaction from probability
    # level. I guess I need to calculate validity from some maths? possible - if yes, then how
    # accurate/likely probability distruibution for observations? Algorithm one in PowerGraph paper

    def __init__(self) -> None:
        self.consensus_threshold : float = 0.6 # TODO - tune

    def data_is_valid(obs: dict) -> bool:
        """
        Validate observation data for physical and logical consistency.
        Works with a CRTBP-based model.

        Args:
        - obs: Observation data dictionary.

        Returns:
        - True if data is valid, False otherwise.
        """

        # Check observer position
        r_obs = np.array(obs.get("observer_state_eci", {}).get("r_m", []), dtype=float)
        if r_obs.shape != (3,) or not np.isfinite(r_obs).all():
            return False

        # Ensure at least one measurement exists
        has_measurement = any(key in obs for key in ["range_m", "az_el_rad", "ra_dec_rad", "los_eci"])
        if not has_measurement:
            return False

        # Check range
        if "range_m" in obs:
            rng = obs["range_m"]
            if not (isinstance(rng, (int, float)) and np.isfinite(rng)):
                return False
            if rng < 1e5 or rng > 5e7:  # 100 km to 50,000 km
                return False

        # Check LOS vector
        if "los_eci" in obs:
            los = np.array(obs["los_eci"], dtype=float)
            if los.shape != (3,) or not np.isfinite(los).all():
                return False
            norm = np.linalg.norm(los)
            if not np.isclose(norm, 1.0, atol=1e-3):
                return False
            if "range_m" in obs:  # can form satellite position
                sat_pos = r_obs + rng * los

        # Check az/el
        if "az_el_rad" in obs:
            az, el = obs["az_el_rad"]
            if not (0 <= az < 2*np.pi and -np.pi/2 <= el <= np.pi/2):
                return False

        # Check RA/Dec
        if "ra_dec_rad" in obs:
            ra, dec = obs["ra_dec_rad"]
            if not (0 <= ra < 2*np.pi and -np.pi/2 <= dec <= np.pi/2):
                return False

        # Check covariance matrix
        if "R_meas" in obs:
            R = np.array(obs["R_meas"], dtype=float)
            if R.ndim != 2 or R.shape[0] != R.shape[1]:
                return False
            # Must be symmetric and positive semidefinite
            if not np.allclose(R, R.T, atol=1e-8):
                return False
            eigvals = np.linalg.eigvalsh(R)
            if np.any(eigvals < -1e-12):
                return False

        # LEO altitude bound check (200 to 2000km)
        # only possible with range + LOS
        if sat_pos is not None:
            h = np.linalg.norm(sat_pos) - R_EARTH
            if not (200e3 <= h <= 2000e3):
                return False

        return True

    def nis_to_score(nis: float, dof: int) -> float:
        """
        Convert NIS into a normalised [0,1] correctness score.
        1 = high agreement, perfect fit
        0 = low agreement, extreme outlier

        Args:
        - nis: Normalised Innovation Squared value (>=0).
        - dof: Degrees of freedom of the measurement.

        Returns:
        - Correctness score in [0,1].
        """
        # Ensure valid inputs
        nis = max(0.0, float(nis))
        dof = max(1, int(dof))

        # Compute CDF (probability measurement is <= observed NIS)
        cdf = chi2.cdf(nis, df=dof)

        # Flip it so high NIS → low score, low NIS → high score
        score = 1.0 - cdf
        return float(score)

    def get_correctness_score(self, dag: DAG, result: ODProcessingResult) -> float:
        """
        Determine the correctness score for transactional data being added to the DAG.
        This is done by comparing the data provided by a satellite with historical data in the DAG.
        TODO - change
        Returns a correctness score between 0 (inconsistent) and 1 (high agreement)

        """

        target_id = result.target_id
        nis = result.nis
        dof = result.dof
        matches = []

        # If we've never seen this target_id before -> neutral score
        for tx_list in dag.ledger.values():
            for tx in tx_list:
                try:
                    past_data = json.loads(tx.tx_data)
                    if past_data.get("target_id") == target_id:
                        matches.append(past_data)
                except (json.JSONDecodeError, TypeError):
                    continue

        if not matches:
            # Neutral value is the expected NIS for the given dof
            # Expected[NIS] = dof, so neutral score = nis_to_score(dof, dof)
            return self.nis_to_score(dof, dof)

        # Otherwise calculate correctness based on this measurement
        return self.nis_to_score(nis, dof)


    def estimate_accuracy(self, sat: EarthSatellite) -> float:
        """
        Rough placeholder accuracy metric.
        TODO: Replace with actual noise/uncertainty model.
        """
        # Use a dummy normal distribution around ideal speed 7.8 km/s
        ideal_speed = 7.8
        velocity = sat.at(sat.epoch).velocity.km_per_s
        speed = np.linalg.norm(velocity)

        deviation = abs(speed - ideal_speed)

        if deviation < 0.3:
            return 1.0
        if deviation < 0.6:
            return 0.7
        if deviation < 1.0:
            return 0.5
        return 0.2

    def calculate_consensus_score(self, correctness: float,
                                  accuracy: float, reputation: float) -> float:
        """
        Weighted score. Tuneable weights.
        TODO - tune
        """
        return 0.7 * correctness + 0.3 * accuracy + 0.2 * reputation


    def proof_of_inter_satellite_evaluation(self, dag: DAG,
                                            sat_node: SatelliteNode,
                                            transaction: Transaction) -> bool:
        """
        Returns a bool of it consensus has been reached
        NOTE: Assume one witnessed satellite per transaction
        """
        # 1)  Turn transaction data into an EarthSatellite object
        # for validation using wsg4 and skyfield
        sat = build_earth_satellite_list_from_str(load.timescale(),
                                                  transaction.tx_data,
                                                  sat_node.is_malicious)[0]

        # 2) If the list is empty, there is no data that can be valid
        if not transaction.tx_data:
            # Reduce node reputation for providing no or invalid data
            sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_negative(
                sat_node.reputation, sat_node.exp_pos
                )
            return False

        # 2a) If we have data, submit a transaction
        dag.add_tx(transaction)

        # 3) TODO Check we have enough data to be bft (3f + 1)
        # if not, consensus cannot be reached.
        # Length of DAG is 2 by default with genesis transactions. These are dummy data,
        # so we must have at least 4 real transactions on top to be BFT (2 + 4 =  total)
        if not dag.has_bft_quorum():
            return False

        # If we have valid data, try to reach consensus on it
        if self.data_is_valid(sat):
            # 4) TODO Check if satellite has been witnessed before
            #4a if yes, does this data agree with other data/ is it correct?
            # Assign correctness score -> affects transaction
            #  This is going to be very tricky. How do I get this data?? Where do I
            # store it? Do I want this to tie in to how parents are selected?
            # TODO - 0.5 setting is from a recent reference - find and add here
            correctness_score = self.get_correctness_score(sat, dag)

            # 5) TODO is sensor data accurate (done regardless of previous witnessing).
            # Assign accuracy score -> affects transaction and node reputation
            # Again, might be tricky. Probability distribution here? Like in the PowerGraph paper?
            accuracy_score = self.estimate_accuracy(sat)

            # 6) TODO calculate consensus score - node reputation,
            # accuracy and correctness all factor
            # Need to add in a time decay factor
            # Need to develop an equation - this will take some reading and tuning

            consensus_score = self.calculate_consensus_score(correctness_score,
                                                            accuracy_score,
                                                            sat_node.reputation)

            sat_node.reputation = sat_node.rep_manager.decay(sat_node.reputation)

            # 7) if consensus reached - strong node (maybe affects node reputation?),
            # else weak node (like IOTA)
            if consensus_score >= self.consensus_threshold:
                transaction.metadata.consensus_reached = True
                sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_positive(
                    sat_node.reputation, sat_node.exp_pos
                )
                transaction.metadata.is_confirmed = True
                return True

        # If data is invalid, or consensus score is below threshold
        # the transaction is rejected and the node's reputation is penalised.
        transaction.metadata.consensus_reached = False
        sat_node.reputation, sat_node.exp_pos = sat_node.rep_manager.apply_negative(
            sat_node.reputation, sat_node.exp_pos
        )
        transaction.metadata.is_rejected = True
        return False


        # if all history is strong - strong edge connection, else weak edge connection.
        # I guess I can't update edge connections or tx strength retroactively so..
        # this bit is the same as tangle, so i may just try and run this consensus
        # on a tangle like system or simulator like in references

        # TODO - steps 4-6 are the key bits, work here. Get equations and find some
        # tuning evidence. Need to transform into a DAG from a blockchain after that

        # Reputation is tracked and updated throughout
        # Transaction score is fixed once submitted
