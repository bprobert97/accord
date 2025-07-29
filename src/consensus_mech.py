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
from skyfield.api import EarthSatellite, wgs84
from .dag import DAG
from .satellite_node import SatelliteNode
from .transaction import Transaction
from .utils import load_json_data

class ConsensusMechanism():
    """
    The Proof of Inter-Satellite Evaluation (PoISE) consensus mechanism.
        
    TODO - need to check: if physically possible, how many times its been seen (maybe new param?)
    """
    # TODO - proof of Location XYO paper, and bidirectional heuristics (need to check for invalid, or valid but incorrect/inaccurate)(4.3, proof of proximity)
    # TODO - format for data? class for verification/consensus? Need some calcs
    # TODO TODAY - SLAM DRONE PAPER, and consensus on position, doppler shift?? what data so I have to choose from?
    # DAGmap - map consensus could be something to build upon? does it have a ground truth?
    # PowerGraph - consensus for trust level, calculates validity of transacion from probability level. I guess I need to calculate validity from some maths? possible - if yes, hen how accurate/likely
    # probability distruibution for observations? Algorithm one in PowerGraph paper
    
    def __init__(self):
        self.consensus_threshold : float = 0.6 # TODO - tune
        self.reputation_step: float = 0.1 # TODO - tune
    
    def data_is_valid(self, sat: EarthSatellite) -> bool:
        """
        Check if received data is valid, i.e. physically and logically possible for a LEO satellite
        Assume data comes in a standard TLE format from a Celestrak JSON
        Checks validity to reduce computation effort in consensus for data that is impossible
        Assumes that TLE data is received and transformed correctly
        TODO - find out what witness location data will look like for comparison and consensus
        To prevent all of SpaceX agreeing that they didn't see any satellites over Ukraine, nope, no way, they were in geostationary orbit above the North Pole 
        TODO - Josh idea, think about seeding the initial conditions with ground station data until I have built up enough of a map. Could also be GNSS/GPS data if that works. Dont really mind           
        """
        # Use sgp4 propogation to get altitude and velocity
        # More suitable than keplerian motion
        # The Two-Body-Keplerian orbit propagator is the less precise because 
        # it simplifies the gravitational field of the Earth as spherical and 
        # ignores all other sources of orbital perturbations. The SGP4 
        # orbit propagator enhances the accuracy by considering the 
        # oblateness of the Earth and the effects of atmospheric drag.
        epoch = sat.epoch
        state_vector = sat.at(epoch) # works in ITRF frame
        altitude_km = (wgs84.height_of(state_vector)).km
        velocity = state_vector.velocity.km_per_s
        speed_kmps = np.linalg.norm(velocity)

        # Circular orbit: Min speed 6.9 km/s
        # Elliptical orbit: Max speed 10.07 km/s
        # Plus buffer for error
        if not (6.5 <= speed_kmps <= 10.5):
            return False

        # For LEO, satellites should have an inclination between 0 and 180 degrees
        # Inclination is initially provided in radians
        if not (0 <= (sat.model.inclo * 180 / np.pi) <= 180):
            return False
        
        # See https://rhodesmill.org/skyfield/earth-satellites.html#detecting-propagation-errors
        # If any elements don't make sense, the position is returned as [nan, nan, nan]
        # e.g. if eccentricity is not between 0 and 1
        if np.nan in state_vector.xyz.km:
            return False
        
        # Altitude should be in LEO range of 200km to 2000km above the Earth's surface
        if not (200 <= altitude_km <= 2000):
            return False

        return True
    
    def get_correctness_score(self, sat: EarthSatellite, dag: DAG) -> float:
        """
        Determine the correctness score for transactional data being added to the DAG.
        This is done by comparing the data provided by a satellite with historical data in the DAG.
        TODO - change
        Returns a correctness score between 0 (inconsistent) and 1 (high agreement)

        """
        # Try to match OBJECT_ID in historical transactions (basic implementation)

        matches = []

        for tx_list in dag.ledger.values():
            for tx in tx_list:
                try:
                    past_data = json.loads(tx.tx_data)
                    if isinstance(past_data, dict) and past_data.get("OBJECT_ID") == sat.model.satnumid:
                        matches.append(past_data)
                except Exception:
                    continue
        
        if not matches:
            return 0.5 # Neutral if not seen before
        
        # Very basic check — mean motion deviation, inclination deviation
        past = matches[-1]
        motion_dev = abs(sat.model.no_kozai - float(past["MEAN_MOTION"]))
        incl_dev = abs((sat.model.inclo * 180 / np.pi) - float(past["INCLINATION"]))

        correctness_score = 1.0
        if motion_dev > 0.1:
            correctness_score -= 0.3 # TODO - tune
        if incl_dev > 5:
            correctness_score -= 0.3 # TODO - tune
        return max(0.0, min(1.0, correctness_score))
    
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
        elif deviation < 0.6:
            return 0.7
        elif deviation < 1.0:
            return 0.5
        else:
            return 0.2

    def calculate_consensus_score(self, correctness: float, accuracy: float, reputation: float) -> float:
        """
        Weighted score. Tuneable weights.
        """
        return 0.4 * correctness + 0.4 * accuracy + 0.2 * reputation                
    
    def proof_of_inter_satellite_evaluation(self, dag: DAG, sat_node: SatelliteNode, transaction: Transaction) -> bool:
        """
        Returns a bool of it consensus has been reached  
        NOTE: Assume one witnessed satellite per transaction   
        """
        # 1)  Get the data 
        od_data = load_json_data("od_data.json")
        
        # 2) TODO Check if data is valid, if not - ignore. Consensus cannot be reached on invalid data. If yes, add to DAG
        # If the list is empty, there is no data that can be valid
        if len(od_data) == 0:
            # Reduce node reputation for providing no data that can be checked
            sat_node.reputation -= self.reputation_step
            return False
        
        for sat in od_data:
            if not self.data_is_valid(sat):
                # Reduce node reputation for providing invalid data
                sat_node.reputation -= self.reputation_step
                return False

        # 3a) If we have enough, valid data, submit a transaction with the string of data
        # TODO - might be able to optimise this to reduce string length
        # Serialize data for transaction
        tx_data_dict = {
        "OBJECT_NAME": sat.name,
        "OBJECT_ID": sat.model.satnum,
        "EPOCH": str(sat.epoch.utc_iso()),
        "MEAN_MOTION": (sat.model.no_kozai / (2 * np.pi)) * 1440, # Convert rads/min into revs/day
        "ECCENTRICITY": sat.model.ecco,
        "INCLINATION": sat.model.inclo * 180 / np.pi
        }

        transaction.tx_data = json.dumps(tx_data_dict)
    
        dag.add_tx(transaction)
        
        # 3) TODO Check we have enough data to be bft (3f + 1)
        # if not, consensus cannot be reached. 
        # Length of DAG is 2 by default with genesis transactions. These are dummy data,
        # so we must have at least 4 real transactions on top to be BFT (2 + 4 =  total)
        if len(dag.ledger) < 6:
            return False
        
        # 4) TODO Check if satellite has been witnessed before
        #4a if yes, does this data agree with other data/ is it correct? Assign correctness score -> affects transaction
        #  This is going to be very tricky. How do I get this data?? Where do I store it? Do I want this to tie in to how parents are selected?
        correctness_score = self.get_correctness_score(tx_data_dict, dag)
        
        # 5) TODO is sensor data accurate (done regardless of previous witnessing). Assign accuracy score -> affects transaction and node reputation
        # Again, might be tricky. Probability distribution here? Like in the PowerGraph paper?
        accuracy_score = self.estimate_accuracy(sat)
        
        # 6) TODO calculate consensus score - node reputation, accuracy and correctness all factor
        # Need to develop an equation - this will take some reading and tuning

        consensus_score = self.calculate_consensus_score(correctness_score, accuracy_score, sat_node.reputation)

        # if consensus reached - strong node (maybe affects node reputation?), else weak node (like IOTA)
        if consensus_score >= self.consensus_threshold:
            transaction.consensus_reached = True
            sat_node.reputation += self.reputation_step
            return True
        else:
            transaction.consensus_reached = False
            sat_node.reputation -= self.reputation_step
            return False
        
        # 7) TODO if consensus score above threshold, consensus reached. Else not.
     
        # if all history is strong - strong edge connection, else weak edge connection. I guess I can't update edge connections or tx strength retroactively so..
        # this bit is the same as tangle, so i may just try and run this consensus on a tangle like system or simulator like in references 

        # TODO - steps 4-6 are the key bits, work here. Get equations and find some tuning evidence. Need to transform into a DAG from a blockchain after that