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

class SatelliteNode():
    """
    A class representing a node in the network, in this case a LEO satellite. 
    This does NOT represent a node in the ledger - these are transactions
    """
    def __init__(self, id: str) -> None:
        self.id: str = id
        # Reputation starts at 0, affected by validity and accuracy
        self.reputation: float = 0 #TODO - need to consider how this affects consensus. If reputation low, does it get allowed? or does it affect consensus score?