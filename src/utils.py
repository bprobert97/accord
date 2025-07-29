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
from skyfield.api import EarthSatellite, load
import astropy.constants as ac

# Global Variables for consensus
# TODO - May tune these to optimise performance
CONFIRMATION_THRESHOLD = 1
REJECTION_THRESHOLD = -1
CONFIRMATION_STEP = 0.1

# Global astrophysical values
G = ac.G # Gravitational Constant in m3 / (kg s2)
M = ac.M_earth # Mass of the Earth in kg
R = float(ac.R_earth.to("km").value) # Radius of the Earth in km

def load_json_data(file_name: str) -> list:
    """
    Turns json data in a file into a Python dict.
    JSON data must be in the Celestrak format
    https://rhodesmill.org/skyfield/earth-satellites.html
    """
    with load.open(file_name) as f:
        data = json.load(f)

    ts = load.timescale()
    satellite_list = [EarthSatellite.from_omm(ts, fields) for fields in data]
    return satellite_list