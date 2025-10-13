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

RANGE_VAR_FLOOR: float = 25.0 ** 2       # 625 m^2
ANGLE_VAR_FLOOR: float = (1.7e-3) ** 2   # ~2.89e-6 rad^2
LOS_VAR_FLOOR: float = (1e-3) ** 2       # 1e-6 (unitless vector component variance)


def load_json_data(file_name: str = "sim_output.json") -> list[dict]:
    """
    Load data from a JSON file.

    Args:
    - file_name: The name of the JSON file to read data from.
      Must contain a list of observation dicts.

    Returns:
    - A list of dictionaries, each entry is one observation.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data
