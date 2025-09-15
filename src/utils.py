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


def build_tx_data(file_name: str = "od_data.json", index: int = 0) -> dict:
    """
    Build transaction data from a JSON file.

    Args:
    - file_name: The name of the JSON file to read data from.
    As a minimum, the data must have a 'target_id' field.
    - index: The index of the data entry to retrieve from the JSON file.
    Default is 0 (the first entry).

    Returns:
    - A dictionary containing the data from the JSON file.

    """
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data[index]
