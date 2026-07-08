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

from typing import List
from src.filters.filter_interface import ObservationRecord

def validate_observation_records(records: List[ObservationRecord],
                                 n_satellites: int,
                                 expected_step: int) -> None:
    """
    Validates a list of ObservationRecord objects against expected parameters.

    Args:
    - records: The list of ObservationRecord objects to validate.
    - n_satellites: The total number of satellites in the simulation.
    - expected_step: The step index the records are expected to belong to.
    """
    max_records = n_satellites * (n_satellites - 1)
    assert 0 < len(records) <= max_records
    for record in records:
        assert record.step == expected_step
        assert record.nis >= 0
        assert record.dof == 2
