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
from json import JSONDecodeError
from typing import Optional
from skyfield.api import EarthSatellite, load, Timescale
import numpy as np
from sgp4.api import Satrec, WGS72 # type: ignore[import-untyped]

def corrupt_satellite(sat: EarthSatellite, mean_motion_factor: float = 0.8) -> EarthSatellite:
    """
    mean motion factor of >1.05 or <0.73 makes the data invalid
    1.0 is the same.
    1.01 - 1.04 affects correctness
    0.74-0.93 affects accuracy and correctness
    0.94-0.99 affects correctness
    """
    # Copy fields from the original model
    m = sat.model

    # Build a new Satrec with modified mean motion
    corrupted_model = Satrec()
    corrupted_model.sgp4init(
        WGS72,
        'i',
        int(m.satnum),
        float(m.jdsatepoch - 2433281.5),  # epoch in days since 1949-12-31
        float(m.bstar),
        float(m.ndot),
        float(m.nddot),
        float(m.ecco),
        float(m.argpo),
        float(m.inclo),
        float(m.mo),
        float(m.no_kozai) * mean_motion_factor,  # corrupted mean motion
        float(m.nodeo)
    )

    # Create an EarthSatellite instance without calling __init__
    corrupted_sat = EarthSatellite.__new__(EarthSatellite)
    corrupted_model.intldesg = sat.model.intldesg
    corrupted_sat.model = corrupted_model
    corrupted_sat.name = sat.name
    corrupted_sat.epoch = sat.epoch
    corrupted_sat.target = sat.target

    return corrupted_sat


def build_tx_data_str(satellite_data: EarthSatellite) -> str:
    """
    Construct a and serialise a string of data from an
    EarthSatellite object, to be used to populate the transaction
    data (tx_data) field of a Transaction object.

    Arguments:
    - satellite_data: The EarthSatellite object (as defined by the
    skyfield library) to be turned into a string

    Returns:
    - A string of data to be added in a transaction. The string is
    in a dict format. Some items are converted from rads/min into revs/day
    """
    # International designator formatting
    intldesg = str(satellite_data.model.intldesg)
    year = int(intldesg[:2])
    launch = intldesg[2:]

    # Launch years possible: 1957-1999 or 2000-present
    # TODO Code will work until 2057
    if year < 57:
        full_year = 2000 + year
    else:
        full_year = 1900 + year

    motion_ddot = (satellite_data.model.nddot / (2 * np.pi)) * (1440 ** 2)

    # When the second derivative of motion is 0, Celestrak formats it to
    # an int rather than a float
    if motion_ddot == 0:
        motion_ddot = int(motion_ddot)

    # Epoch formatting
    epoch = str(satellite_data.epoch.utc_datetime().\
                replace(tzinfo=None).isoformat(timespec='microseconds'))

    return json.dumps({
                "OBJECT_NAME": satellite_data.name,
                "OBJECT_ID": f"{full_year}-{launch}",
                "EPOCH": epoch,
                "MEAN_MOTION": round((satellite_data.model.no_kozai / (2 * np.pi)) * 1440, 8),
                "ECCENTRICITY": round(satellite_data.model.ecco, 6),
                "INCLINATION": round(satellite_data.model.inclo * 180 / np.pi, 4),
                "RA_OF_ASC_NODE": round(satellite_data.model.nodeo * 180 / np.pi, 4),
                "ARG_OF_PERICENTER": round(satellite_data.model.argpo * 180 / np.pi, 4),
                "MEAN_ANOMALY": round(satellite_data.model.mo * 180 / np.pi, 3),
                "EPHEMERIS_TYPE": satellite_data.model.ephtype,
                "CLASSIFICATION_TYPE": satellite_data.model.classification,
                "NORAD_CAT_ID": satellite_data.model.satnum,
                "ELEMENT_SET_NO": satellite_data.model.elnum,
                "REV_AT_EPOCH": satellite_data.model.revnum,
                "BSTAR": round(satellite_data.model.bstar, 7),
                "MEAN_MOTION_DOT": round((satellite_data.model.ndot / (2 * np.pi)) * (1440 ** 2),
                                         8),
                "MEAN_MOTION_DDOT": motion_ddot
            })

def build_earth_satellite_list_from_str(ts: Timescale, data: str,
                                        make_faulty: bool) -> list[Optional[EarthSatellite]]:
    """
    Construct an EarthSatellite object from a string of data to be
    used for data validation.

    Arguments:
    - ts: The timescale, used to build the satellite's epoch time
    - data: A string of data (ideally in a dict format) to be used to populate
    the attributes of the EarthSatellite object.
    - make_faulty: A boolean flag about whether the data generated should be faulty, to
    represent a malicious node

    Returns:
    - A list of EarthSatellite objects
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string provided to satellite parser: {e}") from e

    # If a single dict was passed, wrap it in a list
    if isinstance(data, dict):
        data = [data]

    # Validate it's a list of dicts
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise TypeError("Expected a list of dicts after parsing satellite data")

    sat_list = [EarthSatellite.from_omm(ts, fields) for fields in data]

    if make_faulty:
        for i, sat in enumerate(sat_list):
            sat_list[i] = corrupt_satellite(sat)

    return sat_list

def load_json_data(file_name: str, faulty_data: bool) -> list[Optional[EarthSatellite]]:
    """
    Turns json data in a file into a Python dict.
    JSON data must be in the Celestrak format
    https://rhodesmill.org/skyfield/earth-satellites.html

    If faulty_data is True, this function will generate faulty data for testing
    """
    try:
        with load.open(file_name) as f:
            data = json.load(f)

    except JSONDecodeError:
        return []

    ts = load.timescale()
    satellite_list = build_earth_satellite_list_from_str(ts, data, faulty_data)
    return satellite_list
