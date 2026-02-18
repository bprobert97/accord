# pylint: disable=protected-access, too-many-locals, too-many-positional-arguments, too-many-arguments
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
import numpy as np
from numpy.typing import NDArray
from src.logger import get_logger

logger = get_logger()

# ----------------------- Constants -----------------------
MU_EARTH = 3.986004418e14  # m^3/s^2
RE = 6378e3
# ---------------------------------------------------------

def generate_random_keplerian_elements(seed: int) -> tuple[float, float, float,
                                                           float, float, float]:
    """
    Generates a set of random but valid Keplerian elements for a LEO satellite.

    Arguments:
    - seed: An integer seed for reproducibility.
    Returns:
    - A tuple containing (a, e, i, raan, argp, ta)
    """
    # Use a seed for reproducibility in tests and demos
    rng = np.random.default_rng(seed)

    altitude = rng.uniform(180e3, 2000e3)
    a = RE + altitude

    e = rng.uniform(0, 0.05)
    i = rng.uniform(0, np.pi) # inclination in radians
    raan = rng.uniform(0, 2 * np.pi)
    argp = rng.uniform(0, 2 * np.pi)
    ta = rng.uniform(0, 2 * np.pi)
    return a, e, i, raan, argp, ta

def keplerian_to_cartesian(a: float, e: float, i: float, raan: float,
                           argp: float, ta: float) -> NDArray[np.float64]:
    """
    Converts Keplerian elements to a Cartesian state vector (position and velocity).
    Args:
    - a: Semi-major axis
    - e: Eccentricity
    - i: Inclination
    - raan: Right Ascension of the Ascending Node
    - argp: Argument of Periapsis
    - ta: True Anomaly
    Returns:
    - 6-element Cartesian state vector [px, py, pz, vx, vy, vz]
    """
    # Position and velocity in the perifocal frame
    r = a * (1 - e**2) / (1 + e * np.cos(ta))

    p_pqw = r * np.array([np.cos(ta), np.sin(ta), 0])

    # Check for division by zero or invalid values
    sqrt_val = MU_EARTH * a * (1 - e**2)
    sqrt_val = max(sqrt_val, 0)

    v_pqw_mag = np.sqrt(sqrt_val) / r
    v_pqw = v_pqw_mag * np.array([-np.sin(ta), e + np.cos(ta), 0])

    # Rotation matrix from perifocal to ECI frame
    ci = np.cos(i)
    si = np.sin(i)
    craan = np.cos(raan)
    sraan = np.sin(raan)
    cargp = np.cos(argp)
    sargp = np.sin(argp)

    rot_matrix = np.array([
        [craan*cargp - sraan*sargp*ci, -craan*sargp - sraan*cargp*ci, sraan*si],
        [sraan*cargp + craan*sargp*ci, -sraan*sargp + craan*cargp*ci, -craan*si],
        [sargp*si, cargp*si, ci]
    ])

    # Transform position and velocity to ECI frame
    p_eci = rot_matrix @ p_pqw
    v_eci = rot_matrix @ v_pqw

    return np.hstack([p_eci, v_eci])
