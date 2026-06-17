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
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from src.logger import get_logger

logger = get_logger()

# ----------------------- Constants -----------------------
MU_EARTH = 3.986004418e14  # m^3/s^2
RE = 6378e3
# ---------------------------------------------------------

@dataclass
class KeplerianElements:
    """
    A dataclass to hold Keplerian elements for a satellite.
    """
    a: float
    e: float
    i: float
    raan: float
    argp: float
    ta: float

def generate_random_keplerian_elements(seed: int) -> KeplerianElements:
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

    e = 0.0
    i = rng.uniform(0, np.pi) # inclination in radians
    raan = rng.uniform(0, 2 * np.pi)
    argp = rng.uniform(0, 2 * np.pi)
    ta = rng.uniform(0, 2 * np.pi)
    return KeplerianElements(a, e, i, raan, argp, ta)

def generate_walker_delta_constellation(t: int, p: int, f: int, a: float,
                                         i: float) -> list[KeplerianElements]:
    """
    Generates Keplerian elements for a Walker Delta constellation i: T/P/F.
    Args:
    - t: Total number of satellites
    - p: Number of planes
    - f: Phase factor (0 to p-1)
    - a: Semi-major axis (assuming circular orbits, e=0)
    - i: Inclination (radians)
    Returns:
    - List of KeplerianElements for each satellite
    """
    if t % p != 0:
        raise ValueError("Total number of satellites T must be divisible by number of planes P.")

    s = t // p  # Satellites per plane
    e = 0.0     # Circular orbit
    argp = 0.0  # Arbitrary for circular orbit

    constellation = []
    for plane_idx in range(p):
        raan = (2 * np.pi / p) * plane_idx
        for sat_idx in range(s):
            # True Anomaly (for circular orbit, ta = Mean Anomaly)
            ta = (2 * np.pi / s) * sat_idx + (2 * np.pi * f / t) * plane_idx
            # Normalise ta to [0, 2*pi]
            ta %= (2 * np.pi)
            constellation.append(KeplerianElements(a, e, i, raan, argp, ta))

    return constellation

def keplerian_to_cartesian(kep_elements: KeplerianElements) -> NDArray[np.float64]:
    """
    Converts Keplerian elements to a Cartesian state vector (position and velocity).
    Args:
    - kep_elements: A KeplerianElements instance containing the orbital elements.
    Returns:
    - 6-element Cartesian state vector [px, py, pz, vx, vy, vz]
    """

    # Position and velocity in the perifocal frame
    r = kep_elements.a * (1 - kep_elements.e**2) / (\
        1 + kep_elements.e * np.cos(kep_elements.ta))

    p_pqw = r * np.array([np.cos(kep_elements.ta),
                          np.sin(kep_elements.ta), 0])

    # Check for division by zero or invalid values
    sqrt_val = MU_EARTH * kep_elements.a * (1 - kep_elements.e**2)
    sqrt_val = max(sqrt_val, 0)

    v_pqw_mag = np.sqrt(sqrt_val) / r
    v_pqw = v_pqw_mag * np.array([-np.sin(kep_elements.ta),
                                  kep_elements.e + np.cos(kep_elements.ta), 0])

    # Rotation matrix from perifocal to ECI frame
    ci = np.cos(kep_elements.i)
    si = np.sin(kep_elements.i)
    craan = np.cos(kep_elements.raan)
    sraan = np.sin(kep_elements.raan)
    cargp = np.cos(kep_elements.argp)
    sargp = np.sin(kep_elements.argp)

    rot_matrix = np.array([
        [craan*cargp - sraan*sargp*ci, -craan*sargp - sraan*cargp*ci, sraan*si],
        [sraan*cargp + craan*sargp*ci, -sraan*sargp + craan*cargp*ci, -craan*si],
        [sargp*si, cargp*si, ci]
    ])

    # Transform position and velocity to ECI frame
    p_eci = rot_matrix @ p_pqw
    v_eci = rot_matrix @ v_pqw

    return np.hstack([p_eci, v_eci])
