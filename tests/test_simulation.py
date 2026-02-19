"""
Tests for simulation.py
"""
import unittest
import numpy as np
from src.simulation import generate_walker_delta_constellation

class TestSimulation(unittest.TestCase):
    """
    Test cases for simulation.py
    """
    def test_generate_walker_delta_constellation(self) -> None:
        """
        Test the generation of a Walker Delta constellation.
        """
        t = 10
        p = 2
        f = 1
        a = 7000e3
        i = np.radians(45)

        constellation = generate_walker_delta_constellation(t, p, f, a, i)

        # Check the number of satellites
        self.assertEqual(len(constellation), t)

        # Check that T must be divisible by P
        with self.assertRaises(ValueError):
            generate_walker_delta_constellation(11, 2, 1, a, i)

        # Check some elements for the first satellite
        sat0 = constellation[0]
        self.assertEqual(sat0[0], a)
        self.assertEqual(sat0[1], 0.0)
        self.assertEqual(sat0[2], i)
        self.assertEqual(sat0[3], 0.0) # Plane 0, RAAN=0
        self.assertEqual(sat0[5], 0.0) # Sat 0 in Plane 0, TA=0

        # Check plane 1 RAAN
        sat5 = constellation[5] # First satellite in second plane (T=10, P=2, S=5)
        self.assertAlmostEqual(sat5[3], np.pi) # Plane 1, RAAN = (2*pi/2)*1 = pi

        # Check phasing
        # M_ps = (s * 2*pi/S) + (p * 2*pi * F / T)
        # For p=1, s=0, F=1, T=10, S=5:
        # M_10 = (0 * 2*pi/5) + (1 * 2*pi * 1 / 10) = 2*pi/10 = pi/5
        self.assertAlmostEqual(sat5[5], np.pi / 5)

    def test_walker_delta_f0(self) -> None:
        """
        Test the generation of a Walker Delta constellation with F=0.
        """
        t = 4
        p = 2
        f = 0
        a = 7000e3
        i = np.radians(45)

        constellation = generate_walker_delta_constellation(t, p, f, a, i)

        # Plane 0, Sat 0: TA=0
        self.assertAlmostEqual(constellation[0][5], 0.0)
        # Plane 1, Sat 0: TA = (0 * 2*pi/2) + (1 * 2*pi * 0 / 4) = 0
        self.assertAlmostEqual(constellation[2][5], 0.0)

if __name__ == "__main__":
    unittest.main()
