# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v1.0] - 2025-11

### Summary
This is the first full release of the PoISE consensus mechanism. This has been released to accompany a presentation at the 2026 SciTech Forum.

### Added
- Pytest unit tests for all code.
- Moved plotting into its own file.
- Correctness score now accounts for average NIS, following chi-squared statistics
- Reputation is now based on chi-squared statistics for bounds, and previous behaviour
- Simulates faulty and malicious nodes in accord_demo.py
- Uses a simpler EKF implementation with Filterpy
---

## [v0.2.1] - 2025-10

### Summary
This is a patch release of the PoISE consensus mechanism, providing key bug fixes.

### Added
- Bug fix in od_filter.py to allow LOS only measurements to be processed correctly.
- Additional plots for chi2 functions and consensus scores.
---

## [v0.2] - 2025-09

### Summary
This is the second proof-of-concept release of the **PoISE consensus mechanism**, using MATLAB for generating satellite simulation data and a second order extended kalman filter with a state transition tensor for orbit determination calculations.

### Added
- Replaced TLE data inputs with simulated sensor data created by a MATLAB script.
- Implemented a Second Order Extended Kalman Filter with State Tranisiotn Tensor for permising Orbit Determination Calculations.
- Added the Normalised Innovation Score (NIS) to the consensus mechanism calculations.
- Simulation of both **good** and **malicious data submissions** for a small network of satellites.
- Generation of plots showing:
  - **Chi Squared probability distributions**
- Addition of logging.
- Additional configuration for linting and type checking tools.
- Created new diagrams describing the orbit determination process.

---

## [v0.1] - 2025-08

### Summary
This is the first proof-of-concept release of the **PoISE consensus mechanism**.

### Added
- Support for propagating orbital data using **TLE data**, with the help of **sgp4** and **skyfield**.
- Simulation of both **good** and **malicious data submissions** for a single satellite.
- Generation of plots showing:
  - **DAG structure**
  - **Satellite reputation**
- Implementation of **asynchronous message communications** using `asyncio`.

---
