# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
