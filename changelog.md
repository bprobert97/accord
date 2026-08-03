# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [4.2] - 2026-08

### Summary
Adds functionality for a Cubature Kalman Filter and implements a bug fix to reputation code.

### Added
- Cubature Kalman Filter implementation.
- New unit tests.
- New args for running the demos (diable-logging and walker-delta).
- Better plots with larger text.
- Bug fix for reputation snowballing up or down due to broadcasting.
- Correction to DOF score calculation to allow for 6 DOF max in a [0,1] bound.
---

## [4.1] - 2026-07

### Summary
This is a bug fix release.

### Added
- Fixed bug that caused the seed in the DEFAULT_CONFIG to be overwritten when doing monte carlo runs in parallel.
- Added new file for generating plots comparing performance of different estimation filters.
---

## [4.0] - 2026-07

### Summary
This release introduces the ability to choose between multiple Kalman Filter implementations in a plug and play style.

### Added
- Changes to figures to make plots more readable/
- Introduction of a UKF implementation and the ability to switch between Kalman Filters in simulations.
---

## [3.2] - 2026-06

### Summary
This release performs a major refactoring to decentralise the DAG and ConsensusMechanism classes to each SatelliteNode, instead of having centralised class instances that SatelliteNodes Interact with.

### Added
- Remove centralised DAG and ConsensusMechanism class instances and put them on board each SatelliteNode
- Change how satellites are clustered for EKF truth generation to make it more efficient and base din orbital physics
- Change how stats are calculated
- Fix get_parents() function and make has load on Transaction initialisation
- Get chi squared plots to use curved text
---

## [3.1.1] - 2026-06

### Summary
This release refactors the codebase without changing any of the underlying functionality. This is to improve readability.

### Added
- Refactor code to remove pylint errors.
- Make boxplot colours more readable.
- Minor bug fix - add ISL_range_m to data that is saved from a run.
---

## [3.1] - 2026-05

### Summary
This release introduces minor changes ahead of writing a journal paper about ACCORD.

### Added
- Script to visualise orbits in a gif
- Walker delta simulation ability
- Persistence of excitation term
- Ability to identify which falty satellites go undetected
- More specific file name for MC data
- Colourblind friendly palettes for figure generation
- Max and Min DOF checks in consensus
---

## [3.0] - 2026-03

### Summary
This release introduces the ability to run Monte Carlo simulations of the PoISE consensus mechanism. The largest test to date is 40 MC runs of random constellations of 400 satellites, which takes roughly 6.5 hours to run when using 4 CPU cores in parallel. This release also adds a new streamlit app for data inspection, hosted at https://accord-demo.streamlit.app/.

### Added
- Monte carlo simulation ability
- Streamlit app hosting
- Monte carlo metrics and the ability to compare runs with different initial conditions
- Minor updates to figures
---

## [2.2] - 2026-03

### Summary
This release makes minor changes related to the calculation and visualisation of the expected and empirical medians of the NIS distributions.

### Added
- A function to calculate median percentiles, to show how the empircal data varies from the expected theoretical values.

---

## [2.1] - 2026-03

### Summary
This release makes minor changes ready for submitting this work to the 29th International Conference on Information Fusion in Trondheim.

### Added
- Box plot, in place of an old violin plot.

---

## [2.0] - 2026-02

### Summary
This release increases the simulation size to a random constellation of 400 satellites.

### Added
- Additional plots for constellation mapping.
- Ability to simulate constellations of up to 400 satellites.
- Abiliity to configure Inter Satellite Link (ISL) ranges to simulate connectivity changes.

---

## [v1.1] - 2025-12

### Summary
This release includes minor updates to make the codebase tidier in preparation for the 2026 AIAA SciTech forum.

### Added
- Test coverage metrics in the CI.
- Updates to diagrams in the design directory.
- Normalised reputation to be between 0 and 1 instead of 0 and 100.

### Removed
- The references directory and files.

---

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
