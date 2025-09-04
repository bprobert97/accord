# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
