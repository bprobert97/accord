# ACCORD Project Context

Autonomous Cooperative Consensus Orbit Determination (ACCORD) is a decentralized framework for satellite operations in Low Earth Orbit (LEO). It uses a Directed Acyclic Graph (DAG)-based Distributed Ledger Technology (DLT) and a reputation-based Byzantine Fault Tolerant (BFT) consensus mechanism called Proof of Inter-Satellite Evaluation (PoISE).

## Project Overview

- **Core Technology:** Python 3.13
- **Main Components:**
    - `src/consensus_mech.py`: Implementation of the PoISE consensus mechanism.
    - `src/dag.py`: Distributed ledger structure using a DAG.
    - `src/filter.py`: Orbit determination logic (Extended Kalman Filter).
    - `src/reputation.py`: Satellite reputation management.
    - `src/satellite_node.py`: Satellite network node representation.
    - `src/simulation.py`: Orbital dynamics and simulation helpers.
- **Demos:**
    - `accord_demo.py`: Main demonstration of the framework.
    - `mc_demo.py`: Monte Carlo simulation for performance evaluation.
    - `streamlit_app.py`: Web-based visualization of results.

## Building and Running

### Environment Setup

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   - Windows: `pip install -r requirements_windows.txt`
   - Linux/macOS: `pip install -r requirements.txt`

### Running Demos

- **Standard Demo:** `python accord_demo.py`
- **Monte Carlo Simulation:** `python mc_demo.py --num-runs 10`
- **Streamlit App:** `streamlit run streamlit_app.py`
- **Orbit Visualization:** `python src/visualise_orbits.py`

### Testing and Quality

- **Run Tests:** `pytest tests/`
- **Coverage:** `pytest --cov=src tests/`
- **Linting:** `pylint src/ tests/ accord_demo.py mc_demo.py streamlit_app.py`
- **Type Checking:** `mypy .`

## Development Conventions

- **Coding Style:** Adheres to PEP 8 (enforced via Pylint). Configuration is in `.pylintrc`.
- **Type Safety:** Uses static typing (checked via Mypy). Configuration is in `mypy.ini`.
- **Testing:** Uses `pytest` for unit testing. Configuration is in `.coveragerc`.
- **Documentation:** README.md provides a high-level overview and installation guide.
- **CI/CD:** GitHub Actions (`.github/workflows/main.yml`) runs Pylint, Mypy, and Pytest on every push.
- **Data Management:** Large simulation data is stored in `sim_data/` as `.npz` files. Git LFS is recommended if tracking these files.
