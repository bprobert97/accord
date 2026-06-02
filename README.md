# Introduction to ACCORD
Autonomous Cooperative Consensus Orbit Determination (ACCORD) is a decentralised framework for satellite operations in Low Earth Orbit (LEO). It addresses the limitations of centralised coordination, such as single points of failure and trust bottlenecks, by leveraging a Directed Acyclic Graph (DAG)-based Distributed Ledger Technology (DLT).

In ACCORD, satellites use on-board sensors to observe one another and collaboratively determine their orbital positions. These observations are submitted to the distributed ledger, where consensus is achieved through Proof of Inter-Satellite Evaluation (PoISE) - a novel, reputation-based, Byzantine Fault Tolerant (BFT) consensus mechanism. Unlike traditional blockchain systems, PoISE does not rely on financial incentives or intermediaries. Instead, it validates orbit data by evaluating mutual observations and assessing the trustworthiness of both the data and the observing satellites.

This decentralised approach enhances data integrity, trust, and resilience across heterogeneous constellations. As more satellites join the network, ACCORD scales naturally, enabling secure and autonomous satellite operations—even in zero-trust environments.

![Satellite Constellation Animation](images/orbits.gif)

This code is licensed under a GNU General Public License v3.0.

# Current Status

The project is currently at TRL 0. The PoISE consensus mechanism is in the early phases of development by [Beth Probert](https://pureportal.strath.ac.uk/en/persons/beth-probert), a PhD student at the University of Strathclyde's Applied Space Technology Laboratory. Once development of PoISE is completed, the rest of the ACCORD framework shall be developed around this consensus mechanism. By incorporating smart contracts in future development, the system will enable autonomous decision-making, allowing satellites to self-govern, coordinate tasks, and reroute services in real time.

## Related Publications

* [B. Probert, R. A. Clark, E. Blasch, and M. Macdonald, “Cooperative Orbit Determination for Trusted, Autonomous, and Decentralised Satellite Operations,” in AIAA SCITECH 2026 Forum, in AIAA SciTech Forum. Orlando, Florida: American Institute of Aeronautics and Astronautics, Jan. 2026. doi: 10.2514/6.2026-0825](https://arc.aiaa.org/doi/10.2514/6.2026-0825)

* [B. Probert, R. A. Clark, E. Blasch, and M. Macdonald, “A Review of Distributed Ledger Technologies for Satellite Operations,” IEEE Access, vol. 13, pp. 123230–123258, 2025, doi: 10.1109/ACCESS.2025.3588688](https://ieeexplore.ieee.org/document/11079570)

## Citation
If you use this work, please cite it as:
> B. Probert, bprobert97/accord: v3.1. (May 28, 2026). Python. University of Strathclyde, Glasgow. [DOI: 10.5281/zenodo.17816885](https://doi.org/10.5281/zenodo.17816885)


# Repository Layout

<pre>
📁 accord/
│
├── 📁 .github/workflows/           # GitHub Workflow files
│   └── main.yml                     # CI configuration for github: Pylint, Mypy and Pytest
│
├── 📁 design/                      # Design documents, Jupyter notebooks and PlantUML diagrams
│
├── 📁 images/                      # Image assets
│
├── 📁 src/                        # Main source code
│   └── __init__.py                # Empty file, for module creation
│   └── consensus_mech.py          # Code for the PoISE consensus mechanism
│   └── dag.py                     # Code for the Directed Acyclic Graph ledger structure
│   └── filter.py                  # Code for the orbit determination calculations
│   └── logger.py                  # Code for the app logger
│   └── mc_comparison.py           # Code for generating comparison plots for different Monte Carlo data sets
|   └── plotting.py                # Code for plotting simulation results
│   └── reputation.py              # Code for the satellite reputation manager
│   └── satellite_node.py          # Code representing a satellite in the network
│   └── simulation.py              # Helper functions for generating and converting satellite orbital elements.
│   └── transaction.py             # Code representing a transaction submitted by a satellite
│   └── visualise_orbits.py        # Code for generating a gif of truth orbits
│
├── 📁 tests/                     # Unit tests, written with pytest
|
├── .codespellrc             # Codespell configuration file
├── .coveragerc              # Pytest coverage configuration file
├── .gitignore               # Files/folders to ignore in Git
├── .pylintrc                # Pylint configuration file
├── accord_demo.py           # Demonstration of ACCORD
├── changelog.md             # Release change log
├── CONTRIBUTING.md          # Guidance on contributing to the project
├── LICENSE.MD               # License file
├── mc_demo.py               # Monte Carlo Simulation of ACCORD
├── mypy.ini                 # Mypy configuration
├── README.md                # Project overview
├── requirements.txt         # List of python package dependencies for Linux and CI
└── requirements_windows.txt   # List of python package dependencies for Windows

</pre>

# Installation

Make sure Python 3.13 is installed on your system before proceeding with the installation.
Follow these steps to set up the project in a Python virtual environment:

1. **Clone the repository**
   ```bash
   git clone https://github.com/bprobert97/accord.git
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   On Windows:

   ```bash
   venv\Scripts\activate
   ```
   On macOS/Linux:

   ```bash
   source venv/bin/activate
   ```
4. **Install dependencies**

   On Windows:

   ```bash
   pip install -r requirements_windows.txt
   ```

   On macOS/Linux:

   ```bash
   pip install -r requirements.txt
   ```

# Usage

Note: if you wish to fork the repository using Git, you will need to have [Git LFS](https://git-lfs.com/) installed in order to commit the large `.npz` data files that the simulations produce, or add the `.npz` files to you `.gitignore`.

## Local Demo
To run the ACCORD demo, either:
* In VSCode, right click `accord_demo.py` and select `Run Python File in Terminal`
* In a terminal, execute `py accord_demo.py`

To run the Monte Carlo Simulation:
* In a terminal, run `py mc_demo.py`
* You can also use the following arguments:
   * `--num-runs`: The number of Monte Carlo runs you wish to do (default: 10)
   * `--threshold`: Detection threshold for KPIs (default: 0.4)
   * `--fpr-offset`: False Palsitive Rate offset percent to ignore initialisation effects of the Extended Kalman Filter (default: 0.2)
   * `--recalculate`: Recalculate KPIs from saved data stored in `./sim_data/mc_results/mc_results.npz`
   * Example: `py mc_demo.py --num-runs 10 --threshold 0.2`

## Configuration & Execution

Before executing any simulations, the environment parameters must be defined via the `FilterConfig` dataclass.

**Setting the ISL Distance**
The maximum Inter-Satellite Link (ISL) distance is the core constraint that dictates network topology, observation frequency, and DAG formation.

1. Locate the `DEFAULT_CONFIG` instantiation in the simulation script (e.g., `accord_demo.py` or `mc_demo.py`).
2. Update the `ISL_range_m` parameter to reflect your desired communication threshold (e.g., 1000e3m or 2000e3m).
3. Ensure your Monte Carlo random seeds are set within this config to guarantee deterministic and reproducible runs.
