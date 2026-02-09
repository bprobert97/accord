# Introduction to ACCORD
Autonomous Cooperative Consensus Orbit Determination (ACCORD) is a decentralised framework for satellite operations in Low Earth Orbit (LEO). It addresses the limitations of centralised coordination, such as single points of failure and trust bottlenecks, by leveraging a Directed Acyclic Graph (DAG)-based Distributed Ledger Technology (DLT).

In ACCORD, satellites use on-board sensors to observe one another and collaboratively determine their orbital positions. These observations are submitted to the distributed ledger, where consensus is achieved through Proof of Inter-Satellite Evaluation (PoISE) - a novel, reputation-based, Byzantine Fault Tolerant (BFT) consensus mechanism. Unlike traditional blockchain systems, PoISE does not rely on financial incentives or intermediaries. Instead, it validates orbit data by evaluating mutual observations and assessing the trustworthiness of both the data and the observing satellites.

This decentralised approach enhances data integrity, trust, and resilience across heterogeneous constellations. As more satellites join the network, ACCORD scales naturally, enabling secure and autonomous satellite operations—even in zero-trust environmen

This code is licensed under a GNU General Public License v3.0.

# Current Status

The project is currently at TRL 0. The PoISE consensus mechanism is in the early phases of development by [Beth Probert](https://pureportal.strath.ac.uk/en/persons/beth-probert), a PhD student at the University of Strathclyde's Applied Space Technology Laboratory. Once development of PoISE is completed, the rest of the ACCORD framework shall be developed around this consensus mechanism. By incorporating smart contracts in future development, the system will enable autonomous decision-making, allowing satellites to self-govern, coordinate tasks, and reroute services in real time.

## Related Publications

* [B. Probert, R. A. Clark, E. Blasch, and M. Macdonald, “A Review of Distributed Ledger Technologies for Satellite Operations,” IEEE Access, vol. 13, pp. 123230–123258, 2025, doi: 10.1109/ACCESS.2025.3588688](https://ieeexplore.ieee.org/document/11079570)


# Repository Layout

<pre>
📁 accord/
│
├── 📁 .github/workflows/              # GitHub Workflow files
│   └── main.yml                       # CI configuration for github: Pylint and Mypy
│
├── 📁 design/                      # Design documents, Jupyter notebooks and PlantUML diagrams
│
├── 📁 images/                         # Image assets
│
├── 📁 src/                        # Main source code
│   └── __init__.py                # Empty file, for module creation
│   └── consensus_mech.py          # Code for the PoISE consensus mechanism
│   └── dag.py                     # Code for the Directed Acyclic Graph ledger structure
│   └── filter.py                  # Code for the orbit determination calculations
│   └── logger.py                  # Code for the app logger
|   └── plotting.py                # Python script for plotting NIS, correctness, reputation and consensus scores
│   └── reputation.py              # Code for the satellite reputation manager
│   └── satellite_node.py          # Code representing a satellite in the network
│   └── simualtion.py              # Code for simulating a satellite network
│   └── transaction.py             # Code representing a transaction submitted by a satellite
│
├── 📁 tests/                     # Unit tests, written with pytest
|
├── .codespellrc             # Codespell configuration file
├── .coveragerc              # Pytest coverage configuration file
├── .gitignore               # Files/folders to ignore in Git
├── .pylintrc                # Pylint configuration file
├── accord_demo.py           # Demonstration of ACCORD
├── changelog.md             # Release change log
├── LICENSE.MD               # License file
├── mypy.ini                 # Mypy configuration
├── README.md                # Project overview
├── requirements.txt         # List of python package dependencies
└── requirements_linux.txt   # List of python package dependencies for Linux and CI

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
   pip install -r requirements.txt
   ```

   On macOS/Linux:

   ```bash
   pip install -r requirements_linux.txt
   ```

# Usage

To run the ACCORD demo, either:
* In VSCode, right click `accord_demo.py` and select `Run Python File in Terminal`
* In a terminal, execute `py accord_demo.py`
