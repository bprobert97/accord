![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/bprobert97/335ad3bc77d4305c53eeb8715939592e/raw/accord-coverage.json)

# Autonomous Cooperative Consensus Orbit Determination (ACCORD)

ACCORD is a decentralised framework for satellite operations in Low Earth Orbit (LEO). It addresses the limitations of centralised coordination, such as single points of failure and trust bottlenecks, by leveraging a Directed Acyclic Graph (DAG)-based Distributed Ledger Technology (DLT).

In ACCORD, satellites use on-board sensors to observe one another and collaboratively determine their orbital positions. These observations are submitted to the distributed ledger, where consensus is achieved through Proof of Inter-Satellite Evaluation (PoISE)—a novel, reputation-based, Byzantine Fault Tolerant (BFT) consensus mechanism. Unlike traditional blockchain systems, PoISE does not rely on financial incentives or intermediaries. Instead, it validates orbit data by evaluating mutual observations and assessing the trustworthiness of both the data and the observing satellites.

This decentralised approach enhances data integrity, trust, and resilience across heterogeneous constellations. As more satellites join the network, ACCORD scales naturally, enabling secure and autonomous satellite operations—even in zero-trust environments.

![Satellite Constellation Animation](images/orbits.gif)

This code is licensed under a [GNU General Public License v3.0](LICENSE.MD).

## Installation

Make sure Python 3.13 is installed on your system before proceeding with the installation. Follow these steps to set up the project in a Python virtual environment:

1. **Clone the repository**
   git clone https://github.com/bprobert97/accord.git
   cd accord

2. **Create a virtual environment**
   python -m venv venv

3. **Activate the virtual environment**
   * **Windows:** venv\Scripts\activate
   * **macOS/Linux:** source venv/bin/activate

4. **Install dependencies**
   * **Windows:** pip install -r requirements_windows.txt
   * **macOS/Linux:** pip install -r requirements.txt

## Usage & Demonstration

### Local Demo
To run the single-run ACCORD demo, execute the script in your terminal. By default, this runs using the Extended Kalman Filter (EKF):
* `python accord_demo.py`

To run the demo using the Unscented Kalman Filter (UKF), pass the --filter-type argument:
* `python accord_demo.py --filter-type ukf`

To run the demo with a Walker-Delta topology instead of a random topology, pass the --walker-delta argument:
* `python accord_demo.py --walker-delta True`

*(Note: You can also run these directly in VSCode by right-clicking the file and selecting `Run Python File in Terminal`, appending any arguments as needed.)*

### Monte Carlo Simulation
To run the Monte Carlo Simulation, execute:
* `python mc_demo.py`

You can customise the simulation and KPI extraction using the following arguments:
* `--filter-type`: Which orbit determination filter to use, either `ckf` or `ekf` or `ukf` (default: `ekf`).
* `--num-runs`: The number of Monte Carlo runs you wish to execute (default: 40).
* `--threshold`: Reputation detection threshold for KPIs (default: 0.5).
* `--fpr-offset`: False Positive Rate offset percent to ignore the initialisation effects of the filter (default: 0.2).
* `--start-step`: The simulation step to start plotting from, useful for viewing steady-state convergence (default: 0).
* `--recalculate`: Recalculate KPIs from previously saved data without regenerating the entire simulation.
* `--disable-logging`: Disable logging for the Monte Carlo runs. This helps improve performance for simulations with large amounts of data, such as high ISL ranges or Walker-Delta topologies. Empty log files will still be created for each run to help you track progress.

*Example:* `python mc_demo.py --num-runs 10 --threshold 0.3 --filter-type ukf`

## Configuration

Before executing any simulations, the environment parameters must be defined via the FilterConfig dataclass in the respective demo script.

The simulation framework locally caches data separated by filter type to save processing time on subsequent executions. If any of the core parameters change between runs (such as the number of satellites), the simulation will automatically detect the configuration mismatch, overwrite the stale data, and collect fresh data from scratch. If the default config remains unchanged, the simulation will seamlessly load the pre-existing saved data.

**Setting the ISL Distance**
The maximum Inter-Satellite Link (ISL) distance is the core constraint that dictates network topology, observation frequency, and DAG formation.
1. Locate the DEFAULT_CONFIG instantiation in the simulation script (accord_demo.py or mc_demo.py).
2. Update the ISL_range_m parameter to reflect your desired communication threshold (e.g., 1000e3 or 2000e3).
3. Ensure your Monte Carlo random seeds are set within this config to guarantee deterministic and reproducible runs.

## Contributing

If you wish to contribute to the ACCORD framework, please note that external contributors must fork the repository to submit changes, rather than requesting direct branch access. Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

**Note on Git LFS:** When forking, ensure you have [Git LFS](https://git-lfs.com/) installed to handle the large `.npz` data files produced by the simulations, or simply add `*.npz` to your `.gitignore` to prevent committing local data. See CONTRIBUTING.md for full developer guidelines and repository layout details.

## Current Status & Roadmap

The project is currently at TRL 0. The PoISE consensus mechanism is in the early phases of development. Once development of PoISE is completed, the rest of the ACCORD framework will be developed around this consensus mechanism. By incorporating smart contracts in future development, the system will enable autonomous decision-making, allowing satellites to self-govern, coordinate tasks, and reroute services in real time.

## Related Publications

* [B. Probert, R. A. Clark, E. Blasch, and M. Macdonald, “Cooperative Orbit Determination for Trusted, Autonomous, and Decentralised Satellite Operations,” in AIAA SCITECH 2026 Forum, Jan. 2026. doi: 10.2514/6.2026-0825](https://arc.aiaa.org/doi/10.2514/6.2026-0825)
* [B. Probert, R. A. Clark, E. Blasch, and M. Macdonald, “A Review of Distributed Ledger Technologies for Satellite Operations,” IEEE Access, vol. 13, pp. 123230–123258, 2025, doi: 10.1109/ACCESS.2025.3588688](https://ieeexplore.ieee.org/document/11079570)

## Citation
If you use this work, please cite it as:
> B. Probert, bprobert97/accord: v4.2. (Aug. 03, 2026). Python. Zenodo. doi: [10.5281/zenodo.21774210](https://zenodo.org/records/21774210).

